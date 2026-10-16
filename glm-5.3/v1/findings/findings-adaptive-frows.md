# findings-adaptive-frows — the concurrent draft window allocated by ACTIVE streams (r2 Zig engine)

PROGRESS LOG (append-only). Lane: fixer, dispatched 2026-10-16. Target: on a parallel-8 DSpark serve a
single active stream got frows = min(9, 32/8) = 4 → 3 drafts (c1 code 53.1 tok/s, t/r 2.72 vs the p1
serve's 55.9 / 3.11 — the 32-row window budget wasted when few streams are active). Fix: the per-round
draft cap by the round's LIVE stream count. All gates run live on .111-.118.

## §0 PROGRESS LOG
- [x] §1 DESIGN (what N means at every site, the budget proof) — below
- [x] §2 THE PATCH (depthOf by live; the cut_streams AUDIT; tests) — below
- [x] §3 BUILD + DISTRIBUTION (rc=0 all 3 stages; 102/102; md5s; all 8 nodes verified)
- [x] §4 AOT VERDICT — NO extension needed (the evidence, both design questions answered)
- [x] §5 LIVE A/B (two boots: CUT_STREAMS=8 and the sane CUT_STREAMS=4; the table; the c4 finding)
- [x] §6 THE SMARTS INVARIANT — fixed-prompt greedy shas MATCH (both prompts, both boots)
- [x] §7 RESTORE PROCEDURES

## §1 DESIGN — what N means at each site, before → after

`mdraft.frows(sp, N)` (cuda_mdraft.zig:44-46) is UNCHANGED as a function: rows a stream's verify window
takes when N streams share the 32-row decode window = min(maxDepth+1, 32/N). What changed is WHICH N
each call site passes:

| Site | N before | N after |
|---|---|---|
| `mdraft.fits(sp, N)` ← MultiDrafter.init / coverage | configured parallel | UNCHANGED (configured — pool/pass feasibility) |
| `drafterRows(n, specs)` ← Multi.init :646 → `mu.frows` | configured | UNCHANGED (configured — the widest window share; receipt) |
| `windowRowsWith(n, k, fr)` ← Multi.init :647 → `mu.rows` | configured | UNCHANGED (configured — buffer/staging sizing, prewarm widths 1..rows) |
| `Multi.depthOf(which)` :807-815 → the per-round room cap | **CONFIGURED `mu.N`** — THE BUG (c1 on p8: 3 drafts) | **the round's LIVE count** (`n_live`, passed from step :1243) → frows(live)−1 |
| `Multi.step` cut/dcut threshold :1229/:1231 | `n_live >= cut_streams` (ACTIVE) | UNCHANGED — see the §2 audit |
| `Multi.prewarm` cap :1462 | `mu.frows` (configured) | UNCHANGED (capture position headroom only) |

Adaptive cap at live streams 1..8 (DSpark block 8, maxDepth 8): drafts 8, 8, 8, 7, 5, 4, 3, 3
(frows(live)−1; frows = min(9, 32/live)). 1 active → full depth-8 chains; 8 active → today's 3.

**The budget proof (constraint 1: n_active × max(k+1, frows) ≤ 32, REALLOCATE never widen):** every live
item takes at most max(k+1, frows(live)) window rows; live × max(k+1, frows(live)) ≤ 32 given the boot's
own `windowRowsWith` refusal pinned N × max(k+1, frows(N)) ≤ 32 and frows is monotone non-increasing in
its argument: if k+1 ≥ frows(live) the boot forced N(k+1) ≤ 32 so live(k+1) ≤ 32; else
live × frows(live) ≤ live × ⌊32/live⌋ ≤ 32. ∿ `tables()`' WindowTooWide backstop (:277) is now
unreachable by construction. Encoded as a host test ("adaptive frows: the per-round draft cap follows the
round's LIVE streams and the window budget always holds", cuda_multi.zig tests).

The graph-set selection needed NO change: `Multi.prewarm` :1493-1498 already captures EVERY verify width
1..mu.rows (=32 on this boot) × key bucket × both picks — the deep 9-row c1 shape is inside the set the
engine plans (both patched boots captured the same 400 graphs as the unpatched p8 boot).

## §2 THE PATCH (backups *.bak alongside; diff: zig-adaptive-frows.diff, 144 lines)

- `cuda_multi.zig` — THE FIX: `depthOf(mu, which, n_live)` caps by `mdraft.frows(b.sp, max(n_live,1)) − 1`;
  `step` passes `n_live` (the lead computes the plan; followers apply its `room` — rank-consistent by
  construction). Doc comments updated (file header, Settings.cut_streams, depthOf). New host test (the
  cap curve + the budget invariant at k ∈ {0,2}).
- `cuda_mdraft.zig` — `frows` doc: the two meanings of N (configured for sizing, live for the per-round
  cap). Test extended: frows(ds,1)=9, frows(ds,3)=9, frows(ds,8)=4.
- `cuda_engine.zig` — boot receipts now honest: "draft cut {d} from {d} **active** streams" and
  "up to {d} drafts a stream at {d} streams **(adaptive: by the round's active streams)**".
- **The cut_streams AUDIT (task claim vs code):** the threshold comparison `n_live >= mu.set.cut_streams`
  (cuda_multi.zig:1229 plan.cut, :1231 plan.dcut) has counted the round's LIVE streams since the ORIGINAL
  TP4 port (git log -S: f156915f; unchanged through 7843a145 and the deployed §12 binary 5f19d512 —
  verified live: the CUT4 boot's c1 keeps floor 0.3 while c4 gets 0.6). The measured "stock floor bound
  c1 on a p8 boot" was the DEPTH cap (depthOf's configured mu.N → 3 drafts), not the threshold; the
  CUT_STREAMS=8 in the boot script was a workaround for the wrong suspect. The sane semantics
  ("from 4 ACTIVE streams") are the stock behavior — CUT_STREAMS=4 is safe and (see §5) RECOMMENDED.
  No code change needed; docs clarified.

## §3 BUILD + DISTRIBUTION
- build-r2.sh (3-stage, RAM-guarded 9 GiB, -j4, zig 0.17.0, sm121): **rc=0 ALL 3 STAGES**
  (fatbins 27 s; native+lib+generate 78 s; host tests 11 s). test-glm53 **102/102** (101 + the new
  adaptive-frows test), test-cuda-host rc=0. (The "344/346" failure summary in build-r2.log predates
  this lane — an earlier era's attempt; today's stages append nothing on success.)
- Artifacts (md5, distributed to ALL 8 nodes .111-.118, rsync zig-out, every node verified identical):
  - tf-glm53-generate `a187975be888de9fef8dede415d67897`
  - libtensorfold_glm53.so `0d5bb053b329933b1696907302f7766f`
  - tensorfold-native `e105562304b2db37a797d18b8294d9c2`
  - glm53_kv8.fatbin `0b0e3e901ba0f73c65a87515b95b7c35` UNCHANGED (§12's — no .cu change); kv8repro
    `c331f3aafa85ab26472a89cf3e22f60f` unchanged.
- NOTE: the nodes' ~/TensorFold-glm53-r2 SOURCE trees are stale (pre-kv8, .111's cuda_multi.zig differs
  from the worktree) — only zig-out was ever synced after the kv8 era. Running is artifact-only (the
  container mounts the tree for zig-out); left as-is, recorded here.

## §4 AOT VERDICT — **NO EXTENSION NEEDED** (both constraint-2 questions answered)
1. Does the captured graph set cover the low-active deep shapes? YES: multi.prewarm captures every verify
   width 1..mu.rows(=32) in every key bucket with both picks — the p8 boot's 400 graphs (both patched
   boots: "400 concurrent decode graphs captured in 52.1 s", identical to unpatched). A c1 round's 9-row
   window replays a boot-captured graph; positions come from device tables, so one graph serves every
   position mix. The drafter's own graphs don't depend on depth at all (the dspark-capture ground truth:
   the block pass is always whole; depth only trims the host chain — its only device effect is the verify
   width). Non-contiguous DSpark slot masks capture on first use by design (runGraph: miss → eager →
   captured).
2. Does the boot-time capture enumerate the adaptive shape set? YES — the engine captures what it plans,
   and prewarm's width loop is 1..mu.rows REGARDLESS of adaptivity (the adaptive widths are a subset).
   The AOT PACK (~/aot-world8-32768-dcp1-dspark) also already covers the Triton side: the dspark capture
   extension ran every runner width dspark-verify{16, 1..9} (the 9-row deep window IS one of them), and
   widths 10..32 share the same variant classes (R is a grid dim, not a variant key — proven live by the
   working 32-row c8 windows on the same pack). Empirical gate: both patched boots loaded and served with
   ZERO NoKernelPack/missing-variant errors; c1 (9-row) and c8 (32-row) both at full speed. No
   run-aot-dspark-ext.sh rerun, no restage.

## §5 LIVE A/B (greedy 1024, thinking on, the CSV-parser + beekeeper prompts; cluster .111-.118,
kv8 32K×8, k=0, DSpark-original conf 0.3; unpatched = the §12 binary 5f19d512, measured 2026-10-16
morning on the same serve: c1 code 53.1 t/r 2.72 acc 0.742, prose t/r 1.96)

| gate | target | unpatched p8 | patched CUT_STREAMS=8 | patched CUT_STREAMS=4 (RECOMMENDED) |
|---|---|---|---|---|
| c1 code tok/s (×2) | ~55-56 (t/r ~3.1) | 53.1 (t/r 2.72) | **54.3 / 54.8 (t/r 3.11, acc 0.707)** | **54.5 (t/r 3.11)** |
| c1 prose tok/s (×2) | ~43.9 | 43.9 (t/r 1.96) | **43.1 / 43.5 (t/r 1.97)** | **43.5 (t/r 1.97)** |
| c2 code agg | ~62 | 62.0 | 61.4 | 61.3 |
| c4 code agg | ~96 | 96.7 | **93.2-93.4 ← the regression** | **96.7 / 97.0** |
| c8 code agg | 113-130 | 130.4 | 130.4 | 130.6 |
| coherence | no loops, acc bands | — | PASS (prose 0.626, code 0.707, no loops) | identical at c1 (shas match) |

- **c1 CLOSED TO THE P1 PROFILE:** t/r 3.11 and acc 0.707 are EXACTLY the p1 serve's drafter-matrix
  values — the window slice is no longer the limiter. The residual ~2% tok/s vs p1's 55.9 is the
  multi-decoder's per-round host path (plan/message/applyRound at p8 config), not drafting.
- **THE REGRESSION FOUND (and resolved):** with the boot's CUT_STREAMS=8, c4 dipped to 93.3 (−3.4%,
  consistent ×3): at 4 active the adaptive cap is 7 drafts → verify windows double (16→32 rows; measured
  t/r 3.10) and on bandwidth-bound GB10 the extra rows cost more than the depth pays (the k3-matrix
  lesson). NOT an engine bug — the "nocut" boot config disabled the designed remedy: with the SANE
  stock CUT_STREAMS=4 (active-based, §2), the drafter confidence floor rises 0.3→0.6 from 4 ACTIVE
  streams, trims the mid-concurrency chains, and c4 recovers to 96.7-97.0 while c1/c2 keep the floor 0.3
  (n_live < 4) and their wins. c8 identical under both (floor 0.6 at 8 active either way).
- Prose t/r barely moves (1.96→1.97): the 0.3 confidence cut already trims prose chains early — depth
  pays on code, as the drafter matrix predicted.
- Code acc 0.742→0.707 is the denominator effect of deeper chains (more low-confidence drafts drafted),
  NOT a quality change — 0.707 is the p1 full-depth value from the matrix; the healthy band.

## §6 THE SMARTS INVARIANT — PASSED
Fixed-prompt greedy sha256, patched vs the LIVE unpatched serve (captured before teardown, ×2 each,
deterministic): prose `75dd870c5ac8ba72` MATCH, code `2e7afc473225097f` MATCH — on BOTH patched boots
(CUT8 ×2 each, CUT4 ×1 each). Drafting is output-preserving under the deeper windows: byte-identical
greedy output at c1 with depth 8 vs depth 3.

## §7 RESTORE PROCEDURES
- **The serve (one boot):** `ssh .111 'setsid nohup ~/drop_caches_loop.sh & ' && ~/dspark-boot-p8-nocut.sh
  GLM-5.3-dspark` (the CUT_STREAMS=8 "nocut" config; kill the loop once "serving" appears, ~400 s).
  The recommended config is the new `~/dspark-boot-p8-cut4.sh` (identical but CUT_STREAMS=4) — it is
  what is LIVE now. The drop_caches loop during load is REQUIRED on .111 (rank 0 = the NFS server; its
  page cache refills during the load and the cache-fit check 2-4 s after load-complete fails otherwise).
- **The engine (full revert):** in the worktree `cp cuda_mdraft.zig.bak cuda_mdraft.zig &&
  cp cuda_multi.zig.bak cuda_multi.zig && cp cuda_engine.zig.bak cuda_engine.zig` (backups alongside the
  patched files), `bash investigations/glm53-exl3-tp8/build-r2.sh`, then rsync zig-out to .111-.118 and
  md5-verify (the §3 pattern). The unpatched §12 md5s to expect: generate ac435384…, lib bc9021f7…,
  native 5f19d512….
