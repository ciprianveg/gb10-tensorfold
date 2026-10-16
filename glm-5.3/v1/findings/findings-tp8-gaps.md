# TP8 GAP ENUMERATION — drowzeys r2 Zig glm53 engine, world 4 → 8 (explorer, read-only code analysis)

## PROGRESS LOG
- [x] MISSION-LOG.md read (state: TP4 smoke OOM-demoted; TP8 = main line; r2 built at /home/ciprian/eugr/TensorFold-glm53-r2, tip 00c3bb11; sidecars DONE; mesh copies running)
- [x] Scope 1: world/rank plumbing (cuda_engine.zig, rendezvous, nccl, native/glm53_cuda.zig) — DONE
- [x] Scope 2: the wire (cuda_roce.zig, forward exchanges, indexer replicate) — DONE
- [x] Scope 3: splits (split.zig, headq.zig, MoE distribution, DCP cuda_multi.zig) — DONE
- [x] Scope 4: graphs + AOT (cuda_coverage.zig, aot.zig, capture_aot.py/oracle.py, the Python-tree requirement) — DONE
- [x] Scope 5: MTP/DSpark drafters + sidecars world assumptions — DONE
- [x] Scope 6: serving glue world limits — DONE
- [x] Scope 7: memory at world 8 — DONE (budget table below)
- [x] Exactness-contract flags — DONE (§8)
- [x] MISSION-LOG.md journaled

All paths relative to /home/ciprian/eugr/TensorFold-glm53-r2/ unless absolute. Read-only analysis; no file in the engine tree was modified.

---

## 1. WORLD/RANK PLUMBING — verdict: rank-generic; ONE default constant, env override exists

(A) already rank-generic:
- `zig/src/cuda/rendezvous.zig:13` — `pub const max_world = 16;` and `:97` `if (world > max_world or rank >= world) return error.TooManyRanks;` — world 8 passes with margin.
- `zig/src/cuda/nccl.zig:108-111` — `Communicator.init` is `ncclCommInitRank(&c, @intCast(world), id, @intCast(rank))` — no world constant anywhere in the wrapper; all_gather/reduce_scatter shapes parameterized (L119-136).
- `zig/src/families/glm_moe_dsa/cuda_engine.zig:167` — `o.world = try envInt(usize, "TF_GLM53_WORLD", world_default);` — the boot env override EXISTS (the TP8 launch needs only `TF_GLM53_WORLD=8`).
- `cuda_engine.zig:372` — `if (o.rank >= o.world) return error.BadRank;`; `:386` rendezvous.open(rank, world); `:391` NCCL init(rank, world) — all parameterized.
- `cuda_engine.zig:404,414` — the startup exactness checks take world: `c.check(o.world, b.cfg.hidden, b.cfg.layers)` / `c.check(o.world, ..., b.cfg.vocab, ...)` — re-validated at 8 automatically.
- `cuda_engine.zig:815-818, 1018-1020` — allGather buffers sized `cuda.rendezvous.max_world * 8`, sliced `boot.o.world * 8` — generic.
- Reuse-state files are world-keyed: `cuda_engine.zig:871` `Head = { magic, version, rank, world, dcp, n, key, identity, views, carry, bytes }` and `:949` refuses a state whose `head.world != b.o.world` — stale world-4 states are AUTO-REFUSED at world 8 (correct behavior; fresh states needed).
- `native/glm53_cuda.zig` — the serve/follow split is world-generic in code: `:459`/`:670` `if (h.boot.o.world == 1) return;` (the only rank-count branch), `:235` watchdog "rank {d} has gone", `:1265` startup line prints opts.world. **The header's "at TP4 / ranks 1..3" (L1-2,14,19) and `cuda_weights.zig:13`'s "rank * V/4" are STALE DOC PHRASING, not code** (the code at cuda_weights.zig:759-766 divides by `world`).
- `cuda_engine.zig:246-249` — DCP: `d = if (o.dcp != 0) o.dcp else if (o.context > dcp_auto) o.world else 1; if (d != 1 and d != o.world) return error.BadDcp;` — DCP degrees are {1, world} only; at world 8 DCP=8 engages past `dcp_auto` tokens.

(B) needs work:
- `cuda_engine.zig:33` — `pub const world_default: usize = 4;` — (B-trivial, hours): nothing to change for a boot (env override), but every doc/test defaults to 4; optionally flip after TP8 validates. NOT a blocker.

---

## 2. THE WIRE — verdict: runtime generic to 16; the ENVELOPE and the per-collective traffic change character at 8

(A) rank-generic:
- `zig/src/families/glm_moe_dsa/cuda_roce.zig:135` — `if (world < 2 or world > 16) return error.RoceUnavailable;` — **world 8 explicitly inside the supported band**.
- `cuda_roce.zig:45` `pub const max_hcas = 2;` + `:141` `n_hca = @min(set.hcas.len, max_hcas)` — two RoCE rails maximum BY DESIGN (TF_GLM53_ROCE_HCA names); world 8 does NOT need more rails — the one-shot's layout scales with world (`roce_layout(world, slot_bytes)` L75-77) and the rails stripe per-HCA flags (`flag[src][slot][hca]`, header L6). More legs, not more rails.
- `cuda_roce.zig:8-10` — the sum is FIXED RANK ORDER in fp32 (rank 0's value, then + rank 1, ...), verified against NCCL at startup ("windows of 3 and 32 rows of 6144, every rank's sum bit-equal... else every rank refuses") — the verify re-runs at world 8 and pins the 8-addend order.
- `cuda_roce.zig:14, 171-197` — setup over the TP world's TCP rendezvous; per-rank Config exchange (`.world` field L88) with mismatch refusal (L188); start_everywhere (all ranks get RoCE or none) — generic.
- `cuda_forward.zig:47,55,1295` — `max_world = 16`, `gathered: [max_world * 4]f32`, `vals: [max_world]f32` — the head-record exchange holds 8 ranks.
- `cuda_forward.zig:4-6, 760-762` — `fused.gather`: world 1 identity; RoCE one-shot for decode windows (R ≤ b.small) into `b.red`; otherwise NCCL `comm.all_gather(b.part[:R], b.gath[:world R D])` — the branch selection is rows-threshold-based, not world-based.
- **The indexer "replicate-don't-merge" is CONFIRMED STRUCTURAL and world-independent**: `split.zig known()` marks `indexer.wq_b` (EXL3) `.rep` and `indexer.wk.weight / weights_proj.weight / k_norm.{weight,bias}` in the rep list — every rank holds the FULL indexer projections at any world; per-rank top-k then gathered candidates (cuda_forward.zig:30 header; the DCP path "the indexer's per-rank top-k, gathered candidates and one [merge]" L29-30). Nothing to change at world 8.
- Exchange count: 2 `f.gather` call sites per layer (`cuda_forward.zig:746,752` g1/g2) × 78 layers = **156/window** + the head records — matches the mission's ~156; the count is world-independent (each is one collective at any world).

(B) needs work / changes character:
- **(B-1, hours-config + measurement) The RoCE slot envelope at world-8 batched windows.** `cuda_roce.zig:46-47`: `max_bytes = 1 << 20` (1 MiB) with the comment "decode windows up to 32 rows x 6144 fp32 = 768 KB; the head's records and sampled candidates fit". At world 8 the c8 window is 8 streams × (k+1) rows: 32 rows fp32 = 768 KB fits; **9-draft windows (72 rows) = 1.72 MB EXCEED the envelope → those windows take the NCCL ring branch** (`cuda_forward.zig:23` "wider windows over NCCL"). Options, all config: (a) accept the split (narrow windows RoCE, wide NCCL — the mimo M25/M27 precedent says which transport carries which shape is a measured property), (b) `TF_GLM53_ROCE_BF16` (Config.wire bit 0, `cuda_roce.zig:125` "changes the bits (each partial rounded to bf16 before the fp32 rank-order sum)") halves bytes → 85 rows fit, (c) raise max_bytes (kernel/ABI constant — riskier, touches the fatbin). DECISION ITEM for the build order: measure the NCCL-vs-RoCE crossover at world 8 on the cluster.
- **(B-2, measurement, no code) Per-collective traffic doubles-ish at world 8**: the one-shot's per-rank RDMA writes = (world−1) × slot_bytes per collective (each rank writes its full partial into every peer's recv slot) — 7 peers vs 3 at world 4 = 2.33× per-rank wire bytes per collective, while the ARITHMETIC per rank halves (hidden 6144 partial per rank is rows×6144×4 B regardless of world — it does NOT halve; what halves is the COMPUTE per rank, not the partial size). Net: wire pressure per rank RISES ~2.3× on the RoCE path at the same rows. The two rails stripe it; whether the c8 aggregate regresses vs TP4 is a MEASURED question (the cyclesim TP8 band already brackets this: c8 ~94-126 occupancy-prior vs 150-250 weight-bound).

---

## 3. THE SPLITS — verdict: fully rank-generic; MoE is TP-split (no EP in this family); DCP generic

(A) rank-generic:
- `zig/src/families/glm_moe_dsa/split.zig` (170 lines, read in full): `rule(name)` → {rep,row,dim1} by tensor name; `checkCut(name, kind, shape, world)` — `shape[axis] % world == 0` + the whole-Hadamard guard `((shape[axis]/world) % (128/16) == 0)` for trellis and `(shape[0]/world) % 128 == 0` for suh/svh; `cutBytes(..., rank, world)` generic. **World is a parameter everywhere; no 4 anywhere.** The in-file test even pins a kv_b cut at world 4 ("a rank's 16 heads are 56 whole blocks", L~168) — at world 8: 1792/8 = 224 columns = 28 whole blocks ✓; svh 28672/8 = 3584 % 128 = 0 ✓.
- Replicated (world-independent) by rule: q_a_proj, kv_a_proj_with_mqa, indexer wq_b (+ the plain indexer sidecars), eh_proj/enorm/hnorm, embed_tokens, model.norm, layernorms, gates, e_score_correction_bias (split.zig known() rep list). Row-split: down/o_proj/kv_b (inputs), lm_head.weight.
- `headq.zig:49-50` — `Cut.of`: `if (world == 0 or rank >= world or vocab % world != 0 or vocab % had != 0) return error.UnevenVocab;` — **at world 8: 154880 % 8 = 0 ✓, 154880 % 128 = 0 ✓**; part = 19360 = 151.25 blocks → the SHARED-BLOCK mechanism (documented at L3-8 for world 4: "a rank holds the blocks that cover its share — 303 of them, one shared with a neighbor... a token's logit is the same bits whichever rank computes it") applies identically: 152 blocks, one shared. **There is NO 64-block BadSplit floor on the vocab cut** — the only `BadSplit` in headq.zig is `:245` `if ((K / 16) % (sk * wk) != 0)` — the K-dimension warp split of the tile step, world-independent (K is the head's INPUT dim 6144, never cut by world).
- **MoE distribution: TP-SPLIT of every expert, not EP.** split.zig cuts gate/up `.dim1` (output tiles+svh) and down `.row` (input rows+suh) for EVERY expert — each rank holds a slice of all 256 experts; `cuda_weights.zig:522` `const I = c.expert_width / L.o.world;` (2048/8 = 256 intermediate per rank per expert). The family has NO EP path (`grep ep_size|expert_parallel` in families/glm_moe_dsa: zero hits; `ep.zig` belongs to the Flash family `families/glm/`). 256 experts × 8 ranks works unchanged — the grouped-kernel plan is per-rank shapes.
- **DCP (`cuda_multi.zig`)**: `mu.world = w.world` (`:668`), `Layout.make(rows, n, k, w.world)` (`:686`), `lse_all` world-sized (`:520,550`), `mu.broken = mu.world > 1` guards (`:1211,1278`), the decode ranks count `if (comm != null or fast != null) mu.world else 1` (`:1570,1631`) — all parameterized. DCP=world at 8 shards the KV keys 8 ways (memory ÷8) with the log-sum-exp merge at the heads' owners (cuda_forward.zig:29-30) — generic math, UNTESTED at 8 (no world-8 receipt exists anywhere — mission log).

(B) needs work:
- **(B-3, hours) `cuda_multi.zig:1872` — a TEST fixture hardcodes `draft.Spec{ ..., .V = 38720, ..., .world = 4 }`** (the Phase-4c test's df spec; V=38720=154880/4). Production Specs come from `ofDSpark`/`ofDFlash` which take world+vocab_part (`cuda_draft.zig:71-76`) — so this is test-only, but the test suite should gain a world-8 case (V=19360) when TP8 lands, or it silently keeps validating only the world-4 arithmetic.

---

## 4. GRAPHS + AOT — verdict: the BIGGEST world-8 work item; shapes are world-dependent; the capture tool already takes --world

(A) rank-generic:
- `zig/src/cuda/aot.zig:16-28` — the AOT pack lookup is by sm-capability folder + aot.json; variants match by (function, constexprs, specialization class, launch options) — **the pack is keyed by SHAPE, not by world directly** (fix-2's bits-blind finding stands: our 4.04bpw pack rides the published packs at the same shapes).
- Graph capture (`cuda/graph.zig`, engine `:275` boot-shape key) — the shape tuple includes world+dcp (`o.world, o.context, o.k, ...` L275) so graphs re-key per world automatically; the captured graph REPLAY includes the RoCE epoch in device memory (`cuda_roce.zig` header: "the sequence number lives in device memory (the epoch), so CUDA graphs replay them") — world-independent mechanism.

(B) needs work:
- **(B-4, THE critical path — days incl. verification) A fresh world-8 AOT capture is REQUIRED.** The per-rank SHAPES halve at world 8 and every Triton variant is shape/constexpr-keyed:
  - `zig/src/families/glm_moe_dsa/cuda_coverage.zig:8-19` (the enumerate contract, verbatim): drives "window rows R = 1..max_rows... both residual sums (**world 1: RoCE / ring / one rank; world W: the NCCL all-gather**)"; "the prompt GEMM (x3prefill._gemm) for **every EXL3 linear shape of every rank**"; "DCP 1 and DCP = world **with every rank's RANK constexpr**"; Phase 4b rows-windows and 4c concurrent drafters. `enumerate(p, sp: Space)` with `Space.world` (`:44`), `Space.dcps` (`:56`), `Space.streams` (`:68`); the DCP rank loop `for (0..ranksOf(dcp)) |rank|` covers RANK 0..7 (`:78,110-113`).
  - The capture tool ALREADY supports it: `tools/glm53/oracle.py:268-269` — `--world` ("the TP world whose cut is loaded (serving: 4)") and `--rank`; `--record` packs aot.json+cubins via `tools/zig/triton_aot_manifest.py` (oracle.py:285-287,301-305), and the EXL3 CUDA kernels are recorded through the wrapped ext entry points (`rec.wrap(x3linear._ext(), ...)`, `x3experts`).
  - **BUT the capture runs ONE rank** (`load_engine(model, a.layers, a.rank, a.world)` with an identity `LocalComm`, oracle.py:104-118,173-190) — **the DCP RANK 1..7 constexpr variants are NOT recorded by a rank-0-only run.** Two honest options: (a) run the capture once per rank (`--world 8 --rank R` for R=0..7 — 8 GPU capture runs, each loading a rank's cut) and merge packs, or (b) for the FIRST TP8 smoke keep DCP=1 (short context: DCP engages only past `dcp_auto`, cuda_engine.zig:248) so no DCP variants are needed, then decide. Verify at implementation time whether aot.pickVariant treats a missing-RANK launch as refused (aot.zig: "a launch that leaves them open while captured variants differ in them is refused").
  - **The Python-tree requirement (verified): the r2 worktree has NO Python src/** (ls: assets, build.zig, docs, glm53-sidecar, packaging, tools, zig, zig-out — no src/tensorfold). `oracle.py:177-198` imports `tensorfold.families.glm_moe_dsa.{config,cuda.weights,cuda.fused}` + `tensorfold.cuda.exl3.{linear,experts}` — **the Python glm_moe_dsa family must be materialized from the fetched `remotes/drowzeys/glm53-tp4-2026-10-07` branch** (it has src/tensorfold/families/glm_moe_dsa/ with cuda/ — verified in round 3) as a second worktree, with `--tools` pointing at a tree holding `tools/zig/triton_aot_manifest.py` (oracle.py:273-274 help text). Capture needs: one free GB10, the pack (local on .111 ✓), TRITON_CACHE_DIR fresh, torch+triton in the sglang:kimi-k3-p6 image (the mimo pattern).
  - What MUST be re-captured for world 8 (the checklist): every EXL3 linear/GEMM variant at the world-8 per-rank slices (gate/up N/world, down K/world, q_b/kv_b output slices, o_proj input slices); residualAdd with world=8 slots; router/absorb/expand/rmsnorm at the same row classes (row classes are world-independent — R 1..128); the indexer select buckets (T classes incl. ceil(T/8) DCP ends if DCP armed); head/records (world-sized buffers — the [world,4] records shape class); MTP + drafter rows at the world-8 vocab share (V=19360); the 4b/4c concurrent shapes if --parallel is served.

---

## 5. MTP/DSPARK DRAFTERS + SIDECARS — verdict: rank-generic; the sidecars split cleanly at 8

(A) rank-generic:
- `cuda_draft.zig:71-76` — `ofDSpark(c, s, world, vocab_part, vocab_off)` / `ofDFlash(c, world, vocab_part, vocab_off, rsel)`: H = heads/world, KV = kv_heads/world, I = inter/world, V = vocab_part — **the drafter itself is TP-split by the same world**; buffers `sp.world`-sized (`:288,304,324,328`), the cand exchange `[world, sendWords]` (`:159,504,569`).
- `cuda_weights.zig:759-766` — the reduced draft-vocab head: `q = draft_vocab / world`, `if (L.o.draft_vocab % world != 0 ...) return error.BadDraftVocab`, last rank also carries the final 128 SPECIALS rows — rank-count-independent edge, generic. (The `V/4` at `:13` is stale doc; the code divides by world.)
- The bf16 lm_head sidecar (the drafting unblock, mission log): loaded via `L.part("lm_head.weight")` with `shape[0] != w.vocab_part` check (`cuda_weights.zig:733-741`) — a plain world-cut of 154880 rows → 19360 rows/rank at 8. The sidecar file itself is the FULL tensor (each rank cuts it) — **no sidecar rebuild needed for world 8**.
- eh_proj/enorm/hnorm are `.rep` (split.zig) — replicated at any world; the eh_proj sidecar (151 MB) lands on every rank unchanged.
- `cuda_mdraft.zig` (the concurrent drafters): `mdraft.enumerate(p, ds, sp.streams)` — streams-parameterized; world enters via the specs. `drafterRows(9, ...)` → DrafterTooWide (`cuda_multi.zig:1880`) bounds drafter STREAMS at 8 — `windowRowsWith(8, 2, 4)` passes in the tests (world 8 exercised in the test arithmetic).

(B) needs work:
- (B-5, hours) Add world-8 drafter rows to the AOT capture (the draft heads' per-rank row ranges change: V/world) — subsumed by B-4; and add the world-8 test fixture (B-3).

---

## 6. SERVING GLUE — verdict: world-generic; the snapshot/identity machinery already worlds

(A) rank-generic:
- `native/glm53_cuda.zig:1079` — the snapshot shape key includes `o.world, b.dcp, o.k, o.draft_vocab, ...` — learned states/learned-pairs re-key per world (stale world-4 states refused, cuda_engine.zig:949).
- The follow protocol, watchdog, stop votes, --parallel messages: all rank-count loops over `o.world` (L457-459, 644, 667-670, 1187+); the world-1 guards are the only special cases.
- The startup probe's prompt-pass bits are part of the identity (`native/glm53_cuda.zig:10`) — re-validated per world at boot.

(B) needs work:
- (B-6, hours, docs-only) The stale TP4 doc phrasings (native/glm53_cuda.zig:1-2,14,19 "ranks 1..3"; cuda_weights.zig:13 "V/4"; split.zig:4 "Phase 1 loads rank 0 of world 4"; cuda_forward.zig header examples) — correct them when TP8 lands so the next reader doesn't inherit the assumption.

---

## 7. MEMORY AT WORLD 8 — the budget flips from weight-bound to comfortable; the c8 wire envelope becomes the binding shape constraint

Per-rank arithmetic at world 8 (weights from the pack: 360 GB total; head 0.95 GB EXL3-8-bit; dspark-ft2 2.4 GB bf16 drafter; eh_proj sidecar 151 MB):
| item | world 4 (the OOM) | world 8 |
|---|---|---|
| weights/rank | ~90 GB (4.04bpw uniform) | **~45 GB** |
| lm_head share | 238 MB | 119 MB (EXL3) / +bf16 sidecar rows 1.9/8 ≈ 238 MB if head_host |
| drafter (TP-split) | 600 MB | **300 MB** (dspark H/KV/I ÷ 8, cuda_draft.zig:72) |
| eh_proj + replicated sidecars | 151 MB + indexer/norms | same (replicated, world-independent) |
| exchange buffers | gath world×R×D fp32: 4×72×6144×4 ≈ 7 MB | 8×72×6144×4 ≈ 14 MB (rows 72) — trivial |
| KV pool | per rank full (DCP 1) or ÷4 (DCP=world past dcp_auto) | per rank full or ÷8 — **the KV headroom ~doubles vs TP4 at DCP=world** |
| graphs + AOT cubins | unchanged | unchanged (row-class keyed) |
| TOTAL vs 116 GB MemAvail (measured idle, .111-.118) | ~90 + KV + overhead → OOM at the measured shapes | **~46-50 GB + KV + overhead → tens of GB free** |

- The comfortable shapes at world 8: the context limit is now the KV pool + DCP policy, not weights; streams: 8 concurrent streams × (k+1) rows fit the row classes (max_window_rows 128, cuda_coverage.zig:41). The c8 window (72 rows at k=8) exceeds the RoCE fp32 envelope (B-1) — either narrower drafts, the bf16 wire, or NCCL-ring for the wide windows.
- The world-4 shape table (port doc) scales ~2× in the weight-derived dimensions (context per rank at DCP, resident streams); treat all world-4 memory receipts as upper bounds at world 8 EXCEPT the wire envelope (B-1/B-2) and the per-collective RDMA fan (7 peers).

---

## 8. EXACTNESS CONTRACTS AT WORLD 8 — what must hold, what legitimately changes

- **The keyed seed is rank-independent**: `sampling.zig:46-54` seedFor = sha256("id,id,...|salt") first 8 bytes; `:69-71` uniform = splitmix64(seed, position, id) — no world/rank inputs. Holds at 8 unchanged.
- **Every rank computes its own picks** (native/glm53_cuda.zig:21 "every rank computes the picks itself, so no results message is needed") — the deterministic-sampling contract is per-rank local + the gathered records; the head-record tie rule is "the lowest rank of the maxima" (cuda_forward.zig:12) — rank-ORDER-based, generic at 8.
- **Per-mode determinism holds WITHIN world 8** (the RoCE rank-order fp32 sum is fixed-order, verified against NCCL at startup, cuda_roce.zig:8-10; the checks run per-boot at the boot's world, cuda_engine.zig:404,414).
- **Bits will differ from world 4 and from drowzeys' published TP4 receipts**: the fp32 rank-order sum has 8 addends instead of 4, and every GEMM slice changes shape — this is the mimo campaign's characterized non-property class (cross-shape bit-equality), NOT a broken contract; nothing in the r2 tree pins world-4 bits (grep: no test fixes world-4 outputs). The acceptance/envelope distribution may shift (the M22/M28 envelope-lottery lesson) — the Stage-2 gates should re-baseline the per-mode determinism + isolation AT world 8 rather than compare bits against TP4.
- The RoCE health machinery (poison flag + healthyEverywhere all-gather, cuda_roce.zig:11-13,119-125) is world-generic — fail-stop semantics survive at 8.

---

## THE BUILD-ORDER WORK LIST (B items, ranked)

1. **B-4: the world-8 AOT capture** (days, critical path): materialize the Python glm_moe_dsa tree from the fetched glm53-tp4-2026-10-07 branch (second worktree, zero download); run `capture_aot.py --model <pack> --world 8 [--rank R]` on a free GB10 with a fresh TRITON_CACHE_DIR; verify the pack against cuda_coverage.zig's enumerate contract (the source test) BEFORE boot; decide the DCP-RANK question (per-rank captures vs DCP=1 first smoke). Distribution: the pack folder rides zig-out to all 8 nodes (md5).
2. **B-1: the c8 wire-envelope decision** (hours of config + one measurement session): 32-row fp32 windows fit RoCE 1 MiB; 72-row windows don't — pick bf16-wire vs NCCL-ring-for-wide vs raised max_bytes; measure the crossover at world 8 (the M25/M27 lesson: transport-per-shape is measured, not assumed).
3. **B-2: the wire-plan measurement** (rides the same session): per-rank RDMA fan 7 peers × slot_bytes; whether c8 aggregate holds vs the cyclesim TP8 bracket (~94-126 occupancy vs 150-250 weight-bound).
4. **B-3 + B-5: test hygiene** (hours): the world-8 fixture in cuda_multi.zig's Phase-4c test (V=19360), world-8 drafter rows in the capture.
5. **B-6 + B-trivial: docs** (hours): the stale "TP4/ranks 1..3/V/4/303-blocks" phrasings; optionally world_default.
6. **Fleet boot config** (hours): TF_GLM53_WORLD=8 + TF_GLM53_MASTER on all 8 ranks via the tp8_run.sh-class launcher (the glm53-preflight tripwire + the exporter-step guard from findings-fixer1-tooling §C apply verbatim); TF_GLM53_AOT points at the fresh world-8 pack.

## TOP 3 RISKS
1. **AOT variant incompleteness at world 8** — a missing shape/constexpr variant (DCP RANK 1..7, an unvisited row class) surfaces as error.NoKernelPack / a refused launch AT BOOT on the cluster, after the build is otherwise done. Mitigation: the coverage source-test against the new pack before distribution; the DCP=1-first smoke.
2. **The c8 wire character change** — wide windows silently taking NCCL ring (or the bf16 wire changing sum bits) could move the c8 aggregate outside the calibrated band; the transport-per-shape map must be measured at world 8, not inherited from TP4 receipts.
3. **No world-8 receipt exists anywhere** (mission log: "NO TP8 receipt exists anywhere") — every TP8 expectation (cyclesim c1 45.5 [30-68], c8 bracket) is extrapolation; the exactness/envelope gates must be re-baselined at world 8 before any performance claim.

## AOT-CAPTURE REQUIREMENTS FOR WORLD 8 (the one-paragraph answer)
`capture_aot.py` wraps `oracle.py --record`, which takes `--world` and `--rank` (oracle.py:268-269) — **run it with `--world 8`** against the local pack on one free GB10, with the Python `tensorfold.families.glm_moe_dsa` tree materialized from the fetched `drowzeys/glm53-tp4-2026-10-07` branch (the r2 worktree has no Python src — verified), `TRITON_CACHE_DIR` fresh, torch/triton from the sglang:kimi-k3-p6 image. The recorded set must cover: every EXL3 linear + prompt-GEMM shape at the world-8 per-rank slices, residualAdd world-8 slots, the head/MTP/drafter rows at V=19360, and — if DCP will be armed — the DCP=8 RANK 0..7 variants, which a single rank-0 run does NOT record (oracle is one rank with an identity comm): either 8 per-rank capture runs merged, or keep DCP=1 for the first smoke (DCP engages only past dcp_auto). Validate the pack with cuda_coverage.zig's enumerate contract before fleet distribution.
