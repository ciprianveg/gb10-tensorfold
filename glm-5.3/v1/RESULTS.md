# RESULTS — GLM-5.3-4.04bpw-h8-exl3 world-8 serve (accumulating)

Engine: drowzeys r2 Zig 1.0.7 (patched: dtype-round-cast for our pack's 6 F16 classes) on
8× GB10 (.111-.118), single-rail RoCE (rocep1s0f1; rail 1 down cluster-wide — follow-up).
Pack: MikeRoz 4.04bpw mul1 h8 + bf16 sidecars (drafting). Serving: 32K ctx, MTP k=2 drafts on,
prompt reuse on, DCP=1.

## Boot & gates (2026-10-15)
- Boot 3: 78/78 layers in 311-315 s; RoCE one-shot verified bit-equal (single rail); 65 decode
  graphs; serving 127.0.0.1:8080 on .111. Parallel-8 boot: 8 streams accepted (windows ≤24 rows,
  2.95 GiB cache each; draft cut 0.6), RoCE bit-equal at 3- and 32-row windows.
- Exactness: keyed contract PASSES — seed 777 twice → identical token_sha (9f807893a817),
  778 → moves (0b9d7757dd2e). Greedy = exact mode.
- Memory: ~24 GB free/node at 32K×8 (floor 2 GB). Cache cost measured: 2.95 GiB / 32K tokens
  = 92-97 KB/token (bf16 MLA latent + DSA indexer state; no fp8 cache in this engine — see below).

## First generations
- Greedy 60-tok: 50.7 tok/s, TTFT 0.18 s, acceptance 60.4%, 2.22 tok/round.
- Sampled 50-tok: 53.5-53.9 tok/s.

## Standard bench round 1 (drowzeys bench/, 512 toks, seed 1729)
- tfbench single-stream NO-CONTEXT (sampled): prose 22.32, code 21.84 (engine-stats measure —
  2× lower than conc_chat's wall-clock at C1; artifact to reconcile, conc_chat is client-truth)
- tfbench 32K-context: INVALID (orchestrator error: 653 KB filler ≈ 160K tokens > 32K window —
  rerun pending with a ~140 KB filler)
- Concurrent ladder, sampled (aggregate tok/s):
  prose C1 44.1 / C2 59.4 / C4 81.0 / C8 97.3
  code  C1 45.4 / C2 62.1 / C4 85.3 / C8 98.4
- Concurrent ladder, greedy:
  prose C1 44.2 / C2 59.5 / C4 87.9 / C8 125.1
  code  C1 46.4 / C2 64.7 / C4 110.0 / C8 148.5

## vs drowzeys' TP4 published receipts (2.75bpw, DFlash2/DSpark drafters, 4× GB10)
- Their ladder (sampled): chat C1 38.3 / C2 54.3 / C4 65.6; code C1 41.6 / C2 49.9 / C4 62.6.
  OURS at every point higher (C1 +15%, C2 +9-24%, C4 +24-37%), and we extend to C8 (97-148)
  where they stop at 4 streams. Their one-stream headline (39.6/57.8) is their README table
  (image 2026-10-04-opt) — config: DSpark drafter; our MTP A/B pending.
- CAVEAT: drafter differs (ours MTP k=2, theirs DFlash2/DSpark-ft2) — the DSpark-ft2 A/B is
  queued and is the honest like-for-like.

## Context ceiling (DCP=1, current engine)
- Formula: (116 GB usable − 47.5 weights − ~21 overhead) / 96 KB/token ≈ 500K (defaults)
  → ~690K with prompt-cache 0. The engine AUTO-enables DCP past 200K context (their recipe);
  900K+ is DCP territory. 8-bit KV cache would halve 96 → ~48 KB/token → ~1M+ at DCP=1.
- User's NVFP4/b12x comparison: 600K@dcp1 / 1.2M@dcp2 with ~12 GB/rank MORE weights —
  implies their cache layout ≈ 45-55 KB/token (fp8 KV or tighter MLA pooling).

## Queue
1. **8-bit KV cache (fp8, gated lossy env)** — BUILT 2026-10-15 (host-gated; findings-kv8.md):
   TF_GLM53_KV8=1, e4m3 latent + 2 fp32 scales (584 B/row), indexer K cache stays bf16 → 50.6 KiB/token
   (53.6% of bf16) → DCP=1 ceiling ~500K → ~890-940K. Staged native quant/dequant around the UNCHANGED
   Triton kernels — the existing world-8 AOT pack stays valid, NO recapture. Build rc=0 all 3 stages,
   host tests 346/346. Remaining: artifact distribution + the live kv8 boot + quality gates (§7 of
   findings-kv8.md); the decode-speed path = the §6 in-kernel Triton variants (a capture extension).
   2. MTP k=3 test (after fp8-KV; needs an AOT capture extension for the k=3 shapes — the current
   pack validates k∈{2,0} only).
3. DSpark-ft2 drafter A/B (the honest vs-drowzeys comparison).
4. 32K-context bench rerun (~140 KB filler).
5. Rail 1 investigation (2nd RoCE rail = wire headroom), LAN exposure, tfbench measure artifact.

## Measurement correction (2026-10-15 late)
- The tfbench 22 tok/s readings were an ARTIFACT: our engine build does not emit the decode_s
  stats field their engine emits (ours: None), so tfbench's fallback window halved the rate.
  Direct 512-token greedy re-test: engine reports 43.8 tok/s, client wall 11.9 s — consistent.
  C1 truth ≈ 44 tok/s (conc_chat wall-clock). tfbench results above superseded by conc_chat.

## fp8 landscape (the 8-bit KV question)
- TF_GLM53_VERIFY_HEAD=fp8 EXISTS in drowzeys' Python glm_moe_dsa (headq.py:13 — e4m3 head,
  lossy, "every logit changes"; engine.py:130 — may free the bf16 share). Head only.
- capture.py has the e4m3 quant pattern (absmax/448 per row, torch.float8_e4m3fn).
- NO fp8 KV CACHE anywhere in the lineage (drowzeys Zig, Python 0.6.6 glm_moe_dsa; r2 tree
  has fp8 only under nemotron). The cache itself is bf16 (92-97 KB/token measured).
- Build precedent: the Flash port's kv8 design (round-2 findings: e4m3 + fp32 scale per
  token-head, kv8.cu write + cp.async 8-bit tile read kernels, +27% decode at 128K, quality
  gates clean p=0.73/0.38). Porting that design to glm_moe_dsa = the 8-bit KV path.

## Standard bench round 2 — 1024-token ladder (steady state)
- Sampled agg: prose C1 41.8 / C2 57.0 / C4 78.3 / C8 93.6; code C1 46.0 / C2 62.3 / C4 82.4 / C8 95.8
- Greedy agg:  prose C1 42.0 / C2 57.7 / C4 89.5 / C8 125.7; code C1 47.7 / C2 64.7 / C4 108.0 / C8 144.8
- Single-stream greedy 1024 ×3: 42.3-42.6 tok/s, wall 24.2 s, acceptance 0.453, 1.91 tok/round.
- ACCEPTANCE "WARM-UP" FINDING (inverted): short runs catch a high-acceptance window
  (60-tok: 0.604 acc / 2.22 t-r; 256-tok: 2.20) because openings are predictable; the 1024-tok
  steady state settles at 0.453 / 1.91. So 512-tok ladder numbers were ~3-5% flattered; the
  1024 numbers above are the honest steady state. Per-stream at C8: 11.9-15.8 tok/s/user.

## KV8 v1 LIVE BOOT (ctx 26000×8, binary 53fad867, SIDE=0) — MECHANISM BOOTS, GENERATION BROKEN
- Boot receipts GOOD: "8-bit KV on: e4m3 latent + 2 fp32 scales a token, index keys bf16 —
  50.6 KB a token (53% of bf16)", 1.25 GiB/stream (26000 tokens), RoCE bit-equal, HTTP serving.
- 32K×8 REFUSED alloc ("needs 11.4 GiB, 15.3 free, keeping 6 GiB spare") — free-memory
  ANOMALY under investigation (bf16 fitted 23.6 GiB on the same box; kv8 reports ~14-32 GiB
  less free despite half-size caches + 327 MiB staging). Also 46.7 vs 50.6 KB/token math gap.
- GENERATION BROKEN (the gate did its job): all prompts degenerate garbage from token ~1
  ("iku the the M M M M…", "ippi....ed…", digits/END soup), acceptance 0.4-6.5%, one
  3-token stop ("then…" + EOS). Coherence/quality gate: FAILED — v1's GPU data path
  (quant-on-write / dequant-on-read / staging layout) has a real bug. Host tests (byte math,
  cacheFit) did not cover it. Bisect running: bf16 mode on the same binary.

## With-context prefill test (bf16 serve, ctx 32768×8, FIXED filler ~29K tokens; probe: .111:~/stdbench/ctx-probe.py)
- COLD 30,236-token prompt: TTFT 21.02 s (prefill 20.99 s) ≈ 1,440 tok/s prefill at TP8;
  decode 50.0 tok/s, acceptance 0.74 (summarization accepts far better than creative prose).
- CACHE-HIT repeat: cached 30236/30236, TTFT 0.03 s, prefill 0.00 s — the prompt cache
  eliminates re-prefill entirely; wall 5.1 s for a 256-token completion.
- Supersedes the old Phase-2 "all zeros" (the 4.7×-oversized filler; the ~118 KB rebuild fits).
- Note: 256-token samples show empty content (thinking mode reasons first) — perf numbers unaffected.

## MTP-3 (k=3) TEST — VERDICT: NET LOSS vs k=2 at single-stream greedy (2026-10-16)
- Boot: k=3 parallel-1, ctx 32768, bf16 cache, 94 decode graphs captured at boot (k=3 shapes
  graph fine; AOT pack not extended — not needed for this test). NOTE: k=3 at parallel 8 was
  REFUSED by the same free-memory accounting anomaly as kv8 (16.0 GiB free vs ~30 on default
  config — second sighting, queued to the fixer's cache-compute scope).
- Steady state (greedy 1024, beekeeper, ×3 — identical): 39.9-40.0 tok/s, acc 0.357,
  2.07 tok/round, 494 rounds, 1482 drafted.
- vs k=2 (42.3-42.6 tok/s, acc 0.453, 1.91 t/r, 537 rounds):
  tok/round +8.4% (2.07 vs 1.91) but tok/s −5.7% (40.0 vs 42.4).
- WHY: the marginal 3rd draft accepts poorly (acc 0.453→0.357) while each verify round costs
  ~15% more (wider windows read more expert bytes per round — GB10 is bandwidth-bound).
  Per-round yield grew less than per-round cost.
- DECISION: k=2 stays the serving config. The k=3 AOT capture extension is NOT worth
  pursuing for speed. (k=3 could still win on workloads with higher deep-acceptance —
  e.g. code/repetitive — but the standard prose steady state says no.)

## KV8 v1 — FIXED AND VERIFIED (2026-10-16, binary 5f19d512, canonical 32768×8)
- Root cause chain (2 bugs, both fixed by the kv8 lane): (1) staging zeroed the RoPE angle
  (fixed: glm53_kv8_stage true-angle tables); (2) dequant mode 1 wrote the block layout while
  the non-ROWS attention reads keys flat — every prompt row but row 0 unread (fixed: hybrid
  dst = i while pr < K, r*K+i past K). GPU-proven by the new kv8repro gate (worst rel 6.1% =
  plain e4m3 noise, both prompt-like and decode-like shapes).
- ALL §11.2 GATES PASS at 32768×8 (1.58 GiB/stream, 53% of bf16):
  coherence: "The capital of France is Paris." (acc 88.6%); beekeeper reasoning coherent.
  keyed-seed: 777×2 → f11921dbc9b7, 778 → e3c6cd7434f7 (deterministic within kv8).
  steady-state greedy 1024 ×3: 41.4 tok/s, acc 0.491, 1.98 t/r — vs bf16 42.4/0.453/1.91.
  ⇒ THE 8-BIT KV CACHE COSTS ~2.4% DECODE SPEED AND HALVES CACHE MEMORY. RAM floor 15 GiB.
- Bits differ vs bf16 (lossy e4m3, documented); determinism holds within kv8 mode.

## THE 900K CEILING RUN — SERVING AT DCP=1 WITH KV8 (2026-10-16 night)
- Boot: ctx 900000, parallel 1, kv8, DCP=1 PINNED (TF_GLM53_DCP=1 beats the auto-raise),
  MTP_DENSE=0, reserve 3 (user: "2-3 is enough"), 109 decode graphs, loaded 352.6s.
- THE PAGE-CACHE FIGHT (3 failed boots → solved): .111 is rank 0 AND the NFS server; the
  load refills its server-side cache (~40-45 GiB) and the check fires 2-4s after load-complete
  (a marker-triggered drop loses the race). SOLUTION: continuous drop_caches every 8s on .111
  DURING the load — race-free, the check saw ~48 GiB free. Workers stay clean (O_DIRECT reads).
- DECODE AT 900K: 42.5-42.6 tok/s greedy 1024 ×3 (acc 0.492, t/r 1.98) — FASTER than 32K kv8
  (41.4) and equal to 32K bf16 (42.4). ZERO decode cost at 900K context. ≤5% rule: PASSED.
- Cold 30K prefill: 33.1s ≈ 915 tok/s (vs 1,440 at 32K ctx — the capacity-scaled kv8 staging
  1287 MiB + indexer cost PREFILL; decode unaffected). Cache-hit: 0.02-0.03s TTFT (reuse works
  at 900K). RAM floor: 11 GiB free.
- vs the NVFP4 comparison: their max_model_len 450000 (dcp2) — WE SERVE 2× THEIR CONTEXT at
  DCP=1 with equal decode speed.

## THE DRAFTER MATRIX (32K ctx, parallel 1, kv8, greedy 1024; k2 baselines from the 900K serve)
- k2:  prose 42.6 tok/s (acc 0.492, t/r 1.98) | code 50.1 tok/s (acc 0.700, t/r 2.40)
- k3:  prose 36.5-38.5 (acc 0.372, t/r 2.12)      | code 50.5 (acc 0.605, t/r 2.81)
- k4:  prose 34.8 (acc 0.290, t/r 2.16)        | code 47.5-47.8 (acc 0.515, t/r 3.06)
- k5:  prose 31.1 (acc 0.234, t/r 2.17)        | code 44.3 (acc 0.439, t/r 3.19)
- FINAL MTP VERDICT (SPLIT): k=2 wins PROSE decisively (42.6 vs k3's 36.5-38.5). On CODE,
  k=3 edges k=2 (50.5 vs 50.1, t/r 2.81 vs 2.40 — code's high acceptance pays for the third
  draft; k3-code x2 identical). k4/k5 lose everywhere. So: k2 = the best all-round config;
  k3 = a code-specialist option (+0.8% code, −14% prose). DSpark-ft2/orig pending the capture.

## THE DRAFTER MATRIX — FINAL (32K, parallel 1, kv8, greedy 1024; all x2 identical)
| drafter  | prose tok/s (acc) | code tok/s (acc) |
|----------|---|---|
| MTP k2   | 42.6 (0.492) | 50.1 (0.700) |
| MTP k3   | 36.5-38.5 (0.372) | 50.5 (0.605) |
| MTP k4   | 34.8 (0.290) | 47.5-47.8 (0.515) |
| MTP k5   | 31.1 (0.234) | 44.3 (0.439) |
| DSpark   | **43.9 (0.626)** | **55.6-55.9 (0.707)** |

- WINNER: DSpark ORIGINAL (models11/GLM-5.3-dspark — the .11 vLLM solution's drafter) on BOTH
  prose (+3.1% vs k2) and code (+11.6% vs k2). ft2 second. Matches drowzeys' claimed
  +5%/+12% (their TP4). MTP k2 = the best MTP rung; deeper MTP loses everywhere on GB10.
- The DSpark A/B took real debugging: the staged pack needed distribution to all 8 nodes
  (the peers' -v mount auto-created root-owned empty dirs → FileNotFound → PeerClosed),
  and the drafter needed a direct bind mount (the models11 NFS submount under /root/models
  is unreliable inside the peers' containers under .11's load).

## FINAL RECORDED BENCH (run2 warm; DSpark-original at 32K x 8, kv8; log: .111:~/final-bench-results.log)
- DSpark-p8 ladder (agg tok/s): prose sampled 41.6/55.8/70.9/84.7 (c1/c2/c4/c8);
  prose greedy 43.1/56.4/79.0/111.3; code sampled 46.2/61.5/76.6/89.5; code greedy 45.6/62.0/96.7/130.4
- vs the MTP-k2 1024-ladder (bf16): k2 WINS concurrency (prose greedy C4/C8 89.5/125.7 vs 79.0/111.3;
  code greedy C8 144.8 vs 130.4); DSpark-p8's C1 is capped by the concurrent draft mode (3 drafts/stream).
- THE SPLIT VERDICT: single-stream → DSpark-original (prose 43.9, code 55.9 — the matrix winner);
  concurrent (c2-c8) → MTP k2. Both ladders recorded for the comparison.
- Prefill (DSpark serve): cold 30K = 32.3s (936 tok/s), cache-hit TTFT 0.02-0.03s, decode 45.0 tok/s @30K ctx.

## THE 1M CEILING — SERVING (2026-10-16, ~05:20)
- Boot: ctx 1000000, parallel 1, kv8 (50.6 KB/token), DCP=1 PINNED, reserve 3, k=0,
  DSpark-original drafter (222 MiB, direct-mounted), 189 decode graphs, loaded 362s.
- The k=0 (no MTP layer) freed the MTP buffers — the pool grew enough for 1M at DCP=1
  (the 900K k2 boot was the memory ceiling with MTP aboard; DSpark removed the constraint).

## THE TOOLS-API BENCH (10 scenarios, temp 0, thinking ON both)
- Tools API: BOTH SUPPORTED, all scenarios passed (ours 10/10, theirs 10/10) —
  receipts: eval/results/tools_ours.json + tools_nvfp4.json.
- (An intelligence-question bench was also run during the campaign but is kept
  out of the repo; only the tools bench is published.)

## THE ADAPTIVE-FROWS PATCH — one DSpark config for every concurrency (2026-10-17)
The bug: the concurrent draft mode capped each stream's drafts by the CONFIGURED
parallelism — frows = min(maxDepth+1, 32/N) — so a single stream on a p8 boot got
only 3 drafts (c1 code 53.1 tok/s, t/r 2.72 vs the p1 serve's 3.11).
The fix: cuda_multi.zig depthOf(which, n_live) caps by the round's LIVE stream
count (curve: 8,8,8,7,5,4,3,3 drafts at 1..8 live streams). Diff:
zig-adaptive-frows.diff; full A/B + gates: findings-adaptive-frows.md.

| greedy 1024, agg tok/s | stock (pre-patch) | patched CUT8 | patched CUT4 (FINAL) |
|---|---|---|---|
| c1 code (x2) | 53.1 (t/r 2.72) | 54.3 / 54.8 (t/r 3.11, acc 0.707) | **54.5 (t/r 3.11)** |
| c1 prose (x2) | 43.9 (t/r 1.96) | 43.1 / 43.5 (t/r 1.97) | **43.5 (t/r 1.97)** |
| c2 code | 62.0 | 61.4 | **61.3** |
| c4 code | 96.7 | 93.2-93.4 (the regression) | **96.7 / 97.0** |
| c8 code | 130.4 | 130.4 | **130.6** |
| c2/c4/c8 prose | — | — | **56.7 / 78.7 / 111.4**|

- FINAL CONFIG (live serve): ~/dspark-boot-p8-cut4.sh — DSpark-original depth 8,
  adaptive frows, CUT_STREAMS=4, kv8, 32K x8. One serve for single-stream AND
  concurrent workloads.
