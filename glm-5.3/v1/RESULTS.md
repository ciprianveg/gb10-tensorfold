# RESULTS — GLM-5.3-4.04bpw-h8-exl3 world-8 serve (accumulating)

Engine: starts from drowzeys r2 Zig 1.0.7 (patched: dtype-round-cast for our pack's 6 F16 classes) on
8× GB10 (.111-.118), single-rail 100GBot connectivity.
Pack: MikeRoz 4.04bpw mul1 h8 + bf16 sidecars (drafting). Serving: 32K ctx, MTP k=2 drafts on,
prompt reuse on, DCP=1.

## First generations
- Greedy 60-tok: 50.7 tok/s, TTFT 0.18 s, acceptance 60.4%, 2.22 tok/round.
- Sampled 50-tok: 53.5-53.9 tok/s.

## Standard bench round 1 (drowzeys bench/, 512 toks, seed 1729)

- Concurrent ladder, sampled (aggregate tok/s):
  prose C1 44.1 / C2 59.4 / C4 81.0 / C8 97.3
 
- Concurrent ladder, greedy:
  prose C1 44.2 / C2 59.5 / C4 87.9 / C8 125.1

## With-context prefill test
- COLD 30,236-token prompt: TTFT 21.02 s (prefill 20.99 s) ≈ 1,440 tok/s prefill at TP8;
  decode 50.0 tok/s, acceptance 0.74 (summarization accepts far better than creative prose).
- CACHE-HIT repeat: cached 30236/30236, TTFT 0.03 s, prefill 0.00 s — the prompt cache
  eliminates re-prefill entirely; wall 5.1 s for a 256-token completion.


## THE 900K CEILING RUN — SERVING AT DCP=1 WITH KV8 (2026-10-16 night)
- Boot: ctx 900000, parallel 1, kv8, DCP=1 PINNED (TF_GLM53_DCP=1 beats the auto-raise),
  MTP_DENSE=0, reserve 3 (user: "2-3 is enough"), 109 decode graphs, loaded 352.6s.
- DECODE AT 900K: 42.5-42.6 tok/s greedy 1024 ×3 (acc 0.492, t/r 1.98)
- Cold 30K prefill: 33.1s ≈ 915 tok/s (vs 1,440 at 32K ctx — the capacity-scaled kv8 staging
  1287 MiB + indexer cost PREFILL; decode unaffected). Cache-hit: 0.02-0.03s TTFT (reuse works
  at 900K). RAM floor: 11 GiB free.

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

## THE 1M CEILING — SERVING (2026-10-16, ~05:20)
- Boot: ctx 1000000, parallel 1, kv8 (50.6 KB/token), DCP=1 PINNED, reserve 3, k=0,
  DSpark-original drafter (222 MiB, direct-mounted), 189 decode graphs, loaded 362s.
- The k=0 (no MTP layer) freed the MTP buffers — the pool grew enough for 1M at DCP=1
  (the 900K k2 boot was the memory ceiling with MTP aboard; DSpark removed the constraint).

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

- FINAL CONFIG: DSpark-original depth 8, adaptive frows, CUT_STREAMS=4, kv8.
  One serve for single-stream AND concurrent workloads (the repo's start-1m.sh boots
  this drafter stack at 1M context).
