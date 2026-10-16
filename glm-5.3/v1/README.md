# GLM-5.3 EXL3 TP8 on 8× DGX Spark (GB10)

> **Mission:** the first world-8 (TP8) serve of GLM-5.3 full anywhere — the MikeRoz
> EXL3 4.04bpw pack on drowzeys' TensorFold r2 Zig engine, distributed across eight
> GB10 DGX Sparks (.111–.118, single-rail RoCE, 100 Gbit — 200 Gbit should add a
> small further gain, especially in prefill), plus an 8-bit KV cache that carries
> it to a 900K-token context at zero decode cost.
>
> **Entry point:** this file. Numbers: RESULTS.md. Full technical trail: the
> `findings/findings-*.md` files. Engine diffs: `patches/`. Scripts: `scripts/`.
> Build/image/boot guide: BUILD-AND-IMAGE.md.

## What we started from


| Piece                       | Source                                                                                                                                                                                                                                                                                                                               | Link / ID                                                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| Engine                      | **drowzeys'** TensorFold r2 Zig CUDA port for GLM-5.3 (family `glm_moe_dsa`, TensorFold 1.0.7) — the only public full-GLM-5.3 CUDA implementation                                                                                                                                                                                   | branch[`glm53-tp4-2026-10-12-r2`](https://github.com/drowzeys/TensorFold) (tip `00c3bb11`); our worktree: `~/eugr/TensorFold-glm53-r2` |
| Container image             | drowzeys' published serve image (mlx-serve +`libtensorfold_glm53.so` C ABI, AOT toolchain)                                                                                                                                                                                                                                           | `ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-12-r2-mlx-serve`                                                         |
| Bench methodology + recipes | drowzeys' TP4 recipe repo (bench conditions, receipts we compare against)                                                                                                                                                                                                                                                            | [`drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark`](https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark)            |
| Model pack                  | **MikeRoz'** GLM-5.3 EXL3 4.04bpw, h8 head (`GlmMoeDsaForCausalLM`, 76 shards, 360 GB)                                                                                                                                                                                                                                               | [`MikeRoz/GLM-5.3-4.04bpw-h8-exl3`](https://huggingface.co/MikeRoz/GLM-5.3-4.04bpw-h8-exl3)                                            |
| Drafting sidecars           | The pack lacks bf16`eh_proj.weight` (MTP load) and bf16 `lm_head.weight` (all drafters) — we built the 2.05 GB `model-sidecar-bf16.safetensors` from the pack's own EXL3 tensors via the engine's own reference decoder (`tools/glm53/sidecar.zig`), wired in index-driven with **no engine patch** (findings/findings-sidecars.md) | local artifact; mechanism documented in this dir                                                                                       |

Starting state: drowzeys' engine had run this model at **TP4 on 4× GB10 only** ("no
other machine or rank count has been run"), with their 2.75bpw pack. No TP8 serve of
GLM-5.3 existed anywhere — all TP8 numbers in the wild were extrapolations.

## What we added

1. **The first world-8 (TP8) serve of this model** — 8× GB10
   Boot infrastructure: `start-1m.sh` (the winning 1M-context DSpark serve). Exactness
   gate passes: keyed-seed contract (seed 777 twice → identical `token_sha`, 778 →
   moves), greedy exact, RoCE one-shot collectives verified bit-equal at 3- and
   32-row windows.
2. **The world-8 AOT kernel pack** — captured and validated (154 variants / 19
   functions, 4/4 validations zero-missing at (parallel 1,8) × (k 2,0)); one rank-0
   capture suffices at DCP=1 (findings/findings-aot-world8.md). Pack staged in
   `aot-world8/pack/` (aot.json sha256 `0190e384…`).
3. **The dtype-round-cast loader patch** — our pack stores six bf-named tensor
   classes as F16; `cuda_weights.zig` now round-casts them to bf16 at load
   (`bf16Round`, the same schema the capture compiled against — the AOT pack stays
   valid, no recapture). Diff: `zig-dtype-patch.diff`.
4. **The bf16 drafting sidecars** (above) — drafting unblocked on first boot; MTP
   k=2 worked immediately (60.4% acceptance at first light).
5. **The 8-bit (fp8 e4m3) KV cache for `glm_moe_dsa`** — `TF_GLM53_KV8` (default
   OFF, lossy, gated): e4m3 MLA latent + 2 fp32 scales per token-row, DSA indexer
   keys stay bf16 → **50.6 KiB/token = 53% of bf16**. Staged native quant/dequant
   around **unchanged** Triton kernels — AOT-pack-safe, no recapture. Two bugs
   root-caused and fixed (the staging's zeroed RoPE angle; the dequant's mode-1
   block-vs-flat prompt layout), with a GPU self-consistency regression gate
   (`kv8repro`, `zig build kv8repro`) proving staged rows bit-identical and the
   roundtrip clean. Verified live: **~2.4% decode cost, half the cache memory**
   (41.4 vs 42.4 tok/s; 1.58 vs 2.95 GiB/stream at 32K×8). Diff: `patches/zig-kv8.diff`;
6. **The oracle.py DSpark capture chain** — `TF_GLM53_DSPARK`-gated capture steps
   that record the drafter's Triton launches (`_group_sums`, `_prep_kernel`,
   `_block_attn_kernel`, …), which neither the window chains nor the static
   enumerator can see; the pack() step now refuses a dspark capture missing its
   drafter kernels. Diff: `patches/oracle-dspark.diff`; findings/findings-dspark-capture.md.
7.

## What we borrowed — CREDITS

Named individuals and projects whose work this mission built on (borrowed designs
are credited where used; the implementation choices and bugs are ours):

- **ashhart** — the creator of TensorFold itself: the engine this whole mission
  runs on (the project's lead author, 889 commits — the Zig CUDA server, the
  family architecture, the AOT pack system, the drafting and DCP machinery we
  built on).
- **drowzeys** — the r2 Zig engine and its whole GLM-5.3 CUDA port (the base we
  generalized to world 8), the TP4 recipe repo and bench methodology (our bench
  conditions mirror theirs for comparability), the published serve image, the
  DSpark **ft2 fine-tune** of the RedHatAI speculator (the drafter for the pending
  DSpark A/B), and the nemotron in-tree fp8 precedent inside TensorFold.
- **ajensenwaud** — the MiMo Flash port's kv8 (patches 0063–0067,
  [`ajensenwaud/MiMo-V2.6-2x-DGX-Spark-TensorFold`](https://github.com/ajensenwaud/MiMo-V2.6-2x-DGX-Spark-TensorFold)):
  the design reference for our 8-bit KV cache (e4m3 + fp32 scale per token-head,
  quant-on-write / dequant-on-read, roundtrip quality gates).
- **RedHatAI** — [`RedHatAI/GLM-5.3-speculator.dspark`](https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark),
  the DSpark speculator base that drowzeys fine-tuned (ft2).
- **MikeRoz** — the GLM-5.3-4.04bpw-h8-exl3 pack we serve.
- **vLLM and sglang projects** — the MLA fp8-KV patterns mined as design
  references (per-layer scales, quantize-on-save, in-kernel dequant, the DCP-path
  fp8 assertion we mirrored in our v1 gate); the reference mining is
  findings/findings-kv8.md §8/§11.
- **turboderp-org / exllamav3** — the EXL3 format and kernels the engine's weight
  path is built on.

## How to reproduce

Cluster: the 8 GB10 nodes (the start script targets .111–.118; adjust the IPs
in `start-1m.sh` for your own layout). Weights on NFS — one copy, every node sees it.

```bash
# 1. Pull the published image (engine + AOT pack baked in)
docker pull ghcr.io/ciprianveg/gb10-tensorfold-glm53:v1

# 2. Boot — THE winning serve: 1M-token context, DSpark depth-8, kv8, DCP=1
#    (rank r on node N; rank 0 serves 127.0.0.1:8080)
./start-1m.sh

# 3. Verify
ssh <head-node> 'docker logs -f glm53-w8-r0'   # wait for "serving glm53-w8 at http://127.0.0.1:8080/v1"

# 4. Tools-API bench (key via BENCH_API_KEY env — no keys stored in this repo)
BENCH_API_KEY=<your-key> python3 eval/tool_bench.py --preset ours
```

Building from source (the image's engine layer) and the full image story:
BUILD-AND-IMAGE.md. The kv8 GPU regression gate lives in the engine worktree:
`zig build kv8repro` (runs on any single GB10; no model, no engine).

## RESULTS

All numbers greedy unless noted, thinking ON, 32K ctx, steady-state
1024-token rounds (the honest measure — see the acceptance warm-up note in
RESULTS.md). Full logs: RESULTS.md.

**The final config's ladder** (DSpark depth 8 + adaptive frows +
CUT_STREAMS=4, kv8, p8 serve — one serve for every concurrency; greedy 1024,
aggregate tok/s):


|       | C1                             | C2   | C4         | C8    |
| ----- | ------------------------------ | ---- | ---------- | ----- |
| code  | **54.5** (t/r 3.11, acc 0.707) | 61.3 | 96.7–97.0 | 130.6 |
| prose | 43.5 (t/r 1.97)                | 56.7 | 78.7       | 111.4 |

Prose t/r barely moves with concurrency (1.97 at c1 — the 0.3 confidence cut
trims prose chains below the depth cap everywhere), so the adaptive cap is a
code-workload win. c1 is closed to the p1 profile (t/r and acceptance exactly
match the p1 drafter-matrix values); the residual ~2% vs p1's 55.9 is the
multi-decoder's per-round host path, not drafting. Greedy output-preserving:
fixed-prompt shas match the unpatched engine.

**MTP k=2 reference ladder** (the pre-DSpark campaign record):


|               | C1   | C2   | C4    | C8    |
| ------------- | ---- | ---- | ----- | ----- |
| prose greedy  | 42.0 | 57.7 | 89.5  | 125.7 |
| code greedy   | 47.7 | 64.7 | 108.0 | 144.8 |
| prose sampled | 41.8 | 57.0 | 78.3  | 93.6  |
| code sampled  | 46.0 | 62.3 | 82.4  | 95.8  |

**vs drowzeys' TP4 published receipts** (their 2.75bpw pack, DFlash2/DSpark
drafters, sampled): chat C1 38.3 / C2 54.3 / C4 65.6; code C1 41.6 / C2 49.9 /
C4 62.6 — ours is higher at every point (C1 +15%, C2 +9–24%, C4 +24–37%) with a
higher-bpw pack, and extends to C8. Caveat: drafter
differs (ours MTP k=2, theirs DFlash2/DSpark) — the DSpark-ft2 A/B is the honest
like-for-like, pending.

**KV8 (8-bit KV cache), verified at 32768×8:** 50.6 KiB/token (53% of bf16),
1.58 GiB/stream; steady-state greedy 41.4 tok/s vs bf16's 42.4 → **~2.4% decode
cost, half the cache memory**. Determinism holds within kv8 mode (keyed-seed
777×2 → identical sha); bits legitimately differ from bf16 (lossy e4m3,
documented).

**The 900K ceiling (kv8, DCP=1, single stream):** decode 42.5–42.6 tok/s at
900,000-token context — equal to 32K bf16, zero decode cost at 900K; ≤5% rule
passed. Cold 30K prefill 33.1 s (~915 tok/s, capacity-scaled staging); cache-hit
TTFT 0.02–0.03 s. The NVFP4 comparison serve: max_model_len 450,000 (dcp2).

**Prefill / cache-hit (bf16 serve, 32K×8):** cold 30,236-token prompt TTFT
21.02 s (~1,440 tok/s prefill at TP8); cached repeat: 30236/30236 cached, TTFT
0.03 s.

**MTP-3:** net loss vs k=2 (40.0 vs 42.4 tok/s, −5.7%); k=2 stays.

## Final numbers (the overnight campaign)

**The drafter matrix** (32K ctx, parallel 1, kv8, greedy 1024, ×2 identical):


| drafter             | prose tok/s (acc)  | code tok/s (acc)       |
| ------------------- | ------------------ | ---------------------- |
| MTP k2              | 42.6 (0.492)       | 50.1 (0.700)           |
| MTP k3              | 36.5–38.5 (0.372) | 50.5 (0.605)           |
| MTP k4              | 34.8 (0.290)       | 47.5–47.8 (0.515)     |
| MTP k5              | 31.1 (0.234)       | 44.3 (0.439)           |
| **DSpark ** | **43.9 (0.626)**   | **55.6–55.9 (0.707)** |

~~Split verdict~~ — **superseded by the adaptive-frows patch**: the original concurrent
draft mode capped each stream's drafts by the CONFIGURED parallelism (`min(maxDepth+1, 32/N)`), so a single stream on a p8 boot got only 3 drafts (c1 code 53.1, t/r 2.72). The
patch (`cuda_multi.zig` `depthOf(which, n_live)`) caps by the round's LIVE stream count
instead — the curve: 8,8,8,7,5,4,3,3 drafts at 1..8 live streams. With CUT_STREAMS=4
(the confidence floor engages at ≥4 live streams; =8 dipped c4 by −3.4%), one config now
serves both regimes: c1 code 54.5 (t/r 3.11 — the p1 profile, was 53.1), c8 code 130.6,
no concurrency point regressed vs stock. Diff: `patches/zig-adaptive-frows.diff`; A/B and gates:
`findings/findings-adaptive-frows.md`. The MTP-k2 ladder it replaced: prose greedy
42.0/57.7/89.5/125.7, code greedy 47.7/64.7/108.0/144.8 (c1/c2/c4/c8).

**The context ceiling**: 1,000,000 tokens SERVING at DCP=1 (kv8 50.6 KB/token, reserve 3,
DSpark drafter, 189 decode graphs, loaded 362 s) — 2.2× the NVFP4 comparison's 450K max.
Decode at 900K ctx = 42.6 tok/s (equal to bf16 at 32K — zero decode cost from either the
8-bit cache or the long context). Cold 30K prefill ~915–936 tok/s; cache-hit TTFT 0.02 s.
Prefill slows with capacity (the kv8 staging + indexer scale with ctx) — the honest cost.

**Quality gates**: keyed-seed determinism within every mode; kv8 lossy vs bf16 (documented,
e4m3 only — the rope path is bf16-exact); GPU self-consistency gate `kv8repro` (worst rel
6.1% = plain e4m3 noise).

**Long-context quality (the 1M serve)**: needle-in-haystack at 845,962 tokens — FOUND.
100K-context prose/code:  gen 38.8–42.0 tok/s at 100K.

**The tools-API bench vs the NVFP4 vLLM serve** (10 scenarios, temp 0, thinking ON
both, identical prompts — eval/results/tools_*.json): both endpoints support the
OpenAI tools API and pass all 10 scenarios (10/10 ours, 10/10 theirs).
