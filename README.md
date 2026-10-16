# gb10-tensorfold

TensorFold inference solutions for NVIDIA **DGX Spark / DGX Spark+ (GB10)** — Blackwell **SM121**.

This repo is the TensorFold companion to [gb10-vllm](https://github.com/ciprianveg/gb10-vllm):
per-model subtrees for building the engine, deploying multi-node serves, and reproducing
benchmarked results — using the **Zig/CUDA TensorFold engine** instead of vLLM.

TensorFold (by [ashhart](https://github.com/ashhart/TensorFold)) is a compact LLM inference
engine with native CUDA servers, AOT-captured kernel packs, and multi-rank tensor parallelism.
On the 8× GB10 cluster it serves GLM-5.3 EXL3 at **2.05× the prose / 1.65× the code speed of
the equivalent NVFP4 vLLM serve**, with **1,000,000-token context** .

## Models index


| Model                      | Subtree                   | Highlights                                                                                                                                                                                                                                           |
| -------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GLM-5.3** (EXL3 4.04bpw) | [glm-5.3/v1](glm-5.3/v1/) | **Recommended** — world-8 TP across 8× GB10 (first TP8 GLM-5.3 serve anywhere), 8-bit KV cache (53% memory, ~2.4% cost), DSpark speculative drafting with adaptive per-round draft caps, 1M-token context at DCP=1, needle found at 845,962 tokens |

Each subtree is self-contained: a model page (`glm-5.3/README.md`), the versioned working
solution (`v1/`), build/boot scripts, engine patches, findings, and bench receipts. Open the
model page for its full guide.

## Architecture

### Hardware

- **GPU**: 1× NVIDIA GB10 (DGX Spark / DGX Spark+) per node — Blackwell **SM121**
- **Cluster**: 8 nodes, RoCE v2 (ConnectX-7, 100 Gbit) fabric for NCCL collectives, unified memory
- **Target arch**: sm_121 (`-Dsm=121` in the Zig build)

> All results in this repo were measured on **100 Gbit** RoCE connectivity. Moving the
> fabric to **200 Gbit** should provide a small further increase in speed — especially
> in prefill, where cross-rank bandwidth matters most.

### Software stack

- **Engine**: [TensorFold](https://github.com/ashhart/TensorFold) — the Zig CUDA server,
  built from source (3-stage RAM-guarded build: fatbins → native+lib → host tests)
- **Base**: the drowzeys r2 TensorFold GLM-5.3 port (see
  [ATTRIBUTION.md](ATTRIBUTION.md) for the full lineage)
- **Runtime container**: [`ghcr.io/ciprianveg/gb10-tensorfold-glm53:v1`](https://github.com/ciprianveg/gb10-tensorfold/pkgs/container/gb10-tensorfold-glm53)
  (CUDA 13.2, linux/arm64, sm_121) — the fast-start image with the engine + AOT pack
  baked in; the from-source path is documented in the
  [build & image guide](glm-5.3/v1/BUILD-AND-IMAGE.md)
- **Weights**: EXL3 format ([exllamav3](https://github.com/turboderp-org/exllamav3)) via NFS
- **Drafting**: DSpark block-drafter (RedHatAI speculator base) — one block pass yields all
  8 draft candidates

### Key technologies


| Technology         | What it does                                                                                            |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| **World-8 TP**     | Tensor parallelism across 8 GB10 nodes over RoCE v2 one-shot collectives                                |
| **kv8**            | 8-bit fp8 KV cache (e4m3 + 2 fp32 scales/token-row) — 53% of bf16 cache memory, ~2.4% decode cost      |
| **DSpark**         | Block-pass speculative drafting — depth-8 chains from a single drafter forward                         |
| **Adaptive frows** | Per-round draft caps that follow the live stream count — one serve config optimal at every concurrency |
| **AOT packs**      | Pre-captured CUDA kernel packs (per shape/window) — instant graph dispatch, no JIT at boot             |
| **DCP**            | Decode context parallelism — pinned DCP=1 for the 1M-context serve                                     |
| **EXL3**           | The weight format + kernels the engine's loader is built on                                             |

## Repository structure

```
gb10-tensorfold/
├── glm-5.3/                  GLM-5.3 EXL3 — see glm-5.3/README.md
│   └── v1/                   The world-8 serve (kv8 + DSpark + adaptive frows + 1M ctx)
│       ├── README.md         The campaign record — what was built, measured, verified
│       ├── RESULTS.md        The full bench receipts
│       ├── BUILD-AND-IMAGE.md  Build the engine, pull/build the image, boot the cluster
│       ├── findings/         Technical findings (kv8, adaptive frows, AOT, DSpark capture)
│       ├── patches/          Engine diffs (kv8, adaptive frows, dtype, oracle dspark)
│       ├── scripts/          Boot / bench / preflight / build scripts
│       └── eval/             The tools-API bench (harness + receipts)
```

## Comparison with the vLLM stack

Same cluster, same model (GLM-5.3), thinking ON, greedy, 1024-token rounds:


|                  | TensorFold (this repo)  | vLLM (gb10-vllm) |
| ---------------- |-------------------------|------------------|
| prose tok/s (c1) | **43.9**                | 26               |
| code tok/s (c1)  | **55.9**                | 40               |
| weights          | EXL3 4.04bpw            | NVFP4            |

Full receipts: [glm-5.3/v1/RESULTS.md](glm-5.3/v1/RESULTS.md).
