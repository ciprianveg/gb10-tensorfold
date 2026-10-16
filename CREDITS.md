# Credits & Attribution

## Core Dependencies

| Component | Source | License | Notes |
|-----------|--------|---------|-------|
| **TensorFold** | [ashhart/TensorFold](https://github.com/ashhart/TensorFold) | MIT | The engine — Zig CUDA server, family architecture, AOT packs, drafting + DCP machinery |
| **TensorFold r2 (GLM-5.3 port)** | [drowzeys/TensorFold](https://github.com/drowzeys/TensorFold) | MIT | The base we generalized to world 8: the GLM-5.3 CUDA port, TP4 recipe + bench methodology, the DSpark ft2 fine-tune, the published serve image |
| **EXL3** | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | MIT | The weight format and kernels the engine's loader is built on |
| **GLM-5.3 EXL3 pack** | [MikeRoz](https://huggingface.co/MikeRoz) (GLM-5.3-4.04bpw-h8-exl3) | — | The weights we serve |
| **DSpark speculator** | [RedHatAI/GLM-5.3-speculator.dspark](https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark) | Apache 2.0 | The drafter base (drowzeys fine-tuned it as ft2) |
| **kv8 design reference** | [ajensenwaud/MiMo-V2.6-2x-DGX-Spark-TensorFold](https://github.com/ajensenwaud/MiMo-V2.6-2x-DGX-Spark-TensorFold) | — | The 8-bit KV cache design reference (e4m3 + fp32 scales, quant-on-write/dequant-on-read, roundtrip gates) |
| **vLLM / sglang** | [vllm-project/vllm](https://github.com/vllm-project/vllm) / [sgl-project/sglang](https://github.com/sgl-project/sglang) | Apache 2.0 | The MLA fp8-KV patterns mined as design references (per-layer scales, in-kernel dequant) |
| **CUDA 13.2** | NVIDIA | Proprietary | GB10/SM121 support |
| **NCCL** | NVIDIA | BSD-3-Clause | RoCE v2 collectives |

## Upstream Projects

| Project | Repository | Branch/Commit |
|---------|------------|---------------|
| TensorFold (upstream) | https://github.com/ashhart/TensorFold | main |
| TensorFold (r2 fork, our base) | https://github.com/drowzeys/TensorFold | r2 (GLM-5.3 port) |
| Our engine branch | local `glm53-tp8-2026-10-16` (to be published on the ciprianveg fork) | world-8 + kv8 + adaptive frows |

See [ATTRIBUTION.md](ATTRIBUTION.md) for the narrative lineage and what is ours vs borrowed.
