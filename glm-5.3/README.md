# GLM-5.3 (EXL3 4.04bpw) on GB10 — TensorFold, world-8

GLM-5.3 inference on 8× NVIDIA DGX Spark (GB10, SM121) using the **TensorFold** Zig/CUDA
engine — tensor parallelism across all 8 nodes over RoCE v2, 8-bit KV cache, DSpark
speculative drafting, and a **1,000,000-token context**.

**Model:** [MikeRoz/GLM-5.3-4.04bpw-h8-exl3](https://huggingface.co/MikeRoz/GLM-5.3-4.04bpw-h8-exl3)
(EXL3, 4.04bpw) — thanks to **MikeRoz** for the quant.
**Drafter:** [RedHatAI/GLM-5.3-speculator.dspark](https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark)
(the original speculator; drowzeys' ft2 fine-tune was also benchmarked — the original wins).

**Why TensorFold here:** on the same cluster, same model, thinking ON, greedy — this serve
runs **2.05× faster on prose and 1.65× on code** than the equivalent NVFP4 vLLM serve, and
serves **1M context vs their 450K**. Full receipts: [v1/RESULTS.md](v1/RESULTS.md).

**Image (fast start, no building):**
[`ghcr.io/ciprianveg/gb10-tensorfold-glm53:v1`](https://github.com/ciprianveg/gb10-tensorfold/pkgs/container/gb10-tensorfold-glm53)
(linux/arm64, sm_121) — the runtime base + built engine + dspark-extended AOT pack, baked
and published. The from-source path (drowzeys' base image + bind-mounted engine/pack) is
documented in [v1/BUILD-AND-IMAGE.md](v1/BUILD-AND-IMAGE.md).

## Versions

| Version | Guide | What |
|---|---|---|
| **v1** (recommended) | [v1/README.md](v1/README.md) | The world-8 serve: TP8 across 8× GB10, kv8 (8-bit KV cache), DSpark depth-8 drafting with adaptive per-round caps, 32K×8 default / 1M-context ceiling at DCP=1. The campaign record — everything built, measured, verified. |

## Quick start

```bash
# 1. Pull the published image (engine + AOT pack baked in, no building)
docker pull ghcr.io/ciprianveg/gb10-tensorfold-glm53:v1

# 2. Boot the cluster — the winning serve: 1M-token context, DSpark drafting, kv8
#    (rank r on node N; rank 0 serves http://127.0.0.1:8080/v1)
cd glm-5.3/v1 && ./start-1m.sh

# 3. Watch it come up (~6 min load)
ssh <head-node> 'docker logs -f glm53-w8-r0'   # wait for "serving glm53-w8 at http://127.0.0.1:8080/v1"
```

Weights live on NFS (one copy, seen by all nodes). The drafter needs a direct bind mount
(the script handles it). Building from source instead of pulling the image, the AOT pack
story, and the load discipline: [v1/BUILD-AND-IMAGE.md](v1/BUILD-AND-IMAGE.md).
