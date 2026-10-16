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

**Image:** [`ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-12-r2-mlx-serve`](https://github.com/drowzeys/TensorFold/pkgs/container/keys-tensorfold-glm53-tp4-dgx-spark)
(linux/arm64, sm_121) as the runtime base — the engine binary and AOT kernel pack are
built from source and bind-mounted. See [v1/BUILD-AND-IMAGE.md](v1/BUILD-AND-IMAGE.md).

## Versions

| Version | Guide | What |
|---|---|---|
| **v1** (recommended) | [v1/README.md](v1/README.md) | The world-8 serve: TP8 across 8× GB10, kv8 (8-bit KV cache), DSpark depth-8 drafting with adaptive per-round caps, 32K×8 default / 1M-context ceiling at DCP=1. The campaign record — everything built, measured, verified. |

## Quick start

```bash
# 1. Build the engine (any GB10 node; ~30 min, no GPU needed for build/tests)
#    engine branch: glm53-tp8-2026-10-16 (world-8 + kv8 + adaptive frows)
cd glm-5.3/v1 && ./scripts/build-r2.sh

# 2. Pull the runtime image on every node (or ship the tar — see BUILD-AND-IMAGE.md)
docker pull ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-12-r2-mlx-serve

# 3. Boot the cluster (rank r on node N; rank 0 serves http://127.0.0.1:8080/v1)
#    The final config: DSpark depth-8, adaptive frows, CUT_STREAMS=4, kv8, 32K×8
./scripts/glm53-w8-kv8-boot.sh          # base kv8 serve
# ...or the DSpark variants — see v1/README.md#how-to-reproduce

# 4. Bench it
./scripts/glm53-bench.sh
```

Weights live on NFS (one copy, seen by all nodes). The drafter needs a direct bind mount.
The full guide — including the AOT pack capture and the preflight tripwire — is in
[v1/BUILD-AND-IMAGE.md](v1/BUILD-AND-IMAGE.md) and [v1/README.md](v1/README.md).
