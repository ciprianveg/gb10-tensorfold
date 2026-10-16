# gb10-tensorfold GLM-5.3 v1 — the fast-start image
# Base: the drowzeys r2 runtime (CUDA 13.2, linux/arm64, sm_121). Engine + AOT pack baked in.
# In-container layout EXACTLY matches the cluster-validated bind-mount boot:
#   /engine/zig-out/... = the engine worktree's zig-out (from scripts/build-r2.sh)
#   /aot                = the AOT kernel pack (aot-world8-32768-dcp1-dspark)
# Build context must contain: engine/ (zig-out) and aot/ (the pack)
FROM ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-12-r2-mlx-serve
COPY engine /engine/zig-out
COPY aot /aot
ENTRYPOINT ["/engine/zig-out/native/bin/tensorfold-native"]
