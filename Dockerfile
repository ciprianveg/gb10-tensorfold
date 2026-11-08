# The gb10-tensorfold serve image — the sm_121/GB10 product form.
#
# Bakes what the serve runs: the TensorFold native server binary and the grammar
# library, over the proven glm53-fp8-final runtime base (the CUDA/NCCL/RoCE
# environment the serve containers use today — see BUILD.md for the base's
# lineage and the source build lane). The model pack, the DSpark drafter pack,
# and the AOT cache pack stay PATH-MOUNTED (they are far larger than the image
# and node-local concerns) — start.sh wires the mounts.
#
# Build context: this Dockerfile plus deploy/ containing
#   tensorfold-native    the native server binary (BUILD.md lane; ~100 MB, gitignored)
#   libtfgrammar.so      the grammar library (lazy-loaded; ~3 MB, gitignored)
#   tensorfold-serve     the entry wrapper (tracked in this repo)
# Build:  docker build -t ghcr.io/ciprianveg/gb10-tensorfold:v1.0.0 .
# Push:   docker push ghcr.io/ciprianveg/gb10-tensorfold:v1.0.0

FROM ghcr.io/ciprianveg/gb10-tensorfold:base-glm53-fp8-final

COPY deploy/tensorfold-native /usr/local/bin/tensorfold-native
COPY deploy/tensorfold-serve /usr/local/bin/tensorfold-serve
COPY deploy/libtfgrammar.so /usr/local/bin/libtfgrammar.so
RUN chmod +x /usr/local/bin/tensorfold-native /usr/local/bin/tensorfold-serve /usr/local/bin/libtfgrammar.so

# The grammar library is dlopen'd lazily beside the binary (candidate 2 of the
# loader's search) — baking it means the structured-output lane needs no mount.
# The lazy load also means free-form serving never touches it.
ENV LD_LIBRARY_PATH=/usr/local/bin:${LD_LIBRARY_PATH}

ENTRYPOINT ["/usr/local/bin/tensorfold-serve"]
