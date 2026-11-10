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

# The --vision decoders, built at the versions the engine itself pins and verifies at
# load (JPEG by the loaded binary's string table, WebP by WebPGetDecoderVersion):
# libjpeg-turbo 3.1.4.1 and libwebp 1.6.0 — the releases Pillow 13.0.0 bundles, so
# decoded pixels match the reference stack byte for byte. A throwaway stage; the
# sonames land beside the binary on the documented loader path (bin/../lib).
FROM ubuntu:24.04 AS decoders
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake make gcc g++ curl ca-certificates libpng-dev libjpeg-dev libtiff-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN curl -fsSL -o jpeg.tar.gz https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.1.4.1.tar.gz \
 && tar xf jpeg.tar.gz \
 && cmake -S libjpeg-turbo-3.1.4.1 -B bjpeg -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
 && cmake --build bjpeg -j"$(nproc)" \
 && cp -L bjpeg/libturbojpeg.so.0 /out-libturbojpeg.so.0
RUN curl -fsSL -o webp.tar.gz https://github.com/webmproject/libwebp/archive/refs/tags/v1.6.0.tar.gz \
 && tar xf webp.tar.gz \
 && cmake -S libwebp-1.6.0 -B bwebp -DCMAKE_BUILD_TYPE=Release -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF \
 && cmake --build bwebp -j"$(nproc)" \
 && cp -L bwebp/libwebpdecoder.so.3 /out-libwebpdecoder.so.3

FROM ghcr.io/ciprianveg/gb10-tensorfold:base-glm53-fp8-final

COPY deploy/tensorfold-native /usr/local/bin/tensorfold-native
COPY deploy/tensorfold-serve /usr/local/bin/tensorfold-serve
COPY deploy/libtfgrammar.so /usr/local/bin/libtfgrammar.so
COPY --from=decoders /out-libturbojpeg.so.0 /usr/local/lib/libturbojpeg.so.0
COPY --from=decoders /out-libwebpdecoder.so.3 /usr/local/lib/libwebpdecoder.so.3
RUN chmod +x /usr/local/bin/tensorfold-native /usr/local/bin/tensorfold-serve /usr/local/bin/libtfgrammar.so

# The grammar library is dlopen'd lazily beside the binary (candidate 2 of the
# loader's search) — baking it means the structured-output lane needs no mount.
# The lazy load also means free-form serving never touches it.
ENV LD_LIBRARY_PATH=/usr/local/bin:${LD_LIBRARY_PATH}

ENTRYPOINT ["/usr/local/bin/tensorfold-serve"]
