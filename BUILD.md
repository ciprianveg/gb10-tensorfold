# BUILD.md — building the binary and the image

The GHCR image ships a prebuilt binary; this document is the source lane: how
that binary is produced, and the gotchas that cost real debugging hours.

## What the serve runs

- `tensorfold-native` — the TensorFold native server, built from this repo's
  `zig/` sources (the GLM-5.3 lane). ~100 MB with symbols as shipped.
- `libtfgrammar.so` — the grammar library (structured output). It is
  dlopen'd **lazily** on the first structured request and sits beside the
  binary; free-form serving never loads it.
- `tensorfold-serve` — the entry wrapper (`deploy/tensorfold-serve`, tracked);
  parses the flags and execs the native binary. The image's ENTRYPOINT.

The model pack, the DSpark drafter pack, and the AOT cache pack are NOT in the
image — they are path-mounted by `start.sh` (see README).

## Prerequisites

- **Target**: NVIDIA DGX Spark (GB10, sm_121), aarch64. The binary is built
  ON an aarch64 GB10 node — cross-building x86 -> aarch64 for this stack is
  not a lane anyone has run; don't start there.
- **Zig**: the pinned toolchain from `.zig-version` (aarch64-linux tarball).
- **nvcc**: CUDA 13.x with sm_121 support (`-Dsm=121` below).
- **The GLM53 lane flag**: the GLM-5.3 family code is gated behind
  `TF_BUILD_GLM53_LANE=1` at build time.

## The build

```bash
TF_BUILD_GLM53_LANE=1 <zig> build native -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
# -> zig-out/native/bin/tensorfold-native
```

A cold build takes minutes to tens of minutes (the CUDA objects dominate);
incremental is fast. Verify the result by md5 — the shipped class is pinned
in the release notes and the binary is identical on every node of a fleet
(`start.sh`'s image gate enforces the image-level equivalent).

**One build tree per node**: the build tree must live on the node that runs
the build — building over NFS or rsyncing a build tree between nodes picks up
stale artifacts and wrong-arch objects. Keep the sources in git, the build
tree local, and treat any `zig-out` not produced on this node as suspect.

## The AOT cache pack (why AOTDIR is a required mount)

The serve boots from a pre-captured AOT pack (the captured decode graphs and
their fatbins — `aot.json` plus the fatbin blobs, tens of GB). The pack must
exist **on every node** before boot: `docker -v` of a missing path silently
mounts an empty directory, and the failure surfaces ~10 minutes later as a
pack-load error far from the cause. `start.sh` refuses the boot instead.

The pack is produced by the AOT capture lane (`zig` tooling in this repo);
it is node-format-stable but version-tied to the binary class — a new binary
class wants a fresh capture, or at minimum a verified boot against the old
pack before trusting deep-context behavior.

## The base image

The runtime base (`ghcr.io/ciprianveg/gb10-tensorfold:base-glm53-fp8-final`)
is the Ubuntu 24.04 + CUDA/NCCL/RoCE userspace + the python HTTP environment
the serve containers have run in production: the IB verbs devices, the NCCL
tunables' supported set, and the entry conventions all assume it. The product
Dockerfile layers the binary, the grammar library, and the entry wrapper over
that base and nothing else — if you rebuild the base from scratch you own the
NCCL/IB surface too; the supported path is to consume it from GHCR.

## Building the product image

```bash
# deploy/ holds the build context binaries (gitignored — fetch or build them):
#   deploy/tensorfold-native   (the build lane above)
#   deploy/libtfgrammar.so     (the grammar library)
#   deploy/tensorfold-serve    (tracked in this repo)
docker build -t ghcr.io/<you>/gb10-tensorfold:vX.Y.Z .
docker push  ghcr.io/<you>/gb10-tensorfold:vX.Y.Z
```

Then point `start.sh` at it with `IMAGE=...`.
