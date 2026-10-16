# Build, image, and boot — the working v1 stack

How to reproduce the GLM-5.3 world-8 TensorFold serve on 8× DGX Spark. Three layers:

1. **The engine** — built from source (Zig + nvcc), bind-mounted into the container
2. **The AOT kernel pack** — pre-captured CUDA kernels per shape/window, bind-mounted
3. **The runtime image** — the CUDA 13.2 container everything runs inside

## 0. Preflight

```bash
./scripts/glm53-preflight.sh        # validates CLUSTER_NODES/COPY_HOSTS, checks the pack
```

The tripwire hard-rejects wrong-cluster configs. Cluster layout: rank r on node N,
rank 0 (the head) serves `http://127.0.0.1:8080/v1`. Weights on NFS — one copy, every
node sees it; never copy packs between nodes.

## 1. Build the engine (~30 min, no GPU needed, <10 GiB RAM guarded)

Requirements: Zig 0.17 toolchain (aarch64), CUDA toolkit with nvcc, the engine source.

```bash
# engine branch: glm53-tp8-2026-10-16 (world-8 + kv8 + adaptive frows)
# on top of drowzeys/TensorFold r2 (the GLM-5.3 port)
git clone https://github.com/drowzeys/TensorFold && cd TensorFold
git checkout <r2>            # then apply our branch (see patches/ — or use our fork)

./scripts/build-r2.sh        # 3 stages: fatbins (sm121) → native+lib+generate → host tests
```

The build is RAM-guarded (a watchdog kills it past 9 GiB) and runs on any node — no GPU
needed for build or tests. Output: `zig-out/native/bin/tensorfold-native` + libs + fatbins.

The engine diffs in `../patches/` are the campaign's changes:
`zig-kv8.diff` (the 8-bit KV cache), `zig-adaptive-frows.diff` (the per-round draft cap),
`zig-dtype-patch.diff`, `oracle-dspark.diff` (the DSpark AOT capture chain).

## 2. The AOT kernel pack

The engine dispatches through pre-captured kernel packs (per shape/window) — no JIT at
boot. The base pack (`aot-world8-32768-dcp1`) covers the world-8 windows at 32K context;
the DSpark extension adds the drafter kernels (164 cubins total).

- Capture flow + the pack pipeline: `../findings/findings-aot-world8.md` and
  `../findings/findings-dspark-capture.md` (static list-variants cannot see drafter
  launches — the oracle dspark chain is required; diff in `../patches/oracle-dspark.diff`)
- The pack must exist on **every** node at the same path (bind-mounted `-v <pack>:/aot:ro`)
- Distribute from the head via rsync. If a node is missing the dir, docker auto-creates an
  empty root-owned one and the rank dies with FileNotFound → remove it, rsync again.

## 3. The runtime image

**The fast-start path — no building:** the complete v1 stack (runtime base + built engine +
dspark-extended AOT pack) is published as a GHCR image:

```
ghcr.io/ciprianveg/gb10-tensorfold-glm53:v1
```

```bash
docker pull ghcr.io/ciprianveg/gb10-tensorfold-glm53:v1
```

The image bakes in the engine at `/engine` and the AOT pack at `/aot`, with the native
binary as ENTRYPOINT — so the boot drops the engine/pack bind-mounts entirely (see the
fast-start block below). Structure-verified (binary runs, pack loads); the bind-mount
path below remains the cluster-validated reference boot.

**Fast start (the final v1 config, rank 0 shown):**

```bash
docker run -d --name glm53-w8-r0 --gpus all --network host --ipc host \
  --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --memory 110g \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_HCA=rocep1s0f1 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_CROSS_NIC=0 \
  -v /home/ciprian/models:/root/models:ro \
  -v /path/to/GLM-5.3-dspark:/dspark:ro \
  -e TF_GLM53_RANK=0 -e TF_GLM53_WORLD=8 -e TF_GLM53_MASTER=<head-ip>:29581 \
  -e TF_GLM53_AOT=/aot -e TF_GLM53_HEAD_Q=pack -e TF_GLM53_HCAS=rocep1s0f1 \
  -e TF_GLM53_KV8=1 -e TF_GLM53_DRAFT_CUT_STREAMS=4 -e TF_GLM53_DSPARK=/dspark \
  ghcr.io/ciprianveg/gb10-tensorfold-glm53:v1 \
  serve /root/models/models111/GLM-5.3-4.04bpw-h8-exl3 --name glm53-w8 --context 32768 --parallel 8
```

**The from-source path** (what v1 was built and validated with): the drowzeys published
image as the runtime base, engine + pack bind-mounted:

```bash
IMG=ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-12-r2-mlx-serve

# on the head:
docker pull $IMG

# ship to the workers (23 GB tar — faster than 8× pulls over one uplink):
docker save $IMG | gzip > /tmp/tf-img.tar.gz
for n in 112 113 114 115 116 117 118; do
  scp /tmp/tf-img.tar.gz ciprian@192.168.178.$n:/tmp/
  ssh ciprian@192.168.178.$n 'gunzip -c /tmp/tf-img.tar.gz | docker load'
done
```

**Build the image yourself** (the thin layer — this is how the published one was built;
Dockerfile in `../docker/Dockerfile`):

```dockerfile
FROM ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-12-r2-mlx-serve
COPY engine /engine          # the built zig-out (from scripts/build-r2.sh)
COPY aot /aot                # the AOT pack (aot-world8-32768-dcp1-dspark)
ENTRYPOINT ["/engine/native/bin/tensorfold-native"]
```

```bash
docker build -t gb10-tensorfold-glm53:v1 .
```

**Push it to GHCR** so others can reproduce your exact build:

```bash
echo $GHCR_PAT | docker login ghcr.io -u <your-user> --password-stdin
docker tag gb10-tensorfold-glm53:v1 ghcr.io/<your-user>/gb10-tensorfold-glm53:v1
docker push ghcr.io/<your-user>/gb10-tensorfold-glm53:v1
```

(GHCR PAT needs `write:packages`. The image is linux/arm64 — build on a GB10. If your
base layers match an already-pushed image, the push only uploads your thin layers.)

## 4. Boot the cluster

```bash
# rank 0 first (opens the rendezvous port + serves HTTP), then the followers
./scripts/glm53-w8-boot.sh                  # bf16 cache, undrafted — the base serve
./scripts/glm53-w8-kv8-boot.sh              # + TF_GLM53_KV8=1 (8-bit KV cache)
./scripts/glm53-w8-kv8-ceiling-dcp1.sh      # the 900K/1M ceiling (DCP=1 pinned)
./scripts/glm53-w8-dspark-boot.sh           # + the DSpark drafter (needs the extended pack)
```

The **final v1 config** (one serve for every concurrency): DSpark-original depth 8,
adaptive frows, `TF_GLM53_DRAFT_CUT_STREAMS=4`, kv8, 32K×8 — see the README's results
section for why this exact combination.

Container flags that matter (already in the scripts): `--device /dev/infiniband`,
`--ulimit memlock=-1`, `--cap-add IPC_LOCK`, `--network host`, and the NCCL RoCE v2 env
(`NCCL_IB_HCA=rocep1s0f1`, `NCCL_IB_ROCE_VERSION_NUM=2`).

**The drafter needs a direct bind mount** (`-v <dspark-dir>:/dspark:ro -e TF_GLM53_DSPARK=/dspark`)
— the NFS submount is unreliable inside peer containers.

**Load discipline (NFS-served packs):** the head is rank 0 AND the NFS server — its page
cache refills during the ~400 s load and the cache-fit check fires 2–4 s after
load-complete. Drop caches on the head every ~8 s *during* the load:
`sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`. Workers stay clean (O_DIRECT reads).

## 5. Verify + bench

```bash
docker logs -f glm53-w8-r0       # wait for "serving glm53-w8 at http://127.0.0.1:8080/v1"
./scripts/glm53-bench.sh         # prose + code, c1 + c8 (drowzeys-comparable conditions)
```

Expected (greedy, thinking ON, 1024-token rounds, final config):
code c1/c2/c4/c8 = **54.5 / 61.3 / 96.7 / 130.6** aggregate tok/s; prose
**43.5 / 56.7 / 78.7 / 111.4**. Full record: `../RESULTS.md`.
