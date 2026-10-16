#!/bin/bash
# glm53-w8-kv8-boot.sh — WORLD-8 GLM-5.3 serve with the 8-BIT (fp8 e4m3) KV CACHE.
# Same proven boot as glm53-w8-boot.sh (RANK_FLAGS + nccl_env + AOT pack + NFS pack view)
# plus -e TF_GLM53_KV8=1 (lossy, gated; default OFF) and -e TF_GLM53_SIDE=0 (the side stream
# defaults ON ("af") and build 53fad867… refuses the combination; the auto-off build drops the
# need for this). Expect boot receipt "8-bit KV on … 50.6 KB a token" and ~1.58 GiB/stream
# caches (vs 2.95 bf16). AOT pack: UNCHANGED (kv8 v1 = staged native quant/dequant).
set -u
IMG=ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-12-r2-mlx-serve
MASTER=192.168.178.111:29581
RANK_FLAGS="--gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --memory 110g --memory-swap 110g"
NCCL="-e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_CROSS_NIC=0"

start_rank() {
  local r=$1 n=$((111+$1))
  ssh -o ConnectTimeout=10 ciprian@192.168.178.$n "docker rm -f glm53-w8-r$r 2>/dev/null; \
    docker run -d --name glm53-w8-r$r $RANK_FLAGS $NCCL \
      -v /home/ciprian/models:/root/models:ro \
      -v /home/ciprian/TensorFold-glm53-r2:/engine:ro \
      -v /home/ciprian/aot-world8-32768-dcp1:/aot:ro \
      -e TF_GLM53_RANK=$r -e TF_GLM53_WORLD=8 -e TF_GLM53_MASTER=$MASTER \
      -e TF_GLM53_AOT=/aot -e TF_GLM53_HEAD_Q=pack \
      -e TF_GLM53_HCAS=rocep1s0f1 -e TF_GLM53_KV8=1 -e TF_GLM53_SIDE=0 \
      --entrypoint /engine/zig-out/native/bin/tensorfold-native \
      $IMG serve /root/models/models111/GLM-5.3-4.04bpw-h8-exl3 --name glm53-w8 --context 32768 --parallel 8 \
    && echo 'r$r started on .$n'"
}

# rank 0 first (opens the rendezvous port + serves HTTP), then the followers
start_rank 0
sleep 20
for r in 1 2 3 4 5 6 7; do start_rank $r; done
echo "all 8 ranks launched — watch: ssh ciprian@192.168.178.111 'docker logs -f glm53-w8-r0'"
