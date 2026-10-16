#!/bin/bash
# glm53-w8-dspark-boot.sh — DSpark drafter A/B (bf16, k=0, MTP unloaded, confidence 0.3, single stream) of GLM-5.3-4.04bpw-h8-exl3 on the drowzeys r2 Zig engine.
# Undrafted, greedy, ctx 32768, parallel 1. Rank r on node .(110+r); rank 0 (.111) serves.
# Container flags: the PROVEN run_phase2a_cluster.sh RANK_FLAGS + nccl_env pattern
# (--device /dev/infiniband, memlock=-1, IPC_LOCK, network host, NCCL RoCE v2 env).
# AOT pack: ~/aot-world8-32768-dcp1 (validated 4/4 zero-missing). Pack: the NFS view
# ~/models/models111/... (one copy, NFS-served from .111 — see CLUSTER-STORAGE-MAP.md).
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
      -e TF_GLM53_HCAS=rocep1s0f1 -e TF_GLM53_K=0 -e TF_GLM53_DSPARK=/root/models/models11/GLM-5.3-dspark -e TF_GLM53_DSPARK_POLICY=confidence -e TF_GLM53_DSPARK_CONFIDENCE=0.3 \
      --entrypoint /engine/zig-out/native/bin/tensorfold-native \
      $IMG serve /root/models/models111/GLM-5.3-4.04bpw-h8-exl3 --name glm53-w8 --context 32768 --parallel 1 \
    && echo 'r$r started on .$n'"
}

# rank 0 first (opens the rendezvous port + serves HTTP), then the followers
start_rank 0
sleep 20
for r in 1 2 3 4 5 6 7; do start_rank $r; done
echo "all 8 ranks launched — watch: ssh ciprian@192.168.178.111 'docker logs -f glm53-w8-r0'"
