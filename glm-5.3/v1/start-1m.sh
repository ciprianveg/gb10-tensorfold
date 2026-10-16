#!/bin/bash
# start-1m.sh — THE v1 START: the winning GLM-5.3 serve — 1,000,000-token context,
# DSpark drafting, 8-bit KV cache, DCP=1 — on 8x DGX Spark using the published v1
# image (engine + AOT pack baked in: ghcr.io/ciprianveg/gb10-tensorfold-glm53:v1).
# Rank r on node .(111+r); rank 0 (the head) serves http://127.0.0.1:8080/v1.
# Usage: ./start-1m.sh [drafter-name]   (default GLM-5.3-dspark — the original speculator)
set -u
IMG=ghcr.io/ciprianveg/gb10-tensorfold-glm53:v1
DRAFT_NAME=${1:-GLM-5.3-dspark}
HOST_DRAFT=/home/ciprian/models/models11/$DRAFT_NAME
MASTER=192.168.178.111:29581
RANK_FLAGS="--gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --memory 110g --memory-swap 110g"
NCCL="-e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_CROSS_NIC=0"

start_rank() {
  local r=$1 n=$((111+$r))
  ssh -o ConnectTimeout=10 ciprian@192.168.178.$n "docker rm -f glm53-w8-r$r 2>/dev/null; \
    docker run -d --name glm53-w8-r$r $RANK_FLAGS $NCCL \
      -v /home/ciprian/models:/root/models:ro -v $HOST_DRAFT:/dspark:ro \
      -e TF_GLM53_RANK=$r -e TF_GLM53_WORLD=8 -e TF_GLM53_MASTER=$MASTER \
      -e TF_GLM53_AOT=/aot -e TF_GLM53_HEAD_Q=pack -e TF_GLM53_HCAS=rocep1s0f1 \
      -e TF_GLM53_KV8=1 -e TF_GLM53_SIDE=0 -e TF_GLM53_K=0 -e TF_GLM53_DCP=1 -e TF_GLM53_CACHE_RESERVE_GB=3 \
      -e TF_GLM53_DSPARK=/dspark -e TF_GLM53_DSPARK_POLICY=confidence -e TF_GLM53_DSPARK_CONFIDENCE=0.3 \
      $IMG serve /root/models/models111/GLM-5.3-4.04bpw-h8-exl3 --name glm53-w8 --context 1000000 --parallel 1 \
    && echo 'r$r started on .$n'"
}

# rank 0 first (opens the rendezvous port + serves HTTP), then the followers
start_rank 0
sleep 20
for r in 1 2 3 4 5 6 7; do start_rank $r; done
echo "1M-context DSpark boot launched (drafter: $DRAFT_NAME)"
echo "watch: ssh ciprian@192.168.178.111 'docker logs -f glm53-w8-r0'"
