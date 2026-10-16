#!/bin/bash
# dspark-boot.sh DRAFTER_DIR — DSpark drafter boot for the A/B: 32K ctx, parallel 1, kv8,
# k=0 (MTP unloaded), confidence 0.3, the EXTENDED AOT pack (~/aot-world8-32768-dcp1-dspark).
# Usage: ./dspark-boot.sh /root/models/models11/GLM-5.3-dspark        (the original)
#        ./dspark-boot.sh /root/models/models11/GLM-5.3-speculator.dspark-ft2  (the ft2)
set -u
DRAFT_NAME=${1:?usage: dspark-boot.sh <drafter-name>}
DRAFT=/dspark
HOST_DRAFT=/home/ciprian/models/models11/$DRAFT_NAME
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
      -v /home/ciprian/aot-world8-32768-dcp1-dspark:/aot:ro -v $HOST_DRAFT:/dspark:ro \
      -e TF_GLM53_RANK=$r -e TF_GLM53_WORLD=8 -e TF_GLM53_MASTER=$MASTER \
      -e TF_GLM53_AOT=/aot -e TF_GLM53_HEAD_Q=pack \
      -e TF_GLM53_HCAS=rocep1s0f1 -e TF_GLM53_KV8=1 -e TF_GLM53_SIDE=0 -e TF_GLM53_K=0 \
      -e TF_GLM53_DSPARK=$DRAFT -e TF_GLM53_DSPARK_POLICY=confidence -e TF_GLM53_DSPARK_CONFIDENCE=0.3 -e TF_GLM53_DRAFT_CUT_STREAMS=4 \
      --entrypoint /engine/zig-out/native/bin/tensorfold-native \
      $IMG serve /root/models/models111/GLM-5.3-4.04bpw-h8-exl3 --name glm53-w8 --context 32768 --parallel 8 \
    && echo 'r$r started on .$n'"
}

start_rank 0
sleep 20
for r in 1 2 3 4 5 6 7; do start_rank $r; done
echo "dspark boot launched: $DRAFT"
