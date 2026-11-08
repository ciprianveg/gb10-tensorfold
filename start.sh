#!/usr/bin/env bash
# start.sh — the gb10-tensorfold 8-node launcher (the distilled product form).
#
# Boots one serve container on every node in the fleet, wired for the RoCE
# fabric and the model mounts. One command:
#
#   bash start.sh                      # THE DEFAULT: the pool-P4 form — DCP1, 4 slots,
#                                      #   a 600K shared KV pool (200K a request's cap),
#                                      #   the 262144 graph bucket floor
#   bash start.sh pool                 #   same, explicit
#   bash start.sh capacity             #   the DCP2 4-slot form: a 1.2M total KV pool, 450K a stream's max context
#   bash start.sh stop                 #   the clean fleet teardown
#
# THE POOL IS THE BEHAVIOR: mixed-length streams, first-fit claims, evict-oldest-idle,
# honest over-cap refusal at the API (the serve stays alive). The knobs, env-overridable:
#   PARALLEL   the concurrent slots          (default 4)
#   POOL_AGG   the shared pool's aggregate   (default 600000; set EMPTY for the uniform
#              pre-division path — the slots divide the window evenly, no pool)
#   CONTEXT    a request's KV cap            (default 200000)
# NOTE: plain DCP1/P4 WITHOUT the pool refuses at the prewarm budget (139 graphs, 8.7 GiB
# needed vs 6.7 free — measured 2026-11-08): the pool pins and the P4 form go together.
#
# Parameters (env-overridable):
#   NODES    the node IPs, space-separated (default: 192.168.178.111..118)
#   PORT     the serve port       (default 5901)
#   IMAGE    the container image  (default ghcr.io/ciprianveg/gb10-tensorfold:latest)
#   PACK     the model pack path  (default /home/ciprian/models/models11/GLM-5.3-NVFP4-CSF)
#   DSPARK_PACK  the drafter pack (default /home/ciprian/models/models11/GLM-5.3-dspark)
#   AOTDIR   the AOT cache pack, must exist on EVERY node
#            (default /home/ciprian/aot-world8-nvfp4-serve-copack-ord)
#
# Requirements: 8x GB10 (DGX Spark) with docker, a RoCE fabric (the NCCL envs
# below assume one HCA per node; adjust HCAS/GID for your fabric), the models
# share mounted at the same path on every node, and ssh access to every node.
set -u

PORT=${PORT:-5901}
IMAGE=${IMAGE:-ghcr.io/ciprianveg/gb10-tensorfold:latest}
PACK=${PACK:-/home/ciprian/models/models11/GLM-5.3-NVFP4-CSF}
DSPARK_PACK=${DSPARK_PACK:-/home/ciprian/models/models11/GLM-5.3-dspark}
AOTDIR=${AOTDIR:-/home/ciprian/aot-world8-nvfp4-serve-copack-ord-e3m}
NODES=${NODES:-"192.168.178.111 192.168.178.112 192.168.178.113 192.168.178.114 192.168.178.115 192.168.178.116 192.168.178.117 192.168.178.118"}
ARM=${1:-pool}

# The forms: the pool-P4 default (the measured form), the DCP2 capacity split, the teardown.
# ${VAR-default} (unset-only) so an EMPTY env var is a meaningful override (POOL_AGG= = no pool).
case "$ARM" in
  pool)      DCP=1 PARALLEL=${PARALLEL-4} CONTEXT=${CONTEXT-400000} POOL_AGG=${POOL_AGG-600000} MIN_BUCKET=${MIN_BUCKET-262144};;
  capacity)  DCP=2 PARALLEL=${PARALLEL-4} CONTEXT=${CONTEXT-450000} POOL_AGG=${POOL_AGG-1200000} MIN_BUCKET=${MIN_BUCKET-262144};;
  stop)
    for n in $NODES; do
      ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" \
        'docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1; pkill -x tensorfold-native 2>/dev/null; true'
    done
    echo "fleet down on: $NODES"; exit 0;;
  *) echo "unknown arm: $ARM (pool | capacity | stop)" >&2; exit 2;;
esac

MASTER=$(echo $NODES | awk '{print $1}')
WORLD=$(echo $NODES | wc -w)

# --- the gates (each one names a real failure class; all are cheap) -------------------------
for n in $NODES; do
  # The AOT pack must exist on every node: docker -v of a missing path silently
  # mounts an EMPTY directory and the boot dies ~10 minutes in, far from the cause.
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" "test -f $AOTDIR/aot.json" 2>/dev/null \
    || { echo "REFUSED: the AOT pack $AOTDIR is absent on $n (rsync it there first)"; exit 9; }
  # Free memory: the model load needs headroom on every rank.
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" \
    'av=$(free -g | awk "/^Mem:/{print \$7}"); [ "$av" -ge 100 ] || { echo "avail=${av}GiB < 100"; exit 9; }' 2>/dev/null \
    || { echo "REFUSED: $n preflight (free memory)"; exit 9; }
done

# The image must be present and IDENTICAL on every node (a stale or missing
# image on one rank wedges the rendezvous with no error anywhere). On absence
# or mismatch, stream it from the master node.
want=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$MASTER" "docker images --format {{.ID}} $IMAGE" 2>/dev/null | tr -d '\n')
[ -n "$want" ] || { echo "REFUSED: $MASTER has no $IMAGE — pull it first (docker pull $IMAGE)"; exit 13; }
for n in $NODES; do
  img=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" "docker images --format {{.ID}} $IMAGE" 2>/dev/null | tr -d '\n')
  if [ "$img" != "$want" ]; then
    echo "IMAGE: $n has ${img:-NONE} != $want — restoring from $MASTER (streams the image, minutes)"
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$n" 'docker load' \
      < <(ssh -o BatchMode=yes -o ConnectTimeout=15 "$MASTER" "docker save $IMAGE") \
      || { echo "REFUSED: $n image restore failed"; exit 13; }
    # docker load renames a superseded same-tag image to <none> — prune the orphan,
    # or every image update leaks a 25.5 GB layer set per node.
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$n" \
      'docker images --filter dangling=true --format "{{.ID}}" | xargs -r docker rmi >/dev/null 2>&1; true'
  fi
done
echo "image gate: $IMAGE $want uniform on all $WORLD nodes"

# --- the boot: one detached container per node ------------------------------------------------
# The stale-container clean first (the golden's clean_ranks class): a prior boot's dead or live
# containers own the serve-r names — without the clean the run dies on the name conflict.
for n in $NODES; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" 'docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1; true'
done

idx=0
for n in $NODES; do
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" "docker run -d --name serve-r$idx --gpus all --network host --shm-size=2g \
    --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK \
    --user \$(id -u):\$(id -g) \
    -v /home/ciprian/models:/home/ciprian/models \
    -v $DSPARK_PACK:/dspark:ro \
    -v $AOTDIR:/aot:ro \
    -e NCCL_SOCKET_IFNAME=enp1s0f1np1 -e NCCL_CUMEM_ENABLE=0 \
    -e NCCL_IB_HCA=rocep1s0f1 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_CROSS_NIC=0 \
    -e NCCL_MAX_NCHANNELS=4 -e NCCL_BUFFSIZE=16777216 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_SPLIT_DATA_ON_QPS=0 \
    -e TF_GLM53_RANK=$idx -e TF_GLM53_WORLD=$WORLD -e TF_GLM53_MASTER=$MASTER:29581 \
    -e TF_GLM53_BIND=$PACK -e TF_GLM53_AOT=/aot \
    -e TF_GLM53_KV8=1 -e TF_GLM53_KV8_V3=1 -e TF_GLM53_DCP=$DCP -e TF_GLM53_GRAPHS=1 -e TF_GLM53_PREWARM=1 \
    -e TF_GLM53_GRAPH_MAX_ROWS=8 -e TF_GLM53_PREWARM_MB_GRAPH=20 -e TENSORFOLD_MEMORY_RESERVE_GIB=8 \
    -e TF_GLM53_K=0 -e TF_GLM53_PARALLEL=$PARALLEL -e TF_GLM53_PROMPT_SP=1 \
    -e TF_GLM53_PROMPT_ROWS=4096 -e TF_GLM53_PROMPT_ROWS_SHORT=4096 \
    -e TF_GLM53_ROCE=1 -e TF_GLM53_HCAS=rocep1s0f1 -e TF_GLM53_GID=3 -e ROCE_PROXY_CPU=19 \
    -e TF_GLM53_PREWARM_END_FREE_GB=6 -e TF_GLM53_TIMEOUT=5400 \
    -e TF_GLM53_DSPARK=/dspark -e TF_GLM53_DSPARK_POLICY=confidence -e TF_GLM53_DSPARK_CONFIDENCE=0.20 -e TF_GLM53_DSPARK_DEPTH=8 \
    -e TF_GLM53_PROMPT_REUSE=1 -e TF_GLM53_CACHE_GIB=4 -e TF_GLM53_CACHE_ENTRIES=32 \
    -e TF_GLM53_ROCE_HEALTH_LOCAL=1 \
    -e TF_GLM53_POOL_AGG=$POOL_AGG -e TF_GLM53_GRAPH_MIN_BUCKET=$MIN_BUCKET \
    $IMAGE '$PACK' --port $PORT --context $CONTEXT --parallel $PARALLEL --dcp $DCP --drafts on --kv8-v3 on --graphs on" \
    || { echo "REFUSED: rank $idx container failed to start on $n"; exit 3; }
  idx=$((idx+1))
done

echo "serve: $WORLD ranks started (arm=$ARM, port $PORT, context $CONTEXT)"
echo "  logs:  docker logs serve-r<idx> on each node (the master's serve-r0 carries the HTTP surface)"
echo "  poll:  ssh $MASTER 'docker logs serve-r0 2>&1 | grep -E \"SERVE up|BIND|error\" | tail'"
echo "  ready: bash bench.sh $MASTER $PORT"
