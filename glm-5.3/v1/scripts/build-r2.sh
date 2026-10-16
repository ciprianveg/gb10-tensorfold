#!/bin/bash
# RAM-guarded r2 build — Stage 0. Constraint: <10 GiB RAM, no GPU needed for build/tests.
set -o pipefail
export PATH=$HOME/zig-toolchain/zig-aarch64-linux-0.17.0:$PATH
cd /home/ciprian/eugr/TensorFold-glm53-r2
LOG=/home/ciprian/eugr/spark-vllm-docker/investigations/glm53-exl3-tp8/build-r2.log
RAM_LIMIT_KIB=9437184  # 9 GiB
watchdog() {
  local mainpid=$1
  while kill -0 "$mainpid" 2>/dev/null; do
    local rss
    rss=$(ps -eo rss=,comm= | awk '$2 ~ /zig|nvcc|cc1|clang|zig-cc/ {s+=$1} END {print s+0}')
    if [ "$rss" -gt "$RAM_LIMIT_KIB" ]; then
      echo "$(date -Is) WATCHDOG: build-tree RSS ${rss}KiB > ${RAM_LIMIT_KIB}KiB — killing build" >> "$LOG"
      pkill -TERM -P "$mainpid"; kill -TERM "$mainpid" 2>/dev/null
      echo "$(date -Is) ABORT: RAM limit" >> "$LOG"; exit 9
    fi
    sleep 30
  done
}
echo "$(date -Is) BUILD START zig=$(zig version) nvcc=$(nvcc --version | tail -1)" >> "$LOG"
echo "$(date -Is) STAGE 1/3: fatbins (nvcc kernel packs, sm121)" >> "$LOG"
zig build -j4 -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=safe fatbins >> "$LOG" 2>&1 &
B=$!; watchdog $B & W=$!
wait $B; RC=$?; kill $W 2>/dev/null
echo "$(date -Is) STAGE 1 rc=$RC" >> "$LOG"
[ $RC -ne 0 ] && exit $RC
echo "$(date -Is) STAGE 2/3: native + glm53-lib + tf-glm53-generate" >> "$LOG"
zig build -j4 -Dfatbins=zig-out/fatbin -Dsm=121 -Doptimize=safe native glm53-lib tf-glm53-generate >> "$LOG" 2>&1 &
B=$!; watchdog $B & W=$!
wait $B; RC=$?; kill $W 2>/dev/null
echo "$(date -Is) STAGE 2 rc=$RC" >> "$LOG"
[ $RC -ne 0 ] && exit $RC
echo "$(date -Is) STAGE 3/3: host tests (no GPU): test-glm53 test-cuda-host" >> "$LOG"
zig build -j4 -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=safe test-glm53 test-cuda-host >> "$LOG" 2>&1
RC=$?
echo "$(date -Is) STAGE 3 rc=$RC" >> "$LOG"
echo "$(date -Is) BUILD DONE rc=$RC" >> "$LOG"
exit $RC
