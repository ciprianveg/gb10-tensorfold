#!/usr/bin/env bash
# serve-prod.sh — THE PROD LAUNCHER (both tiers, one command each).
# Lineage: the golden serve8-best-cfg.sh (v5.3's line, §128.27ea onward) — this wrapper only maps
# the arm names to the golden's env pins; every pin, expect, and the md5 gate live in the golden.
#
# CREDITS: the engine is TensorFold (Ash Hart's upstream framework). The pool-semantics design
# and the dense/head quantization patterns are Mia's.
#
# THE ARMS (the serve picked by need — both tiers ship; the measured rows live in
#            BENCHMARK-RECORD.md, not here):
#   speed        DCP1, 200K x 2  — the <=2-stream speed tier (the deep form: 600K x 1).
#   capacity     DCP2, 200K x 4  — the 4-stream split form (the uniform capacity standing serve).
#   pool-1m      DCP2, 450K x 4, POOL_AGG=1000000 — the unequal-request pool form (the 450K
#                per-request cap at 4 streams).
#   pool-600k-d1 DCP1, 450K x 4, POOL_AGG=600000 — the DCP1 4-stream pool form (the packed
#                un-split plane; the 600K cap, the 450K per-request cap).
#   bar-12m      DCP2, 600K x 2, MIN_BUCKET=262144 — the 1.2M aggregate bar.
#                (PROBE-gated: the uniform band past 1.05M is policy, not memory.)
#
# OPERATOR NOTES:
#   - The pool's semantics: admission by actual usage, first-fit packing, release-reuse of done
#     streams' rows, evict-oldest-idle under pressure; an over-capacity claim is REFUSED at the
#     API (never a crash, never a wedge) — the workers stay alive; retry after traffic drains.
#   - The first-fit fragmentation is real: a large claim can refuse while smaller concurrent
#     claims hold mid-plane rows (the rows fit in sum, not contiguously) — submit the large
#     request first when packing a full plane.
#   - The fills serialize at the shared ~1000 tok/s rate under concurrency (expected).
#   - The md5 x8 gate: the golden refuses on any binary mismatch.
#   - Every env here is arm-overridable (the golden's ${VAR:-} class) — see the golden's header.
set -euo pipefail
cd "$(dirname "$0")"
ARM=${1:-capacity}
case "$ARM" in
  speed)        exec env DCP=1 PARALLEL=2 bash serve8-best-cfg.sh "${2:-5901}" "${3:-200000}" ;;
  capacity)     exec env DCP=2 PARALLEL=4 bash serve8-best-cfg.sh "${2:-5901}" "${3:-200000}" ;;
  pool-1m)      exec env DCP=2 PARALLEL=4 POOL_AGG=1000000 PROBE=1 POOL_LOG=1 bash serve8-best-cfg.sh "${2:-5901}" "${3:-450000}" ;;
  pool-600k-d1) exec env DCP=1 PARALLEL=4 POOL_AGG=600000 PROBE=1 POOL_LOG=1 MIN_BUCKET=262144 END_FREE_GB=5 PREWARM_BREAKER_GB=5 CACHE_GIB=1 STREAMS_RESERVE_GB=2 bash serve8-best-cfg.sh "${2:-5901}" "${3:-450000}" ;;
  bar-12m)      exec env DCP=2 PARALLEL=2 MIN_BUCKET=262144 PROBE=1 END_FREE_GB=5 PREWARM_BREAKER_GB=5 CACHE_GIB=1 STREAMS_RESERVE_GB=2 bash serve8-best-cfg.sh "${2:-5901}" "${3:-600000}" ;;
  *) echo "unknown arm: $ARM (speed | capacity | pool-1m | pool-600k-d1 | bar-12m)" >&2; exit 2 ;;
esac
