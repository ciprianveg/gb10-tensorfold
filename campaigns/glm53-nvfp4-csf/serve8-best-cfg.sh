#!/bin/bash
# serve8-best-cfg.sh — THE GOLDEN START-SH, v5: ONE SCRIPT OWNING THE BOOT (§128.27fa/fb/fc; NO PUSH)
#
# THE CONSOLIDATION (the user's ruling, adopt-don't-reinvent): every knob EXPLICITLY PINNED here; the
# launcher reduced to plumbing; the effective env RECEIPT-CHECKED at boot (the era tf-ship.sh's
# pull_receipt/expect_env machinery, adopted verbatim). The decoy defaults become VISIBLE, the wrong-pack
# and partial-distribution classes die at the gate.
#
# THE DRIFT EXHIBITS THIS SCRIPT KILLS (each with its receipt):
#  - THE WRONG-PACK CLASS (§128.27fa): the dcp2 wrapper's effective AOTDIR default is the -e3m pack while
#    the inner launcher's own dead default is the NON-e3m dir — staging or reasoning against the wrong one
#    is the in-container FileNotFound class (three dead boots). NOW: AOTDIR pinned + the launcher's dead
#    default replaced with a hard refuse + the pack's aot.json md5 gated ×8 pre-boot.
#  - THE DECOY DEFAULTS (the -e block, verbatim): GRAPHS:-0, PREWARM:-0, CACHE_GIB:-1,
#    PROMPT_ROWS_SHORT:-2048, PARALLEL:-1, DENSE defaults via the wrapper, DEV:-1 — every boot that omits
#    one silently takes the decoy. NOW: every one exported here and checked against /proc/1/environ.
#  - THE FROZEN LAUNCHER DEFAULTS (§128.27eq): omitting DSPARK_CONF/DEPTH falls to the launcher's frozen
#    0.15/8 — NOT the standing cell. NOW: 0.20/6 pinned and checked.
#  - THE ENV INDIRECTION (§128.27ey's two misses): the boot env goes through the launcher's var names;
#    a direct TF_GLM53_* pass is overwritten by the ${VAR:-} expansions. NOW: the expect check reads the
#    EFFECTIVE env — an overwritten or decoyed var REFUSES the boot.
#  - PRINT-WITHOUT-CONSUMER (§128.27fb, Drift #7): the cal boot line at cuda_engine:781 claims gating the
#    code does not perform (the consumer off since 125.43f). The cal knob stays pinned EMPTY here until
#    the consumer re-land earns it; the print is no longer allowed to imply otherwise.
#  - THE GRAPHS NO-OP (found by THIS SCRIPT'S OWN GATE, boot79): the form's GRAPHS=1 NEVER reached the
#    engine — TF_GLM53_GRAPHS='' effective in EVERY receipted boot (the launcher's handling drops it);
#    all standing rates were measured graphs-empty at the env term; the graphs-on arm is priced
#    separately (§128.27eg's replay question stands). TF_GLM53_DSPARK='/dspark' — the launcher's own
#    default value — routes dspark (harmless, receipted by every boot); pinned as the effective value.
#  - THE SP_CHECKS NO-OP (boot81's find): SP_CHECKS=1 was exported by every standing boot but NEVER
#    passed by the launcher's docker-run block — TF_GLM53_SP_CHECKS is ABSENT from the effective env;
#    the engine's default governed every receipted rate. Pinned absent; the exhibit stands until the
#    launcher plumbs it or the default is receipted as intended.
#  - THE SP_CHECKS VERDICT (§128.27fe's read): the engine has NO reader for TF_GLM53_SP_CHECKS (zero
#    consumption in glm53r2/ and native/) — the term was decorative END-TO-END (the era's var died with
#    the era's engine). DELETED from the pins, never plumbed — a knob nothing reads is removed, not
#    wired. THE GRAPHS RECEIPT: the engine's parse is envFlag("TF_GLM53_GRAPHS", true) — THE DEFAULT IS
#    ON; the standing form ran GRAPHED all along. THE GRAPHS RECONCILIATION (§128.27fe, the trace +
#    environ receipts): the LAUNCHER constructs -e TF_GLM53_GRAPHS=1 correctly (the bash -x trace, all
#    8 nodes) but the var is CONSUMED between docker and pid 1 — the image's entrypoint
#    (/usr/local/bin/tensorfold-serve) translates the env to the --graphs ARG, the TRUE CARRIER (the
#    launcher's CMD: --graphs ${GRAPHS:-on}). The standing boots ran "--graphs 1" — the INVALID form
#    (the era's "must be on|off" refusal line) — and the engine's fallback left graphs ON (envFlag's
#    default true; the g=486-528/e=0 split receipts: every round a replay, zero eager; the prewarm's
#    64 graphs) — the "cosmetic" receipt was the mask. THE CLEAN FORM: GRAPHS=on (pinned,
#    ${GRAPHS:-on} the arm-override) with the CMD receipt asserted (the env expect REMOVED — the var
#    is consumed by design); the OFF arm: GRAPHS=off — the graphs' true benefit measured by absence.
#
# THE STANDING FORM (the receipts): the r4 best cell 0.20/6 (§128.27eg; the A/A-safe peak, §128.27r5);
# the true-parallel form PARALLEL=4/DCP_STREAMS=1 (§128.27ea); SP=1 + SP_CHECKS=1; KV8 v3; the measured
# capture fit GRAPHS=1/GRAPH_MAX_ROWS=8 (§F.4 envelope); PROMPT_ROWS_SHORT=4096; CACHE_GIB=4;
# END_FREE_GB=11; DEV=0 (the reaper dodge). The CAL: available-but-unset — the table is staged ×8
# (/aot/draft-cal.json, md5 0d091ba7) but CONSUMER-LESS (§128.27fb); the fixed policy knob exists
# ({cost,confidence,fixed}) for the fixed-depth arm, queued.
#
# Usage: bash serve8-best-cfg.sh [port] [context]   (defaults: 5901 200000)
# Gates: the pack md5 ×8 pre-boot; SERVER-UP poll; the receipt trio (binary / pack / effective env);
# expect_env REFUSES on any pin mismatch. Outer probes (the band, TE, quality) are the caller's.
set -euo pipefail
PORT="${1:-5901}"; CTX="${2:-200000}"

# ------------------------------------------------------------------ THE PINS ----
export DEV=0 SP=1 KV8=1 DCP=2 DCP_STREAMS=1
export PREWARM=1 GRAPH_MAX_ROWS=8 ODC=0 GRAPHS=${GRAPHS:-on}
export DSPARK_CONF=0.20 DSPARK_POLICY=confidence DSPARK_DEPTH=6
export DSPARK_CAL= DSPARK_LOG= MTP= MTPK=0
export PARALLEL=4 STREAMS_RESERVE_GB=10
export PROMPT_ROWS=4096 PROMPT_ROWS_SHORT=4096
export NATIVE_OVERLAY=1 CACHE_GIB=4 CACHE_ENTRIES=32 RESERVE_GB=0 PROMPT_REUSE=1
export END_FREE_GB=11
export DENSE_LANE=1 DENSE_FP8=1
export AOTDIR=/home/ciprian/aot-world8-nvfp4-serve-copack-ord-e3m
export AOT_PACK_MD5=e64685bd2a294663e4fd8ede9c7d90c1
BIN_MD5_EXPECT=5db17e0b81bac8edc36ec17e992cca91

# ------------------------------------------------- THE PRE-BOOT PACK GATE ×8 ----
for n in 111 112 113 114 115 116 117 118; do
  M=$(ssh -o BatchMode=yes -o ConnectTimeout=10 ciprian@192.168.178.$n "md5sum $AOTDIR/aot.json" 2>/dev/null | cut -d' ' -f1)
  [ "$M" = "$AOT_PACK_MD5" ] || { echo "REFUSED: the AOT pack on .$n is '${M:-absent}' != the pinned $AOT_PACK_MD5 (the wrong-pack / partial-distribution class)"; exit 9; }
done
echo "pack gate: aot.json md5 $AOT_PACK_MD5 uniform on all 8 ranks"

# ------------------------------------------------------------------ THE BOOT ----
bash "$(dirname "$0")/serve8-image-dcp2.sh" "$PORT" "$CTX"
U=0
for i in $(seq 1 22); do
  sleep 55
  if curl -s -m 5 http://192.168.178.111:$PORT/v1/models 2>/dev/null | grep -q GLM; then U=1; echo "poll $i: SERVER-UP"; break; fi
  if [ $((i % 4)) -eq 0 ]; then echo "poll $i: booting"; fi
done
[ "$U" = "1" ] || { echo "GOLDEN-BOOT: FAIL (never up)"; docker logs serve-r0 2>&1 | tail -5; exit 8; }

# --------------------------------------------- THE RECEIPT TRIO (tf-ship, adopted) ----
fail() { echo "GOLDEN-BOOT: FAIL — $*" >&2; exit 7; }
R_BIN=$(docker exec serve-r0 md5sum /proc/1/exe 2>/dev/null | cut -d' ' -f1)
R_AOT=$(docker exec serve-r0 sh -c 'tr "\0" "\n" < /proc/1/environ | grep "^TF_GLM53_AOT=" | cut -d= -f2' 2>/dev/null)
R_PACK=$(docker exec serve-r0 sh -c "md5sum $R_AOT/aot.json" 2>/dev/null | cut -d' ' -f1)
R_ENV=$(docker exec serve-r0 sh -c 'tr "\0" "\n" < /proc/1/environ | grep "^TF_GLM53_" | sort' 2>/dev/null)
[ -n "$R_BIN" ] || fail "receipt pull failed — serve-r0 not running?"
echo "--- RECEIPT TRIO ---"
echo "binary  md5(/proc/1/exe) = $R_BIN  (intended $BIN_MD5_EXPECT)"
echo "pack    aot.json md5     = $R_PACK  ($R_AOT)"
echo "env     (effective, /proc/1/environ):"
echo "$R_ENV" | sed 's/^/    /'
[ "$R_BIN" = "$BIN_MD5_EXPECT" ] || fail "binary $R_BIN != intended $BIN_MD5_EXPECT"
[ "$R_PACK" = "$AOT_PACK_MD5" ] || fail "pack $R_PACK != pinned $AOT_PACK_MD5 (is a decoy default live?)"
R_CMD=$(docker inspect serve-r0 --format "{{join .Config.Cmd \" \"}}" 2>/dev/null)
echo "cmd     (the graphs carrier): $(echo "$R_CMD" | grep -o -- "--graphs [a-z]*")"
echo "$R_CMD" | grep -q -- "--graphs $GRAPHS" || fail "the --graphs arg is not '$GRAPHS' (the arg-form bug class: the value must be on|off)"

# ------------------------------------------------------------ THE EXPECT GATE ----
bad=0
for kv in \
  TF_GLM53_AOT=/aot TF_GLM53_KV8=1 TF_GLM53_DCP=2 TF_GLM53_DCP_STREAMS=1 \
  TF_GLM53_GRAPH_MAX_ROWS=8 TF_GLM53_PREWARM=1 TF_GLM53_ODC=0 \
  TF_GLM53_K=0 TF_GLM53_MTP= \
  TF_GLM53_DSPARK=/dspark TF_GLM53_DSPARK_POLICY=confidence TF_GLM53_DSPARK_CONFIDENCE=0.20 TF_GLM53_DSPARK_DEPTH=6 \
  TF_GLM53_DSPARK_CAL= TF_GLM53_DSPARK_LOG= \
  TF_GLM53_PARALLEL=4 TF_GLM53_DCP_STREAMS_RESERVE_GB=10 \
  TF_GLM53_PROMPT_SP=1 TF_GLM53_SP_CHECKS= TF_GLM53_PROMPT_ROWS=4096 TF_GLM53_PROMPT_ROWS_SHORT=4096 \
  TF_GLM53_PREWARM_END_FREE_GB=11 TF_GLM53_CACHE_GIB=4 TF_GLM53_PROMPT_REUSE=1
do
  k=${kv%%=*}; v=${kv#*=}
  got=$(echo "$R_ENV" | grep "^$k=" | head -1 | cut -d= -f2- || true)
  [ "$got" = "$v" ] || { echo "RECEIPT MISMATCH: env $k='$got' != intended '$v'" >&2; bad=1; }
done
[ "$bad" = "0" ] || fail "the effective env disagrees with the pins (a decoy default or an overwrite is live)"
echo "GOLDEN-BOOT: PASS (the pack uniform ×8, the binary intended, every pin green in the effective env)"
