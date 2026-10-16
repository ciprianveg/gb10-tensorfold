#!/usr/bin/env bash
# glm53-preflight.sh — GLM-5.3 EXL3 TP8 mission preflight tripwire (Stage 0).
#
# PURPOSE: make it mechanically impossible to aim mission tooling at the WRONG
# cluster. The reserved cluster (192.168.177.11-18 / 192.168.178.11-18) runs the
# NVFP4 GLM-5.3 job and must NEVER be touched by this mission — no ssh, no
# checks, no commands. The target cluster is 192.168.177.111-118 /
# 192.168.178.111-118 ONLY.
#
# What it validates:
#   (a) every node in CLUSTER_NODES (and COPY_HOSTS when present) is inside
#       192.168.177.111-118 / 192.168.178.111-118;
#   (b) HARD-REJECTS (exit 1, loud) any target in the reserved ranges
#       192.168.177.11-18 / 192.168.178.11-18 — before any ssh would happen;
#   (c) the model pack directory exists with the 76 shards + index + configs
#       (local path configurable; --node mode checks the pack on a given
#       .111-.118 node over ssh, read-only).
#
# Node-list sources, in precedence order:
#   1. --nodes "ip,ip,..."            (explicit override)
#   2. $CLUSTER_NODES                 (poisoned-environment tripwire test hook)
#   3. CLUSTER_NODES= line in the env file (default: .env-glm53-tp8 next to
#      this script's parent repo; override with --config FILE)
#
# Usage:
#   ./glm53-preflight.sh                          # nodes + local pack check
#   ./glm53-preflight.sh --node 192.168.177.111   # remote pack check on one node
#   ./glm53-preflight.sh --config /path/to/env --pack /path/to/pack
#   ./glm53-preflight.sh --skip-pack              # node ranges only
#
# Exit codes: 0 = all checks pass; 1 = any check fails (reserved-range hits are
# loud and immediate).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE_DEFAULT="/home/ciprian/eugr/spark-vllm-docker/.env-glm53-tp8"
PACK_DEFAULT="/home/ciprian/models/models11/GLM-5.3-4.04bpw-h8-exl3"
NODE_PACK_DEFAULT='~/models111/GLM-5.3-4.04bpw-h8-exl3'
EXPECTED_SHARDS=76
REQUIRED_FILES=(model.safetensors.index.json config.json generation_config.json
                quantization_config.json tokenizer.json tokenizer_config.json)

say()  { printf '[glm53-preflight] %s\n' "$*"; }
die()  { printf '[glm53-preflight] *** %s ***\n' "$*" >&2; exit 1; }
warn() { printf '[glm53-preflight] WARN: %s\n' "$*" >&2; }

# --- IP classification (numeric octet math — NEVER string-prefix matching:
# --- .17 is reserved while .117 is a target; "192.168.177.1" is neither). -----

valid_ipv4() {  # ip -> 0 valid
  local ip=$1 o
  [[ "$ip" =~ ^([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})\.([0-9]{1,3})$ ]] || return 1
  for o in "${BASH_REMATCH[@]:1}"; do
    (( 10#$o <= 255 )) || return 1
  done
  return 0
}

ip_class() {  # ip -> echoes: reserved | target | foreign ; dies on malformed
  local ip=$1 a b c d
  if ! valid_ipv4 "$ip"; then
    die "malformed IP in node list: '$ip' — refusing (no ssh attempted)"
  fi
  IFS=. read -r a b c d <<<"$ip"
  a=$((10#$a)); b=$((10#$b)); c=$((10#$c)); d=$((10#$d))
  if (( (a == 192 && b == 168 && (c == 177 || c == 178)) )); then
    if (( d >= 11 && d <= 18 )); then
      echo reserved; return 0
    fi
    if (( d >= 111 && d <= 118 )); then
      echo target; return 0
    fi
  fi
  echo foreign; return 0
}

check_node_list() {  # label, comma-separated ips -> 0 ok, exits 1 on any violation
  local label=$1 raw=$2 ip cls any=0
  [ -n "$raw" ] || die "$label is empty — nothing to validate (misconfiguration?)"
  IFS=',' read -ra _ips <<<"$raw"
  for ip in "${_ips[@]}"; do
    ip="$(trim "$ip")"
    [ -n "$ip" ] || continue
    cls=$(ip_class "$ip")
    case "$cls" in
      reserved)
        printf '[glm53-preflight] *** HARD REJECT: %s contains %s — RESERVED for the NVFP4 GLM-5.3 job (192.168.177/178.11-18). NEVER ssh, check or run anything there. Fix the config before continuing. ***\n' "$label" "$ip" >&2
        exit 1
        ;;
      target)
        say "$label: $ip OK (target cluster .111-.118)"
        ;;
      foreign)
        warn "$label: $ip is OUTSIDE the target ranges 192.168.177.111-118 / 192.168.178.111-118"
        any=1
        ;;
    esac
  done
  (( any == 0 )) || die "$label has nodes outside the allowed .111-.118 target ranges"
}

trim() { printf '%s' "$1" | tr -d '[:space:]'; }

# --- env-file parsing (no sourcing — a poisoned file must not execute) --------

env_value() {  # file, key -> value (empty when absent)
  local file=$1 key=$2 line
  [ -r "$file" ] || return 0
  line=$(grep -E "^${key}=" "$file" | tail -1 | cut -d= -f2-) || return 0
  printf '%s' "$(trim "$line")"
}

# --- pack checks ---------------------------------------------------------------

check_pack_files() {  # pack-dir, expected-shards, shard-glob-dir (local) — via
                      # the same rules as the remote check; echoes problems.
  local dir=$1 want=$2 prob=""
  local shard of ofwant n
  [ -d "$dir" ] || { echo "pack dir missing: $dir"; return 1; }
  n=$(find "$dir" -maxdepth 1 -name 'model-*-of-*.safetensors' -type f | wc -l)
  if (( n != want )); then
    prob+="shard count $n != $want; "
  fi
  # every shard must declare the same "of-000NN" total, and it must equal want
  ofwant=$(find "$dir" -maxdepth 1 -name 'model-*-of-*.safetensors' -type f \
           | sed -n 's/.*-of-\([0-9]*\)\.safetensors$/\1/p' | sort -u)
  if [ "$(printf '%s\n' "$ofwant" | grep -c .)" -ne 1 ] || (( 10#$ofwant != want )); then
    prob+="inconsistent shard 'of-NNNNN' totals (got: $(echo $ofwant)); "
  fi
  local f
  for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -s "$dir/$f" ]; then
      prob+="missing/empty $f; "
    fi
  done
  if [ -n "$prob" ]; then
    echo "$prob"
    return 1
  fi
  return 0
}

check_pack_local() {  # pack-dir -> 0 ok
  local dir=$1 out
  say "checking local pack: $dir (expect ${EXPECTED_SHARDS} shards + index + configs)"
  if out=$(check_pack_files "$dir" "$EXPECTED_SHARDS"); then
    say "local pack OK: $dir"
  else
    die "local pack FAILED: $dir — $out"
  fi
}

check_pack_node() {  # node, pack-path -> 0 ok (read-only ssh; node already validated)
  local node=$1 path=$2 out rc
  say "checking pack on node $node: $path (read-only ssh)"
  if ! out=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$node" bash -s -- "$path" "$EXPECTED_SHARDS" <<'REMOTE'
set -eu
path=$1; want=$2
# expand a leading ~ to $HOME (quoted args do not tilde-expand remotely)
path=${path/#\~\//"$HOME"/}
prob=""
[ -d "$path" ] || { echo "pack dir missing: $path"; exit 1; }
n=$(find "$path" -maxdepth 1 -name 'model-*-of-*.safetensors' -type f | wc -l)
if [ "$n" -ne "$want" ]; then prob+="shard count $n != $want; "; fi
ofwant=$(find "$path" -maxdepth 1 -name 'model-*-of-*.safetensors' -type f \
         | sed -n 's/.*-of-\([0-9]*\)\.safetensors$/\1/p' | sort -u)
nof=$(printf '%s\n' "$ofwant" | grep -c . || true)
if [ "$nof" -ne 1 ] || [ "$((10#$ofwant))" -ne "$want" ]; then
  prob+="inconsistent shard 'of-NNNNN' totals (got: $ofwant); "
fi
for f in model.safetensors.index.json config.json generation_config.json \
         quantization_config.json tokenizer.json tokenizer_config.json; do
  [ -s "$path/$f" ] || prob+="missing/empty $f; "
done
if [ -n "$prob" ]; then echo "$prob"; exit 1; fi
echo OK
REMOTE
  ); then
    die "pack check on $node FAILED (ssh or remote check failed): $path"
  fi
  [ "$out" = "OK" ] || die "pack check on $node FAILED: $path — $out"
  say "pack on $node OK: $path"
}

# --- CLI ------------------------------------------------------------------------

CONFIG="$ENV_FILE_DEFAULT"
NODES_ARG=""
PACK="$PACK_DEFAULT"
NODE_MODE=""
NODE_PACK="$NODE_PACK_DEFAULT"
SKIP_PACK=0

while [ $# -gt 0 ]; do
  case "$1" in
    --config)    CONFIG="$2"; shift 2 ;;
    --nodes)     NODES_ARG="$2"; shift 2 ;;
    --pack)      PACK="$2"; shift 2 ;;
    --node)      NODE_MODE="$2"; shift 2 ;;
    --node-path) NODE_PACK="$2"; shift 2 ;;
    --skip-pack) SKIP_PACK=1; shift ;;
    -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

# Resolve the node list: --nodes > $CLUSTER_NODES > env file. Never source it.
if [ -n "$NODES_ARG" ]; then
  CLUSTER_NODES_RESOLVED="$NODES_ARG"
  CLUSTER_NODES_SRC="--nodes"
elif [ -n "${CLUSTER_NODES:-}" ]; then
  CLUSTER_NODES_RESOLVED="$CLUSTER_NODES"
  CLUSTER_NODES_SRC="\$CLUSTER_NODES (environment)"
else
  if [ ! -r "$CONFIG" ]; then
    die "no --nodes, no \$CLUSTER_NODES, and env file unreadable: $CONFIG"
  fi
  CLUSTER_NODES_RESOLVED=$(env_value "$CONFIG" CLUSTER_NODES)
  CLUSTER_NODES_SRC="$CONFIG"
  [ -n "$CLUSTER_NODES_RESOLVED" ] || die "no CLUSTER_NODES in $CONFIG"
fi
say "node list source: $CLUSTER_NODES_SRC"

# (a)+(b): validate CLUSTER_NODES — hard-reject reserved, require target-only.
check_node_list "CLUSTER_NODES" "$CLUSTER_NODES_RESOLVED"

# COPY_HOSTS gets ssh'd/scp'd by the copy tooling — same tripwire when present.
if [ -n "$NODES_ARG" ]; then
  COPY_HOSTS_RESOLVED="$NODES_ARG"
elif [ -n "${COPY_HOSTS:-}" ]; then
  COPY_HOSTS_RESOLVED="$COPY_HOSTS"
else
  COPY_HOSTS_RESOLVED=$(env_value "$CONFIG" COPY_HOSTS)
fi
if [ -n "$COPY_HOSTS_RESOLVED" ]; then
  check_node_list "COPY_HOSTS" "$COPY_HOSTS_RESOLVED"
else
  say "COPY_HOSTS not set — skipped"
fi

# --node mode: validate the requested node, then read-only remote pack check.
if [ -n "$NODE_MODE" ]; then
  cls=$(ip_class "$NODE_MODE")
  case "$cls" in
    target)   say "--node $NODE_MODE OK (target cluster)" ;;
    reserved) die "HARD REJECT: --node $NODE_MODE is RESERVED (192.168.177/178.11-18) — never ssh there" ;;
    *)        die "--node $NODE_MODE is outside the allowed .111-.118 target ranges" ;;
  esac
  check_pack_node "$NODE_MODE" "$NODE_PACK"
  say "NODE-MODE PREFLIGHT: PASS"
  exit 0
fi

# (c): local pack check.
if (( SKIP_PACK )); then
  say "--skip-pack: local pack check skipped"
else
  check_pack_local "$PACK"
fi

say "PREFLIGHT: PASS — nodes in .111-.118 target ranges only; pack verified."
