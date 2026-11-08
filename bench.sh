#!/usr/bin/env bash
# bench.sh — the gb10-tensorfold deploy verification (short, self-contained).
#
#   bash bench.sh [host] [port]     # default host = this machine's serve master, port 5901
#
# Three checks, in order:
#   1. /health          — the serve answers.
#   2. the speed probe  — a 1500-token single-HTML Snake game, greedy; reports tok/s.
#                         The speed tier's receipted class is ~44-49 tok/s (DCP1, 200K window);
#                         other tiers/contexts read lower — this is a sanity read, not a benchmark.
#   3. the determinism spot-check — the same Snake prompt and a fixed prose prompt, twice each:
#                         greedy output must be byte-identical across runs (the sha match).
set -u
HOST=${1:-192.168.178.111}
PORT=${2:-5901}
BASE="http://$HOST:$PORT"

SNAKE="Create a complete, playable Snake game in a single HTML file using JavaScript and the canvas API. Include score tracking, increasing speed, and game-over handling. Write the full code."
PROSE="Write a detailed, vivid short story about a lighthouse keeper who finds a message in a bottle during a storm. Use rich descriptive language throughout."

gen() { # gen <prompt> <max_tokens> -> sets OUT, TOK, WALL
  local body r t0 t1
  body=$(python3 -c 'import json,sys; print(json.dumps({"model":"glm53","messages":[{"role":"user","content":sys.argv[1]}],"max_tokens":int(sys.argv[2]),"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}))' "$1" "$2")
  t0=$(date +%s.%N)
  r=$(curl -s -m 900 "$BASE/v1/chat/completions" -H "Content-Type: application/json" -d "$body")
  t1=$(date +%s.%N)
  OUT=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' <<<"$r" 2>/dev/null) || { echo "FAIL: no completion — $r" | head -c 300; exit 3; }
  TOK=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["usage"]["completion_tokens"])' <<<"$r")
  WALL=$(python3 -c "print(f'{$t1-$t0:.1f}')")
}

echo "== 1/3 health"
curl -s -m 5 "$BASE/health" || curl -s -m 5 "$BASE/v1/models" | head -c 120; echo

echo "== 2/3 the speed probe (1500-token Snake, greedy)"
gen "$SNAKE" 1500
SHA1=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.read().encode()).hexdigest()[:12])' <<<"$OUT")
echo "   tokens=$TOK wall=${WALL}s -> $(python3 -c "print(f'{$TOK/$WALL:.1f}')") tok/s"

echo "== 3/3 the determinism spot-check (each prompt twice, greedy must repeat byte-identically)"
gen "$SNAKE" 1500
SHA2=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.read().encode()).hexdigest()[:12])' <<<"$OUT")
gen "$PROSE" 1024
P1=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.read().encode()).hexdigest()[:12])' <<<"$OUT")
gen "$PROSE" 1024
P2=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(sys.stdin.read().encode()).hexdigest()[:12])' <<<"$OUT")

echo "   snake: $SHA1 vs $SHA2 -> $([ "$SHA1" = "$SHA2" ] && echo IDENTICAL || echo MOVED)"
echo "   prose: $P1 vs $P2 -> $([ "$P1" = "$P2" ] && echo IDENTICAL || echo MOVED)"
if [ "$SHA1" = "$SHA2" ] && [ "$P1" = "$P2" ]; then
  echo "VERDICT: PASS (serving, speed read above, deterministic at greedy)"
else
  echo "VERDICT: FAIL (non-deterministic at greedy — capture the two outputs and the serve logs)"
fi
