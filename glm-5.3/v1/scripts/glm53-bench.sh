#!/bin/bash
# glm53-bench.sh — category bench for the world-8 serve (drowzeys-comparable conditions:
# thinking ON, greedy, 32K ctx). Categories: prose + code. Shapes: c1 (solo) + c8 (8 concurrent).
set -u
EP=http://127.0.0.1:8080/v1/chat/completions
PROSE=("Explain the difference between tensor parallelism and pipeline parallelism in LLM inference."
"Write a clear paragraph describing how speculative decoding speeds up autoregressive generation."
"Summarize the tradeoffs between quantization precision and inference throughput."
"Describe how RDMA over Converged Ethernet differs from TCP for collective communication.")
CODE=("Write a Python function implementing binary search with type hints and a docstring."
"Write a CUDA kernel that performs an element-wise ReLU on a float32 tensor."
"Refactor this loop into a list comprehension: result=[];for x in items: result.append(x*2)"
"Write a bash script that finds the three largest files under a directory recursively.")
req() { # $1=prompt
  curl -s $EP -H "Content-Type: application/json" \
    -d "{\"model\":\"glm53-w8\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"temperature\":0,\"max_tokens\":256}" \
  | python3 -c "import json,sys; r=json.load(sys.stdin); t=r['tensorfold']; s=r.get('speculative',{}); print(f\"{t['tokens_per_second']:.1f}|{s.get('acceptance_rate',0):.2f}|{s.get('tokens_per_round',0):.2f}|{t['time_to_first_token']:.2f}|{r['usage']['completion_tokens']}\")"
}
conc() { # $1=prompt $2=tag
  req "$1" | sed "s/^/$2|/"
}
echo "=== C1 PROSE (solo, mean of 4)"
for p in "${PROSE[@]}"; do req "$p" | sed 's/^/prose|/'; done
echo "=== C1 CODE (solo, mean of 4)"
for p in "${CODE[@]}"; do req "$p" | sed 's/^/code|/'; done
echo "=== C8 PROSE (8 concurrent)"
for p in "${PROSE[@]}" "${PROSE[@]}"; do conc "$p" c8prose & done; wait
echo "=== C8 CODE (8 concurrent)"
for p in "${CODE[@]}" "${CODE[@]}"; do conc "$p" c8code & done; wait
