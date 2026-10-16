# findings-eval-harness.md — Phase 4 prep: the GLM-5.3 apples-to-apples eval harness

**Date:** 2026-10-16 (late) · **Lane:** fixer · **Status:** BUILT, deployed, NVFP4-smoked; ours-smoke pending the 900K serve

## What was built

A compact, no-heavy-deps (stdlib urllib only, python3.12-native) benchmark harness
comparing OUR GLM-5.3 EXL3 TensorFold TP8 serve against the user's NVFP4 GLM-5.3
vLLM serve. Lives in `investigations/glm53-exl3-tp8/eval/` (repo copy) and
`.111:~/eval/` (runnable copy — REQUIRED for ours, which binds 127.0.0.1:8080).

| File | Role |
|---|---|
| `eval_common.py` | Shared OpenAI-compatible client (streaming SSE + non-stream fallback), endpoint presets, pinned sampling, reasoning-field handling, content cleaning |
| `intell_eval.py` | 54 fixed questions: math 15 (GSM8K-style numeric check), logic 12 (verifiable word/number), factual MCQ 15 (letter match), coding 12 (fence-stripped code exec'd in a sandboxed namespace against deterministic unit tests, SIGALRM-bounded, whitelisted imports only). Per-category + total scores, TTFT, wall, tokens |
| `tool_bench.py` | Tools-API probe (honest verdict: supported / ignored / unsupported / endpoint-down) + 10 scenarios; checks name_ok, args_ok (valid JSON + expected key/values), followup_ok (second turn with canned tool result contains the expected fact) |
| `run_comparison.py` | Runs both evals on both endpoints — one subprocess per endpoint in parallel, sequential within; writes `results/COMPARISON.md` + `comparison_raw.json` + per-eval JSONs |

## The mirrored config (apples-to-apples, "as in the nvfp4 yaml")

Live NVFP4 serve identified as **`recipes/8x-spark-cluster/glm53-nvfp4-b12x-dcp2.yaml`**
— confirmed by `/v1/models`: id `GLM-5.3-NVFP4`, root `/root/models/models11/glm53-nvfp4-vision`
(exactly the b12x recipe's model path), max_model_len 450000, fingerprint
`vllm-0.30.1rc1.dev723+g6bbad6acd...-tp8`. Mirrored per-request settings, pinned
identically on BOTH endpoints:

- **temperature 0.0, top_p 1.0, seed 42** (the yaml pins `--seed 42` but no
  temperature; greedy temp-0 is our documented determinism choice for scoring runs)
- **Thinking ON for both**: ours by serve default (enable_thinking,
  `reasoning_content` separate); NVFP4 per its serve flags (`--reasoning-parser
  glm45`, `--default-chat-template-kwargs {"reasoning_effort":"high",
  "clear_thinking":false}`). Only `content` is scored, never the reasoning field.
- **Reasoning-field quirk found by probe:** this vLLM build emits reasoning in the
  **`reasoning`** field, NOT `reasoning_content` (both handled; inline `<think|thinking>`
  blocks also stripped defensively). `completion_tokens_details.reasoning_tokens`
  confirms thinking actually ran (e.g. 20 reasoning toks on the train-speed probe).
- **Tools:** NVFP4 has glm47 tool-call parser + auto-tool-choice per the yaml —
  probed WORKING (clean `tool_calls`, valid JSON args, follow-up turn OK).
- Same fixed question sets, same max_tokens per eval (intell 4000, tools 2500),
  sequential requests per endpoint, endpoints parallel.

## Verification done

- Offline self-tests: all answer checkers (number/fraction/currency, word, MCQ,
  code extraction) pass; all 12 coding reference solutions pass their embedded
  tests; infinite-loop code correctly times out in the sandbox; tool-args checker
  (subset match, numeric coercion, case-insensitive strings, extra keys tolerated,
  invalid JSON rejected) passes.
- **NVFP4 smoke from .111 (live): PASS** — intell 3/3 (m01 math, q05 MCQ, c04
  coding sandbox), tools probe "supported", t01+t03 full-pass (name/args/follow-up).
  Streaming works: avg TTFT 0.40 s, usage-based token counts.
- **Endpoint-down path:** verified honest (ours currently down → "endpoint-down:
  Connection refused" verdicts, run completes, COMPARISON.md marked N/A).
- Bug found & fixed during smoke: SIGALRM sandbox requires the main thread →
  run_comparison now runs each endpoint's evals as a **subprocess** (parallel per
  endpoint, sequential within), which also isolates crashes.

## Not done / pending

- **OURS smoke + tools-API verdict:** the .111 serve's 900K boot FAILED
  (`CacheDoesNotFit`; rank-0 log warns 90.8 GiB held by a stale occupant — the
  known §10/§12 class issue, boot lane's scope, not touched per no-GPU-work rule).
  Once serving: `ssh ciprian@192.168.178.111 'python3 ~/eval/run_comparison.py --smoke --only ours'`
- **Full comparison:** NOT run (final config not chosen — user triggers).

## The full-run command (for the user)

```bash
ssh ciprian@192.168.178.111 'python3 ~/eval/run_comparison.py'
# then bring results back to the repo:
scp -r ciprian@192.168.178.111:~/eval/results /home/ciprian/eugr/spark-vllm-docker/investigations/glm53-exl3-tp8/eval/
```

Tuning knobs: `--max-tokens 4000` (intell), `--tool-max-tokens 2500`,
`--timeout 900`, `--only ours|nvfp4`, `--smoke`. Estimated full-run wall:
~1-2 h per endpoint (54 thinking-ON questions + 10 tool scenarios ×2 turns),
endpoints in parallel.
