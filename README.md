# gb10-tensorfold

TensorFold serving GLM-5.3 (NVFP4) on a fleet of NVIDIA DGX Spark (GB10)
nodes — one image, one launcher, one verification script.

The engine is **TensorFold** (Ash Hart's upstream framework — the DSA
attention, the DCP cache planes, the draft/verify speculative spine this
repo's tree carries, plus this campaign's pool plane). The pool-semantics
design and the dense/head quantization patterns are **Mia**'s. See
`CREDITS.md`.

## What you need

- 8x DGX Spark (GB10) with docker, ssh access to every node.
- A RoCE fabric between them (one HCA per node; the launcher's NCCL envs
  name the common one-HCA form — adjust `HCAS`/`GID` and the NCCL vars in
  `start.sh` for your fabric).
- The model pack, the DSpark drafter pack, and the AOT cache pack, at the
  same paths on every node (see `BUILD.md` for the paths' roles; the
  defaults are `start.sh`'s `PACK` / `DSPARK_PACK` / `AOTDIR` variables).

## Quickstart

```bash
# on the master node (the first of your node list):
docker pull ghcr.io/ciprianveg/gb10-tensorfold:latest
# what's inside the image: PATCHES.md (the upstream base, every code
# delta from it, the layer manifest — read it before the first boot)

bash start.sh            # the speed tier: DCP1, 2 slots, 200K context
# ... the boot takes ~10-20 minutes (the model load dominates; watch:
#   ssh <master> 'docker logs serve-r0 2>&1 | grep -E "SERVE up|BIND|error" | tail'

bash bench.sh <master> 5901   # the deploy verification: health, tok/s, determinism
```

The serve speaks the OpenAI-compatible surface (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`) plus `/health`, on the port `start.sh` was
given (default 5901).

## The tier arms

| Arm | Form | Use |
|---|---|---|
| `speed` (default) | DCP1, 2 concurrent slots, 200K context | the fast tier — the best single- and double-stream latency/throughput form |
| `pool-600k` | DCP1, 4 slots, 600K aggregate pool (450K a request) | the 4-slot form at DCP1 — requests claim what they need; an over-capacity claim is refused at the API, the serve stays alive |
| `capacity` | DCP2, 4 slots, 200K context each | the split-plane form — 4 uniform slots |

```bash
bash start.sh pool-600k
bash start.sh capacity
bash start.sh stop        # the clean fleet teardown
```

Measured receipts for the `speed` tier and the deep-context quality gates
live in `campaigns/glm53-nvfp4-csf/BENCHMARK-RECORD.md`.

## Settings (the knobs, their defaults, and why)

`start.sh` pins these; every one is env-overridable. The defaults are the
receipted production configuration — change one only with a reason.

| Knob | Default | Why |
|---|---|---|
| `DCP` | 1 (`speed`), 2 (`capacity`) | decode-context-parallel mode; DCP1 = the un-split KV plane (fastest tiers), DCP2 = the split plane (4 uniform slots) |
| `PARALLEL` | 2 (`speed`), 4 (pool/capacity) | concurrent request slots |
| `CONTEXT` | 200000 (`speed`) | the KV window per slot |
| `POOL_AGG` | 600000 (`pool-600k`) | the shared KV pool aggregate — requests claim what they need, not a pre-division |
| `MIN_BUCKET` | 0; 262144 in `pool-600k` | the decode-graph capture's bucket floor; the deep-context capture bound |
| `TF_GLM53_DSPARK_CONFIDENCE` | 0.20 | the drafter's confidence floor — the receipted accept-rate/yield balance |
| `TF_GLM53_DSPARK_DEPTH` | 8 | the drafter's max depth — the sweep's winner (depth-8 with the MTP layers off) |
| `TF_GLM53_DSPARK_POLICY` | confidence | drafts accepted by per-token confidence |
| `TF_GLM53_K` | 0 | the MTP layers stay off — the dspark drafter owns speculation (K-layers were sha-identical, pure load) |
| `TF_GLM53_KV8_V3` | 1 | the 656B KV rows (the exact-bf16 rope + the latent block) |
| `TF_GLM53_GRAPHS` | 1 | captured decode graphs — eager is measurably slower |
| `TF_GLM53_PREWARM` | 1 | the graph capture runs at boot, not on first request |
| `TF_GLM53_GRAPH_MAX_ROWS` | 8 | the captured verify-window rows |
| `TF_GLM53_PREWARM_MB_GRAPH` | 20 | the prewarm's memory budget for capture |
| `TENSORFOLD_MEMORY_RESERVE_GIB` | 8 | the engine's reserve: the prewarm floor + the measured variance band |
| `TF_GLM53_PREWARM_END_FREE_GB` | 6 | the prewarm stops with this much free — the deep-form fit floor |
| `TF_GLM53_CACHE_GIB` | 4 | the prompt-reuse cache plane (multi-turn prefix reuse) |
| `TF_GLM53_PROMPT_REUSE` | 1 | the multi-turn cache on |
| `TF_GLM53_PROMPT_ROWS` / `_SHORT` | 4096 / 4096 | the prompt chunk rows |
| reasoning `effort` | `none` | the request-side reasoning-effort default; `none minimal low medium high xhigh max` all accepted, anything else is a clean 400 |
| `TF_GLM53_ROCE` / `HCAS` / `GID` | 1 / `rocep1s0f1` / 3 | the RoCE fabric wiring — adjust to your fabric |

## Known issues and limitations (honest, per the receipts)

- **Structured output is not enforced yet.** `response_format` is unsupported
  and `tool_choice: "required"`/named returns a clean 400 — the token-grammar
  lane is built but parked pending one mid-round collective. Agent clients
  must use `tool_choice: auto` with tightly-typed tool schemas; the model
  calls tools when appropriate and emits well-formed call JSON, but nothing
  constrains the output tokens yet.
- **Multi-turn resume trades small-turn latency for correctness.** A grown
  conversation re-computes the tail of the kept context at the resume (the
  exact-bits form); big turns get fast resume, small turns pay a visible
  latency price. The trade was ruled in favor of score and determinism.
- **Prose generation is entropy-bound around ~22-25 tok/s.** Not a defect:
  the drafter, the KV form, and the graph lane were each tested and
  exonerated; low-entropy prose simply accepts fewer drafts than code.
  Code generation reads ~2x prose.
- **Four uniform slots do not fit at DCP1.** The un-split plane's memory
  ceiling refuses 4 uniform DCP1 slots; use the `pool-600k` arm (requests
  claim what they need) or the DCP2 `capacity` arm for 4-stream serving.

## How the numbers were measured

Every row in `BENCHMARK-RECORD.md` carries its class (what was generated),
denominator (how many requests/streams/runs), and distribution (the spread
across runs). All speed and quality rows are **greedy, temperature 0**,
single-prompt classes unless stated: the code rows are 1500-token
single-HTML Snake games, the prose rows 1024-token stories, the concurrency
rows are N identical code streams started together. Determinism is a gate,
not a hope: the same prompt at greedy must be byte-identical across runs
and across concurrent-vs-sequential — `bench.sh` spot-checks exactly this
on every deploy, and the campaign's receipts held it across binaries.

## Notes that save hours

- **The AOT pack must exist on every node before boot** — `start.sh` checks;
  a missing pack mounts as an empty directory and fails ~10 minutes in,
  far from the cause.
- **The image must be identical on every node** — `start.sh` checks and
  auto-restores from the master; a stale image on one rank wedges the
  rendezvous with no error anywhere.
- **A boot that hangs past the load is usually a rank that never came up** —
  check `docker ps -a` on every peer before reading anything else.
- Greedy (temperature 0) is deterministic — `bench.sh`'s spot-check verifies
  it on every deploy.
- The drafter (`--drafts off`) is a different numeric lane — same model,
  different texts; don't compare outputs across it.

## Building from source

See `BUILD.md` — the zig lane, the sm_121 toolchain, the AOT pack rules, and
the base-image lineage.
