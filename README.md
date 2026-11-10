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

bash start.sh            # THE DEFAULT: the pool-P4 form — DCP1, 4 slots, a 600K shared KV pool, 400K a stream's max
# ... the boot takes ~10-20 minutes (the model load dominates; watch:
#   ssh <master> 'docker logs serve-r0 2>&1 | grep -E "SERVE up|BIND|error" | tail'

bash bench.sh <master> 5901   # the deploy verification: health, tok/s, determinism
```

The serve speaks the OpenAI-compatible surface (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`) plus `/health`, on the port `start.sh` was
given (default 5901).

## The forms

The default is the **pool-P4 form**: DCP1, 4 concurrent slots, a 600K shared
KV pool (400K a stream's max context), the 262144 graph bucket floor. **The pool is the behavior**: mixed-
length streams, first-fit claims, evict-oldest-idle, and an honest over-cap
refusal at the API — the serve stays alive. **The capacity is the pool
total** (600K aggregate): what the fleet can hold at once, claimed as the
streams actually need. `CONTEXT` (400K by default) is a single request's
claim bound — raise it toward `POOL_AGG` and one deep stream can claim the
whole pool. The knobs, env-overridable at the `start.sh` invocation:
`PARALLEL` (slots), `POOL_AGG` (the pool's aggregate; set it **empty** for
the uniform pre-division path — the slots divide the window evenly, no
pool), and `CONTEXT` (a request's claim bound).

| Form | Shape | Use |
|---|---|---|
| `pool` (default) | DCP1, 4 slots, 600K shared pool, 400K a stream's max | the measured best today: 4 concurrent streams with per-request claims — c4 85.7/87.4/76.3 (code/structured/prose) |
| `capacity` | DCP2, 4 slots — **1.0M total KV pool, 450K a stream's max context** | the split-plane form: the maximum-capacity pool — four streams claiming from 1.0M, each up to 450K deep |

**The pool pins and the P4 form go together**: plain DCP1/P4 *without* the
pool refuses at the prewarm budget (139 graphs, 8.7 GiB needed vs 6.7 free —
measured 2026-11-08). For the 2-slot speed form, run `PARALLEL=2 POOL_AGG=
bash start.sh`.

```bash
bash start.sh capacity       # the DCP2 maximum-capacity form: the 1.0M pool
bash start.sh stop           # the clean fleet teardown
```

## Structured output

`response_format: {"type": "json_schema"}` and `tool_choice: "required"` or
named are **supported**: the grammar library (libtfgrammar, xgrammar v0.2.8)
ships in the image and lazy-loads at the **first structured request** —
free-form serving never loads it, and free-form bits are byte-identical with
the lane armed (the boot's anchors gate receipts this). Token picks are
constrained per-position under the compiled schema, server-side; agent
clients get guaranteed-well-formed call JSON with `tool_choice: "required"`.

The honest cost: a masked round's token pick rides a host-side rendezvous
exchange across the ranks (sub-millisecond per token typical; a fully
masked long generation pays it per token, per stream). Free-form rounds pay
nothing.

## Performance (the pool-P4 form, same-day 2026-11-08, greedy)

Every number in this section and in the benchmark record was measured on a
live 8-node fleet — none is a projection; `bench.sh` reproduces the probes.

Decode, aggregate across concurrent streams (512-token generations,
2 runs a cell, best reported; per-stream in the record):

| streams | code | JSON-output* | prose |
|---|---|---|---|
| c1 | 47.5 | 53.1 | 27.6 |
| c2 | 66.5 | 77.1 | 49.8 |
| c4 | 85.7 | 87.4 | 76.3 |

\* **JSON-output, not JSON-constrained**: these are free-form generations
prompted to produce JSON — no grammar, no schema enforcement. The
structured-output lane (`response_format` with a json_schema) is supported
as of the v1.0.3 binary class, but these rows were measured free-form —
grammar-constrained serving contributes nothing to them.

The Snake-1500 probe (the record's own class): **49.0 tok/s**. Prefill,
two distinct uncached ~8K prompts at once: **1282 tok/s** aggregate.
Replies are deterministic at greedy (byte-identical repeats), and
4-at-once equals 1-at-a-time on the same requests.

**Link speed**: these rows were measured with the ranks connected at
**100 Gb/s** (one CX7 rail). At **200 Gb/s** (both rails) prefill runs
**~30% higher — 1600 tok/s** at the same 2×8K class; decode is
link-insensitive at these depths.

**The pool, concretely**: the fleet holds 600K tokens of KV at once,
claimed as streams need — one 400K conversation beside a 100K and a 90K,
or four growing together; a claim the pool cannot place yet is refused
honestly at the API and the serve stays alive.

Full rows — class, denominator, distribution, both forms, the quality
gates (the 128K-depth needle, the anchor sets) — in
[`campaigns/glm53-nvfp4-csf/BENCHMARK-RECORD.md`](campaigns/glm53-nvfp4-csf/BENCHMARK-RECORD.md).

**This repo carries only the product's docs**: this README (the quickstart,
the forms, the settings, the known issues), **the benchmark record**
([`campaigns/glm53-nvfp4-csf/BENCHMARK-RECORD.md`](campaigns/glm53-nvfp4-csf/BENCHMARK-RECORD.md)
— every measured row with its class and distribution), `PATCHES.md` (the
provenance: the upstream base and every code delta from it), `BUILD.md`,
the credits, and the third-party notices. The TensorFold *framework's* own
documentation — its format recipes, install runbook, and feature docs —
lives at the upstream repo (`asHHart/TensorFold`); nothing in this product
depends on it. This repo serves the GLM-5.3 **NVFP4-CSF** pack on the GB10
fleet, full stop.

## The engine work (beyond the TensorFold base)

The engine work this repo adds on top of TensorFold 1.2.1; every other
delta is listed one line at a time in `PATCHES.md`:

- **The glm53r2 native CUDA family** — the NVFP4-CSF pack loader, MLA/DSA
  latent attention, the sparse indexer, the KDA linear-attention layers:
  the native serving path for GLM-5.3.
- **DCP2 (decode-context-parallel 2)** — the decode-time split-plane KV
  form, the maximum-capacity mode (DCP1, the un-split plane, predates
  this work).

The **shared-KV pool** — max-not-division: requests claim what they
need (first-fit, evict-oldest-idle, an honest over-cap refusal at the
API); mixed-length streams share one aggregate (600K at DCP1/P4, 1.0M
at DCP2) — is **Mia's design** (see the credits), implemented here for
the glm53r2 engine.
- **KV8 v3** — the 656B KV rows: exact-bf16 rope plus the latent scale
  block.
- **DSpark** — the adaptive speculative drafter (confidence policy,
  conf 0.20, depth 8).
- **The fp8-e4m3 dense lane** — the four attention projections at
  128-block quantized-compute.
- **Resume/tail-recompute** — grown conversations resume without
  re-reading: 297K-token turns in ~4-5.6 s (was a 344 s re-read).
- The prompt cache + reuse, the AOT compile lane with the copack, and
  the RoCE fabric health checks.

## Settings (the knobs, their defaults, and why)

`start.sh` pins these; every one is env-overridable. The defaults are the
receipted production configuration — change one only with a reason.

| Knob | Default | Why |
|---|---|---|
| `DCP` | 1 (`pool`), 2 (`capacity`) | decode-context-parallel mode; DCP1 = the un-split KV plane (the pool form), DCP2 = the split plane (4 uniform slots) |
| `PARALLEL` | 4 | concurrent request slots |
| `CONTEXT` | 400000 (`pool`), 450000 (`capacity`) | a single request's claim bound (the pool admits claims up to `POOL_AGG`; raise `CONTEXT` for deep single streams) |
| `POOL_AGG` | 600000 (`pool`), unset (`capacity`) | the shared KV pool's aggregate — the fleet's total KV capacity, claimed as needed, not a pre-division |
| `MIN_BUCKET` | 262144 (`pool`), unset (`capacity`) | the decode-graph capture's bucket floor; the deep-context capture bound (the pool pins and the P4 form go together) |
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

- **Structured output costs a per-token exchange on masked rounds.** With
  `response_format` (json_schema) or `tool_choice: "required"`/named active,
  each masked round's token pick rides a host-side rendezvous exchange
  across the ranks (sub-ms per token typical; a fully-masked long generation
  pays it per token, per stream). Free-form rounds pay nothing.


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

- **The first deploy on a fresh fleet pays the image distribution** — about
  13 minutes end to end: `start.sh`'s image gate streams the 25.5 GB image
  from the master to each worker sequentially, then boots. This is a
  once-per-node cost (the image persists); subsequent boots skip straight
  to the launch.
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
