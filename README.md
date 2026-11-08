# gb10-tensorfold

TensorFold serving GLM-5.3 (NVFP4) on a fleet of NVIDIA DGX Spark (GB10)
nodes — one image, one launcher, one verification script.

> **Status (2026-11-08).** The image is published and the registry
> manifests verified (`v1.0.0` / `latest`; digests in `PATCHES.md`). The
> one remaining deploy step — the fleet smoke test (`start.sh` +
> `bench.sh` on the nodes) — is **deferred while the cluster is in use
> for other tasks**; nothing below needs it to be true. When the cluster
> frees: `docker pull` on the master, `bash start.sh`, `bash bench.sh` —
> that is the whole verify. (A rebuild, if ever needed, is the
> `docker build` in `BUILD.md`, run on the GB10 host when free; the
> build-context binaries and their lane are documented there.)
>
> **Validation, as it stands.** Verified without the fleet: the registry
> manifests (all three tags resolve); the fresh-clone hygiene (no keys,
> no stray benchmark classes in the product files); the baked entry
> wrapper (the drafter pins and the exec line as documented). **Deferred
> — the serve-adjacent live runs only**: `start.sh`'s fleet boot and
> `bench.sh`'s probes (health, tok/s, determinism) need the cluster
> free; they are the user's one command away and are the only unverified
> claims in this repo. Every number quoted in the README and the
> benchmark record was measured on the running fleet — none was taken
> from this packaging window.

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

bash start.sh            # THE DEFAULT: the pool-P4 form — DCP1, 4 slots, a 600K shared KV pool
# ... the boot takes ~10-20 minutes (the model load dominates; watch:
#   ssh <master> 'docker logs serve-r0 2>&1 | grep -E "SERVE up|BIND|error" | tail'

bash bench.sh <master> 5901   # the deploy verification: health, tok/s, determinism
```

The serve speaks the OpenAI-compatible surface (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`) plus `/health`, on the port `start.sh` was
given (default 5901).

## The forms

The default is the **pool-P4 form**: DCP1, 4 concurrent slots, a 600K shared
KV pool, the 262144 graph bucket floor. **The pool is the behavior**: mixed-
length streams, first-fit claims, evict-oldest-idle, and an honest over-cap
refusal at the API — the serve stays alive. **The capacity is the pool
total** (600K aggregate): what the fleet can hold at once, claimed as the
streams actually need. `CONTEXT` (200K by default) is a single request's
claim bound — raise it toward `POOL_AGG` and one deep stream can claim the
whole pool. The knobs, env-overridable at the `start.sh` invocation:
`PARALLEL` (slots), `POOL_AGG` (the pool's aggregate; set it **empty** for
the uniform pre-division path — the slots divide the window evenly, no
pool), and `CONTEXT` (a request's claim bound).

| Form | Shape | Use |
|---|---|---|
| `pool` (default) | DCP1, 4 slots, 600K shared pool | the measured best today: 4 concurrent streams with per-request claims — c4 85.7/87.4/76.3 (code/structured/prose) |
| `capacity` | DCP2, 4 uniform slots, 200K each | the split-plane form — 4 uniform slots, no pool |

**The pool pins and the P4 form go together**: plain DCP1/P4 *without* the
pool refuses at the prewarm budget (139 graphs, 8.7 GiB needed vs 6.7 free —
measured 2026-11-08). For the 2-slot speed form, run `PARALLEL=2 POOL_AGG=
bash start.sh`.

```bash
bash start.sh capacity       # the DCP2 split form
bash start.sh stop           # the clean fleet teardown
```

Measured receipts for both forms and the deep-context quality gates
live in `campaigns/glm53-nvfp4-csf/BENCHMARK-RECORD.md`.

**About `docs/`**: that tree is the upstream TensorFold framework's
documentation, kept for provenance (PATCHES.md maps every delta from it).
Its EXL3 / EXLR8 GLM-5.3 recipes describe the *framework's* serving of
other weight formats on other hardware (two-Mac EXLR8, the Flash TR3
checkpoints) — **not this product**. This repo serves the GLM-5.3
**NVFP4-CSF** pack on the GB10 fleet; the NVFP4 path is the only one the
image, `start.sh`, and the receipts below describe.

## Settings (the knobs, their defaults, and why)

`start.sh` pins these; every one is env-overridable. The defaults are the
receipted production configuration — change one only with a reason.

| Knob | Default | Why |
|---|---|---|
| `DCP` | 1 (`pool`), 2 (`capacity`) | decode-context-parallel mode; DCP1 = the un-split KV plane (the pool form), DCP2 = the split plane (4 uniform slots) |
| `PARALLEL` | 4 | concurrent request slots |
| `CONTEXT` | 200000 | a single request's claim bound (the pool admits claims up to `POOL_AGG`; raise `CONTEXT` for deep single streams) |
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
