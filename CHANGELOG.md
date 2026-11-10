# Changelog

## v1.0.4 — the serve entry accepts --alias and --api-key

The image's entry wrapper now forwards `--alias` (extra model ids the
serve answers to, repeatable) and `--api-key` (the key clients must
present) to the native serve. Previously any unknown flag's value was
taken as the model path — passing `--api-key <key>` made the key the
pack path and the boot died on it.

## v1.0.3 — structured output landed

Image: `ghcr.io/ciprianveg/gb10-tensorfold:v1.0.3` (also `latest`),
manifest digest `sha256:09ae8af63afcff3e4627e9d3466c3b056906f3dc01668474c2a1ac87e7862a55`;
binary class `c5cef43b…`, grammar library `1a37c255…`.

**`response_format` (json_schema) and `tool_choice: "required"`/named are
supported.** The grammar lane is real and verified: the server half
(`response_format`/`guided_*` → the compiled Spec, the compiler handoff,
the tool_choice-required enforcement), the lane core (the grammar walk, the
per-round mask pool, the admit first-mask), and the RoCE-safe masked-pick
gather — the records over the rendezvous channel, host-reduced,
bit-identical on every rank; **no eager GPU-side collective mid-round**.
The pre-take protocol holds the pick and the commit to ONE matcher
transition. The library (libtfgrammar, xgrammar v0.2.8) ships in the image
and lazy-loads at the first structured request — **free-form serving never
loads it**, and free-form bits are byte-identical with the lane armed (the
anchors gate receipts this at every boot).

**The honest cost**: a masked round's token pick rides a host-side
rendezvous exchange across the ranks (sub-millisecond per token typical; a
fully-masked long generation pays it per token, per stream). Free-form
rounds pay nothing.

**The engine tree now corresponds to the shipped binary** (the `c5cef43b…`
class): the zig/ tree is the source the binary was built from — the grammar
lane (PATCHES items 11-14), both its fix cycles, and the inert fp8
experiment arms (default OFF; the standing ruling: the head and the shared
experts remain bf16).

## v1.0.2 — the pool-P4 default

**The pool is the default.** `start.sh` now boots the measured-best form
directly: DCP1, 4 slots, a 600K shared KV pool (200K a request's cap), the
262144 graph bucket floor — mixed-length streams, first-fit claims,
evict-oldest-idle, honest over-cap refusal at the API (the user's ruling,
2026-11-08; the era's receipts: the unequal four-for-four at the cap, the
over-cap honest evict-refit, the alive gate). `PARALLEL`, `POOL_AGG` and
`CONTEXT` are env-overridable knobs; an empty `POOL_AGG` selects the uniform
pre-division path. Note: plain DCP1/P4 *without* the pool refuses at the
prewarm budget (139 graphs, 8.7 GiB needed vs 6.7 free — the pool pins and
the P4 form go together). The benchmark record carries the same-day pool-P4
rows (c4 85.7/87.4/76.3 code/structured/prose — the rows the 2-slot form
cannot reach; Snake-1500 at 49.0; prefill 2×8K at 1282 aggregate; the
pool-form anchor set of record).

**The fp8 arms: experimental, default OFF, not advertised.** The head arm
(`TF_GLM53_HEAD_FP8`, the full-vocab head at fp8-e4m3 128-block
quantized-compute): implemented and quality-clean, but speed-null at the
record class (48.9 vs 48.7 armed/unarmed — the head's per-round weight read
is ~1% of the round). The dense-full arm (`TF_GLM53_DENSE_FP8_FULL`):
implemented and **found inert in the A/B gates** — a no-op, not a feature
(the shared experts' compute lives inside the fused MoE block, not the
packed dense linears). The user's standing ruling: **the head and the
shared experts remain bf16.**

## v1.0.1 — the guide test's harvest

The first live deploy-through-the-README found and fixed a boot-fatal entry
bug: the image's `tensorfold-serve` wrapper exec'd the binary with the old
positional CLI (`serve PACK PORT CONTEXT`) while the binary speaks the
argparse form — every rank refused at launch. The wrapper now execs the
binary's actual CLI (`serve PACK --host 0.0.0.0 --port --context
--parallel`, the form the production overlay wrapper always used).
`start.sh` also gained the pre-boot stale-container clean (a re-boot after
a failed or prior boot died on the container-name conflicts), and the
README now states the first-deploy cost honestly: ~13 minutes for the
image distribution to a fresh fleet (the master streams the 25.5 GB image
to each worker; a once-per-image, once-per-node cost) — plus the topology
note that only the master pulls from the registry; the workers receive the
image from the master by design.

## v1.0.0 — the glm53-nvfp4-csf-final release

The GLM-5.3 (NVFP4) TensorFold campaign's final state, packaged for deploy.

**The engine** (this tree): the TensorFold GLM-5.3 lane with the campaign's
pool plane — Mia's pool semantics implemented (max-not-equal-division,
admission by actual usage, evict-oldest-idle) — and the six-defect
root-cause arc fixed and receipted (the stale-view line, the wire-admit
fatality, the null-rung and bucket trims, the dcp>1 gate, the claim-path
rank divergence). The libtfgrammar structured-output port is in-tree and
parked (see README: known issues).

**The image** (`ghcr.io/ciprianveg/gb10-tensorfold:v1.0.0`): the final
binary class baked with the grammar library over the proven GB10 runtime
base — pull-and-boot, the model/drafter/AOT packs path-mounted.

**The measured receipts** (`campaigns/glm53-nvfp4-csf/BENCHMARK-RECORD.md`):
the DCP1-best rows — code 47.31/49.08 tok/s, prose 25.50, the JSON ceiling
57.2, the 2-stream aggregate 70.89, prefill 1181 — with the deep-context
quality gates (needles pristine to 360K+, greedy determinism byte-identical).

**The packaging**: `start.sh` (the 8-node launcher, three tier arms, the
functional gates), `bench.sh` (the deploy verification), `BUILD.md` (the
source lane and the gotchas), this README.

Credits: `CREDITS.md` — Ash Hart (the upstream TensorFold framework) and
Mia (the pool-semantics design and the dense/head quantization patterns).
