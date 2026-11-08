# Changelog

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
