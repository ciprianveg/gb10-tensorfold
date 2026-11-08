# Changelog

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
