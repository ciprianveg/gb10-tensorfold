# PATCHES.md — what this tree and image contain, against the upstream

The one-line-per-change provenance: the base, every code delta from it, and
the shipped image's manifest. The full engineering history (the 427-commit
lineage with the receipts) is the campaign's private record; this is its
public distillation.

## The upstream base

This tree builds on **TensorFold** (Ash Hart's upstream,
`github.com/ashhart/TensorFold`) at fork-point commit
**`76aafcf274e60a5e31a59ce145b2f71f7b366e9d`** ("native cuda: --drafter
reaches a family that loads its drafter itself"). Within the delta below,
the campaign also adopted upstream's **v1.2.1** `server/` and `core/`
wholesale (upstream tag `1db3edbe`) before layering its own changes.

## The change list (every code delta from the base, one line each)

| # | Change | What it is |
|---|---|---|
| 1 | The v1.2.1 wholesale adoption | upstream's `server/` + `core/` at tag `1db3edbe`, taken whole as the working core |
| 2 | The `glm53r2` family adapter | ours, not upstream's: the GLM-5.3 lane's engine adapter (the multi-stream rounds, the fill lane, the serving surface) for GB10 |
| 3 | The pool plane | Mia's semantics implemented: max-not-equal-division packing, admission by actual usage, release-reuse, evict-oldest-idle — a shared KV pool with honest refusal at the API |
| 4 | The stale-view fix | the pool's views are rebuilt after every claim (they were built once at init over an empty ledger — the deep-context corruption root cause) |
| 5 | The admission-fatality fix | a request-class refusal at the wire's admit no longer kills the engine; the collective-class errors stay fatal |
| 6 | The idle heartbeat | the leader speaks a no-op on the wire every 240 s so idle followers don't hit their 600 s receive timeout and exit |
| 7 | The copy-engine fence | a stream sync between the host-to-device record upload and the gather that reads it (the cross-engine visibility class) |
| 8 | The resume/keeps machinery | the tail-recompute form: a grown conversation resumes from the kept prefix, re-computing the tail for exact bits (the multi-turn fast resume; see README: known issues for its trade) |
| 9 | The bucket + null-rung graph trims | the decode-graph set bounded for the 600K envelope; the shallow capture-transition anomaly trimmed |
| 10 | The drafter config class | the adaptive dspark drafter pinned: depth 8, confidence floor 0.20, the MTP layers off (the receipted sweep's winners) |
| 11 | The libtfgrammar structured-output port | **in-tree, PARKED**: grammar-masked token picks end-to-end at the library level, pending one mid-round collective (see README: known issues). Not in the shipped binary. |
| 12 | The debug instruments | env-gated and off by default: the pool logging, the round dump, the profilers. The parked port's stage-mark logging (item 11) is unconditional but only reachable on the structured path — it retires when the port resumes. |

## The image manifest (`ghcr.io/ciprianveg/gb10-tensorfold`)

| Layer | Content |
|---|---|
| The base container | the proven GB10 runtime: Ubuntu 24.04, the CUDA/NCCL/RoCE userspace, the python HTTP environment the serve containers have run in production (`base-glm53-fp8-final`) |
| `tensorfold-native` | the campaign's **(c)-form binary class** (`74039aa42dba…`) — the free-form-proven form: the base + items 1-10 above, WITHOUT the parked port. Built from this tree at the pre-port source state. |
| `libtfgrammar.so` | present, dlopen'd lazily beside the binary; the structured-output lane it serves is parked — free-form serving never loads it |
| `tensorfold-serve` | the entry wrapper (`deploy/tensorfold-serve`, tracked here) |

Path-mounted at boot (not in the image — see `start.sh`): the model pack
(`GLM-5.3-NVFP4-CSF`), the dspark drafter pack, and the AOT cache pack
(the captured decode graphs; `BUILD.md` explains why it must exist on every
node). The image is ~25 GB before those mounts; the mounts are the model
itself.
