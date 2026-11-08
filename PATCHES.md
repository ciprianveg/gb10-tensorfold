# PATCHES.md — what this tree and image contain, against the upstream

The one-line-per-change provenance: the base, every code delta from it, and
the shipped image's manifest. The full engineering history (the 427-commit
lineage with the receipts) is the campaign's private record; this is its
public distillation. **The cherry-pick map at the bottom serves the
partial-feature contributor: what travels, how far, and by which path.**

## The upstream base

This tree builds on **TensorFold** (Ash Hart's upstream,
`github.com/ashhart/TensorFold`) at fork-point commit
**`76aafcf274e60a5e31a59ce145b2f71f7b366e9d`** ("native cuda: --drafter
reaches a family that loads its drafter itself"). Within the delta below,
the campaign also adopted upstream's **v1.2.1** `server/` and `core/`
wholesale (upstream tag `1db3edbe`) before layering its own changes.

## The change list (every code delta from the base, one line each)

The **Travels** column is the upstreamability tag:

- **ENGINE-PATTERN** — the fix *class* is engine-general (any multi-node
  family hits it), but the code is seated in our family's wire/CUDA lanes,
  which do not exist upstream. The *pattern* travels; the patch does not.
- **OURS-FAMILY** — family-specific feature work; travels as a
  directory-level pick from this tree (see the cherry-pick map).
- **CAMPAIGN-INTERNAL** — instruments and debug surface; keep or drop
  freely when grafting.

| # | Change | What it is | Travels |
|---|---|---|---|
| 1 | The v1.2.1 wholesale adoption | upstream's `server/` + `core/` at tag `1db3edbe`, taken whole as the working core | (upstream's own) |
| 2 | The `glm53r2` family adapter | ours, not upstream's: the GLM-5.3 lane's engine adapter (the multi-node serve wire, the multi-stream rounds, the fill lane, the CUDA copy-engine lanes) for GB10 | OURS-FAMILY |
| 3 | The pool plane | Mia's semantics implemented: max-not-equal-division packing, admission by actual usage, release-reuse, evict-oldest-idle — a shared KV pool with honest refusal at the API | OURS-FAMILY |
| 4 | The stale-view fix | the pool's views are rebuilt after every claim (they were built once at init over an empty ledger — the deep-context corruption root cause) | ENGINE-PATTERN |
| 5 | The admission-fatality fix | a request-class refusal at the wire's admit no longer kills the engine; the collective-class errors stay fatal | ENGINE-PATTERN |
| 6 | The idle heartbeat | the leader speaks a no-op on the wire every 240 s so idle followers don't hit their 600 s receive timeout and exit | ENGINE-PATTERN |
| 7 | The copy-engine fence | a stream sync between the host-to-device record upload and the gather that reads it (the cross-engine visibility class) | ENGINE-PATTERN |
| 8 | The resume/keeps machinery | the tail-recompute form: a grown conversation resumes from the kept prefix, re-computing the tail for exact bits (see README: known issues for its trade) | OURS-FAMILY |
| 9 | The bucket + null-rung graph trims | the decode-graph set bounded for the 600K envelope; the shallow capture-transition anomaly trimmed | ENGINE-PATTERN |
| 10 | The drafter config class | the adaptive dspark drafter pinned: depth 8, confidence floor 0.20, the MTP layers off (the receipted sweep's winners) | OURS-FAMILY |
| 11 | The libtfgrammar structured-output port | **in-tree, PARKED**: grammar-masked token picks end-to-end at the library level, pending one mid-round collective (see README: known issues). Not in the shipped binary. | CAMPAIGN-INTERNAL |
| 12 | The debug instruments | env-gated and off by default: the pool logging, the round dump, the profilers. The parked port's stage-mark logging (item 11) is unconditional but only reachable on the structured path — it retires when the port resumes. | CAMPAIGN-INTERNAL |

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

## The cherry-pick map — partial features, honestly

**The seat audit first, because it changes the mechanics.** A literal
`git cherry-pick` of our fixes onto Ash's base is **not possible for any
row above**: every ENGINE-PATTERN fix is implemented inside
`zig/src/families/glm53r2/` (the family's own serve wire and CUDA lanes),
and that directory does not exist upstream — the family is ours. Verified
against upstream's current main: no `V2Kind` (our wire protocol) anywhere
outside `glm53r2/`; no copy-engine lane in `core/`; and upstream's own
`zig/src/cluster/` subsystem (grown since our fork point) is a different
design — gossip membership with its own heartbeats (`beat/suspect/dead`)
and cluster admission — not a long-poll serve wire, so our fixes have no
seat there to patch. A "ready" cherry-pick branch against Ash's base would
be theater; this map is the honest artifact.

**Path A — the directory graft (the real partial-feature form).** Take
whole directories from THIS tree: `zig/src/families/glm53r2/` carries the
adapter, the pool plane, the resume machinery, and all four hardening
fixes already in it. It imports upstream's `core/` + `server/`
interfaces, so the graft onto an upstream checkout is the directory plus
the family registration in the build wiring. This is how the whole lane
travels.

**Path B — the pattern port (the fix classes).** Read the named commits
and re-implement the pattern at your own seat (`git show <sha>` on this
branch):

| Fix | The commits to read |
|---|---|
| The idle heartbeat (240 s no-op under the wire mutex; followers ignore-and-loop) | `1f5804a708`, the sleep-idiom follow-up `56444b5ca9` |
| The copy-engine fence (the upload→gather stream sync) | the fence hunk inside `0cb34fe47a` (it rides the port commit; the hunk is the `followerV2` upload/gather sync) |
| The admission-fatality fix (the request-class returns without breaking the world; the collective-class stays fatal) | `fbfbdd5bed` (the claim-path coherence gate) + `eb8de09481` (the leader's mirror) |
| The buffer-bound trims (the capture bucket bound; the null-rung trim) | `61f0604ca4` (the bucket trim), the null-rung refinement referenced by `391cc5665a` |
| The stale-view rebuild (the views rebuilt after every claim) | inside the pool-plane edits — `74dc68afb7` and the pool series it completes |

**Path C — the true upstream PR.** If a fix pattern is worth offering Ash:
branches belong on the contributor's **fork of upstream**
(`ciprianveg/TensorFold`) — branch off **Ash's current main**, one fix per
branch, the pattern re-implemented clean against upstream's own
`cluster/`/`server/` design (a port, priced as design work — not a pick),
re-committed with plain messages carrying no campaign content. After the
fork's one-click "Sync fork" the base sits server-side, so the
workflow-free commits push with a standard token and the PR opens from
there. **This repo (`gb10-tensorfold`) stays the product repo** — the
squash and the image; the fork is the contribution vehicle, and the two
never share refs.
