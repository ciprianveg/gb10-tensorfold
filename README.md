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
