# Pinned r2 (EXL3/DCP) snapshot — the source this family ports from

**Pin:** r2 `70e68f665aa7fadb5956b21ff8c137365cec91a3` (`70e68f66`), working tree **dirty (40 files)**
at snapshot time — these copies ARE the pins (the working tree moves under the DCP2 lane).

| file | why | md5 |
|---|---|---|
| `cuda_dcp.zig` | sharding math + exchanges → our `dcp.zig` | see `dcp-sync-pins.txt` |
| `cuda_forward.zig` | `attentionDcp`, `selectDcp`, `State.initDcp` → the open port | see pins |
| `cuda_kernels.zig` | the launch wrappers (kv8, fp8 scorer, top-k) → our `kv8.zig`/`indexer.zig` | see pins |

Copies are byte-identical to r2's working tree at the pin (no edits here — do not edit them; re-copy
on a re-sync). The standing rule (SEAM-DCP2.md §5): every later r2 DCP2 improvement is re-ported by
re-pinning and re-applying, never silently diverged. `../../../../../../spark-vllm-docker`'s
`investigations/.../dcp-sync-check.sh` reports drift in both directions.

Snapshot 2026-10-20 (round 3 re-pin; r2's working tree was dirty: 41 files). `glm53_decode.cu` joined
the set here as the source of the compact radix select (ROUND 3, findings-dcp2-build §I), which is ported
(adapted) into our `glm53_select.cu`. Everything in this directory is byte-identical to r2's working tree
at the pin; do not edit — re-copy on a re-sync.
