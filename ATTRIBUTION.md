# Attribution & Credits

This project builds on the following open-source works. Full credit to their authors.

## Foundational Work

**ashhart/TensorFold** is the engine this whole repo runs on — the Zig CUDA server, the
family architecture, the AOT kernel-pack system, and the drafting and DCP machinery we
built on. Full credit to **ashhart** as TensorFold's creator and lead author.

**drowzeys/TensorFold** (the r2 fork) pioneered GLM-5.3 serving on TensorFold for DGX
Spark. This project builds directly on their work:

- The whole GLM-5.3 CUDA port (the `glm_moe_dsa` family) and its EXL3 weight path
- The TP4 recipe repo and bench methodology (our bench conditions mirror theirs for
  comparability)
- The published serve image (`ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark`) —
  the runtime base of our `ghcr.io/ciprianveg/gb10-tensorfold-glm53:v1` image, which
  adds our built engine and AOT kernel pack on top
- The DSpark **ft2 fine-tune** of the RedHatAI speculator
- The nemotron in-tree fp8 precedent inside TensorFold

Full credit to **drowzeys**. See [drowzeys/TensorFold](https://github.com/drowzeys/TensorFold)
for the r2 lineage.

**ajensenwaud/MiMo-V2.6-2x-DGX-Spark-TensorFold** — the MiMo Flash port's kv8 design
(e4m3 + fp32 scale per token-head, quant-on-write / dequant-on-read, roundtrip quality
gates) was the design reference for our 8-bit KV cache.

## What is ours (the v1 campaign contributions)

Built on top of the above, in this repo's engine branch (`glm53-tp8-2026-10-16`) and the
published image:

- **World-8 generalization** — the first TP8 GLM-5.3 serve on eight DGX Sparks: the
  multi-rank orchestration, RoCE one-shot collectives verified bit-equal, the page-cache
  load discipline for NFS-served packs (the continuous `drop_caches` technique for the
  rank-0/NFS-server box)
- **The world-8 AOT kernel pack** — captured and validated (154 variants / 19 functions,
  zero-missing across all windows)
- **The dtype-round-cast loader patch** — our pack stores six bf-named tensors that the
  stock loader mis-typed; the patch round-casts them on load
- **The bf16 drafting sidecars** — the pack lacks bf16 `eh_proj.weight` (MTP load) and
  bf16 `lm_head.weight` (all drafters); we built the 2.05 GB sidecar from the pack's own
  EXL3 tensors via the engine's reference decoder, wired in index-driven with no engine
  patch
- **kv8** — the 8-bit fp8 KV cache for `glm_moe_dsa` (staged native quant/dequant around
  unchanged Triton kernels, AOT-pack-safe; two root-caused bugs fixed: the RoPE angle
  zeroing in staging, the dequant block-vs-flat layout)
- **The DSpark AOT pack extension** — the drafter capture chain (static list-variants
  cannot see drafter launches), the needs→sweep→validate→stage pipeline, the 164-cubin pack
- **Adaptive frows** — the per-round draft cap that follows the round's live stream count
  (`depthOf(which, n_live)`): one serve config optimal at every concurrency, c1 closed to
  the p1 profile; plus the concurrent draft mode tuning (CUT_STREAMS)
- **The 1M-context serve** — DCP=1 pinning, the 900K/1M ceilings, needle found at 845,962
  tokens twice
- **The published fast-start image** — `ghcr.io/ciprianveg/gb10-tensorfold-glm53:v1`:
  the drowzeys runtime base + our built engine + the dspark-extended AOT pack, baked with
  the exact cluster-validated layout
- **The benches and receipts** — the drafter matrix (MTP k2–k5 + both DSpark drafters),
  the MTP-3/4/5 net-loss verdicts, the concurrent ladders, the long-context suite, the
  tools-API bench, and the apples-to-apples speed comparison against the equivalent
  NVFP4 vLLM serve
- **Tooling** — spark-tools improvements (exl3 inspect JSON cap, source-manifest
  Zig-layout support), the `glm53-preflight.sh` cluster tripwire

The implementation choices and bugs are ours; the borrowed designs are credited where used.
