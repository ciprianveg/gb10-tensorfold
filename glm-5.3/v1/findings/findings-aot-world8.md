# AOT world-8 capture — findings & receipts

Task: produce + validate the world-8 Triton AOT kernel pack (ctx 32,768, DCP=1) for
GLM-5.3-4.04bpw-h8-exl3 on the drowzeys r2 Zig 1.0.7 engine, staged for the first
world-8 boot on .111.

## PROGRESS LOG (append-only)

- [x] P0. Findings file created (this file).
- [x] P1. Capture-set determination written (§1) — ONE rank-0 capture, NOT 8 (evidence below).
- [x] P1b. Tooling read: oracle.py (469 ln), capture_aot.py (thin --record wrapper),
      sweep_aot.py (264 ln), run_phase1_node.sh (docker pattern), image/serve (gate),
      cuda_coverage.zig (enumerate + tileShapes), glm53_generate.zig (list-variants).
- [x] P2. Runner written: `aot-world8/run-aot-world8.sh` + `aot-world8/dump-tiles-w8.py` (head copy; scp to .111).
- [x] P3. .111: tiles-w8.json dumped from the reference at world 8 (§3.1).
- [x] P4. .111: oracle rank-0 world-8 capture DONE (§3.2).
- [x] P5. .111: list-variants needs enumerated (parallel 8) (§3.3).
- [x] P6. .111: sweep_aot merged pack assembled (§3.4).
- [x] P7. Validation: 4× list-variants against merged pack — 0 missing (§4).
- [x] P8. Staged: .111:~/aot-world8-32768-dcp1/ + head investigations/glm53-exl3-tp8/aot-world8/ (§5).
- [x] P9. MISSION-LOG.md journaled.

## §1 Capture-set determination

**Verdict: ONE capture — `oracle.py --record --world 8 --rank 0 --layers 4 --long 20000` —
plus a sweep of the engine's own enumeration. Eight per-rank captures are NOT needed at
DCP=1.**

Evidence:

1. The engine's launch universe is defined by `cuda_coverage.zig`'s `Space` struct
   (:42-69): world, heads, vocab_part, kva_n, max_rows, capacity, dcps, widths, gemms,
   mtp, draft, drafters, rows_windows, streams — **there is no rank field**. All ranks
   at the same world share one launch-key universe.
2. Per-rank Triton variants exist ONLY as DCP RANK constexprs:
   `cuda_coverage.zig:110-116` — `for (0..ranksOf(dcp)) |rank|` — at `dcp == 1`,
   `ranksOf(1) == 1` → rank 0 only, no per-rank variants enumerated. At world 8 with
   ctx 32,768 the serve gate itself picks DCP=1 (`image/serve:61`: `dcp=1; [ "$ctx" -gt 200000 ] && dcp=0`).
3. Per-rank shape differences at world 8 (headq block offset 0/32, draft-head last-rank
   +128, vocab_part 19360) are either runtime args (class-specialized: 1 / mult-16 /
   other) or non-Triton CUDA fatbin launches (headq linear, EXL3 experts/linear ext).
   The aot variant key is (function, num_warps/num_stages, named constexprs, arg
   specialization classes) — identical for all 8 ranks at DCP=1.
4. The capture itself only supplies TEMPLATES + the variants the reference happens to
   launch; every remaining enumerated need is compiled by `sweep_aot.py` via
   `JITFunction.warmup` (no launch, no memory touched; sweep_aot.py:217) and merged
   (union by hash, sweep_aot.py:240-260). Templates are shape-independent (they pin
   only unnamed constexprs, fp64 float values, num_stages defaults — sweep_aot.py:203-215),
   so even the world-8 oracle run's captured variants are mostly re-derived; what
   matters is that every function the needs reference has SOME captured variant.
5. `_gemm` (prompt GEMM, R > 128 rows) is never launched by the oracle (its long prefix
   uses 128-row chunks = max_window_rows), but the Zig launch names ALL its constexprs
   (`cuda_triton.zig:108`: K, N, BM, BK, GROUP, HAS_BIAS=false, SCALE) → the sweep
   compiles it purely from needs. The only template-less float is SCALE=HAD_SCALE
   (prefill.py:11, 1/sqrt(128)); `_gemm` uses it in fp32 arithmetic (`y * SCALE`,
   acc fp32 — prefill.py:41), so `f32_of_bits` double-rounding is bit-exact. Benign;
   sweep will print a WARNING for it — expected.
6. ROWS variants (`_index_scores`/`_kv_write`/`_ik_write`/`_absorb`/... with ROWS=true,
   cuda_triton.zig:251-284) are the SAME functions the oracle captures with ROWS=false →
   templates supply their exact fp64 float constexprs (WSCALE/QSCALE/SCALE).

**Pipeline** (mirrors how the published world-4 packs were built, sweep_aot.py docstring
:2-14 + run_phase3b gate):

```
(1) dump tiles-w8.json   — the reference's own world-8 (k,n) list (oracle.py:196-212 tiles_table)
(2) oracle --record      — rank 0, world 8, 4 layers, long 20000  → cap-r0/ (manifest.json + aot/)
(3) list-variants        — --world 8 --context 32768 --prompt-rows 8192 --window 128 --k 2 --dcp 1
                           --parallel 8 --aot cap-r0/aot --tiles tiles-w8.json → needs.json
(4) sweep_aot.py         — --needs needs.json --template cap-r0 --out sweep --merged aot-world8-32768-dcp1
(5) validate             — list-variants --aot merged: 0 missing, for (parallel,k) ∈ {1,8}×{2,0}
```

**Why a synthesized tiles table is exact**: `list-variants` requires tile tables
(glm53_generate.zig:1050+, `error.NoTileTables` — the prompt GEMM's (k,n) shapes come
from them), but `tileShapes` (cuda_coverage.zig:238-248) reads ONLY rows' `[k, n]`
(fields 0,1 of `[k, n, bits, codebook, layout, sk, wk]`), and the enumeration's launch
params come from `exl3.prefillTiles(kn[0], kn[1])` (cuda_coverage.zig:144) — the same
default tiler the table-less engine uses at serve. The published world-4 tables
(image /opt/keys-glm53/ref/tiles-r{0..3}.json, 575 linears each) are PER-RANK with
replication rules (row0 q_a [6144,2048] replicated; row1 kv_a [6144,640] replicated;
row2 q_b [2048,4096] = 16 heads×256 at world 4; row3 o_proj [4096,6144]) → blind halving
is WRONG. Ground truth = the Python reference loaded at world 8: `tiles_table(w)`
(oracle.py:196-212) iterates each layer's linears in canonical order (q_a, kv_a, q_b,
o_proj, mlp.gate/up/down, wq_b) — row 1 = kv_a, whose n feeds kva_n; consecutive
gate/up/down triples feed swigluWidths. Layers 0-3 cover every linear kind (dense 0-2,
MoE 3); the MTP layer is `load_layer(cfg, 78)` + bf16 eh_proj (weights.py:164-176) —
same EXL3 shapes as a dense layer; the draft head is bf16 lm_head rows (fused.py:70,
:529), not EXL3 — no GEMM shapes from it.

**Row-class coverage of the capture**: the oracle runs prompt24/decode1/decode3 (short),
128-row prefix windows to 20,000 tokens, decode1/decode3 at index bucket 32,768, plus
radix-select variants (oracle.py:332-407) → row classes {1, mult-16, other} and the
32,768 bucket exactly match the first boot's shape.

## §2 Environment (verified on .111)

- image: `ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-12-r2-mlx-serve` (loaded, 23.9 GB)
- PYSRC: `~/TensorFold-glm53-py` (image 2026-10-07 source, git 68e67776) → mounted at `/opt/tensorfold/src`
- r2 tree: `~/TensorFold-glm53-r2` (tools/glm53/oracle.py, sweep_aot.py; zig-out/bin/tf-glm53-generate)
- model: `~/models111/GLM-5.3-4.04bpw-h8-exl3` (sidecar wired, index 236,422 entries)
- host: 121 GB RAM (~115 available), 2.1T free disk. Docker pattern per run_phase1_node.sh:
  `--rm --network none --ipc host --gpus all --memory 64g --user $(id -u):$(id -g)`,
  env `PYTHONPATH=/opt/tensorfold/src TORCH_CUDA_ARCH_LIST=12.1`, caches outside /out.

## §3 Receipts

### 3.0 PYSRC version resolution (blocker found + fixed)
- First run failed: `oracle.py:189 r.release()` → `AttributeError: RankReader has no
  release` — the r2 tools expect a newer python than the staged 2026-10-07 PYSRC.
- Verified the 10-07 source carries the FULL r2 Triton ABI anyway: `_index_scores` with
  `DCP/RANK/PACK/BASE/ROWS` constexprs at fused.py:429-434; 75 DCP/ROWS/RANK sites in
  fused.py — identical count to the 10-09 branch (the last python glm branch; the zig
  port deleted the python family from 10-12 onward, and no pushed branch has
  `def release` — drowzeys's paired python was never pushed).
- `release()` is per-layer reader cleanup (memory hygiene); a no-op shim is safe for a
  4-layer capture. load_engine had already passed RankReader(model,0,8)/get/load_layer
  at world 8 when it hit release — the rest of the path works.
- Fix: `oracle-shim.py` (monkeypatch + delegate to oracle.main), used by the runner;
  same shim inlined in dump-tiles-w8.py. No edits to the r2 tree or PYSRC.

### 3.1 tiles-w8.json dump — DONE 14:5x → 15:11
- `dump-tiles-w8.py` in the image, world 8 rank 0, 4 layers: **31 linear rows**.
- First rows (q_a, kv_a, q_b, o_proj): `[[6144, 2048], [6144, 640], [2048, 2048],
  [2048, 6144]]` — q_a/kv_a REPLICATED at world 8 (full 2048/640), q_b = 8 heads × 256
  = 2048 (halved from world-4's 4096), o_proj k = 2048. Confirms the replication-rule
  analysis (§1): blind halving of the published world-4 table would have been WRONG.
- kva_n (enumeration's kv-cache rows) = row 1 [1] = 640 (replicated kv_a).
- First CUDA-extension builds cached under ~/aot-cache (reused by later steps).

### 3.2 oracle rank-0 world-8 capture
- Attempt 1 (15:11): FAILED at a Triton compile — `glue.py:343 router` → `_router_part`:
  `tl.dot(x, tl.trans(w), acc)` got **bf16 x vs fp16 w** (fused.py:1115, the indexer's
  `wk`). Root cause: our pack stores the "bf-named" plain tensors (wk, weights_proj,
  k_norm, norms, kv_b, router gate) as **fp16**; drowzeys's 2.75 pack stored them bf16,
  and the 10-07 python's `bf = lambda n: r.get(n, device)` (weights.py:140) is a
  PASSTHROUGH — the name only documents the expected dtype.
- The zig contract (what the captured cubins must match):
  - `_router_part`: X `*bf16`, W `*bf16` (cuda_triton.zig:77)
  - `_topk` BIAS: `*fp32` (cuda_triton.zig:85); router_bias held fp32
    (cuda_weights.zig:54, :367)
  - `_rmsnorm` W: `*bf16` (cuda_triton.zig:69); `_index_scores` W is an fp32 activation
  - loader: all bf-named tensors via `L.bf16` with an **exactness refusal** for
    fp16→bf16 (cuda_weights.zig:346-359); embed strictly bf16 (:657); kv_b → bf16
    blocks (:49-50). Our pack PASSED loader acceptance → every fp16→bf16 conversion
    the zig loader does is exact → a python cast is lossless and faithful.
- FIX (capture-side only, mirrors the zig loader exactly; backup
  `weights.py.pre-aot-patch` on .111):
  - `bf = lambda n: r.get(n, device).to(torch.bfloat16)`
  - router bias: `r.get(...).float()` (fp32, not bf16)
- Exactness note: the zig engine serves these tensors as bf16 (loader-cast); the
  capture now compiles the same `*bf16`-W variants the engine will request at serve —
  cubin/ABI consistent by construction. The fp32 router bias matches `_topk`'s
  `*fp32` BIAS arg.
- Attempt 2 (15:25, after the dtype patch): **PASS**.
  - `[oracle] 4 layers loaded in 13.3s` (55.9s first time: CUDA extension builds)
  - `[oracle] long prefix 20000 tokens in 3.8s`
  - `[oracle] aot: 41 variants of 15 functions; missing none` (all TRITON_KERNELS)
  - cap-r0/: manifest.json + aot/aot.json + 41 cubins + variants.txt

### 3.3 list-variants needs — DONE 15:25
- `variants: 518128 launches enumerated, 171 distinct needs (rows 1..8192, context
  32768, dcp { 1 }, 1 tile tables, 9 GEMM shapes, SwiGLU widths { 1536, 256 },
  concurrent windows up to 24 rows)`
  - SwiGLU widths {1536, 256} = dense 12288/8 + shared-expert 2048/8 at world 8 ✓
  - concurrent windows ≤ 24 rows = 8 streams × (k+1)=3 verify rows ✓
  - dcp {1} ✓ (ctx 32768 < 200000 → no DCP variants, as determined)
- vs cap-r0: **171 needs, 131 missing** → the sweep's input. Missing classes seen:
  `_gemm` prompt-GEMM shapes (R>128), ROWS variants (`_absorb`/`_expand` RBK=15),
  `_router_part` at the mult-16 row class with BM=16 (the oracle's windows hit BM=16
  only at M class 1), MTP 2D router, head router NE=19360 (vocab/8).
- NOTE: list-variants exits nonzero while variants are missing — expected at
  needs-generation time (the doc is written before the exit); runner fixed to only
  require the needs doc.

### 3.3 list-variants needs
(pending)

### 3.4 sweep + merge — DONE 15:32
- `[sweep] compiled 131/131 in 16 s; 0 failed; no template (defaults used) for:
  ['_cat2', '_gemm', '_group_sums', '_qrope']` — all four have every constexpr named
  by the zig launch (verified for `_gemm`: cuda_triton.zig:108 names K/N/BM/BK/GROUP/
  HAS_BIAS/SCALE; the others are the same class), so template-less warmup compiles
  them exactly from the needs; `_gemm`'s SCALE=HAD_SCALE double-rounding is bit-exact
  in fp32 arithmetic (§1.5).
- `[sweep] merged 154 variants of 19 functions from 2 sets -> /out/final`

## §4 Validation — PASS 15:32
`tf-glm53-generate --mode list-variants --aot <merged>` at world 8, ctx 32768,
prompt-rows 8192, window 128, dcp 1, tiles tiles-w8.json — **zero missing** for every
serving shape:

| parallel | k | needs | verdict |
|---|---|---|---|
| 1 | 2 | 147 | OK, no missing |
| 1 | 0 | 147 | OK, no missing |
| 8 | 2 | 171 | OK, no missing |
| 8 | 0 | 161 | OK, no missing |

(p=8 enumerates the one-stream windows too — 171 ⊇ 147 confirmed by both validating.)

## §5 Staging — DONE
- .111: `~/aot-world8-32768-dcp1/` — aot.json + cubins/ (154 cubins) + SHA256SUMS;
  155 files, 9.5 MB.
- head: `investigations/glm53-exl3-tp8/aot-world8/pack/` (rsynced, sha256 verified
  identical) + `receipts/` (tiles-w8.json, needs-p8.json/log, val-p{1,8}-k{2,0}.log,
  cap-r0 manifest.json + variants.txt, weights-py-aot-capture.patch).
- **aot.json sha256:
  0190e3841816ae610380e82b351d6c6af9f1ed42f1a1c49dd57b7d3372aa97f2**
- Serve wiring (first boot): `TF_GLM53_AOT=<pack dir>` (+ NO tiles table — table-less,
  exl3.plan defaults, which the enumeration used via exl3.prefillTiles). The image's
  serve-side coverage gate is world=4-only (image/serve:58) — at world 8 it will not
  re-check; §4's four validations are the world-8 gate.
- NOT in this pack (by design, first boot at DCP=1, no tree drafters): DCP RANK 1..7
  variants (ctx > 200000), DSpark/DFlash drafter kernels (would need --dspark/--dflash
  specs at enumeration).

## §6 Zig loader dtype patch (the first-boot UnexpectedTensor — boot 2 prerequisite)

The first world-8 boot failed at weights load with UnexpectedTensor — the zig mirror of
the capture-side dtype quirk (§3.2). Pack shard-header scan (all 76 shards, local NFS
copy) confirms **exactly six non-scale F16 classes** (suh/svh excluded,
correct-by-convention — untouched):

| count | tensor class | shape | loader site (patched file cuda_weights.zig) | cast |
|---|---|---|---|---|
| 79 | self_attn.kv_b_proj.weight | [28672,512] | kvB() — was strict .bf16 (:445 pre-patch) | fp16→bf16 round (bf16Bits per element in the wk/wv block split) |
| 76 | mlp.gate.weight | [256,6144] | router_w site — was L.bf16 (exact-only) | bf16Round |
| 22 | self_attn.indexer.wk.weight | [128,6144] | idx.wk site — was L.bf16 | bf16Round |
| 22 | self_attn.indexer.weights_proj.weight | [32,6144] | idx.weights_proj site — was L.bf16 | bf16Round |
| 22 | self_attn.indexer.k_norm.weight | [128] | idx.k_norm_w site — was L.bf16 | bf16Round |
| 22 | self_attn.indexer.k_norm.bias | [128] | idx.k_norm_b site — was L.bf16 | bf16Round |

Verified NOT needing a patch (same scan): e_score_correction_bias is **F32** [256]
(f32vec accepts as-is, :372); all layernorms + model.norm + embed_tokens are **BF16**
(strict paths stay strict); no other non-suh/svh F16 class exists in the pack; only
cuda_weights.zig loads checkpoint names (other .zig files use loaded handles).

The patch (diff: `investigations/glm53-exl3-tp8/zig-dtype-patch.diff`; backup
`cuda_weights.zig.bak` in the tree):
- new `bf16Round()` loader method: bf16 as stored; fp16 → round-to-nearest-even bf16 via
  the engine's own `bf16Bits` (:993-997 pre-patch — "torch's float -> bfloat16 ...
  round to nearest even"); other dtypes still UnexpectedTensor. The five bf-named
  sites above switched to it; `bf16()` itself is UNCHANGED (strict-exact policy kept
  for norms/final_norm/MTP — all bf16-stored in our pack, so they take the as-stored
  path anyway).
- `kvB()` now accepts .f16 and casts per element while splitting the wk/wv blocks
  (byte layout identical fp16/bf16 — only the element decode differs; the bf16Bits
  per-element precedent is the KVB_Q decode path).
- Loud once-per-class-per-process log at every cast site:
  `glm53: <class>: fp16 storage -> bf16 round cast (torch .to(bfloat16); the AOT
  capture compiled these values)` — the exactness-gate receipt.

**Consistency with the capture (the load-bearing invariant)**: the Python capture patch
(§3.2) used torch `.to(torch.bfloat16)` = round-to-nearest-even; the zig `bf16Bits` is
documented as exactly torch's conversion (:993). Both sides produce bit-identical bf16
values, so the served tensors equal the values the captured Triton cubins compiled
against — the AOT pack (§5) remains valid without recapture. Future exactness gates
must use these bf16-cast values (both sides).

**Build**: build-r2.sh (RAM watchdog 9 GiB, -j4, zig 0.17.0, nvcc 13.0 sm121) —
rc=0 all 3 stages (16:06-16:09): fatbins 71 s → native+glm53-lib+tf-glm53-generate
81 s → host tests (test-glm53, test-cuda-host) 11 s. Fresh artifacts:
zig-out/native/bin/tensorfold-native, zig-out/lib/libtensorfold_glm53.so,
zig-out/bin/tf-glm53-generate. NOT yet distributed (orchestrator's next step; the
running fleet still has the old engine — every rank must get the new native+so before
the relaunch, else the load fails exactly as before).
