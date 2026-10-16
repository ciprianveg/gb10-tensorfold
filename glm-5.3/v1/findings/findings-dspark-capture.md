# DSpark drafter AOT capture — the missing drafter kernels

Task: make the AOT oracle capture exercise the DSpark drafter so its Triton
kernels land in the pack. Host-side tooling only (no GPU work done here); the
real capture re-run happens on .111 (orchestrator-coordinated).

## §1 The problem

The world-8 GLM-5.3 serve (Zig engine, AOT-only for Triton) boots with the
DSpark drafter (`glm53-w8-dspark-boot.sh`: `TF_GLM53_K=0`, MTP unloaded,
`TF_GLM53_DSPARK=/root/models/models11/GLM-5.3-dspark`,
`TF_GLM53_DSPARK_POLICY=confidence`, `TF_GLM53_DSPARK_CONFIDENCE=0.3`,
parallel 1) and dies at runtime:

```
error: no captured Triton variant of _group_sums for this launch (warps 2, stages null):
       X: *bf16, XS: *fp32, x_stride: i32 30720
```

`_group_sums` is `glm5_next/cuda/qmm.py:106` — the 64-input group sums the
drafter's 4-bit matmuls need. `x_stride 30720` = the drafter's tap input width
(5 tap layers x hidden 6144): the launch is `SparkDrafter._taps_compute`'s
`qmm.matmul(self.tap_in[:n], self.fc)` (dspark.py:280, `_taps_compute`). The
target never uses qmm — the kernel can only come from the drafter.

The capture pipeline (`~/aot-work/run-aot-dspark-ext.sh` on .111, phase
"capture") ran `oracle.py --record --layers 4 --world 8 --rank 0 --long 20000`
with the dspark env set and completed "41 variants of 15 functions; missing
none" — but recorded NO drafter kernels: oracle.py drives explicit target
window chains only; the speculative draft/verify loop never runs, so a drafter
loaded via env never launches anything. The static needs enumeration cannot
see drafter launches either (list-variants with the drafter env: 147 needs,
0 missing — yet the boot dies), so sweep_aot cannot close the gap: **the only
way in is a capture that actually runs the drafter.**

## §2 The serve's draft step (what the capture must mirror)

Python reference (the Zig engine ports these exactly):

- **Context (taps)**: `runner.py:639 _add_taps` -> `add_taps`
  (dflash.py:85 `TapColumns.add_taps`): decode rounds chunk at the drafter's
  block (kept rows 1..9 -> chunks of <= 8); prompt chunks feed whole windows,
  chunked by `add_taps` at `tap_in`'s **64 rows**. The Zig runner
  (cuda_runner.zig:606-620) says the same: "decode: a block, each pass a
  captured graph; prompts: tap_rows (64), eager past a block".
- **Draft**: `runner.py:879` `dr.propose(tok, room, sampling, conf, **kw)` ->
  `SparkDrafter.propose` (dspark.py:328) -> `candidates` (dspark.py:312) ->
  `_block_compute` (dspark.py:290): ONE block pass of `block` = 8 rows
  (embed -> 3 draft layers -> norm -> head top-k -> gather -> confidence).
- **Verify**: `runner.py:885` `self._verify(R, P, ...)` — the target verifies
  [pending, drafts] in one window; the runner captures verify graphs at every
  width in `runner.widths` = 1..block+1 (= 9) plus the copy widths {8, 16}
  (copy drafts default ON, `copies.py` TF_GLM53_COPY_DRAFTS default "1").

The authoritative launch universe is the Zig drafter's own static enumerator,
`cuda_draft.zig:1218 enumerate` (the "probe mode" behind cuda_coverage):

```zig
for (1..tap_rows + 1) |n| try dr.tapsCompute(dv, n);   // tap passes, n = 1..64
try dr.blockCompute(dv);                                // the block pass
```

DSpark's Triton kernels (cuda_draft.zig:19-21, cross-checked against Python):
`_group_sums` (x_stride 30720 fc / 6144 kv / 512 o), `_prep_kernel`
(dflash2.py:51; rows is a specializeable arg -> the n=1 / div-16 / generic
classes), `_block_attn_kernel` (dspark.py:58, N=block constexpr), glue
`_rmsnorm` (SUMS), `_embed_b16`, `_swiglu`, plus `_residual_add` (see §5).
The 4-bit matmuls themselves (`qmm.matmul` -> the shared qmm CUDA extension)
and the head/top-k/confidence steps are NATIVE, not AOT — the placeholder
head's shape sets no variant.

**Depth does not affect launch shapes** (task item 3): the block pass always
computes all 8 slots (`_block_attn` N=block, head rows = block); the depth /
confidence cut only slices the candidate array on the host
(`candidates()`'s `[:, :depth]`, dspark.py:325) and the Markov chain's walk
length. One block pass records the drafter's full kernel set. The confidence
policy's effect is only the NUMBER of drafts -> the verify window's width,
which is why the capture runs verify windows at every width instead.

## §3 The oracle.py extension (the design)

`tools/glm53/oracle.py` (r2 tree; backup `oracle.py.bak`, diff
`oracle-dspark.diff`): a third chain, gated on `TF_GLM53_DSPARK` so the
default capture is byte-identical to before. After the long chain, inside the
existing `torch.no_grad()` / Recorder phases:

1. **`dspark-load`**: `DSparkConfig.read` + `check(world)` +
   `tap_plan([None, tap_layers])` -> `w.tap_slot` = {1:0, 19:1, 38:2, 57:3,
   74:4} (engine.py:186's exact wiring; with only 4 layers loaded, layer 1
   writes real taps, the rest stay zero — fine, the pack needs shapes, not
   bits), `settings`/`quant` from the env (the serve's confidence 0.3),
   `headq.prepare(w, drafts=True)` (the serve's q4 draft head; k=0 -> the full
   share; native qmm -> no variant depends on the head's shape), then
   `SparkDrafter(spath, w, capacity=2080, ...)` — engine.py:216's call shape.
2. **`dspark-taps`**: `add_taps` at EVERY chunk size n = 1..64 (zeros as tap
   values), the context growing 1+2+...+64 = 2080 positions monotonically —
   mirroring `cuda_draft.zig enumerate` and covering every specializeable
   rows class (1 / div-16 / generic) for `_group_sums`, `_rmsnorm`,
   `_prep_kernel` at both x_strides (30720, 6144).
3. **`dspark-block`**: one `propose(pending, block, None, confidence,
   policy=...)` — the serve's draft step; records `_embed_b16`,
   `_block_attn_kernel` (N=8, MP=16), `_prep_kernel` (H=8), `_group_sums`
   (x_stride 512), `_swiglu`, `_rmsnorm`.
4. **`dspark-verify{R}`**: target verify windows at every width the runner
   captures — R in (16, 1..9), the copy width first so the 61 rows fit the
   long chain state's P+64 capacity — with `Buffers(w, 16, 32768,
   decode=True)`, exactly the serve's verify buffers (wrows=16), at positions
   past index_topk (T=32768 bucket). Insurance: R=2..9 are the same
   specialization classes as the already-captured R=3, and R=16 was already
   exercised by the copy-drafting MTP serve — but a window is cheap and the
   widths are exactly what the serve replays.
5. **meta.json**: `meta["chains"]["dspark"]` (block, tap layers, quant,
   settings, tap pass sizes, drafts, window table) + a loud
   `[oracle] dspark chain: ...` receipt line so the re-run log shows the
   drafter ran (the previous capture's only symptom was the absence of any
   dspark mention).
6. **pack()**: new `dspark` flag; when set, the missing-kernel gate also
   demands the drafter's own kernel names (`DSPARK_KERNELS`:
   `_group_sums`, `_prep_kernel`, `_block_attn_kernel`, `_embed_b16`,
   `_swiglu`) — a dspark capture that somehow skipped the drafter now FAILS
   instead of packing silently.

Not changed: the short/long chains, fixtures (the dspark windows run with
`taps.prefix = None` and write only scratch positions >= P of the long state;
all fixture clones are taken from [:P]), and any file under
`zig/src/families/glm_moe_dsa/**` or `zig/kernels/**` (untouched, read-only).

## §4 Host verification done (no GPU)

- `python3 -m py_compile oracle.py` — OK (host Python 3.12).
- `oracle.py --help` / no-args error path — OK (argument paths unchanged).
- Dry-run of the whole config/argument path via the torch-free
  `dspark_host` (uv venv + numpy on the host), against the REAL checkpoint
  (`~/models/models11/GLM-5.3-dspark`) and the REAL target config
  (`~/models/models111/GLM-5.3-4.04bpw-h8-exl3`):
  - `DSparkConfig.read`: hidden 6144, heads 64, hd 64, inter 12288, 3 layers,
    block 8, window 2048, causal; tap_layers (1, 19, 38, 57, 74).
  - `check(world=8)`: OK (heads/inter split evenly, 1536 % 64 == 0).
  - `tap_plan([None, tap_layers])`: slot {1:0, 19:1, 38:2, 57:3, 74:4},
    cols [None, None] — DSpark alone takes the one-copy `add_taps` path.
  - `settings`/`quant` with the serve's env: depth 8, policy confidence,
    confidence 0.3, top_k 64, q4.
  - arithmetic: tap capacity 2080; verify rows 16+45=61 from P=20000 end at
    20061 <= the state's 20064 capacity.
- Signatures cross-checked against source: `SparkDrafter.__init__(draft_dir,
  w, capacity, *, settings, quant, tap_cols)`, `propose(pending, depth,
  sampling, confidence, *, policy, round_ms)`, `headq.prepare(w, drafts)`,
  `fused.Buffers(w, rows, score_cols, decode=True)` (16 <= DECODE_ROWS 32),
  `run_window(w, st, b, toks, a, e, T)`.

## §5 The capture re-run procedure (on .111, GPU)

The patched file must reach .111's r2 tree first:

```bash
# head -> .111 (the tools tree the capture mounts as /tf)
scp ~/eugr/TensorFold-glm53-r2/tools/glm53/oracle.py \
    ciprian@192.168.178.111:~/TensorFold-glm53-r2/tools/glm53/oracle.py
```

Then re-run the capture phase of `~/aot-work/run-aot-dspark-ext.sh` (it
already mounts the dspark checkpoint and sets the env; make sure it does NOT
skip: `rm -rf ~/aot-work/cap-r0 ~/aot-work/triton-r0` or its dspark-ext
equivalents first). Equivalent explicit docker command (the
run-aot-world8.sh pattern + the dspark mounts):

```bash
cd ~/aot-work
rm -rf cap-r0-dspark triton-r0-dspark && mkdir -p cap-r0-dspark triton-r0-dspark
docker run --rm --network none --ipc host --gpus all --memory 64g \
  --user "$(id -u):$(id -g)" --entrypoint python3 \
  -v $HOME/models111/GLM-5.3-4.04bpw-h8-exl3:/model:ro \
  -v $HOME/TensorFold-glm53-py/src:/opt/tensorfold/src:ro \
  -v $HOME/TensorFold-glm53-r2:/tf:ro \
  -v $HOME/aot-work:/out -v $HOME/aot-cache:/cache \
  -v $HOME/models11/GLM-5.3-dspark:/dspark:ro \
  -e PYTHONPATH=/opt/tensorfold/src -e TORCH_EXTENSIONS_DIR=/cache/torch_ext \
  -e CUDA_CACHE_PATH=/cache/cuda_cache -e TORCH_CUDA_ARCH_LIST=12.1 \
  -e TRITON_CACHE_DIR=/out/triton-r0-dspark \
  -e TF_GLM53_DSPARK=/dspark -e TF_GLM53_DSPARK_POLICY=confidence \
  -e TF_GLM53_DSPARK_CONFIDENCE=0.3 \
  ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-12-r2-mlx-serve \
  -B /out/oracle-shim.py --model /model --out /out/cap-r0-dspark \
  --record --layers 4 --world 8 --rank 0 --long 20000 --tools /tf
```

Gates on the log: the `[oracle] dspark chain: block 8, 5 tap layers, ...`
receipt; `dspark-*` phases in launches.json; and the pack line must now count
more variants (was 41 of 15) with `missing none` INCLUDING the DSPARK_KERNELS
check. Then MERGE into the serving pack (sweep_aot's union-by-hash):
re-run the sweep/merge + validation steps of the pipeline with the dspark
capture as the template/capture input (or the full run-aot-dspark-ext.sh),
validate with list-variants (0 missing for k=0/parallel=1 AND k=2/parallel=8),
re-stage `~/aot-world8-32768-dcp1`, redistribute nothing else (the pack is
read at boot), and relaunch the dspark boot. Expected first missing-kernel
error GONE: `_group_sums` x_stride 30720 is the tap pass the capture now runs
64 times.

## §6 Risks — what this pass could still miss

- **`_residual_add` at the drafter's shapes**: the ZIG drafter's
  `rowResidual` (cuda_draft.zig:417-419) sums the gathered partials with the
  Triton `_residual_add` (residualAddStride), where the PYTHON
  `SparkDrafter._row` (dspark.py:242-252) uses torch adds — so the capture
  records no drafter-specific `_residual_add` launch. Covered anyway: the
  signature is identical to the target's (X, XOUT, G, RS, D, WORLD, BLOCK),
  the constexprs match (D=6144, WORLD=8, BLOCK=1024, warps 4), rows is
  grid-only, and RS is divisible-by-16 in both paths (target: rows*D;
  drafter: the gather stride) — the target's decode windows already recorded
  that variant hash. Verified against cuda_triton.zig:365-368 and
  glue.py:213-231. Residual risk: if the Zig loader keys on something beyond
  the Triton specialization (the literal RS value), a distinct entry would be
  needed — the boot error message prints literal values, so if the next boot
  dies on `_residual_add` with rs=49152, that is the (small) remaining gap
  and a one-line addition to the oracle chain closes it.
- **The drafter's CUDA graphs**: the serve replays tap/block graphs; graph
  replay launches the same kernels the eager capture recorded (the Recorder
  hooks the Python launcher, and the Zig side replays its own graphs with
  packed cubins). No variant can exist only inside a graph.
- **`_dconv_kernel` / `_dattn_ring`**: DFlash2-only (cuda_draft.zig:19-21);
  DSpark never launches them. Not exercised, not needed for this serve.
- **Markov/confidence host stage**: pure host numpy (dspark_host), no
  Triton; the zeros-context logits make the chain walk on garbage values,
  which cannot crash it (finite q4 biases) and cannot affect the pack.
- **Widths beyond 16**: concurrent (parallel>1) DSpark is refused by the
  engine (engine.py:119-121), so no multi-segment block passes
  (cuda_mdraft.zig) exist in this serve's launch set.
- **The head's top-k (`topkColumns`/`cands`) and qmm matmuls**: native CUDA,
  not AOT — the placeholder [128, 6144] head records nothing wrong and
  nothing missing.
