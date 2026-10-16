# findings-kv8 — 8-bit (fp8 e4m3) KV cache for glm_moe_dsa (r2 Zig engine)

PROGRESS LOG (append-only). Lane: fixer, dispatched 2026-10-15. Target: halve the
per-token cache cost (92-97 KB/token bf16 → ~51 KB/token) → roughly double the DCP=1
context ceiling (~500K → ~940K), per RESULTS.md queue item 1.

## §0 PROGRESS LOG
- [x] §1 SCOPE MAP (every kernel/site touching cache bytes) — persisted below
- [x] §2 DESIGN (format, scales, indexer decision, gate, exactness contract) — persisted below
- [x] §3 IMPLEMENTATION (Zig + native .cu; build wiring) — done, see §5
- [x] §4 GATES: 3-stage build rc=0, host tests rc=0 — see §5
- [x] §5 AOT IMPLICATION list (new-variant needs for the speed path) — see §6
- [x] §7 what remains before a live fp8-cache boot

## §1 SCOPE MAP — the KV cache paths in zig/src/families/glm_moe_dsa/

The cache is `fw.State` (cuda_forward.zig:461): per layer `kc` = bf16 [rows, 576]
(latentWidth = kv_lora 512 + rope 64), `ic` = bf16 [rows, 128] on full-indexer layers
(21 of 78 + the MTP layer's), plus the MTP layer's `mkc`/`mic`. Rows = slots ×
ceil(capacity/dcp)+1. Measured cost (boot receipt, 32K×8): 2.95 GiB/stream =
94.4 KiB/token = 79×1152 B latent + 22×256 B index keys (cuda_dcp.zig test:78 layers,
21 full indexers, mtp+indexer).

ALL cache-touching GPU kernels are TRITON kernels launched from the AOT pack
(cuda_triton.zig) — the compiled cubins bake the `*bf16` pointer type into each
variant. Native CUDA kernels (zig/kernels/cuda/*.cu → fatbins at build time) are a
SEPARATE mechanism (cuda_kernels.zig) and need NO AOT capture.

| Path | Site | Kernel(s) | Touches |
|---|---|---|---|
| Alloc | cuda_forward.zig:492 `State.initSlots` (sole external caller: cuda_runner.zig:316) | — | kc/ic/mkc/mic planes |
| Write (decode+prompt, ALL modes) | cuda_forward.zig:860 `keys()` — centralized; cuda_prompt.zig:506-508 (prompt chunks call the same `kvWrite`/`ikWrite`) | `_kv_write` (DCP/ROWS variants), `_ik_write` | kc, ic |
| Read — attention | cuda_forward.zig:885 `attentionCore` → :912-914 | `_attn_chunks` (DIRECT/ROWS × decode/prompt tilings) | kc |
| Read — attention, DCP | cuda_forward.zig:1233 `attentionDcp` | `_attn_dcp`, `_attn_dcp_rows` | kc |
| Read — DSA selection | cuda_forward.zig:1127 `select` / :1181 `selectDcp` | `_index_scores` (+topk) | ic only |
| Slot copy (parallel N) | cuda_multi.zig:894 `copySlotRows` | cuMemcpy (row bytes) | kc, ic, mkc, mic |
| Reuse-state files | cuda_reuse.zig:600 `rowViews` → cuda_engine.zig:875 `writeState`/`readState` | byte planes; size-validated (`z[0] != v.row` refuses) | all planes |
| Memory accounting | cuda_forward.zig:541-560 bytesFor/bytesForDcp/latentBytes; cuda_dcp.zig:103 tokenBytes; cuda_engine.zig:260 cacheFit, :551 checkCacheFits; cuda_multi.zig:787 bytesPerStream | — | — |
| Boot receipt | cuda_engine.zig:531 ("GiB of caches each") | — | — |
| NOT touching the target cache | cuda_draft.zig / cuda_mdraft.zig (drafter-internal bf16 rings), cuda_score.zig (route taps), l2pf (weights only) | — | — |

`_attn_chunks` exact semantics (Python fused.py:224, verified): row r → p = POS[r]
(ROWS) or POS+r; n = min(p+1, K); key = idx while p < K (contiguous rows BASE_r+0..p),
else TOK[r*K+idx] (all K entries); ROWS adds BASE[r] to the LC pointer. Reuse modes
(mtp_reuse, default 2) attend the kept K-list with slot K-1 patched to the row's own
position — same p≥K shape. This makes a static per-row-block scratch layout possible
(§2.3).

## §2 DESIGN (frozen before implementation)

### 2.1 What quantizes
- **kc/mkc (the MLA latent + shared rope key): fp8 e4m3 + fp32 scales. THE PRIZE:**
  79 planes × (576 B codes + 8 B scales) = 46,136 B/token vs 91,008 bf16.
- **ic/mic (DSA indexer K cache): stays bf16 in v1.** Reasons: (a) the top-k SCORING
  reads all of ic every step — quantized scoring inputs shift the selected SET, which
  changes which keys attention reads (a far bigger output perturbation than value
  precision); (b) no lineage precedent covers the indexer (the Flash kv8 port
  quantized keys+values of plain attention, no DSA); (c) it is only 5.6 of 94.4
  KiB/token. v2 candidate with its own quality gate.
- Per-token total: 51,768 B = **50.6 KiB/token (53.6% of bf16)** → DCP=1 ceiling
  ~500K → ~940K at the same free memory. Index keys in fp8 too (v2) would give 47.8.

### 2.2 Format
- e4m3 codes (torch.float8_e4m3fn == CUDA `__nv_cvt_float_to_fp8` SATFINITE), one code
  per bf16 value; **TWO fp32 scales per token-row per plane**: one over the latent half
  [0,512), one over the rope half [512,576) (the halves have different statistics; a
  shared absmax would waste one). Scale = absmax(half)/448, 1.0 when absmax=0 — the
  capture.py absmax/448 per-row pattern (precedent: Python glm_moe_dsa cuda/capture.py:33).
- Quant/dequant are deterministic → per-mode keyed-seed determinism HOLDS within a
  kv8 config; bits legitimately DIFFER from the bf16-cache serve (lossy, like
  TF_GLM53_VERIFY_HEAD=fp8 — headq.py:13 gating style).

### 2.3 Mechanism — "staged kv8": native quant/dequant around UNCHANGED Triton kernels
The existing world-8 AOT pack (aot.json 0190e384…) stays VALID: every Triton launch
keeps the same variant key (name + constexprs + arg dtypes) — only pointer VALUES
change (staging/scratch are bf16, same `*bf16` args). NO new Triton variants, NO
recapture for v1. (Verified against cuda_triton.zig arg lists and aot.find semantics.)

- **Write**: `_kv_write` writes the window's rows into a bf16 STAGING buffer as rows
  0..R-1 (POS→a zero word, BASE→a zero table — the same device buffers every replay);
  then native `glm53_kv8_quant` reads staging and writes fp8 codes + 2 scales at the
  true destinations, replicating `_kv_write`'s row math (non-ROWS: POS[0]+r; ROWS:
  BASE[r]+POS[r]; DCP=1 only in v1). Staging: max(prompt_rows, decode window rows) × 1152 B.
- **Read (decode, graph-safe)**: native `glm53_kv8_dequant` materializes the needed bf16
  rows into a SCRATCH laid out per-row-block: row r's block at scratch rows [r*K,
  r*K+K); rows with p<K additionally need the contiguous prefix — non-ROWS rows read
  scratch rows 0..p directly, so the prefix lives at scratch rows [0,K) (no collision:
  p<K ⇒ p<K ≤ block start of every row ≥ 1; row 0's block is [0,K) and when row 0 has
  p≥K no row has p<K). Static tables make the Triton kernels read this layout:
  - non-ROWS: TOK' = ramp[R_max*K] (TOK'[r*K+i] = r*K+i);
  - ROWS: BASE' = [r*K] table + TOK' = per-row ramp (TOK'[r*K+i] = i) — the kernel's
    BASE[r] shifts LC to row r's block and key=i lands inside it; p<K ROWS rows read
    r*K+idx — also inside the block. UNIFORM for ROWS.
  - All decisions (p<K vs TOK) are DEVICE-side (POS loads) → CUDA-graph-replay-safe.
  - Benign identical-write races when several p<K rows copy the same prefix rows —
    output deterministic.
- **Read (prompt/wide)**: DIRECT dequant of rows [0, POS[0]+R) into a capacity-sized
  bf16 prompt scratch (POS read on device; grid = scratch rows, early exit) — the
  prompt kernels read keys 0..p with no TOK/BASE, so the scratch is a direct mirror.
  Stateless per chunk (correct across resets/reuse/slot switches); the incremental
  dequant (only new chunk rows) is a documented follow-up.
- **Traffic honesty**: the staged read materializes bf16 the attention then re-reads —
  attention-related DRAM traffic ≈ 2.5× the bf16-cache path (fp8 read + bf16 write +
  bf16 attention read vs bf16 attention read). v1 buys CAPACITY (the prize); the SPEED
  win (the Flash port's +27% at 128K) requires the in-kernel dequant variants (§6).
  Gate defaults OFF; the live bf16 serve is untouched until deliberately enabled.

### 2.4 Gate + contracts
- **TF_GLM53_KV8=1** (envFlag, default OFF, lossy — documented like VERIFY_HEAD fp8).
- v1 refuses at boot: DCP>1 (the `_attn_dcp` read path needs the same trick per rank
  shard — follow-up), TF_GLM53_SIDE letters (the side stream would race the shared
  staging/scratch). L2PF allowed (weights only).
- settingsDigest (rendezvous) includes kv8 → mixed ranks refuse.
- Reuse-state files: kc views become row=576 (codes) + row=8 (scales) planes; the
  existing size validation (`z[0] != v.row`) refuses stale bf16 states automatically.
- Boot receipt adds: `8-bit KV: <KB>/token (fp8 latent + bf16 index keys)`.
- Memory accounting halved for kc: bytesForDcp/latentBytes/cacheFit/bytesPerStream
  take the kv8 row size; the prompt scratch + staging + decode scratch count in
  cacheFit's `extra`.

### 2.5 New native kernels (build-time fatbin, NO AOT)
`zig/kernels/cuda/glm53_kv8.cu`: `glm53_kv8_quant` (grid R: staging row r → codes+
scales at dest_r), `glm53_kv8_dequant` (grid R*K decode / grid N direct: one program
a 576-byte row, 144 u32 words, per-half scale applied by word index). Wired via
zig/build/cuda.zig images + src/cuda/kernels.zig + cuda_kernels.zig (Kernels.kv8_*).

## §5 IMPLEMENTATION + GATES (done)
- 2026-10-15 (fixer): §1-§2 persisted BEFORE code (this file).
- Implementation (all in /home/ciprian/eugr/TensorFold-glm53-r2; diff: zig-kv8.diff; backups *.kv8bak
  alongside the patched files; the earlier dtype patch's cuda_weights.zig.bak untouched):
  - NEW zig/kernels/cuda/glm53_kv8.cu: glm53_kv8_quant (staging row r → plane row dest_r, per-half
    absmax/448, SATFINITE e4m3), glm53_kv8_dequant (mode 0 direct / 1 one-stream / 2 base-table; one
    program a 576-B row, per-half scales by word index).
  - Build wiring: zig/build/cuda.zig images += glm53_kv8 (-O3); zig/src/cuda/kernels.zig blob;
    cuda_kernels.zig: Kernels.kv8_{mod,quant,dequant} + Ops.kv8Quant/kv8Dequant.
  - cuda_forward.zig: State kv8 flag + lat_row (584) + initSlotsKv8 (DCP>1 refused) + kv8-aware
    bytesFor/bytesForDcp/latentBytes (+*Kv8 variants); Kv8 struct (staging, zero/ramp tables, block
    scratch [128×2048 rows], capacity prompt mirror, ramp TOK'/BASE' tables) + quant/dequantDecode/
    dequantPrompt; Forward.kv8; keys() staged write; attentionCore scratch read (wide=direct mirror,
    decode=blocks+ramps). Single-plane layout (codes+scales in one 584-B row) — slot copies, reuse
    views and saves unchanged except the row size.
  - cuda_prompt.zig: frontAttn's kvWrite staged (defensive; SP is off under kv8).
  - cuda_runner.zig: Settings.kv8 → State.initSlotsKv8.
  - cuda_engine.zig: Options.kv8 (TF_GLM53_KV8, envFlag, default OFF) + fromEnv; settingsDigest
    includes kv8 (mixed ranks refuse at rendezvous); boot refusals Kv8NeedsDcp1 / Kv8NoSide; SP off
    under kv8 (warn); Kv8 init (stage_rows = max(prompt_rows, 128), rows_max = max_rows = 128,
    prompt mirror = localCapacity); boot receipt "8-bit KV on … KB a token …% of bf16 … MiB";
    checkCacheFits: halved row + side buffers in `extra`; Runner settings .kv8.
  - cuda_multi.zig: bytesPerStream kv8-aware; copySlotRows uses st.lat_row.
  - cuda_reuse.zig: rowViews uses st.lat_row (stale bf16 state files auto-refused by the size table).
  - Tests: cuda_dcp.zig kv8 token-bytes test (51,768 B/token = 50.55 KiB = 53.55% of bf16; 32K×8:
    2.95 → 1.58 GiB/stream); cuda_engine.zig cacheFit kv8 test (8×32K on 24 GiB: bf16 refused, kv8
    fits; 1-stream on 44 GiB: bf16 refuses 500K, kv8 fits it, kv8 ceiling ~888K ≈ 1.8× bf16's 454K).
- BUILD (build-r2.sh, RAM-guarded -j4, zig 0.17.0, sm121): **rc=0 ALL 3 STAGES** (fatbins 71 s incl.
  the new glm53_kv8.fatbin 26 KB; native+glm53-lib+tf-glm53-generate 77 s; host tests 11 s —
  346/346 PASS). Two en-route fixes: __nv_cvt_fp8_to_float → __nv_fp8_e4m3 struct cast (not in this
  CUDA header); Zig method receiver ordering (v first). Artifacts (md5): tf-glm53-generate
  4c1881105f7e807c88f36ea9e78e190c, libtensorfold_glm53.so b2bd651335adbca56b55bc3ade7952b7,
  tensorfold-native 53fad8674de9667af87815e3c71f6325.
- The EXISTING world-8 AOT pack (aot.json 0190e384…) stays VALID: no Triton launch's variant key
  changes (same kernels, same constexprs, same *bf16 arg types — only pointer values and the native
  kv8 launches differ, and native fatbins are build-time). cuda_coverage.zig unchanged (kv8 is
  env-off in coverage runs).

## §6 AOT IMPLICATION — new Triton variants the SPEED path needs (v2, capture extension)
v1 needs NONE (pack 0190e384… stays valid). The in-kernel-dequant speed path (kv8.cu
two-stage cp.async pattern from the Flash port) needs Python fused.py variants + a
fresh capture (do NOT run now):
- `_kv_write` KV8-out variants (or a `_kv_write8`): × {DCP 1, ROWS t/f} — quantize in
  the write kernel (kills staging).
- `_attn_chunks` KV8: load e4m3 tiles, dequant in-register, scales on scores —
  × {DIRECT t/f, ROWS t/f, decode/prompt tilings} over the captured shape space
  (H=8/rank, LW=512, RD=64, K=2048, CHK 256/2048, KTT 32/64).
- `_attn_dcp`/`_attn_dcp_rows` KV8 (only when kv8×DCP>1 ships).
- `_index_scores` KV8 only if the indexer cache ever quantizes (v2+; own quality gate).
- cuda_coverage.zig must enumerate the kv8 launches then (v1: unchanged — kv8 is
  env-off in coverage runs, and v1 adds no Triton launches).

## §7 WHAT REMAINS BEFORE A LIVE FP8-CACHE BOOT
1. Distribute the rebuilt 3 artifacts to .111-.118 (all 8 ranks must run the same
   build; md5-verify as with the dtype patch).
2. Boot with TF_GLM53_KV8=1 (parallel 8, 32K, DCP=1, MTP k=2) on the EXISTING AOT
   pack — expect the "8-bit KV: 50.6 KB/token" receipt and ~1.62 GiB/stream caches
   (vs 2.95).
3. Gates: keyed-seed determinism WITHIN kv8 (777×2 → same sha; 778 → moves); KL /
   fixed-prompt sha-delta vs the bf16-cache serve (tf-glm53-generate --mode score or
   fixed-prompt comparisons) — bits WILL differ from bf16 (lossy, documented); memory
   ≥2 GB floor; c1/c8 bench vs the bf16 serve (expect some decode regression from the
   staged read — measure; the ceiling experiment: context 500K-900K × 1 stream).
4. (later, speed) the §6 capture extension on .112+.

## §8 vLLM/sglang fp8-KV reference mining (orchestrator, 2026-10-16) — VERDICT: KEEP v1's scheme

**vLLM** (vllm-project/vllm @ main, clone /tmp/opencode/vllm-src; the MLA backends, vllm/v1/attention/backends/mla/):
- Cache dtype: `kv_cache_dtype fp8`/`fp8_e4m3`, per-layer SCALAR scales — flashmla.py:320-323:
  `descale_q=layer._q_scale.reshape(1)`, `descale_k=layer._k_scale.reshape(1)` (single values
  from checkpoint calibration k_scale/v_scale tensors, default 1.0 via models/utils.py's
  cache-scale mapper; deepseek_v2.py adds none itself).
- Read: IN-KERNEL dequant — FlashMLA fp8 native (`flash_mla_with_kvcache_fp8`,
  `get_mla_metadata_dense_fp8`); Triton path: "For FP8 KV cache, we dequantize to BF16 on load
  inside the Triton kernel" (triton_mla.py:221), kernel branches `if dtype.is_fp8()` (no-op
  when bf16 — the SAME cache-dtype-branch pattern as our planned KV8 AOT variants).
- Queries NOT quantized (triton_mla.py:222-224 explicitly tells the common layer to skip it).
- DCP×fp8 constraint: triton_mla.py:295 asserts "causal multi-token decode is not supported
  with DCP" on that path — production evidence that fp8×DCP needs dedicated care. Our v1
  refusing DCP>1 with KV8 (cuda_engine.zig boot gate) is CONSISTENT with their constraint.

**SGLang** (local comparator stack sglang-sm121-gb10/): its fp8 work is WEIGHTS, not KV —
patches/sglang/mimo26-fp8-oproj.diff = block-FP8 linears (128×128 blocks, fp32 amax/448
scales, quantize-at-load, quant_method swap only). Same amax/448 e4m3 idiom we use, but no
DSA/MLA KV-cache fp8 scheme to mine there (their MLA runs bf16 KV on this stack).

**Comparison table:**
| | our v1 | vLLM MLA fp8 | sglang (this stack) |
|---|---|---|---|
| KV dtype | e4m3 | e4m3 | bf16 (fp8 is weights-only) |
| Scale granularity | 2×fp32 PER TOKEN-ROW (latent/rope halves, dynamic absmax/448) | per-layer SCALAR (calibrated, default 1.0) | 128×128 blocks (weights) |
| Read path | staged native dequant v1 (in-kernel v2 planned) | in-kernel (FlashMLA native / Triton branch) | n/a |
| Queries | bf16 | bf16 (explicitly unquantized) | n/a |
| DCP×fp8 | refused (boot gate) | asserted unsupported on that path | n/a |

**Verdict:** v1's per-token-row dynamic scales are STRICTLY FINER than vLLM's production
per-layer static scalars (dynamic absmax adapts to outlier tokens; overhead +8 B/row on 584 B
rows ≈ 1.4%). vLLM proves per-layer suffices in production → ours has margin, not risk.
KEEP v1 unchanged before the live boot. No pre-boot changes needed.

**v2 validation:** vLLM's in-kernel pattern (descale params in kernel signature, cache-dtype
branch no-op for bf16, bf16 accumulate) is exactly the shape of our §6 AOT KV8 variants
(_attn_chunks KV8 ×{DIRECT,ROWS}, _kv_write KV8) — the §6 list stands as written.

## §9 THE FIRST KV8 BOOT ATTEMPT — the SIDE default, the rendezvous decode, the auto-off patch (2026-10-16)

### 9.1 What happened
The first kv8 boot (glm53-w8-kv8-boot.sh, TF_GLM53_KV8=1, parallel 8, 32K, DCP 1, MTP k=2, pack
unchanged) died in seconds on MY OWN gate: `Kv8NoSide`. ROOT CAUSE: `Options.side` DEFAULTS TO
"af" (cuda_engine.zig — TF_GLM53_SIDE is ON unless explicitly disabled), the boot script never
set it, and v1 refused the combination. Gate-design bug: a refusal against a default the
operator never chose. My §2.4/§5 boot instructions should have carried `-e TF_GLM53_SIDE=0`.

### 9.2 What the side stream is (and what off costs)
cuda_decode.zig / cuda_forward.zig: TF_GLM53_SIDE = the decode windows' SECOND CUDA stream
(Phase 3a speed), fork/join-event branches INSIDE the captured decode graphs. Letters: "a" =
the attention key path (the kv_a linear + keys(): the latent/index cache writes) forked beside
the query path; "w" = the alternative split (keys forked after the grouped q_a/kv_a launch);
"f" = the shared-expert MLP forked beside the router + routed experts (ffnPart). It is PURE
SCHEDULING — the same kernels with the same bits on another stream. It is NOT the sidecar head
(headq, unrelated), NOT drafting (MTP/DSpark run their own windows), NOT prompt-SP (separate,
also auto-off under kv8). COST OF OFF: decode overlap only — some speed, ZERO quality change.
The live bf16 serve runs with it on; the kv8 boot loses the overlap until §9.3 lands.

### 9.3 kv8 + side: the honest correction
v1's "the side stream would race the staging" claim was OVER-CONSERVATIVE. Re-analysis:
keys() runs exactly ONCE per layer per window, on exactly one stream; the fork event orders the
side branch after everything main queued; the join (attentionCore, BEFORE the dequant) orders
main's plane reads after the side's quant; v.stage is single-use per layer and stream-serialized;
"f" never touches kv8 buffers. By construction there is no data race — what is missing is GPU
VERIFICATION (the keyed-seed/KL gates), not a fix. Enabling = drop the §9.4 normalization +
one gate run on .112+. vLLM's fp8-MLA reference (§8: queries bf16, dequant-at-read) does not
interact — side is scheduling, not math. NOT v2 territory; it is a small verified flip.

### 9.4 The patch (shipped): auto-off instead of refusal
The operator's seat is right: kv8 is the opt-in capacity mode; the side stream is a speed-only
default; a boot should not require knowing the interaction. bringUp now NORMALIZES the letters
to off when kv8 (parseSide result overridden), logs
  "side stream off (kv8): TF_GLM53_SIDE '<raw>' ignored - its overlap is scheduling only (no
   quality change); the staged kv8 path is not verified against the side fork (findings §9.3)"
and the rendezvous settingsDigest hashes the RAW side setting EMPTY under kv8 (kv8 itself is in
the digest), so ranks differing only in the ignored TF_GLM53_SIDE cannot refuse each other.
The DCP>1 refusal is KEPT — a real capability gap, and vLLM's fp8 MLA carries the same
constraint on its DCP path (§8, triton_mla.py:295). Host test added
("TF_GLM53_KV8 auto-offs the side stream … §9"): kv8 on → digests equal across side settings;
kv8 off → they differ; kv8 in the digest. BUILD rc=0 ALL 3 STAGES; test-glm53 101/101 (the new
test included), test-cuda-host rc=0. NEW md5s: tf-glm53-generate fc613b372f10f6bdd091c774607cef96,
libtensorfold_glm53.so 61e738600a4a0d2861787dbd5858d72e, tensorfold-native
b1b6ca38c17960bd8e8814d808545c9c; glm53_kv8.fatbin UNCHANGED (3eaa3463… — no .cu change).

### 9.5 The rendezvous "garbage peer" lines — decoded, benign, unrelated
The refused 12-byte hello decodes as ASCII: rank field = "/v1/", world field = "mode"(ls) — an
HTTP `/v1/models` request from an OpenAI-style client or health check hitting the RENDEZVOUS
port (29581) instead of the serve port (8080). rendezvous.zig validates magic/world/rank,
logs "refused a peer", closes, KEEPS LISTENING — the same harmless stray-refusal class boot 1
recorded ("a scanner or mangled packet; refused, harmless"). NOT a bug, NOT exposed by the
kv8 refusal; the boot died solely on Kv8NoSide. ACTION: aim the health/monitor curl at the
serve port (127.0.0.1:8080 on .111), not MASTER's rendezvous port.

### 9.6 The boot fix (both options live)
(a) IMMEDIATE, works with the ALREADY-DISTRIBUTED build (native 53fad867…): add
  `-e TF_GLM53_SIDE=0`  (parseSide: "0"/"off" → no letters) — glm53-w8-kv8-boot.sh line 22
  patched accordingly. Boot NOW with this; no redistribution needed.
(b) The auto-off build (§9.4 md5s): boots with or without the SIDE env; distribute on the
next artifact sync (fatbin unchanged — only the 3 binaries).
Expected receipt either way: "8-bit KV on … 50.6 KB a token …" + (with raw SIDE set)
"side stream off (kv8) …", caches ~1.58 GiB/stream at 32K×8.

## §10 THE FREE-MEMORY ANOMALY — CLOSED (2026-10-16): a stale predecessor serve, not engine accounting

THE DATA: kv8 boot (KV8=1, k=2, 32K×8) measured **15.3 GiB free** at the cache check; the k=3 MTP
boot (**bf16, KV8 UNSET**, 32K×8) measured **16.0 GiB**; the default k=2 bf16 boots (original serve
+ the bisect reboot) passed a check needing ≥27.9 GiB free. Apparent pattern: "non-default configs
see ~15-16" — misleading.

THE EXONERATION (code): for the k=3 bf16 boot the check path is BYTE-IDENTICAL to the healthy k=2
boot's — kv8 dormant ⇒ `extra = 0` (cuda_engine.zig checkCacheFits: `if (b.kv8) |*v| v.bytes() else 0`),
`row` = bf16 89,856 B, the same 6 GiB reserve policy (cacheFit, pre-existing, the Python engine's
own), and `free` is a RAW cuMemGetInfo reading (context.zig:60-62) — the engine cannot mis-count it.
The only k-dependent inputs: `capacity = ctx + k + 1` (32771 k2 / 32772 k3 — the +k+1 draft-rows
offset, correct by design, 0.7 MiB of difference) and `row` — nothing config-dependent on the free
side. The 46.7-vs-50.6 KB/token question (closed): the refusal's 11.4 GiB is the GUARD's
latent-only figure (78×584×32772×8 = 11.12 GiB + 334 MiB extra = 11.45; 45,552 B/token — "index
keys and the MTP layer not counted, as Python counts", cuda_forward.zig latentBytes); the receipt's
50.6 KB/token is the FULL bytesForDcp figure (51,768 B). The bf16 guard has the same convention gap
(89,856 guard vs 96,640 full; its 32K×8 guard figure is exactly the 21.9 GiB the k3 refusal
printed). The real kv8 allocation would be 12.64 GiB (1.58 GiB/stream).

THE CAUSE (arithmetic closure): both low readings fit ONE story — a previous serve still held
~40 GiB (caches 23.6 + graphs/rings ~14) when the new boot's check ran:
  121 − ~40(predecessor) − 47.5(weights) − 18(MtpDense, cuda_prompt.zig:633-635, default-on) − ε
  ≈ 15.5 ≈ the 15.3 AND 16.0 measured. The "default configs see ~30" boots ran on clean nodes
  (121 − 47.5 − 18 − misc ≈ 53 free — the k2 bisect's 27.9 requirement passed comfortably). The
  config "pattern" is a sequence artifact: the non-default boots raced a live/not-yet-released
  predecessor; the default boots did not. Page cache is a secondary suspect only (the loader's
  O_DIRECT falls back to cached reads on filesystems that refuse it, cuda_source.zig:85-87/271-272)
  — but page cache would PERSIST across boots, and the bisect boot saw clean memory right after
  kv8 boots, so the predecessor-serve story fits the data better.

THE PATCH (shipped): the refusal now prints "…GiB of **<total>** GiB is free" and, when the device
reports >90 GiB held (a clean node holds ~67-70 at the check), warns: "another serve's caches (or
its page cache) are likely still allocated — check every node (docker ps, nvidia-smi) before
rebooting smaller". OPERATOR RULE for every boot: all 8 nodes clean first (`docker ps` empty of
glm53* containers, engine-free ≈ `free -g`); the boot script should rm -f ALL glm53* names.

THE CEILING MATH: UNCHANGED — the accounting was never wrong. On a clean node the §7 numbers stand
(kv8 single-stream guard ceiling ~890-940K at DCP=1). Two ceiling-run levers, both engine-supported
(the refusal message lists the first): TF_GLM53_MTP_DENSE=0 frees the 18 GiB MtpDense copy (+~350K
tokens of ceiling; drafts unaffected — it only changes the MTP layer's PROMPT rows path); and the
boot script's `--memory 110g` cgroup cap binds before the node does at ~900K (67 fixed + ~46
caches > 110) — raise it for the ceiling run.

## §11 THE CORRECTNESS BREAK — ROOT CAUSE FOUND AND FIXED: the staging redirect zeroed the ROPE ANGLE

SYMPTOMS (live kv8 boot, binary 53fad867, ctx 26000×8): every prompt degenerates to token soup
from ~token 1; acceptance 0.4-6.5% (vs 45-55% bf16); one run stopped at 3 tokens. Bisect: bf16 on
the SAME binary healthy ⇒ isolated to the kv8 data path.

ROOT CAUSE (file:line): **`_kv_write` rotates the rope half of its cache row at the row's TRUE
position** — Python fused.py:139 `ang_p = pg`, :148 `ang = ang_p * INV[i]` (pg = POS + r, or
POS[r] under ROWS). My staging redirect passed POS → a zero word (BASE → a ramp), so every staged
row's rope (64 of the 576 dims) was rotated at angle `r` instead of `true_pos + r`. The latent
half (position-independent RMSNorm) was correct — hence a semi-plausible token 0 — and the rope
dot-products were garbage — hence soup and the drafter-side acceptance collapse (the target's
hidden states the drafter taps are computed over garbage rope scores). The dequant/scratch/ramp
READ machinery was correct all along (verified line-by-line against _attn_chunks' semantics); the
plane simply held wrongly-rotated rope.

THE FIX (shipped, all in the r2 worktree): use the decoupling the ROWS variant of `_kv_write`
ALREADY has — the angle comes from POS[r] ALONE, the write row is BASE[r] + POS[r]:
- NEW native `glm53_kv8_stage` (glm53_kv8.cu): per window builds P'[r] = the row's TRUE position
  (rows_mode: POS[r]; else POS[0] + r) and B'[r] = r − P'[r].
- keys()/frontAttn now run `_kv_write`'s ROWS variant with (P', B') → staging row B'[r]+P'[r] = r,
  rope rotated at the TRUE angle ⇒ **the staged row is bit-identical to what the bf16 cache would
  hold** (same single-rotation rounding — the capture-consistency property), then the quantizer
  (unchanged) takes the true rows. Pack-safe: `_kv_write`'s variant key is (LW, RD, DCP, RANK,
  ROWS) — no R — and the ROWS variant is already in the pack (the parallel-8 serve launches it
  every window).
- Kv8: ramp_base removed; stage_pos/stage_base added (int32[stage_rows]); shared `stageWrite`
  method; cuda_kernels.zig Ops.kv8Stage + Kernels.kv8_stage.
- Host tests unchanged (they cover byte math, not the GPU path — the live boot is this fix's gate).

### §11.1 Reference comparison (the user-requested mining)
- **The Flash port's working kv8** (/tmp/opencode/mimo-flash-port, patches-zig 0063-0067): their
  structural lesson is THE fix's lesson — they do RoPE + quant in ONE native kernel (mimo_qkv8
  "RoPE + q to qo, k / v quantised into the 8-bit cache") — the write path OWNS the rope, so my
  bug class (a staged write losing the position) cannot occur there. Their quant semantics match
  mine: absmax/448 per (token, key head), `to_fp8(v * inv)` SATFINITE e4m3, fp32 scale stored,
  dequant = code × scale on load (0063:115-127, 165-194). Their gate (0063's test note + results/
  zig-z7/kv8): "error against Triton on the same 8-bit values" — they gate the QUANT ROUNDTRIP,
  not vs bf16 — the right gate for us too: after the rope fix the staged row is bf16-exact, so the
  only remaining delta is the e4m3 roundtrip (deterministic).
- **vLLM's write path** (/tmp/opencode/vllm-src): triton_mla.py:221-224 — fp8 KV with BF16 queries
  ("Tell the common layer not to quantize queries"), quantize-on-save with the layer's STATIC
  per-layer scalar scales (§8). Their DSA indexer cache is ALSO fp8 (deepseek_v2.py:727-737:
  "fp8 naive cache, value in fp8 and scale in fp32 per quant_block_size=128", ue8m0 scales) — an
  existence proof that 8-bit INDEXER keys can work (block-128 static scales) — a v2 candidate for
  our ic with its own quality gate; v1's bf16 ic stands.
- Net: all three implementations (ours, Flash, vLLM) agree on e4m3 + absmax/448-class scaling +
  fp32 scales + dequant-at-read; ours is the only one with per-token-row DYNAMIC scales on the
  latent (finer than Flash's per-token-head, far finer than vLLM's per-layer). The bug was never
  in the quant math — it was the staged write losing the position.

### §11.2 The one-boot live test plan (after redistributing the 4 artifacts)
1. All 8 nodes verified clean (docker ps, nvidia-smi; §10 rule).
2. glm53-w8-kv8-boot.sh (SIDE=0 line harmless with the auto-off build; ctx 32768×8, k=2, pack
   unchanged). Expect: "8-bit KV on … 50.6 KB a token", ~1.58 GiB/stream, free ≈ 50+ GiB at the
   check (if ~15 again: STOP, find the stale occupant — the new refusal prints total + warns).
3. Gates: beekeeper greedy 200 — coherent + acceptance ~45-55% (the rope is now bf16-exact; the
   only loss is e4m3); keyed-seed 777×2 → identical sha, 778 → moves (WITHIN kv8); fixed-prompt
   sha/KL vs the bf16 serve (bits WILL differ — lossy, documented); memory floor; c1/c8 bench
   (expect some decode cost from the staged read; measure).
4. Then the ceiling run: --parallel 1, TF_GLM53_MTP_DENSE=0, raised --memory, clean nodes.

## §12 THE PROMPT-PATH BREAK — ROOT CAUSE FOUND AND FIXED: dequant mode 1 wrote BLOCK rows where the attention reads the SHARED PREFIX (2026-10-16)

### §12.1 The isolation
The §11 fix restored decode (beekeeper 52.5%) but prompts stayed garbled at every length
(5-54+). Host-side re-verification of the single-stream chain kept passing on paper, so we built
`kv8repro` (zig/tests/cuda/kv8repro.zig, build step `kv8repro`; no model, no engine — just the
pack + the kv8 fatbin + synthetic kva) to gate the chain against the bf16 path it wraps:
- Gate 1: the stage tables P'[r] = pos + r, B'[r] = r − P'[r].
- Gate 2: the staged ROWS write BIT-IDENTICAL to the bf16 non-ROWS `_kv_write` at the same
  positions (the §11 rope fix, now proven on GPU).
- Gate 3: quant → dequant(mode 1) → scratch matches the bf16 rows within e4m3.
First run on .112: gates 1-2 PASS, gate 3 FAIL — 92% of values past 20% rel, worst 1.0, FIRST
BAD AT SCRATCH ROW 1 (row 0 fine). That signature — only row 0 correct — named the bug.

### §12.2 The root cause (glm53_kv8.cu, glm53_kv8_dequant mode 1)
The non-ROWS `_attn_chunks` reads keys FLAT while p < K: key = idx, TOK never consulted
(fused.py's `key = idx if p < K else TOK[r*K+idx]`). Mode 1's job is to materialize the plane
into the scratch in THAT layout: rows i ≤ p at scratch row i — the shared prefix region [0, K)
(the kernel's own header comment, lines 16-20, says exactly this). The code instead wrote
`dst = r*K + i` unconditionally — the mode-2 BLOCK layout. Every prompt row except (r=0,i=0)
landed where the attention never reads, and the attention read rows nobody wrote (stale
scratch): prompts saw garbage keys for every row past 0. Decode was healthy because under
`--parallel 8` decode runs mode 2, whose block layout IS correct (its attention takes
BASE' = r*K and TOK' = per-row ramps, so both its p<K and TOK reads stay inside the blocks —
cuda_forward.zig's rows_base/rows_tok, :690-695). Mode 1 was reachable only by prompts (small
windows, base == 0 — attentionCore :1088) and single-stream decodes, so only prompts broke.
The engine orchestration (slot views, multi, buffers) was innocent all along.

### §12.3 The fix (one branch, glm53_kv8.cu mode 1)
```cuda
if (pr < K) { if (i > pr) return; src = i; dst = i; }        // shared prefix: key = idx, flat
else        { src = TOK[r*K + i]; dst = r*K + i; }           // past K: TOK' = identity ramp → block
```
Past K the launch passes ramp_tok = the identity table (r*K + i, cuda_forward.zig :688), so the
attention's `key = TOK'[r*K+idx]` = r*K+idx — exactly the block the fixed dequant wrote. Rows
with p < K never consult TOK'. No overlap: the prefix is read only while some query has p < K
(i.e. POS[0]+R ≤ K), and blocks are written only at r*K+i ≥ K. Mode 0 (wide/prompt mirror,
dst = i) and mode 2 were already correct and untouched.

### §12.4 Verification (GPU, .112)
- kv8repro pack --pos 0 --rows 24: ALL GATES PASS (worst rel 0.0612 — plain e4m3 error).
- kv8repro pack --pos 100 --rows 3: ALL GATES PASS (worst rel 0.0604).
- The repro stays in the tree (build step `kv8repro`) as the kv8 chain's regression gate.
Live retest needs a FULL 8-rank reboot (every rank runs prompt windows through mode 1).

### §12.5 The rebuilt artifacts (build-r2.sh rc=0 all stages, 2026-10-15 23:05)
- zig-out/bin/tf-glm53-generate  ac435384ddeb39de25b68cc8c03f9f22
- zig-out/native/bin/tensorfold-native  5f19d512ac7b02b24f3f5219052cda32
- zig-out/lib/libtensorfold_glm53.so  bc9021f79da063ea757cd7d1b8adc4e5
- zig-out/fatbin/glm53_kv8.fatbin  0b0e3e901ba0f73c65a87515b95b7c35
- zig-out/bin/kv8repro  c331f3aafa85ab26472a89cf3e22f60f (the regression gate tool)
Host tests: test-glm53 + test-cuda-host rc=0 (stage 3). zig-kv8.diff refreshed (1425 lines,
now carries glm53_kv8.cu and kv8repro.zig as new-file sections). Retest: FULL 8-rank reboot
(every rank's prompt windows run mode 1), then §11.2's gates.
