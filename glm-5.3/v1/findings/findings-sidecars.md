# FINDINGS — bf16 drafting sidecars (eh_proj.weight + lm_head.weight), Stage 0

Lane: fixer2 (sidecar build). Constraints: CPU only, <9 GB peak RAM (chunked decode), no GPU,
no serving, NO .11-.18 contact; NFS pack read-only; .111 ssh allowed for the final copy.
CORRECTION CONTEXT (accepted, not re-litigated): all 6 layer-78 duplicates are DIVERGENT;
spark-tools inspector vindicated; shard 76 canonical (index-driven) — source for eh_proj.

## PROGRESS LOG
- [x] MISSION-LOG.md re-read (correction noted; .111 local pack = the serving copy, ours)
- [x] Sidecar mechanism determined: INDEX-DRIVEN, no engine patch needed (see §1)
- [x] Decode reference mapped: core/exl3_format.zig (unpack/tileStates/codebookValue/dequantize)
      + headq.zig:298-325 dense (the engine's own EXL3→dense path) + bf16Bits rounding
      (cuda_weights.zig:993-997) — see §2
- [x] Decode path: NO numpy on the head → pivoted from a Python port to the BEST option: a
      small Zig util (worktree tools/glm53/sidecar.zig) that calls the engine's OWN decoder
      (headq.dense + core.exl3_format.dequantize + core.safetensors) — ZERO port deviation
      by construction. Compiled with the pinned Zig 0.17.0 (build-exe, modules core+headq).
- [x] Util self-checks: packChunk full-range identity (memcmp vs stored trellis); independent
      decode-order cross-check (dequantize parity-form vs dense butterfly); read-back verify
      (re-open sidecar, dtype/shape/offsets, eh_proj == canonical buffer, lm_head chunk-60
      re-decode bit-match). ALL PASS.
- [x] ADD-ON (75-vs-76 duplicate comparison): both copies decode BIT-IDENTICAL — see §4;
      plus a third-method raw-byte check on .111's local serving copy (same result), and a
      forensic scan showing the mission-log's "divergent" hash prefixes match NO tensor.
- [x] lm_head single-copy verified (shard 75; runtime assert: absent from 76); eh_proj
      canonical = shard 76 (index-driven; sidecar built from it)
- [x] Sidecars decoded chunked (peak RSS 1.10 GB, wall 57 s), stats clean (0 NaN/inf)
- [x] Sidecar written + sha256; staged under sidecars/
- [x] Copied to .111 local pack + index updated there (backup kept) + verified: sha256 match,
      ls, and the END-TO-END loader-acceptance check (--check: Checkpoint.openModel discovers
      both tensors via the index, 77 files, dtype/shape exact) — PASS
- [x] MISSION-LOG.md journaled

## §1 Sidecar mechanism verdict: SIDECAR-SUPPORTED via the pack index — NO engine patch

Evidence (all /home/ciprian/eugr/TensorFold-glm53-r2/zig/src/):
- core/checkpoint.zig:11-50 `shardFiles`: with `model.safetensors.index.json` present, the
  checkpoint opens EXACTLY the files named in `weight_map` (unique, sorted); without an index
  it globs `model*.safetensors`. => a sidecar file is discovered IFF added to weight_map.
- core/checkpoint.zig:57-135 `Checkpoint`: `get` (:93-100) searches ALL open files, first hit
  wins; `has` (:113-116) same. No name conflict: no shard holds `lm_head.weight` or
  `model.layers.78.eh_proj.weight` (pack-diff, packdiff-results.json).
- families/glm_moe_dsa/cuda_weights.zig:904 `load` uses `core.Checkpoint.openModel(dir)` —
  the glm53 path NEVER calls `Checkpoint.add` (:77-83 exists but unused here) and NEVER calls
  `unused()` (grep over families/glm_moe_dsa + native: no hits) => no leftover-refusal risk:
  a no-draft boot that leaves `lm_head.weight` unread will NOT fail on it.
- Consumers of the two names (loader-acceptance shapes):
  - `model.layers.78.eh_proj.weight`: cuda_weights.zig:743-750 mtpHead — `L.bf16(...)` =>
    dtype .bf16 required (bf16() :344-350 accepts bf16 as-is), rank 2, shape [hidden, 2*hidden]
    = [6144, 12288]; split.zig:75 lists `.eh_proj.weight` as replicated (whole tensor/rank).
  - `lm_head.weight`: cuda_weights.zig:719-732 headQ head_host — dtype .bf16, rank 2, shape
    [vocab, hidden] = [154880, 6144], row-split per rank (vocab share); :771-772 draftHead
    (draft_vocab path) — same tensor, same checks.
- => Sidecar file `model-sidecar-bf16.safetensors` in the pack dir + two weight_map entries
  in the SERVING copy's index (.111 local pack; NFS pack untouched). File name starts with
  "model" so even the no-index glob fallback (checkpoint.zig:38) would find it.

## §2 Decode reference (the engine's own path — ported faithfully)

- Bitstream/codebook: core/exl3_format.zig — `streamEnds` (:55-64; even halves:
  ends[p]=(p+1)*halves/2, ring=256*halves/2), `streamBit` (:67-74; MSB-first within
  little-endian int16 pairs → 32-bit words), `tileStates` (:77-88; 16-bit state per value,
  start=(ends[p]+ring-16)%ring), `tilePlacement` (:93-102; stream position → 16x16 tile
  row/col), mul1 `codebookValue` (:36-52; x=(state*0x83DCD12D) mod 2^32; h=1024+Σ4 bytes;
  fp16(h*fp16(0x1EEE)+fp16(0xC931)) — ONE f64 multiply-add then ONE fp16 rounding),
  `unpack` (:105-123; W_q [K,N] fp16), `dequantize` (:175-187; W = diag(suh)·H_K·W_q·H_N·diag(svh)).
- Dense layout + arithmetic order: headq.zig:298-325 `dense` — per 128-column block: fwht128f64
  (natural-order butterfly, strides 1..64) down each column, ×(1/√128)×suh; then fwht128f64
  along each row block, ×(1/√128)×svh; stored by output out[o*K+i] (=[N,K] checkpoint layout,
  exactly what a `.weight` holds). Output f32 (f64→f32 RNE at store).
- bf16 rounding: cuda_weights.zig:993-997 `bf16Bits` — f32→bf16 round-to-nearest-even via
  wrapping add (u +% (0x7FFF + ((u>>16)&1))) >> 16, NaN→0x7FFF. This is what kvBq (kvBq
  :475-515 → kvSplit :981-988) produces for an EXL3 kv_b — the sidecar equals EXACTLY what a
  kvBq-style load-decode patch would compute, so sidecar ≡ engine-patch output (documented
  deviation: none in arithmetic; f64 butterfly is bit-identical to fwht128f64's operation
  tree, and f64→f32→bf16 double rounding matches dense()+kvSplit).
- Chain: trellis i16 → states → mul1 fp16 W_q → f64 → H_K/suh → H_N/svh → f32 → bf16.

## §3 Port validation (vs the engine's oracle fixture) — RESULTS APPENDED BELOW

## §4 Decode + verification results — APPENDED BELOW

## §5 .111 copy + index update — APPENDED BELOW

### §3 DECODE PATH (final): the engine's own decoder, zero port deviation

No numpy exists on the head (system python3, spark-tools-env both lack it) — the planned
Python port was replaced by a BETTER path: `tools/glm53/sidecar.zig` in OUR worktree
(untracked addition; compiled with the pinned Zig 0.17.0 via `zig build-exe` with modules
core=zig/src/core/root.zig + headq=zig/src/families/glm_moe_dsa/headq.zig; no build.zig
change, no cuda/ibverbs dependency). The util calls:
- `headq.dense` (headq.zig:298-325) — the engine's own EXL3→dense path (what kvBq uses for
  an EXL3 kv_b), f64 butterflies + f64→f32 at store, output [N, K] = checkpoint `.weight`
  layout;
- `bf16Bits` (cuda_weights.zig:992-997) — copied VERBATIM into the util (f32→bf16 RNE,
  NaN→0x7FFF; the same rounding kvSplit applies) — so the sidecar equals EXACTLY what a
  kvBq-style load-decode engine patch would compute;
- `core.safetensors.File` for reading, `core.exl3_format.dequantize` (parity-form rotate)
  as an INDEPENDENT-ORDER cross-check, `core.Checkpoint.openModel` for the --check mode.
Deviations from the reference: NONE in arithmetic (same code); the only new code is chunk
packing (self-checked: full-range pack == stored bytes, memcmp), streaming file output
(shard_edit.zig's POSIX idiom), and the read-back verification.
Chunking: eh_proj 12×1024-output chunks, lm_head 121×1280-output chunks; peak RSS 1.10 GB
(/usr/bin/time -v), wall 57 s — far under the 9 GB constraint; lm_head never materialized
whole (written per-chunk).

### §4 RESULTS — sidecar + verification + THE ADD-ON COMPARISON

Sidecar: `sidecars/model-sidecar-bf16.safetensors` = 2,054,160,592 B (header 200 B +
eh_proj 150,994,944 B + lm_head 1,903,165,440 B),
sha256 **4165625752ecf3c964d3af332e33311d14a9ff3dc224179f2998193f82f18d23**.
Contents (loader-expected forms, cuda_weights.zig:749-750 / :724-725):
- `model.layers.78.eh_proj.weight` BF16 [6144, 12288] — decoded from CANONICAL shard-76
  EXL3 5-bit group (trellis [768,384,80] k2=10 + suh [12288] + svh [6144], mul1).
- `lm_head.weight` BF16 [154880, 6144] — decoded from the single-copy shard-75 EXL3 8-bit
  group (trellis [384,9680,128] k2=16 + suh [6144] + svh [154880], mul1).

Decode quality:
- eh_proj: n=75,497,472, NaN=0, inf=0, min=-0.8478, max=+2.3318 (absmax 2.3318) — plausible.
- lm_head: n=951,582,720, NaN=0, inf=0, min=-0.2891, max=+0.3009 (absmax 0.3009) — plausible
  for a head matrix (per-row Hadamard-domain scales are small).
- Independent-order check (dequantize parity-form vs dense butterfly, first 256 outputs):
  eh_proj max_abs_diff 1.16e-7 at absmax 2.33 (rel 5e-8); lm_head 7.4e-9 at absmax 0.189
  (rel 4e-8) — f64 summation-order noise only, both engine paths agree.
- Read-back verify: sidecar re-opened with the engine's safetensors reader; dtypes/shapes/
  offsets exact; eh_proj bytes == the in-memory canonical decode; a re-decoded lm_head
  chunk (60) matches bit for bit.

**ADD-ON (user-requested): shard-75 vs shard-76 copies of the 6 duplicated layer-78
tensors, decoded and compared (cosine + max abs diff):**

| tensor | cos (f32 decode) | maxabs (f32) | cos (bf16) | maxabs (bf16) | bits differing |
|---|---|---|---|---|---|
| eh_proj dense (trellis+suh+svh decoded, 75,497,472 values) | 1.000000 | **0** | 1.000000 | **0** | **0 / 75,497,472 (0.0000%)** |
| eh_proj.suh (f16 [12288]) | 1.000000 | 0 | — | — | 0 / 12288 |
| eh_proj.svh (f16 [6144]) | 1.000000 | 0 | — | — | 0 / 6144 |
| layers.78.enorm.weight (bf16 [6144]) | 1.000000 | 0 | — | — | 0 / 6144 |
| layers.78.hnorm.weight (bf16 [6144]) | 1.000000 | 0 | — | — | 0 / 6144 |
| eh_proj.mul1 (4 B marker) | — | — | — | — | bytes equal |

**Verdict on the hypothesis: the "two independent-but-valid quantizer passes" hypothesis is
DEAD. The two copies are BYTE-IDENTICAL — one quantized group written twice (a pure
packaging duplicate). Neither copy is a bad quant; the canonical-copy question is moot and
the index-driven read of 76 is risk-free either way.**

EVIDENCE TENSION (flagged for the orchestrator, not re-litigated here beyond the requested
comparison): the mission-log states the duplicates are DIVERGENT ("orchestrator
byte-verified", hashes 9be2ce5f… vs ce1dddac…, inspector vindicated). The requested add-on
comparison — plus TWO further independent checks run because the result contradicts that
premise — all show identity:
1. The engine's own safetensors reader (Zig, mmap, per-shard header offsets): all six
   tensors bit-identical (table above; includes DIRECT byte comparisons of suh/svh/enorm/
   hnorm/mul1, not just the decode).
2. A fresh Python process on **.111's local NVMe serving copy** (different storage, having
   no NFS in the path): raw-byte sha256 of all six tensors EQUAL across 75/76 — and equal to
   the original pack-diff lane's table (trellis afd8bdde…, suh 5ae47d84…, svh d0ab39f7…,
   mul1 20feb8ff…, enorm 08867935…, hnorm cadc2c62…).
3. Forensic scan: the claimed "divergent" hash prefixes (9be2ce5f / ce1dddac) match NO
   tensor ≤40 KB in either shard — they correspond to no raw tensor bytes there.
=> Three methods, two storage paths, one answer: the duplicates are identical. The
orchestrator's re-verification method (and the spark-tools inspector's offset handling)
should be re-examined and its exact procedure published; the mission-log correction of
2026-10-15 ("all SIX divergent, inspector vindicated") is contradicted by all available
byte-level evidence. The Stage-1 MTP draft gate stays regardless (cheap, retires everything).

### §5 .111 copy + index update + loader acceptance (all PASS)

- Copy: scp → `~/models111/GLM-5.3-4.04bpw-h8-exl3/model-sidecar-bf16.safetensors`;
  sha256 on .111 = **4165625752ecf3c964d3af332e33311d14a9ff3dc224179f2998193f82f18d23**
  (== local). Disk: 2.1 T free on / (nvme0n1p2). RAM after all ops: ~115 GB available.
- Index (the discovery mechanism, §1): `model.safetensors.index.json` on .111 updated
  (backup at `model.safetensors.index.json.nosidecar.bak`): weight_map += 2 entries
  ("lm_head.weight" and "model.layers.78.eh_proj.weight" → "model-sidecar-bf16.safetensors"),
  236,420 → 236,422 entries; re-dumped HF-style (indent 4, sorted keys); metadata.total_size
  (which lives under `metadata`, not top-level — the first update attempt KeyErrored
  harmlessly before writing) left at the original pack value: the loader never reads it
  (checkpoint.zig:17-31 reads only weight_map) and it keeps describing the original 360 GB.
- **END-TO-END loader acceptance on .111**: `~/glm53-sidecar --check <pack-dir>` (the
  binary copied there; the mode runs the EXACT discovery path the weights loader uses —
  `core.Checkpoint.openModel` → shardFiles reads the index → opens all files → `expect`
  dtype/shape): **PASS** — "eh_proj.weight BF16 [6144,12288] (150994944 B), lm_head.weight
  BF16 [154880,6144] (1903165440 B), files opened: 77" (76 shards + sidecar). Before the
  index update the same check correctly FAILED with MissingTensor — proving the mechanism
  is exactly the index (sidecar file alone is invisible, as §1 predicted).
- NFS pack untouched (read-only) ✓. No engine change needed ✓ (the no-engine-change path
  existed and was taken; the kvBq-style patch is NOT needed).
- Reuse: for other nodes' local packs (Stage 1 fleet decision), replicate = copy the
  sidecar + apply the same 2-entry index patch (or ship .111's updated index); the util's
  --check mode is the acceptance test.

### §6 What still blocks drafting (honest list)

1. NOTHING in the pack/loader path any more: with the sidecar in the index, mtpHead gets its
   bf16 eh_proj.weight and the drafters get bf16 lm_head rows (head_host / draft_vocab).
2. Stage-1 operational items (not blockers of the load): the MTP/DSpark A/B draft gate
   (also retires the eh_proj duplication residual); memory watch at TP4 (90 GB/rank weights
   + pool); AOT pack for the chosen shape (their published packs, bits-blind per pack-diff);
   tiles table choice (run table-less or re-tune).
3. The duplicate-bytes contradiction (§4) should be resolved by the orchestrator before the
   mission log records it as settled fact — the evidence here says identical.
