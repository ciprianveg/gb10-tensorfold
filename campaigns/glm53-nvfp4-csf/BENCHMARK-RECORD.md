# THE GLM-5.3 NVFP4 TENSORFOLD CAMPAIGN — THE BENCHMARK RECORD (DCP1 BEST)
**The shareable deliverable — every number with class/denominator/distribution.**
The form: D8/K0/MTP=dspark, conf 0.20, the bucket+width trims, the pool fixes. The serve: the
golden `serve8-best-cfg.sh` (the md5 ×8 gate; the restoration one command). This record carries
the campaign's best DCP1 rows; the full engineering log (the findings, the defect hunts, the
internal comparisons) is the campaign's private record and is not part of this publish.

## CREDITS

- **Ash Hart** — the upstream TensorFold framework this campaign's fork builds on (the DSA
  attention, the DCP cache planes, the draft/verify speculative spine).
- **Mia** — the pool-semantics design (max-not-equal-division, admission by actual usage,
  evict-oldest-idle) and the dense/head quantization patterns.

## SPEED (DCP1; greedy, temperature 0; the Snake class = 1500-token single-HTML Snake games)

| Row | The receipt (class / denominator / distribution) |
|---|---|
| **C1 code, DCP1 200K×2 (the speed tier)** | **47.31 / 49.08 / 47.92 t/s** (best-of-3 across runs; acceptance ~59%, yield 4.87; the sha 3d5bc78c2333 — the DCP1 tier's numerics class, byte-identical across sequential and concurrent) |
| **C1 code, DCP1 600K×1 (the deep form)** | **50.23 t/s** (the campaign's fastest C1; yield 5.03, acceptance 73% — the deepest context's rooms walk deepest; quality-verified by the 277K/360K needles) |
| rigMark code, DCP1 (the fuller bench; 5 runs, greedy, seed 20260905) | **41.9 t/s** (40.6–44.2; best-of-2 runs; 15/15 output gates) |
| C1 prose (1024 tok, thinking-off) | **25.50 t/s** (the entropy-bound class) |
| Structured (rigMark, JSON ceiling) | **57.2 t/s** (the JSON-constrained generation class; 58.4–61.1 across the final binaries) |
| **C2 code, DCP1 (2×1500 Snake concurrent)** | **69.50–70.89 tok/s aggregate** (34.75–35.45 a stream, makespan 43.2s; both streams byte-identical to the sequential sha — the concurrency isolation CLEAN) |
| rigMark concurrency (256-token code rounds, 3 rounds × C1/C2) | C1 33.6 / **C2 50.7** |
| Prefill, DCP1 | **1181 tok/s** (the 191954-token fill in 162.5s; the 600K form 1045–1100) |

## CAPACITY (DCP1)

| Row | The receipt |
|---|---|
| **The DCP1 600K×1 (the proportional bar — MET)** | Boots and serves (the un-split 33-GiB plane, the trimmed 75-graph walk, the floors 5); **deep-clean to 360K+** (the 277K/360K needles pristine); the Snake 50.23 t/s; prefill 1045–1100 tok/s |

## THE POOL SEMANTICS (CLOSED — Mia's design, our implementation)

Max-not-equal-division; per-request caps by actual usage (need = ceil((L+count+64)/dcp)+1);
first-fit packing (deterministic, no free-GB read); concurrent fills (serialized at the shared
~940–1100 tok/s rate); release-reuse (a later request pristine through freed holes);
evict-oldest-idle (idle kept states go oldest-first; in-flight generations absolute); the honest
edge refusal at the API. **The first-fit fragmentation is real and honest**: a claim needs a
CONTIGUOUS hole — concurrent arrivals that claim mid-plane can refuse a larger later claim whose
rows would fit in sum; the ordered submit (the large claim first) packs the same set.

## THE POOL-P4 ROWS (2026-11-08 — the measured default form, same-day, every row class/denominator/distribution)

All rows same-day 2026-11-08, greedy temperature 0, the pool-P4 form (DCP1, 4
slots, 600K shared pool, 200K caps) unless labeled otherwise. **The ranks
were connected at 100 Gb/s (one CX7 rail) for every row below; at 200 Gb/s
(both rails) prefill measures ~30% higher** (the user's receipt — the
prompt-pass all-gather is the link-bound stage; decode at these depths is
link-insensitive).

- **Snake-1500: 49.0 tok/s** (greedy, `/v1/completions`, the record's probe
  class — the 1500-token single-HTML Snake game; same-day reads 48.4–49.2
  across forms, the spread stated).
- **The concurrency matrix at pool-P4** (512-token generations, greedy, 2 runs
  per cell, the best reported per the convention):

  | streams | code | structured | prose |
  |---|---|---|---|
  | c1 | 47.5* | 53.1 | 27.6 |
  | c2 | 66.5 | 77.1 | 49.8 |
  | c4 | 85.7 | 87.4 | 76.3 |

- ***: the c1-code class read **44–49 tok/s** across the day's clean runs
  (this form, this prompt); two late-session reads (39.2/42.0) fell under
  heavy disk contention on the master and are excluded from the range. The
  cell = the best clean read.
- **Prefill 2×8K** (two distinct uncached prompts, `max_tokens=8`): **1282
  tok/s aggregate** (above the 1181 receipt).
- **The c2 trade, stated honestly**: pool-P4 reads ~5–7% below the P2 form
  per-stream at c2 (66.5 vs the P2-era 69.50–70.89) — both forms' rows
  carried, labeled by form. The P4 pool's c4 rows (85.7/87.4/76.3) are the
  ones the P2 form cannot reach: its c4 runs as 2 waves (72.0).
- **Determinism**: byte-identical greedy repeats, both probes, both forms.
- **Needle**: 128K-depth two-fact retrieval PASS, 105–107 s, both forms.
- **The pool-form anchor set of record** (8/8 reproduced across two same-day
  boots; cross-era continuity with the pool-era binaries):
  `f88320752d 4a56454311 3501b2452b 8490f74474 925cbb37e7 1195aad667 81485737e5 6642922800`.

## QUALITY

- The needles: pristine across forms and depths to **360K+** (the varied numbered-entry class; the
  periodic-filler confound ruled out).
- **The anchors byte-identical per layout class**: the DCP1 tier's class (3d5bc78c2333)
  byte-identical across its forms — the layout-numerics shift known, quality-verified by the deep
  needles at every class.
- Determinism: PASS (the sha-identity across runs, binaries, and concurrent-vs-sequential).
- The deep-quality break root-caused to the pool's stale views — **never the model's limit**.

## CONCURRENCY AND THE TIER GEOMETRY (DCP1, code-class)

- **C2 at DCP1**: 69.50–70.89 agg (the 2×1500 Snake class) — the two streams byte-identical.
- The fills serialize at the shared rate; the rooms adaptive (the per-round confidence-gated walks).
- **The tier**: DCP1 the ≤2-stream speed tier (the fastest C1 rows, prefill 1181, deep-parity).
  All ship as one-command arms (the prod script).

## THE ROOT-CAUSE ARC (the defects found, fixed, and receipted)

1. The stale-view line — the pool's whole corruption (the views built at init over an empty
   ledger, never rebuilt after a claim).
2. The wire-admit fatality — a request-class refusal at the wire's admit killed the engine.
3. The null-rung trim — the shallow capture-transition anomaly at deep forms.
4. The bucket-trim arithmetic — the 600K envelope's graph-set bound.
5. The pool-fit's dcp>1 gate — the pool branch refused DCP1 forms the arithmetic fit (the hoist +
   the guard; the DCP1 600K pool's enabler).
6. The claim-path PoolFull's rank divergence — TWO halves, both fixed: (a) the refusal mutated
   rank0's ledger (the drop + the evict) then aborted the wire replay — the dry-run coherence
   gate (poolHoleDry) decides the fit BEFORE any mutation; (b) the leader's admit catch marked
   the world broken unconditionally — the request-class refusal is symmetric under the gate and
   now mirrors the followers' catch (the collective-class stays fatal). Receipted: the over-cap
   leaves the serve alive AND at full speed; the anchors byte-identical throughout.

## THE SERVING-SETTINGS AUDIT (the official card vs our serve)

- **The sampling defaults**: greedy the served default (temperature 0; the client's temp/top_p
  honored when set); the pack's generation_config (T=1.0/top_p=0.95) matches the official card.
- **No repetition penalty anywhere** in the serve path (the card carries none; exact-token
  copying unharmed). One noted deviation: **top_k=20 is the server's default on the SAMPLED
  path** (moot at greedy — the argmax; live only at client-sampled requests; the card carries
  no top_k).
- **The reasoning-effort vocabulary: richer than the card** — `none | minimal | low | medium |
  high | xhigh | max` all accepted; anything else → the clean 400.
- **The tool-call template: clean both ways** — tools offered: the proper `tool_calls` array
  (the function name + JSON args, content empty); tools NOT offered: plain prose, no raw
  `<tool_call` markup ever reaches the client. The tool-choice handling renders through the
  same path.
- **The tool-definition format: the official form by construction** — the serve's template
  engine is a Jinja2 interpreter rendering the pack's own official `chat_template.jinja`
  (byte-parity-tested against the Python reference), the adapter passing the tools through
  unwrapped: the `<tools>` JSON envelope, the `tool_to_json` rendering, and the
  `{{function-name}}` call format are the trained-on form.
- **The exec-tool class: PASS, nothing to enable** — an execution-shaped tool renders and parses
  like any function; the serve is pure client-side execution (the correct posture). One operator
  note: **`tool_choice: "required"`/named → the clean 400** (the token-grammar enforcement — the
  libtfgrammar lane — is ported but parked; see the parked-work note below; use `auto`, which
  the probes show calls when appropriate).
- **clear_thinking: accepted but INERT** — the kwarg validates and the render drops no prior
  reasoning. **The port is owed**: the render-side strip of prior turns' think blocks when the
  flag is on (the official card's explicit recommendation), priced ~0.5-1 day, byte-identical
  when off.

## THE PARKED WORK AND THE PRICED REMAINDERS (the next-campaign menu)

- **The structured-output port (libtfgrammar) — PARKED, in-tree, one resume step from done**: the
  grammar-masked picks work end-to-end at the library level (the standalone harnesses exonerate
  the walk: the root fill pure, the accepts consistent); the remaining seat is ONE collective —
  the masked pick's 16-byte mid-round record gather (the fast oneshot times out; a raw NCCL op
  wedges between graph replays). The designed fix: the records over the rendezvous channel
  (host-reduced, bit-identical by construction). Until it lands, `tool_choice: required`/named
  and json_schema enforcement stay the clean 400.
- The refusal body's exact JSON (one line at the next pool boot — the status + the PoolFull
  reason receipted).
- The dense-fp8 lever sized (the user's floor ruling: dense layers never below 8-bit — the
  pricing document before any build).
- The 1.2.3 rebase (ruled in) with the upstream column-split (#1784) a target in its own right.
- The era's last-12% wall mystery (the 44.18 vs 49.90 class, unreproduced).
