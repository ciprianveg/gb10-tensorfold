//! GLM-5.3's fused forward for windows of 1 to 128 rows (fused.compute / layer / attention_part / attention_core /
//! ffn_part / head), one stream, one rank of the TP world.
//!
//! Exchanges (fused.gather's all-gather branch): every rank's fp32 partial [world, R, D] in rank order, then
//! glue.residual_add sums the world slots in rank order (WORLD = world), so every rank gets the same bits.
//! Phase 2a: `Forward.comm` set - NCCL all-gather of the real partials on the compute stream (the Python engine's
//! all-gather path for every window: tools/glm53/reference.py gives its engine a communicator without all_reduce /
//! all_to_all, so prompt windows too take this branch). `comm` null - the Phase 1 identity communicator (oracle.py's
//! LocalComm): b.gath zeroed, this rank's partial in its slot. Every kernel computes each row alone, so a row's bits
//! never depend on the window: the outputs match the Python engine's bit for bit when the kernels and launches do.
//! The head (fused.head mode "argmax"): this rank's logits over its vocabulary share (glue.router), its first
//! maximum (host: torch.argmax's rule), the [value, id, 0, 0] records all-gathered, the lowest rank of the maxima.
//!
//! Frozen as tools/glm53/oracle.py freezes the Python side: bf16 latent cache, DCP 1, one stream (TF_GLM53_SIDE=0:
//! the side stream gives the same bits), x3linear.plan tiles, TensorFold's expert layout, the shared expert folded
//! into the routed combine. Selection always goes through the radix select (topk.top_columns over _index_scores'
//! 4-byte order words): the same keys as torch.topk + sort, which the champion uses for decode-sized windows (the
//! oracle checks the two agree).
//! Phase 2b: the device head (`headDevice`: fused.head "argmax" / "local" / "draft" with no host step, so a window
//! replays as a CUDA graph), ids on the device (b.ids, torch.index_select's rows), the MTP layer (`mtpCompute`,
//! fused.mtp_compute with TF_GLM53_MTP_REUSE's keep / reuse) and the decode-class buffers (`BufOpts.decode`).
//! Phase 3a: the served configuration's exchanges - decode windows over the RoCE one-shot (`Forward.fast`: the
//! rank-order sum into b.red, the head's records and sampled candidates gathered over it), wider windows over NCCL's
//! bf16 ring all-reduce (`Forward.ring`, TF_GLM53_PREFILL_REDUCE=ring) - and windows over 128 rows (`Forward.wide`:
//! the prompt GEMM, torch.bmm absorb / expand, prompt experts); the sequence-parallel prompt chunk itself is
//! cuda_prompt.zig's.
//! Phase 3b: decode context parallelism (`Forward.dcp`, cuda_dcp.zig; fused._attention_dcp and select's DCP branch):
//! the caches sharded round robin over the ranks (State.initDcp), each rank's keys written by their owner only
//! (_kv_write / _ik_write DCP, RANK), every rank's queries gathered and all heads attended over this rank's keys, the
//! partials merged by log-sum-exp at the heads' owners; the indexer's per-rank top-k, gathered candidates and one
//! global top-k. DCP 1 keeps every Phase 3a path unchanged. TODO: int8/int4 latent caches, the exact reduce-scatter
//! (TF_GLM53_PREFILL_REDUCE=rs).
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const exl3 = @import("exl3.zig");
const kern = @import("cuda_kernels.zig");
const tri = @import("cuda_triton.zig");
const W = @import("cuda_weights.zig");
const smp = @import("sampling.zig");
const roce = @import("cuda_roce.zig");
const prompt = @import("cuda_prompt.zig");
const dec = @import("cuda_decode.zig");
const dcpm = @import("cuda_dcp.zig");

/// The widest TP world the head's exchange records hold.
pub const max_world: usize = 16;

/// One head pick: the token, this rank's local argmax and the gathered [world, 4] fp32 records (bit-compared with
/// the Python engine's b.amax_all).
pub const Pick = struct {
    token: u32 = 0,
    local: usize = 0,
    world: usize,
    gathered: [max_world * 4]f32 = @splat(0),
};

/// fused.FAST_ROWS: windows up to this many rows use the decode attention tiling (non-decode Buffers).
pub const fast_rows: usize = 16;
/// fused.MAX_ROWS: the widest window of the row-invariant EXL3 linear; wider prompt chunks take the prompt GEMM.
pub const max_rows: usize = 128;
/// fused.PROMPT_ROWS' default: the widest prompt chunk the served path's buffers take.
pub const max_prompt_rows: usize = 8192;
/// fused.DECODE_ROWS: the widest decode-class buffer set.
pub const decode_rows: usize = 32;
/// fused.RADIX_MIN_ROWS: selections of fewer rows (decode windows) take the multi-block top-k (glm53_decode.cu) - the
/// Python engine's torch.topk + sort branch there; the same columns as the one-program radix select.
pub const radix_min_rows: usize = 32;

pub const BufOpts = struct {
    /// fused.Buffers(decode=True): verify and MTP windows
    decode: bool = false,
    /// the MTP layer's scratch (hin, me, mh, mcat, mx32, eh_proj partials)
    mtp: bool = false,
    /// Phase 3a: prompt chunks of more than max_rows rows (the served path's 8192 / 4096-row buffers): the EXL3
    /// linear scratch stays at max_rows rows (wider calls take the prompt GEMM's workspace), plus the sequence-
    /// parallel and ring-reduce buffers (bpart, bred, xo)
    prompt: bool = false,
    /// weights.EXPERTS_IMPL == "shared" (experts_cx.SharedExperts): the decode experts' scratch holds min(rows,
    /// MAX_ROWS) rows and wider windows take the prompt experts kernel (prompt_experts.cu); false ("tf"): the decode
    /// kernel at every width, its scratch for every row
    shared_experts: bool = false,
    /// Phase 3a speed (cuda_prompt.MtpDense, --mtp-dense 1): the MTP layer's prompt rows take cuBLAS for eh_proj and
    /// the dense experts, so the eh_proj router partials / fp32 rows stay at max_rows rows and no decode-kernel
    /// blocks are carved (xs_big)
    mtp_dense: bool = false,
    /// Phase 3b: decode context parallelism over this many ranks (fused.Buffers with w.dcp > 1): partial slots for
    /// every rank's heads, the query / partial exchange buffers, the selection's candidates; the score scratch holds a
    /// rank's ceil(score_cols / dcp) slots
    dcp: usize = 1,
    /// Phase 4: fused.Buffers.taps - the drafters' tap layers' output rows, [rows, taps * D] bf16 (0: no drafter)
    taps: usize = 0,
    /// TF_GLM53_E1_FP8 (section 18): the DCP E1 query gather ships kv8Quant-packed e4m3 rows (584 bytes:
    /// codes + the s0/s1 scales) instead of bf16 - half the wire, the scorer dequantizes in-register
    e1_fp8: bool = false,
};

/// Phase 4: fused.Weights.tap_slot - which target layers' output rows the loaded drafters read and where in
/// Buffers.taps (draft_host.TapPlan's slots); n 0: no drafter.
pub const Taps = struct {
    slot: [256]i16 = @splat(-1),
    n: usize = 0,

    pub fn of(plan: anytype) !Taps {
        var t: Taps = .{};
        for (plan.layers[0..plan.n], 0..) |l, i| {
            if (l >= t.slot.len) return error.Invalid;
            t.slot[l] = @intCast(i);
        }
        t.n = plan.n;
        return t;
    }
};

/// What a window's head computes (fused.head's modes; `none`: no head).
pub const HeadMode = enum { none, argmax, local, draft };

/// fused.Buffers: scratch for windows of up to `rows` rows (sliced [:R]); `score_cols` the widest index range.
pub const Buffers = struct {
    d: *const cuda.Driver,
    rows: usize,
    score_cols: usize,
    small: usize,
    mem: std.ArrayList(cuda.DeviceBuffer) = .empty,
    gpa: std.mem.Allocator,
    x: u64 = 0, // bf16 [rows, D]
    normed: u64 = 0,
    qa: u64 = 0, // bf16 [rows, q_lora]
    qn: u64 = 0,
    kva: u64 = 0, // bf16 [rows, kv_a.n]
    kva_n: usize = 0,
    q: u64 = 0, // bf16 [rows, H * (nope + rope)]
    qlat: u64 = 0, // bf16 [rows, H, kv_lora]
    qrot: u64 = 0, // bf16 [rows, H, rope]
    po: u64 = 0, // fp32 chunk partials
    pm: u64 = 0,
    pl: u64 = 0,
    ol: u64 = 0, // bf16 [rows, H, kv_lora]
    o: u64 = 0, // bf16 [rows, H * v]
    dummy: u64 = 0, // int32 [1] zero
    ik: u64 = 0, // fp32 [rows, index_dim]
    iw: u64 = 0, // fp32 [rows, index_heads]
    iq: u64 = 0, // bf16 [rows, index_heads * index_dim]
    sc: u64 = 0, // int64 [min(rows, 128) * score_cols] (int32 order words here)
    tok: u64 = 0, // int32 [rows, index_topk]
    g: u64 = 0, // bf16 [rows * width]
    u: u64 = 0,
    act: u64 = 0,
    mlog: u64 = 0, // fp32 [rows, E]
    pick: u64 = 0, // int32 [rows, top_k]
    wts: u64 = 0, // fp32 [rows, top_k]
    rpart: u64 = 0, // fp32 router partials [ROUTER_KS, rows, max(E, index_dim)]
    sy: u64 = 0, // fp32 [rows, D]
    part: u64 = 0, // fp32 [rows, D]
    gath: u64 = 0, // fp32 [world * rows * D]: every rank's partial in rank order (fused.gather)
    // the head (allocated when the weights hold one): fused.Buffers fnormed / lpart / amax / amax_all
    head_rows: usize = 0, // min(rows, small)
    fnormed: u64 = 0, // bf16 [head_rows, D]
    lg: u64 = 0, // fp32 [head_rows, vocab_part]
    hpart: u64 = 0, // fp32 [ROUTER_KS, head_rows, vocab_part] (glue.router's partials); a quantized head: its launch's result, fp32 [head_rows, blocks * 128]
    hxh: u64 = 0, // a quantized head (TF_GLM53_HEAD_Q): fp16 [head_rows, D], the launch's rotated input
    amax: u64 = 0, // fp32 [head_rows, 4]: [max logit, global id, 0, 0] a row
    amax_all: u64 = 0, // fp32 [world, head_rows, 4]
    // Phase 2b: device ids, the hidden rows the MTP layer reads and writes, its scratch, the device head
    ids: u64 = 0, // int64 [rows]
    hin: u64 = 0, // bf16 [rows, D]: the hidden rows the MTP layer reads
    hidden: u64 = 0, // bf16 [rows, D]: the window's final (target) or chain (MTP) hidden rows
    ht: u64 = 0, // bf16 [rows, D]: target_hidden_for_mtp of a prompt chunk
    me: u64 = 0, // bf16 [rows, D]
    mh: u64 = 0,
    mcat: u64 = 0, // bf16 [rows, 2 D]
    mx32: u64 = 0, // fp32 [rows, D]
    epart: u64 = 0, // fp32 [ROUTER_KS, rows, D]: eh_proj's router partials
    erows: usize = 0, // rows mx32 / epart hold (BufOpts.mtp_dense: min(rows, max_rows))
    argmax: u64 = 0, // int64 [head_rows]
    gsum: u64 = 0, // fp32 [head_rows, D / 64]: the draft head's input group sums
    head_v: usize = 0, // b.lg holds head_rows rows of up to this many logits (max(vocab_part, draft rows))
    cand_words: usize = 0, // words a rank sends for sampled candidates: head_rows * 2 * max_candidates + 1
    cand_send: u64 = 0, // fp32 [cand_words rounded up to 4]
    cand_all: u64 = 0, // fp32 [world * cand_words rounded up to 4]
    lin: kern.LinearScratch = undefined,
    xs: ?kern.ExpertScratch = null,
    // Phase 3a
    red: u64 = 0, // fp32 [rows, D]: the RoCE sum (decode windows) / the ring all-reduce's bf16 sum widened (prompts)
    bpart: u64 = 0, // bf16 [rows, D]: a prompt chunk's partial for the bf16 ring all-reduce / reduce-scatter
    bred: u64 = 0, // bf16 [rows, D]
    xo: u64 = 0, // bf16 [ceil(rows / world), D]: sequence-parallel own rows of the residual stream
    xs_rows: usize = 0, // rows the decode experts' scratch holds
    pe: ?kern.PromptExpertScratch = null, // prompt experts (SharedExperts.prefill), prompt buffers of the shared impl
    // Phase 3a speed
    slin: kern.LinearScratch = undefined, // TF_GLM53_SIDE: the side stream's EXL3 scratch (min(rows, small) rows)
    srpart: u64 = 0, // the side stream's router partials
    tsel: kern.TopkScratch = .{}, // the multi-block top-k (decode-sized selections, sampled candidates)
    cand_cols: u64 = 0, // int32 [head_rows, max_candidates]: a sampled window's candidate columns
    /// the MTP layer's prompt rows (8-bit experts prompt_experts does not take): the decode kernel in blocks of
    /// xs_big.rows rows, its big buffers inside b.gath (idle while the MTP layer runs: prompt chunks reduce over the
    /// bf16 ring); null: blocks of xs_rows
    xs_big: ?kern.ExpertScratch = null,
    // Phase 3b: decode context parallelism (fused.Buffers with w.dcp > 1; the exchange buffers 0 at DCP 1)
    dcp: usize = 1,
    qpack: u64 = 0, // bf16 [rows, H, lw + rd]: this rank's absorbed queries, packed
    qall: u64 = 0, // bf16 [G, rows, H, lw + rd]: every rank's
    qpack8: u64 = 0, // u8 [rows * H, lw + rd + 8]: the E1 fp8 wire's packed rows (TF_GLM53_E1_FP8; e4m3 codes + 2 fp32 scales)
    qall8: u64 = 0, // u8 [G, rows * H, lw + rd + 8]: the gathered fp8 wire (QALL + its QSC word view)
    osend: u64 = 0, // bf16 [G, rows, H, lw]: normalized partials by destination rank
    lsend: u64 = 0, // fp32 [G, rows, H]: their log-sum-exps
    e3send: u64 = 0, // 126(g) the merged E3 send: [G, rows, H, LW+2] bf16 - per group [o || lse] for ONE gather
    e3recv: u64 = 0, // 126(g) the merged E3 recv: [fan, G, rows, H, LW+2] bf16
    orecv: u64 = 0, // bf16 [fan, G, rows, H, lw] (fan G for decode windows' all-gather, 1 for the all-to-all)
    lrecv: u64 = 0, // fp32 [fan, G, rows, H]
    cnt: u64 = 0, // int32 [rows]: this rank's share of each row's selection
    cand: u64 = 0, // int64 [G, min(rows, 128), index_topk]: every rank's candidate keys
    dmine: u64 = 0, // int64 [min(rows, 128), index_topk]: this rank's candidates (int64 min + 1 padded)
    dsel: u64 = 0, // int64 [min(rows, 128), index_topk]: topk.top_keys' output (radix branch)
    // Phase 4: the drafters' taps (fused.Buffers.taps)
    taps: u64 = 0, // bf16 [rows, tap_n * D]
    tap_n: usize = 0,

    fn alloc(b: *Buffers, bytes: usize) !u64 {
        var m = try cuda.DeviceBuffer.alloc(b.d, @max(bytes, 16));
        errdefer m.free();
        try m.fill8(0, null); // torch.zeros / empty: zero here, so stray reads are deterministic
        try b.mem.append(b.gpa, m);
        return m.ptr;
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, rows: usize, score_cols: usize) !Buffers {
        return initWith(gpa, d, w, rows, score_cols, .{});
    }

    /// `opts.decode` (fused.Buffers(decode=True)): every window these buffers serve is a decode / verify window,
    /// whatever its width: the decode attention tiling and head rows up to max(rows, FAST_ROWS).
    pub fn initWith(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, rows: usize, score_cols: usize, opts: BufOpts) !Buffers {
        const c = w.config;
        if (rows > max_rows and !opts.prompt) return error.WindowTooWide;
        if (rows > max_prompt_rows) return error.WindowTooWide;
        if (opts.decode and rows > decode_rows) return error.WindowTooWide;
        var b: Buffers = .{ .d = d, .gpa = gpa, .rows = rows, .score_cols = score_cols, .small = if (opts.decode) @max(rows, fast_rows) else fast_rows };
        errdefer b.deinit();
        const D = c.hidden;
        const H = w.heads;
        const lw = c.kv_lora;
        const rd = c.rope;
        const qd = c.qkDim();
        b.x = try b.alloc(rows * D * 2);
        b.normed = try b.alloc(rows * D * 2);
        b.qa = try b.alloc(rows * c.q_lora * 2);
        b.qn = try b.alloc(rows * c.q_lora * 2);
        b.kva_n = w.layers[0].kv_a.n;
        b.kva = try b.alloc(rows * b.kva_n * 2);
        b.q = try b.alloc(rows * H * qd * 2);
        b.qlat = try b.alloc(rows * H * lw * 2);
        b.qrot = try b.alloc(rows * H * rd * 2);
        const G = @max(opts.dcp, 1);
        b.dcp = G;
        // DCP: partials for every rank's heads
        const slots = @max(c.index_topk / tri.attn_decode.chk * @min(rows, b.small), c.index_topk / tri.attn_prompt.chk * rows) * G;
        if (G > 1) {
            // The SMALL windows at dcp < world gather the WHOLE world (the in-graph all-gather / RoCE one-shot;
            // captured NCCL p2p bled ~180 MB a graph at the 2x600K mirror - findings-dcp2-build.md §C): their
            // qall holds world blocks (the kernel reads its scored groups' at a base offset into it), the small
            // orecv/lrecv [world][G]. The prompt buffers (rows > small) keep the G-sized subgroup p2p buffers.
            const wide = G != w.world and rows <= b.small;
            const qb: usize = if (wide) w.world else G;
            b.qpack = try b.alloc(rows * H * (lw + rd) * 2);
            b.qall = try b.alloc(qb * rows * H * (lw + rd) * 2);
            if (opts.e1_fp8) {
                b.qpack8 = try b.alloc(rows * H * (lw + rd + 8));
                b.qall8 = try b.alloc(G * rows * H * (lw + rd + 8));
            }
            b.osend = try b.alloc(G * rows * H * lw * 2);
            b.lsend = try b.alloc(G * rows * H * 4);
            b.e3send = try b.alloc(G * rows * H * (lw + 2) * 2);
            const fan: usize = if (rows <= b.small) (if (wide) w.world else G) else 1; // decode windows exchange by all-gather (G x the bytes)
            b.e3recv = try b.alloc(fan * G * rows * H * (lw + 2) * 2);
            b.orecv = try b.alloc(fan * G * rows * H * lw * 2);
            b.lrecv = try b.alloc(fan * G * rows * H * 4);
            b.cnt = try b.alloc(rows * 4);
            const sel_rows = @min(rows, 128);
            b.cand = try b.alloc(G * sel_rows * c.index_topk * 8);
            b.dmine = try b.alloc(sel_rows * c.index_topk * 8);
            b.dsel = try b.alloc(sel_rows * c.index_topk * 8);
        }
        b.po = try b.alloc(slots * H * lw * 4);
        b.pm = try b.alloc(slots * H * 4);
        b.pl = try b.alloc(slots * H * 4);
        b.ol = try b.alloc(rows * H * lw * 2);
        b.o = try b.alloc(rows * H * c.v_dim * 2);
        b.dummy = try b.alloc(4);
        b.ik = try b.alloc(rows * c.index_dim * 4);
        b.iw = try b.alloc(rows * c.index_heads * 4);
        b.iq = try b.alloc(rows * c.index_heads * c.index_dim * 2);
        b.sc = try b.alloc(@min(rows, 128) * dcpm.cdiv(score_cols, G) * 8); // Buffers.score_len
        b.tok = try b.alloc(rows * c.index_topk * 4);
        const width = @max(c.dense_width, c.expert_width * @max(c.shared_experts, 1)) / w.world;
        b.g = try b.alloc(rows * width * 2);
        b.u = try b.alloc(rows * width * 2);
        b.act = try b.alloc(rows * width * 2);
        b.mlog = try b.alloc(rows * c.experts * 4);
        b.pick = try b.alloc(rows * c.top_k * 4);
        b.wts = try b.alloc(rows * c.top_k * 4);
        b.rpart = try b.alloc(tri.router_ks * rows * @max(c.experts, c.index_dim) * 4);
        b.sy = try b.alloc(rows * D * 4);
        b.part = try b.alloc(rows * D * 4);
        b.gath = try b.alloc(w.world * rows * D * 4);
        if (w.hasHead()) {
            const hr = @min(rows, b.small);
            const V = @max(w.vocab_part, if (w.draft) |dh| dh.n else 0);
            b.head_rows = hr;
            b.head_v = V;
            b.fnormed = try b.alloc(hr * D * 2);
            b.lg = try b.alloc(hr * V * 4);
            b.hpart = try b.alloc(hr * 4 * if (w.head_q) |hq| @max(hq.lin.n, V) else tri.router_ks * V);
            if (w.head_q != null) b.hxh = try b.alloc(hr * D * 2);
            b.amax = try b.alloc(hr * 4 * 4);
            b.amax_all = try b.alloc(w.world * hr * 4 * 4);
            b.argmax = try b.alloc(hr * 8);
            b.gsum = try b.alloc(hr * (D / 64) * 4);
            b.cand_words = hr * 2 * smp.max_candidates + 1;
            const padded = std.mem.alignForward(usize, b.cand_words, 4); // fused.small_gather's 16-byte RoCE packs
            b.cand_send = try b.alloc(padded * 4);
            b.cand_all = try b.alloc(w.world * padded * 4);
        }
        // TF_GLM53_SIDE: the side stream's own EXL3 and router scratch (decode windows: up to min(rows, small) rows)
        const srows = @min(@min(rows, b.small), max_rows);
        {
            const sext = w.linearExtents();
            b.slin.xh_bytes = srows * sext.k * 2;
            b.slin.z_bytes = srows * sext.zn * 4;
            for (0..exl3.max_jobs) |j| {
                b.slin.xh[j] = try b.alloc(b.slin.xh_bytes);
                b.slin.z[j] = try b.alloc(b.slin.z_bytes);
            }
            b.srpart = try b.alloc(tri.router_ks * srows * @max(c.experts, c.index_dim) * 4);
        }
        // the multi-block top-k: selections of fewer than radix_min_rows rows over score_cols keys, and the sampled
        // candidates of the head rows over the vocabulary share
        {
            // ROUND 3 (§I): 128 rows so the PROMPT path's 128-row select blocks can take the compact
            // radix (tsel.rows gated them out before - the prompt select ran top_keys, §I's reframe)
            const trows = @max(@max(@min(rows, radix_min_rows), b.head_rows), @min(rows, 128));
            const cols = @max(score_cols, b.head_v);
            const tblocks = (cols + kern.topk_chunk - 1) / kern.topk_chunk;
            b.tsel = .{ .hist = try b.alloc(kern.TopkScratch.histBytes(trows)), .state = try b.alloc(trows * 16), .cnt = try b.alloc(trows * tblocks * 8), .rows = trows, .blocks = tblocks, .list = try b.alloc(trows * kern.topk_compact_cap * 4), .lcnt = try b.alloc(trows * 4), .lflg = try b.alloc(trows * 4), .cap = kern.topk_compact_cap };
            if (b.head_rows > 0) b.cand_cols = try b.alloc(b.head_rows * smp.max_candidates * 4);
        }
        b.red = try b.alloc(rows * D * 4);
        if (rows > b.small) {
            b.bpart = try b.alloc(rows * D * 2);
            b.bred = try b.alloc(rows * D * 2);
            if (opts.prompt) b.xo = try b.alloc((rows + w.world - 1) / w.world * D * 2);
        }
        b.ids = try b.alloc(rows * 8);
        b.hidden = try b.alloc(rows * D * 2);
        if (opts.taps > 0) {
            b.taps = try b.alloc(rows * opts.taps * D * 2);
            b.tap_n = opts.taps;
        }
        if (opts.mtp) {
            b.hin = try b.alloc(rows * D * 2);
            b.ht = try b.alloc(rows * D * 2);
            b.me = try b.alloc(rows * D * 2);
            b.mh = try b.alloc(rows * D * 2);
            b.mcat = try b.alloc(rows * 2 * D * 2);
            b.erows = if (opts.mtp_dense) @min(rows, max_rows) else rows;
            b.mx32 = try b.alloc(b.erows * D * 4);
            b.epart = try b.alloc(tri.router_ks * b.erows * D * 4);
        }
        // EXL3 linear scratch: per job fp16 Xh [rows, K] and fp32 Z [SK, rows, N] (the row-invariant kernel: up to
        // max_rows rows; prompt chunks' wider calls take the prompt GEMM's workspace)
        const ext = w.linearExtents();
        const lrows = @min(rows, max_rows);
        b.lin.xh_bytes = lrows * ext.k * 2;
        b.lin.z_bytes = lrows * ext.zn * 4;
        for (0..exl3.max_jobs) |j| {
            b.lin.xh[j] = try b.alloc(b.lin.xh_bytes);
            b.lin.z[j] = try b.alloc(b.lin.z_bytes);
        }
        // experts.Scratch(ex, rows, slots = top_k); SharedExperts.scratch: min(rows, MAX_ROWS) rows
        const xrows = if (opts.shared_experts) @min(rows, max_rows) else rows;
        b.xs_rows = xrows;
        for (w.layers) |*L| {
            const ex = if (L.experts) |*e| e else continue;
            const P = xrows * c.top_k;
            const gu = try exl3.defaultTiles(ex.dims, ex.width, true);
            const dn = try exl3.defaultTiles(ex.width, ex.dims, false);
            const maxu = @min(P, ex.count);
            const xs: kern.ExpertScratch = .{
                .xg = try b.alloc(P * ex.dims * 2),
                .xu = try b.alloc(P * ex.dims * 2),
                .xd = try b.alloc(P * ex.width * 2),
                .z = try b.alloc(@max(2 * gu.sk * ex.width, dn.sk * ex.dims) * P * 4),
                .y = try b.alloc(P * ex.dims * 4),
                .ids = try b.alloc(maxu * 4),
                .count = try b.alloc(4),
                .members = try b.alloc(maxu * xrows * 4),
                .done_gu = try b.alloc(maxu * ((xrows + 15) / 16) * @max(ex.width / 128, 1) * 4),
                .done_d = try b.alloc(xrows * @max(ex.dims / 128, 1) * 4),
                .rows = xrows,
                .slots = c.top_k,
                .gu = gu,
                .dn = dn,
            };
            // members_buf starts at -1 (torch.full(..., -1))
            try b.d.check(b.d.api.cuMemsetD32_v2(xs.members, 0xFFFFFFFF, maxu * xrows), "cuMemsetD32");
            b.xs = xs;
            // prompt_experts.PromptScratch(ex, min(rows, CHUNK_ROWS), top_k) + the slot-order sums' rows
            if (opts.shared_experts and rows > xrows) {
                const pr = @min(rows, exl3.pe_chunk_rows);
                const n = pr * c.top_k;
                const ipm: usize = 112; // prompt_experts.cu IPM (TF_PE_GPM); Kernels.pe.ipm checks it at launch
                const max_items = n / ipm + @min(n, ex.count);
                b.pe = .{
                    .rows = pr,
                    .slots = c.top_k,
                    .max_items = max_items,
                    .sorted = try b.alloc(n * 4),
                    .items = try b.alloc(3 * max_items * 4),
                    .count = try b.alloc(4),
                    .xd = try b.alloc(n * ex.width * 2),
                    .pairs = try b.alloc(n * ex.dims * 2),
                    .acc = try b.alloc(pr * ex.dims * 4),
                    .out16 = try b.alloc(pr * ex.dims * 2),
                };
            }
            // the MTP layer's prompt rows in bigger blocks of the decode kernel (row-invariant: the same bits as
            // blocks of xs_rows), xg / xu / y / z carved out of b.gath (world * rows * D fp32); the group kernel
            // stages a block's picks in 48 KiB of shared memory (1536 rows of 8 slots)
            if (opts.prompt and opts.mtp and !opts.mtp_dense and opts.shared_experts and rows > xrows and b.gath != 0) {
                const zcols = @max(2 * gu.sk * ex.width, dn.sk * ex.dims);
                const per_slot = std.mem.alignForward(usize, ex.dims * 2, 256) * 2 + ex.dims * 4 + zcols * 4;
                const room = w.world * rows * D * 4;
                var rb = room / (per_slot * c.top_k + 4 * 256);
                rb = @min(@min(rb, rows), 49152 / (4 * c.top_k));
                rb = rb / 16 * 16;
                if (rb > xrows) {
                    const Pb = rb * c.top_k;
                    const maxub = @min(Pb, ex.count);
                    // carved in order: xg, xu, z, y (each 256-byte aligned)
                    var off: usize = 0;
                    var region: [4]u64 = undefined;
                    for ([_]usize{ Pb * ex.dims * 2, Pb * ex.dims * 2, zcols * Pb * 4, Pb * ex.dims * 4 }, 0..) |bytes, j| {
                        region[j] = b.gath + off;
                        off = std.mem.alignForward(usize, off + bytes, 256);
                    }
                    const big: kern.ExpertScratch = .{
                        .xg = region[0],
                        .xu = region[1],
                        .xd = try b.alloc(Pb * ex.width * 2),
                        .z = region[2],
                        .y = region[3],
                        .ids = try b.alloc(maxub * 4),
                        .count = try b.alloc(4),
                        .members = try b.alloc(maxub * rb * 4),
                        .done_gu = try b.alloc(maxub * ((rb + 15) / 16) * @max(ex.width / 128, 1) * 4),
                        .done_d = try b.alloc(rb * @max(ex.dims / 128, 1) * 4),
                        .rows = rb,
                        .slots = c.top_k,
                        .gu = gu,
                        .dn = dn,
                    };
                    if (off > room) return error.ScratchTooSmall;
                    try b.d.check(b.d.api.cuMemsetD32_v2(big.members, 0xFFFFFFFF, maxub * rb), "cuMemsetD32");
                    b.xs_big = big;
                }
            }
            break; // one scratch serves every MoE layer (the same shapes)
        }
        return b;
    }

    pub fn deinit(b: *Buffers) void {
        for (b.mem.items) |*m| m.free();
        b.mem.deinit(b.gpa);
        b.* = undefined;
    }
};

/// fused.State: a 576-wide latent row a token and layer, bf16 index keys on full-indexer layers, the device position.
/// DCP (Phase 3b): position p on rank p % dcp at local row p // dcp; every cache holds `local` = ceil(capacity / dcp)
/// + 1 rows (capacity + 1 at DCP 1).
/// TF_GLM53_KV8: the latent rows (kc / mkc) are e4m3 codes + two fp32 scales instead - `lat_row` 584 bytes, 50.8% of
/// bf16; the index keys stay bf16 (the DSA top-k scoring reads all of them every step: quantized scores would move
/// the selected set, not just the values). The latent planes' readers go through the staged quantize / dequantize of
/// cuda_kernels' kv8 pair (Kv8 below); nothing that copies or saves whole rows changes (one plane a layer still).
pub const State = struct {
    d: *const cuda.Driver,
    gpa: std.mem.Allocator,
    capacity: usize,
    dcp: usize = 1,
    local: usize = 0,
    kc: []u64, // per layer: bf16 [local, latent] - kv8: e4m3 codes + 2 fp32 scales a row (latentRow bytes)
    ic: []u64, // per layer: bf16 [local, index_dim], 0 on layers without an indexer
    pos: u64, // int32 [1]
    mkc: u64 = 0, // the MTP layer's latent rows (0: no MTP layer)
    mic: u64 = 0, // the MTP layer's index keys (0: no indexer there)
    mpos: u64 = 0, // int32 [1]: the MTP window's first position
    mem: std.ArrayList(cuda.DeviceBuffer) = .empty,
    /// Phase 4b (fused.State(slots=N)): each cache holds `slots` streams' rows back to back, `local` rows each (slot s
    /// at rows [s local, (s + 1) local)); `slotView` is one stream alone (the one-stream kernels on its rows). A view
    /// owns only its kc / ic arrays (`view` true: `freeView`, never `deinit`).
    slots: usize = 1,
    view: bool = false,
    /// §128.27az THE POOL (the shared sequence plane): the per-slot row INTERVALS - slot s's rows at
    /// [slot_start[s], slot_start[s] + slot_rows[s]) of the plane (0 rows = the slot free; an eviction
    /// leaves a hole - NO compaction, NO row moves: the actives bit-unaffected by construction). null:
    /// the uniform pre-pool layout (slot s at [s * local, ...), bit-identical). The boundaries are LEDGER
    /// ARITHMETIC on the Multi's mirrored admit/evict ledger - never free-GB reads (the coherence ruling).
    slot_start: ?[]usize = null,
    slot_rows: ?[]usize = null,
    /// TF_GLM53_KV8: one latent row's bytes (584: 576 e4m3 + 2 fp32 scales; bf16: 1152).
    kv8: bool = false,
    kv8_v3: bool = false, // 119.2: the 656-byte rows (TF_GLM53_KV8_V3; v2: 584)
    lat_row: usize = 0,

    /// One latent row's bytes: kv8's e4m3 + scales, else bf16.
    pub fn latentRow(kv8: bool, c: anytype) usize {
        return latentRowV3(kv8, false, c);
    }

    /// 119.2 (TF_GLM53_KV8_V3): kv8 v3's row - the latent half's e4m3 codes with a scale a 128-element
    /// block, the rope half bf16 (exact), LW/128 fp32 scales: kv_lora + rope * 2 + 16 = 656 at 512/64
    /// (v2: 584; +12.3% plane bytes for the rope's exactness and 4x finer latent scales).
    pub fn latentRowV3(kv8: bool, v3: bool, c: anytype) usize {
        if (!kv8) return c.latentWidth() * 2;
        return if (v3) c.kv_lora + c.rope * 2 + 16 else c.latentWidth() + 8;
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, capacity: usize) !State {
        return initDcp(gpa, d, w, capacity, 1);
    }

    /// fused.State(w, capacity) with w.dcp = `dcp`: this rank's shard, ceil(capacity / dcp) + 1 rows a cache.
    pub fn initDcp(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, capacity: usize, dcp: usize) !State {
        return initSlots(gpa, d, w, capacity, dcp, 1);
    }

    /// fused.State(w, capacity, slots): `slots` streams of ceil(capacity / dcp) + 1 rows each. Several streams with
    /// the caches sharded (slots > 1, dcp > 1) are this port's own (docs/glm53-dcp-streams.md): each stream's shard of
    /// this rank back to back, stream s's position p at row s * local + p / dcp on rank p % dcp.
    pub fn initSlots(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, capacity: usize, dcp: usize, slots: usize) !State {
        return initSlotsKv8(gpa, d, w, capacity, dcp, slots, false);
    }

    /// initSlots with TF_GLM53_KV8's latent rows (the Runner's settings carry the flag; DCP 1 only for now).
    pub fn initSlotsKv8(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, capacity: usize, dcp: usize, slots: usize, kv8: bool) !State {
        return initSlotsKv8V3(gpa, d, w, capacity, dcp, slots, kv8, false);
    }

    /// §128.27az THE POOL: the plane pre-sized at the config's max AGGREGATE (rows_total = this rank's
    /// rows of the aggregate cap); the slots' boundaries are set by the Multi's ledger at admit/evict
    /// (fill8(0) of the admitted interval ~60 ms per 12 GiB slot; the eviction frees the interval - no
    /// device alloc/free in the admit path, no fragmentation beyond the hole list). `local` carries the
    /// plane's rows (the per-slot bound lives in slot_rows); the uniform initSlots* stay bit-identical.
    pub fn initPoolV3(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, rows_total: usize, dcp: usize, slots: usize, kv8: bool, v3: bool) !State {
        const c = w.config;
        if (dcp == 0 or (dcp > 1 and (dcp > w.world or w.world % dcp != 0))) return error.Invalid;
        if (slots == 0 or rows_total == 0) return error.Invalid;
        const lat = latentRowV3(kv8, v3, c);
        var s: State = .{ .d = d, .gpa = gpa, .capacity = rows_total * @max(dcp, 1), .dcp = dcp, .local = rows_total, .slots = slots, .kv8 = kv8, .kv8_v3 = v3, .lat_row = lat, .kc = try gpa.alloc(u64, w.layers.len), .ic = try gpa.alloc(u64, w.layers.len), .pos = 0, .slot_start = try gpa.alloc(usize, slots), .slot_rows = try gpa.alloc(usize, slots) };
        errdefer s.deinit();
        @memset(s.slot_start.?, 0);
        @memset(s.slot_rows.?, 0);
        for (w.layers, 0..) |*L, i| {
            s.kc[i] = try s.zeros(rows_total * lat);
            s.ic[i] = if (L.indexer != null) try s.zeros(rows_total * c.index_dim * 2) else 0;
        }
        s.pos = try s.zeros(4);
        if (w.mtp) |*m| {
            s.mkc = try s.zeros(rows_total * lat);
            s.mic = if (m.layer.indexer != null) try s.zeros(rows_total * c.index_dim * 2) else 0;
            s.mpos = try s.zeros(4);
        }
        return s;
    }

    /// initSlotsKv8 with the v3 row (119.2): the planes take the 656-byte rows.
    pub fn initSlotsKv8V3(gpa: std.mem.Allocator, d: *const cuda.Driver, w: *const W.Weights, capacity: usize, dcp: usize, slots: usize, kv8: bool, v3: bool) !State {
        const c = w.config;
        if (dcp == 0 or (dcp > 1 and (dcp > w.world or w.world % dcp != 0))) return error.Invalid; // the parity-mirror's divisors (findings-dcp2-divisor.md §4.1)
        if (slots == 0) return error.Invalid;
        // The kv8+DCP read path IS staged now (the dequant mode-3 shard mirror + the attentionCore
        // dispatch, findings-dcp8.md §1b; gate-10 a/b/c PASS) - the old kv8+dcp refusal is lifted;
        // the serve-path boot gates the composition (the 500Kx4 incremental first).
        const rows = dcpm.localCapacity(capacity, dcp);
        const all = rows * slots;
        const lat = latentRowV3(kv8, v3, c);
        var s: State = .{ .d = d, .gpa = gpa, .capacity = capacity, .dcp = dcp, .local = rows, .slots = slots, .kv8 = kv8, .kv8_v3 = v3, .lat_row = lat, .kc = try gpa.alloc(u64, w.layers.len), .ic = try gpa.alloc(u64, w.layers.len), .pos = 0 };
        errdefer s.deinit();
        for (w.layers, 0..) |*L, i| {
            s.kc[i] = try s.zeros(all * lat);
            s.ic[i] = if (L.indexer != null) try s.zeros(all * c.index_dim * 2) else 0;
        }
        s.pos = try s.zeros(4);
        if (w.mtp) |*m| {
            s.mkc = try s.zeros(all * lat);
            s.mic = if (m.layer.indexer != null) try s.zeros(all * c.index_dim * 2) else 0;
            s.mpos = try s.zeros(4);
        }
        return s;
    }

    /// fused.SlotView: stream `slot` alone - every cache pointer moved to its first row, the positions shared with
    /// the whole State (each window sets them before it runs). Free with `freeView`.
    pub fn slotView(s: *const State, w: *const W.Weights, slot: usize) !State {
        if (slot >= s.slots) return error.Invalid;
        const c = w.config;
        var v = s.*;
        v.view = true;
        v.mem = .empty;
        v.slots = 1;
        const row0: usize = if (s.slot_start) |o| o[slot] else slot * s.local;
        const rows_i: usize = if (s.slot_rows) |r| r[slot] else s.local;
        const lat0 = row0 * s.lat_row;
        const idx0 = row0 * c.index_dim * 2;
        if (s.slot_rows != null) {
            v.local = rows_i;
            v.capacity = rows_i * @max(s.dcp, 1); // localCapacity inverted (the +1 slack absorbed)
            v.slot_start = null;
            v.slot_rows = null;
        }
        v.kc = try s.gpa.alloc(u64, s.kc.len);
        errdefer s.gpa.free(v.kc);
        v.ic = try s.gpa.alloc(u64, s.ic.len);
        for (s.kc, v.kc) |a, *b| b.* = a + lat0;
        for (s.ic, v.ic) |a, *b| b.* = if (a != 0) a + idx0 else 0;
        if (s.mkc != 0) v.mkc = s.mkc + lat0;
        if (s.mic != 0) v.mic = s.mic + idx0;
        return v;
    }

    pub fn freeView(v: *State) void {
        std.debug.assert(v.view);
        v.gpa.free(v.kc);
        v.gpa.free(v.ic);
        v.* = undefined;
    }

    /// §128.27az THE POOL: the first hole of >= `rows` free plane rows (first-fit over the live
    /// intervals; the tail included). Deterministic from the mirrored ledger alone - every rank walks
    /// the same intervals and finds the same hole (the coherence ruling: never a free-GB read).
    pub fn poolHole(s: *const State, rows: usize) ?usize {
        const starts = s.slot_start orelse return null;
        const lens = s.slot_rows orelse return null;
        const Iv = struct { start: usize, end: usize };
        var live: [64]Iv = undefined;
        var n: usize = 0;
        for (starts, lens) |st, ln| {
            if (ln == 0) continue;
            if (n >= live.len) return null; // over max_streams intervals: impossible, guarded
            live[n] = .{ .start = st, .end = st + ln };
            n += 1;
        }
        std.mem.sort(Iv, live[0..n], {}, struct {
            fn lt(_: void, a: Iv, b: Iv) bool {
                return a.start < b.start;
            }
        }.lt);
        var at: usize = 0;
        for (live[0..n]) |iv| {
            if (iv.start - at >= rows) return at;
            at = iv.end;
        }
        if (s.local - at >= rows) return at;
        return null;
    }

    /// §128.27hz+ THE DRY-RUN HOLE: poolHole over the ledger with the EVICTABLE idle intervals
    /// treated as free. The claim-path PoolFull's coherence gate: the fit must be decided BEFORE
    /// any ledger mutation - the drop/evict mutate, and a PoolFull after them aborts the wire's
    /// replay (the handler refused) leaving rank 0's ledger alone moved: the ranks' digests diverge
    /// and every later admit dies RanksOutOfStep (the 350K receipt at the DCP1-P4-POOL boot, the
    /// evict fired then the refuse followed then the serve wedged). Monotone: evicting only frees,
    /// so a hole under the all-idles-evicted layout <=> the real oldest-first cumulative evict
    /// finds a fit; no hole even then => refuse with the ledger UNTOUCHED (rank-coherent).
    pub fn poolHoleDry(s: *const State, rows: usize, evictable: []const bool) ?usize {
        const starts = s.slot_start orelse return null;
        const lens = s.slot_rows orelse return null;
        const Iv = struct { start: usize, end: usize };
        var live: [64]Iv = undefined;
        var n: usize = 0;
        for (starts, lens, 0..) |st, ln, slot| {
            if (ln == 0) continue;
            if (slot < evictable.len and evictable[slot]) continue; // the evictable idle: treated free
            if (n >= live.len) return null; // over max_streams intervals: impossible, guarded
            live[n] = .{ .start = st, .end = st + ln };
            n += 1;
        }
        std.mem.sort(Iv, live[0..n], {}, struct {
            fn lt(_: void, a: Iv, b: Iv) bool {
                return a.start < b.start;
            }
        }.lt);
        var at: usize = 0;
        for (live[0..n]) |iv| {
            if (iv.start - at >= rows) return at;
            at = iv.end;
        }
        if (s.local - at >= rows) return at;
        return null;
    }

    /// §128.27az: a slot's interval claimed (`start`, `rows`) and its cache slices zeroed - the admit's
    /// device work (~60 ms a 12 GiB slot; no alloc; fill32: every slice a whole word multiple). The
    /// ledger write is host-side, mirrored by the same admit sequence on every rank.
    pub fn poolClaim(s: *State, o: anytype, w: *const W.Weights, slot: usize, start: usize, rows: usize) !void {
        const c = w.config;
        const starts = s.slot_start orelse return error.Invalid;
        const lens = s.slot_rows orelse return error.Invalid;
        if (slot >= lens.len or lens[slot] != 0 or rows == 0 or start + rows > s.local) return error.Invalid;
        if (std.c.getenv("TF_GLM53_POOL_LOG") != null and !@import("builtin").is_test) {
            std.log.info("pool: claim slot={d} start={d} rows={d} (plane local={d})", .{ slot, start, rows, s.local });
        }
        const lat = s.lat_row;
        const idx = c.index_dim * 2;
        if (lat % 4 != 0 or idx % 4 != 0) return error.Invalid;
        starts[slot] = start;
        lens[slot] = rows;
        for (s.kc) |a| try o.fill32(a + @as(u64, start) * lat, 0, rows * lat / 4);
        for (s.ic) |a| {
            if (a != 0) try o.fill32(a + @as(u64, start) * idx, 0, rows * idx / 4);
        }
        if (s.mkc != 0) try o.fill32(s.mkc + @as(u64, start) * lat, 0, rows * lat / 4);
        if (s.mic != 0) try o.fill32(s.mic + @as(u64, start) * idx, 0, rows * idx / 4);
    }

    /// §128.27hz++++++++ THE IN-PLACE GROWTH: a slot's interval extended to `rows` when the rows past
    /// its end are free - the grown-turn resume's enabler. The old admit path dropped the slot's kept
    /// states the moment the grown prompt exceeded the old claim's pad (EVERY grown turn does: the
    /// new tokens always beat the +64), destroying the very resume it was about to consume - the T2
    /// receipt: poolDropKept(0), then ReuseOutOfStep hunting the state its own claim had just
    /// destroyed, after=0.02s. Here the kept rows stay put (no move, no re-prefill): only the
    /// extension's rows are zeroed, the ledger grows. Deterministic from the mirrored ledger alone -
    /// every rank walks the same intervals and extends alike, so the fallback path stays
    /// rank-coherent when the extension is blocked.
    pub fn poolExtend(s: *State, o: anytype, w: *const W.Weights, slot: usize, rows: usize) !void {
        const c = w.config;
        const starts = s.slot_start orelse return error.Invalid;
        const lens = s.slot_rows orelse return error.Invalid;
        if (slot >= lens.len or lens[slot] == 0 or rows <= lens[slot] or starts[slot] + rows > s.local) return error.Invalid;
        const from = starts[slot] + lens[slot]; // the extension: [from, starts[slot] + rows)
        const to = starts[slot] + rows;
        for (starts, lens) |st, ln| {
            if (ln == 0) continue;
            if (st >= to or st + ln <= from) continue; // fully past / fully before the extension
            return error.Invalid; // a live interval overlaps the extension
        }
        if (std.c.getenv("TF_GLM53_POOL_LOG") != null and !@import("builtin").is_test) {
            std.log.info("pool: extend slot={d} start={d} rows={d} -> {d} (kept {d})", .{ slot, starts[slot], lens[slot], rows, lens[slot] });
        }
        const lat = s.lat_row;
        const idx = c.index_dim * 2;
        if (lat % 4 != 0 or idx % 4 != 0) return error.Invalid;
        for (s.kc) |p| try o.fill32(p + @as(u64, from) * lat, 0, (to - from) * lat / 4);
        for (s.ic) |p| {
            if (p != 0) try o.fill32(p + @as(u64, from) * idx, 0, (to - from) * idx / 4);
        }
        if (s.mkc != 0) try o.fill32(s.mkc + @as(u64, from) * lat, 0, (to - from) * lat / 4);
        if (s.mic != 0) try o.fill32(s.mic + @as(u64, from) * idx, 0, (to - from) * idx / 4);
        lens[slot] = rows;
    }

    /// §128.27az: a slot's interval released (the eviction / the finish-with-no-kept) - the ledger
    /// shrinks; NO row move, NO compaction (the actives bit-unaffected; the hole reused by poolHole).
    pub fn poolRelease(s: *State, slot: usize) void {
        if (std.c.getenv("TF_GLM53_POOL_LOG") != null and !@import("builtin").is_test and s.slot_rows != null and slot < s.slot_rows.?.len) {
            std.log.info("pool: release slot={d} (was start={d} rows={d})", .{ slot, s.slot_start.?[slot], s.slot_rows.?[slot] });
        }
        if (s.slot_rows) |lens| {
            if (slot < lens.len) lens[slot] = 0;
        }
    }

    /// fused.State.bytes_for: what the caches of `capacity` tokens take (the cache guard's figure).
    pub fn bytesFor(w: *const W.Weights, capacity: usize) usize {
        return bytesForDcp(w, capacity, 1);
    }

    /// bytes_for with TF_GLM53_KV8's latent rows.
    pub fn bytesForKv8(w: *const W.Weights, capacity: usize, kv8: bool) usize {
        return bytesForDcpKv8(w, capacity, 1, kv8);
    }

    /// Runner._check_cache_fits' figure: the target layers' bf16 latent rows (index keys and the MTP layer not
    /// counted, as Python counts) of `slots` streams of `capacity` tokens on this rank.
    pub fn latentBytes(w: *const W.Weights, capacity: usize, dcp: usize, slots: usize) usize {
        return latentBytesKv8(w, capacity, dcp, slots, false);
    }

    /// latentBytes with the latent row TF_GLM53_KV8 holds (584 bytes: 576 e4m3 + 2 fp32 scales).
    pub fn latentBytesKv8(w: *const W.Weights, capacity: usize, dcp: usize, slots: usize, kv8: bool) usize {
        return latentBytesV3(w, capacity, dcp, slots, kv8, false);
    }

    /// latentBytesKv8 with the v3 row (119.2).
    pub fn latentBytesV3(w: *const W.Weights, capacity: usize, dcp: usize, slots: usize, kv8: bool, v3: bool) usize {
        const c = w.config;
        return w.layers.len * latentRowV3(kv8, v3, c) * dcpm.localCapacity(capacity, @max(dcp, 1)) * slots;
    }

    /// fused.State.bytes_for with w.dcp: a rank's shard of the caches.
    pub fn bytesForDcp(w: *const W.Weights, capacity: usize, dcp: usize) usize {
        return bytesForDcpKv8(w, capacity, dcp, false);
    }

    /// bytes_for with TF_GLM53_KV8's latent rows: a rank's shard of the caches.
    pub fn bytesForDcpKv8(w: *const W.Weights, capacity: usize, dcp: usize, kv8: bool) usize {
        return bytesForDcpV3(w, capacity, dcp, kv8, false);
    }

    /// bytes_for with the v3 row (119.2): the cache guard's figure must count the 656-byte rows.
    pub fn bytesForDcpV3(w: *const W.Weights, capacity: usize, dcp: usize, kv8: bool, v3: bool) usize {
        const c = w.config;
        const lat = latentRowV3(kv8, v3, c);
        var per: usize = 0;
        for (w.layers) |*L| per += lat + (if (L.indexer != null) c.index_dim * 2 else 0);
        if (w.mtp) |*m| per += lat + (if (m.layer.indexer != null) c.index_dim * 2 else 0);
        return per * dcpm.localCapacity(capacity, @max(dcp, 1));
    }

    /// prefixes.row_views' m: the local rows a state of n tokens reads on this rank (every rank counts the most any
    /// rank holds, ceil(n / dcp), so every rank saves the same bytes).
    pub fn localRows(s: *const State, n: usize) usize {
        return dcpm.cdiv(n, s.dcp);
    }

    /// prefixes.row_views' mm: the MTP layer's local rows of a state of n tokens (positions < n - 1).
    pub fn localMtpRows(s: *const State, n: usize) usize {
        return dcpm.cdiv(n -| 1, s.dcp);
    }

    /// A fresh request: every cache row and the position zeroed (fused.State's torch.zeros).
    pub fn reset(s: *State) !void {
        for (s.mem.items) |*m| try m.fill8(0, null);
    }

    fn zeros(s: *State, bytes: usize) !u64 {
        var m = try cuda.DeviceBuffer.alloc(s.d, bytes);
        errdefer m.free();
        try m.fill8(0, null);
        try s.mem.append(s.gpa, m);
        return m.ptr;
    }

    pub fn deinit(s: *State) void {
        for (s.mem.items) |*m| m.free();
        s.mem.deinit(s.gpa);
        s.gpa.free(s.kc);
        s.gpa.free(s.ic);
        if (s.slot_start) |o| s.gpa.free(o);
        if (s.slot_rows) |r| s.gpa.free(r);
        s.* = undefined;
    }
};

/// TF_GLM53_KV8's side buffers (findings-kv8.md §2.3 + §11): the staging rows the unchanged `_kv_write` fills
/// through glm53_kv8_stage's tables (the rope rotated at the row's TRUE position - the angle fix; the write row
/// is staging row r), the bf16 scratch rows the unchanged attention
/// kernels read (per-row blocks of K rows: a row's keys at r*K + i, the one-stream prefix at [0, K)), the
/// capacity-sized prompt scratch (a direct mirror of the plane's rows), and the static ramp tables that point the
/// Triton kernels at the scratch layouts. One serves every layer (the MTP layer's plane too); the decode windows'
/// launches are graph-capturable (every decision is a device load).
pub const Kv8 = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    lw: usize, // the latent half (512)
    w: usize, // a latent row's values (576)
    k_sel: usize, // index_topk: keys a row attends (the block width)
    stage: u64 = 0, // bf16 [stage_rows, w]: the window's written rows, 0..R-1
    stage_rows: usize = 0,
    zero_pos: u64 = 0, // int32 [stage_rows] zeros: the dequant's unused BASE / TOK args
    stage_pos: u64 = 0, // int32 [stage_rows]: THE ROPE FIX - the staging write's TRUE position table P'[r]
    stage_base: u64 = 0, // int32 [stage_rows]: the staging write's base table B'[r] = r - P'[r] (row r at staging row r)
    scratch: u64 = 0, // bf16 [rows_max * k_sel, w]: the decode windows' blocks (and one-stream prefix)
    rows_max: usize = 0,
    pscratch: u64 = 0, // bf16 [prompt_rows_cap, w]: the prompt chunks' direct mirror of the plane
    pscratch_rows: usize = 0,
    plane_rows: usize = 0, // DCP + several streams: the slots plane's rows (local * slots) - mode 4's n_max
    local: usize = 0, // the slots plane's rows a slot (mode 4's per-slot bound divisor)
    nmax: u64 = 0, // int32 [8]: mode 4's per-slot absolute mirror bounds (the multi updates them a round)
    dbg_last_kc: u64 = 0, // kv8debug instrument: the last view checked (one mode-3 check a new view)
    dbg_m: usize = 0, // kv8debug instrument: the mode-4 checks since the last prefill chunk
    pt: u64 = 0, // 119.1: the latent plane's 512-row page table (TF_GLM53_KV_PAGES; 0: the direct path)
    v3: bool = false, // 119.2: the 656-byte rows (TF_GLM53_KV8_V3)
    ramp_tok: u64 = 0, // int32 [rows_max * k_sel] = r*K + i: one stream's TOK'
    rows_base: u64 = 0, // int32 [rows_max] = r*K: --parallel's BASE'
    rows_tok: u64 = 0, // int32 [rows_max * k_sel] = i: --parallel's TOK'
    mem: std.ArrayList(cuda.DeviceBuffer) = .empty,

    fn alloc(v: *Kv8, n: usize) !u64 {
        var m = try cuda.DeviceBuffer.alloc(v.d, @max(n, 16));
        errdefer m.free();
        try m.fill8(0, null);
        try v.mem.append(v.gpa, m);
        return m.ptr;
    }

    /// `stage_rows`: the widest window that writes (prompt chunks; decode windows are narrower). `rows_max`: the
    /// widest decode window (its blocks). `prompt_cap`: the local capacity the prompt scratch mirrors.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, c: anytype, stage_rows: usize, rows_max: usize, prompt_cap: usize) !Kv8 {
        var v: Kv8 = .{ .gpa = gpa, .d = d, .lw = c.kv_lora, .w = c.latentWidth(), .k_sel = c.index_topk, .stage_rows = stage_rows, .rows_max = rows_max, .pscratch_rows = prompt_cap };
        errdefer v.deinit();
        if (c.latentWidth() != c.kv_lora + c.rope or c.latentWidth() % 4 != 0) return error.Invalid;
        v.stage = try v.alloc(stage_rows * v.w * 2);
        v.zero_pos = try v.alloc(stage_rows * 4);
        v.stage_pos = try v.alloc(stage_rows * 4);
        v.stage_base = try v.alloc(stage_rows * 4);
        v.nmax = try v.alloc(8 * 4);
        v.scratch = try v.alloc(rows_max * v.k_sel * v.w * 2);
        v.pscratch = try v.alloc(prompt_cap * v.w * 2);
        const ramp_words = rows_max * v.k_sel;
        v.ramp_tok = try v.alloc(ramp_words * 4);
        v.rows_base = try v.alloc(rows_max * 4);
        v.rows_tok = try v.alloc(ramp_words * 4);
        // the static tables (uploaded once; every launch reads them on the device)
        const zb = try gpa.alloc(u8, stage_rows * 4);
        defer gpa.free(zb);
        @memset(zb, 0);
        try v.upload(v.zero_pos, zb);
        // the mode-4 per-slot mirror bounds' safe default: every row (maxInt as an absolute bound) -
        // the multi narrows them per round (findings-dcp8.md section 14)
        const nb = try gpa.alloc(i32, 8);
        defer gpa.free(nb);
        @memset(nb, std.math.maxInt(i32));
        try v.upload(v.nmax, std.mem.sliceAsBytes(nb));
        const rt = try gpa.alloc(i32, ramp_words);
        defer gpa.free(rt);
        for (rt, 0..) |*x, i| x.* = @intCast(i);
        try v.upload(v.ramp_tok, std.mem.sliceAsBytes(rt));
        const rs = try gpa.alloc(i32, rows_max);
        defer gpa.free(rs);
        for (rs, 0..) |*x, i| x.* = @intCast(i * v.k_sel);
        try v.upload(v.rows_base, std.mem.sliceAsBytes(rs));
        for (rt, 0..) |*x, i| x.* = @intCast(i % v.k_sel);
        try v.upload(v.rows_tok, std.mem.sliceAsBytes(rt));
        return v;
    }

    /// 119.1: the latent plane's page table over `plane_rows` rows (TF_GLM53_KV_PAGES=shuffle): 512-row
    /// pages, a fixed-seed Fisher-Yates permutation - deterministic, the SAME table on every layer and
    /// every rank, so the collectives stay coherent and any reader that bypasses it takes wrong bytes
    /// (the serve anchors under it are the bypass proof). One stream only; the multi paths stay direct.
    pub fn pages(v: *Kv8, plane_rows: usize) !void {
        const n = (plane_rows + 511) / 512;
        v.pt = try v.alloc(n * 4);
        const t = try v.gpa.alloc(i32, n);
        defer v.gpa.free(t);
        for (t, 0..) |*x, i| x.* = @intCast(i);
        var s: u64 = 119;
        var i: usize = n - 1;
        while (i > 0) : (i -= 1) {
            s = s *% 6364136223846793005 +% 1442695040888963407;
            const j = @as(usize, @intCast(s >> 33)) % (i + 1);
            const tmp = t[i];
            t[i] = t[j];
            t[j] = tmp;
        }
        try v.upload(v.pt, std.mem.sliceAsBytes(t));
    }

    fn upload(v: *Kv8, dst: u64, from: []const u8) !void {
        // a borrowed span over v.mem's memory: upload only copies (free leaves it alone)
        const m: cuda.DeviceBuffer = .{ .d = v.d, .ptr = dst, .len = from.len, .borrowed = true };
        try m.upload(0, from);
    }

    pub fn deinit(v: *Kv8) void {
        for (v.mem.items) |*m| m.free();
        v.mem.deinit(v.gpa);
        v.* = undefined;
    }

    /// What the side buffers take on the device (the cache guard's `extra`).
    pub fn bytes(v: *const Kv8) usize {
        return (v.stage_rows * v.w * 2) + (v.stage_rows * 8) + (v.rows_max * v.k_sel * (v.w * 2 + 4) + v.rows_max * 4) + (v.pscratch_rows * v.w * 2);
    }

    /// keys()'s write (THE ROPE FIX, §11): `_kv_write` rotates the rope half of its row at the row's TRUE
    /// position (fused.py: ang_p = pg), so the staging redirect must not zero it. The ROWS variant already
    /// decouples the two - the angle comes from POS[r] alone, the write row is BASE[r] + POS[r] - so this builds
    /// P'[r] = the true position and B'[r] = r - P'[r] (glm53_kv8_stage) and runs `_kv_write` ROWS with them:
    /// staging row r, rope at the true angle, the row bit-identical to the bf16 cache's. Then the quantizer
    /// takes the true rows. `rows_mode`: several streams (pos a table, `base` their cache bases); else pos is
    /// the window's first position (a device word), base 0.
    pub fn stageWrite(v: *Kv8, f: *const Forward, b: *const Buffers, L: *const W.Layer, kc: u64, pos: u64, base: u64, R: usize, rows_mode: bool) !void {
        if (R > v.stage_rows) return error.WindowTooWide;
        const rank = f.dcpRank(); // the parity (findings-dcp2-divisor.md §4.2): the write guard's RANK const
        try f.ops().kv8Stage(v.stage_pos, v.stage_base, pos, R, rows_mode, f.dcp);
        try f.t().kvWriteRowsDcp(b.kva, b.kva_n, L.kv_a_norm, v.stage, v.stage_pos, v.stage_base, f.inv, f.w.config.eps, R, f.mla(), f.dcp, rank);
        try v.quant(f, kc, pos, base, R, rows_mode);
    }

    /// keys()'s quantize: staging row r -> the plane's row dest_r (the TRUE pos / base tables).
    pub fn quant(v: *Kv8, f: *const Forward, kc: u64, pos: u64, base: u64, R: usize, rows_mode: bool) !void {
        if (R > v.stage_rows) return error.WindowTooWide;
        try f.ops().kv8Quant(v.stage, kc, pos, base, R, rows_mode, f.dcp, f.dcpRank(), v.lw, v.w, if (rows_mode) 0 else v.pt, v.v3); // 119.1/119.2
    }

    /// attentionCore's read (a decode window): the plane's needed rows as bf16 into the block scratch. `pos`: the
    /// window's positions (a table under --parallel); `base`: the streams' cache bases (0: one stream); `tok`: the
    /// selection (b.tok). Returns (the scratch, the TOK' and BASE' the Triton launch takes instead).
    pub fn dequantDecode(v: *Kv8, f: *const Forward, kc: u64, pos: u64, base: u64, tok: u64, R: usize) !struct { lc: u64, tok: u64, base: u64 } {
        if (R > v.rows_max) return error.WindowTooWide;
        const mode: c_int = if (base != 0) 2 else 1;
        try f.ops().kv8Dequant(kc, v.scratch, pos, if (base != 0) base else v.zero_pos, tok, R, v.k_sel, v.lw, v.w, mode, 1, 0, if (base != 0) 0 else v.pt, v.v3); // 119.1/119.2
        return .{ .lc = v.scratch, .tok = if (base != 0) v.rows_tok else v.ramp_tok, .base = if (base != 0) v.rows_base else 0 };
    }

    /// attentionCore's read (a prompt / wide window): rows [0, POS[0] + R) as a direct bf16 mirror - the wide
    /// kernels read keys 0..p (their selections too: every selected row is below the window's end).
    pub fn dequantPrompt(v: *Kv8, f: *const Forward, kc: u64, pos: u64, R: usize) !u64 {
        try f.ops().kv8Dequant(kc, v.pscratch, pos, v.zero_pos, v.zero_pos, R, v.k_sel, v.lw, v.w, 0, 1, v.pscratch_rows, v.pt, v.v3); // 119.1/119.2
        return v.pscratch;
    }

    /// bf16 bits -> f32 (the low 16 bits of the f32 pattern, zeroed).
    fn bf16ToFloat(bits: u16) f32 {
        return @bitCast(@as(u32, bits) << 16);
    }

    /// attentionDcp's read (findings-dcp8.md section 1): the rank's SHARD as the bf16 view _attn_dcp
    /// consumes - the plane IS the shard (row i = the global position i * DCP + RANK), so the mirror is
    /// direct: pscratch row i = plane row i, up to the window's last owned position / DCP. The pscratch
    /// is sized the full local capacity (the whole shard at parallel == dcp).
    /// Several streams (f.base != 0, findings-dcp8.md section 10's bug B): the window's rows read their
    /// streams' shards anywhere in the slots plane (BASE[r] rows in, the positions a table), so mode 3's
    /// POS[0] + rows bound covers none of the other slots - mode 4 mirrors the WHOLE plane instead
    /// (n_max = plane_rows = local * slots, set by the engine beside the caches).
    pub fn dequantShard(v: *Kv8, f: *const Forward, kc: u64, pos: u64, R: usize) !u64 {
        if (std.c.getenv("TF_GLM53_KV8_DEBUG") != null) {
            std.log.info("kv8debug: dequantShard base={} dcp={} R={} plane_rows={} pscratch_rows={} kc={x}", .{ f.base, f.dcp, R, v.plane_rows, v.pscratch_rows, kc });
        }
        if (f.base != 0) {
            if (v.plane_rows == 0 or v.plane_rows > v.pscratch_rows) return error.WindowTooWide;
            // mode 4: POS = the per-slot bounds array (section 14), K = the slot rows; n_max still caps the grid
            const k_arg: usize = if (v.local != 0) v.local else v.plane_rows;
            try f.ops().kv8Dequant(kc, v.pscratch, v.nmax, v.zero_pos, v.zero_pos, R, k_arg, v.lw, v.w, 4, f.dcp, v.plane_rows, 0, v.v3); // 119.1: the multi stays direct; 119.2: the format rides
            if (std.c.getenv("TF_GLM53_KV8_DEBUG") != null and v.dbg_m < 4) {
                v.dbg_m += 2;
                try v.debugRowCmp(f, kc, 0, "mode4");
                try v.debugRowCmp(f, kc, v.plane_rows / 2, "mode4-deep");
            }
            return v.pscratch;
        }
        try f.ops().kv8Dequant(kc, v.pscratch, pos, v.zero_pos, v.zero_pos, R, v.k_sel, v.lw, v.w, 3, f.dcp, v.pscratch_rows, v.pt, v.v3); // 119.1/119.2
        if (std.c.getenv("TF_GLM53_KV8_DEBUG") != null and R > 0 and kc != v.dbg_last_kc) {
            // one check per NEW view pointer (a new request / slot), so every slot's prefill gets one
            v.dbg_last_kc = kc;
            v.dbg_m = 0;
            try v.debugRowCmp(f, kc, 0, "mode3");
        }
        return v.pscratch;
    }

    /// kv8debug (instrument, findings-dcp8.md section 11 bug E): host-dequant the plane's row `row`
    /// and compare its first values against the mirror's same row, right after a mirror launch.
    /// The mirror is a DEQUANT of the plane; a mismatch means the launch (grid / args / bound) or
    /// the plane content it read is wrong at that row. e4m3fn: bias 7, subnormal m/8 * 2^-6.
    fn debugRowCmp(v: *Kv8, f: *const Forward, kc: u64, row: usize, tag: []const u8) !void {
        const o = f.ops();
        const rb: usize = v.w + 8;
        var plane_row: [584]u8 = undefined;
        var mirror_row: [16]u8 = undefined;
        o.download(std.mem.sliceAsBytes(plane_row[0..rb]), kc + row * rb) catch return;
        o.download(std.mem.sliceAsBytes(mirror_row[0..]), v.pscratch + row * v.w * 2) catch return;
        f.s.synchronize() catch return;
        const s0: f32 = @bitCast(std.mem.bytesToValue(u32, plane_row[v.w..][0..4]));
        const s1: f32 = @bitCast(std.mem.bytesToValue(u32, plane_row[v.w..][4..8]));
        var ok: usize = 0;
        var first_bad: i32 = -1;
        for (0..8) |i| {
            const code = plane_row[i];
            const e: i32 = @as(i32, (code >> 3) & 0xF);
            const m: i32 = @as(i32, code & 7);
            var val: f32 = if (e == 0)
                @as(f32, @floatFromInt(m)) / 8.0 * std.math.pow(f32, 2.0, -6.0)
            else
                @as(f32, @floatFromInt((1 << 3) | m)) / 8.0 * std.math.pow(f32, 2.0, @as(f32, @floatFromInt(e - 7)));
            if (code >> 7 != 0) val = -val;
            if (code == 0x7F or code == 0xFF) val = std.math.nan(f32);
            const want = val * (if (i < v.lw) s0 else s1);
            const bits: u16 = std.mem.bytesToValue(u16, mirror_row[i * 2 ..][0..2]);
            const bf: f32 = bf16ToFloat(bits);
            const scale = @abs(want) + 1e-30;
            if (@abs(bf - want) <= 0.06 * scale) ok += 1 else if (first_bad < 0) first_bad = @intCast(i);
        }
        std.log.info("kv8debug: {s} row {d} R-free plane-vs-mirror first8: {d} ok{s} (s0={e} s1={e})", .{ tag, row, ok, if (first_bad >= 0) " (MISMATCH)" else " (MATCH)", s0, s1 });
    }
};

/// Scoring only (tf-glm53-generate --mode score --score-route): every MoE layer's routed weights and picks of a
/// decode-sized window, copied to the host after fused.route. Records: u32 layer, u32 rows, fp32 [rows, top_k]
/// weights, i32 [rows, top_k] picks. It synchronizes the stream, so nothing captured or served may hold one.
pub const RouteTap = struct {
    gpa: std.mem.Allocator,
    buf: std.ArrayList(u8) = .empty,

    pub fn take(t: *RouteTap, f: *const Forward, b: *const Buffers, L: *const W.Layer, R: usize) !void {
        const n = R * f.w.config.top_k * 4;
        const at = t.buf.items.len;
        try t.buf.resize(t.gpa, at + 8 + 2 * n);
        const head = [2]u32{ @intCast(L.index), @intCast(R) };
        @memcpy(t.buf.items[at..][0..8], std.mem.asBytes(&head));
        try f.ops().download(t.buf.items[at + 8 ..][0..n], b.wts);
        try f.ops().download(t.buf.items[at + 8 + n ..][0..n], b.pick);
        try f.s.synchronize();
    }
};

/// E2 (findings §47): the hybrid FFN hook. When set, a layer whose experts the binder did NOT put in r2's
/// EXL3 `Experts` struct (the 1.1 pack's routed tables are NVFP4, `cuda.experts.Layer`) runs its FFN through
/// this callback instead — the port's audited `cuda.moe.block` (plan + `tf_nvfp4_prompt_*` + shared expert +
/// combine). Type-erased on purpose: the callback belongs to the executable's root module, which owns both the
/// 1.1 family and this one, so this module never imports the port's types.
/// `normed` is the post-attention normed input (b.normed), `out` this rank's fp32 partial (b.part) — the same
/// contract r2's own `mlp`/`experts` branches fill, so the gather and residual add after it are untouched.
pub const HybridFfn = struct {
    ctx: *const anyopaque,
    /// Whether this layer runs the hybrid (the dense layers keep r2's own `mlp` branch).
    has: *const fn (ctx: *const anyopaque, li: usize) bool,
    call: *const fn (ctx: *const anyopaque, li: usize, stream: cuda.Stream, normed: u64, out: u64, rows: usize, prompt: bool) anyerror!void,
};

/// TF_GLM53_DCP_PROMPT_GATHER: a prompt chunk under DCP gathers each target layer's filled cache prefix (latent rows
/// and index keys, every rank's share, put back in position order) and runs DCP 1's selection and attention over it -
/// the head-sharded prompt kernels instead of every rank attending all heads over its quarter of the keys plus the
/// query gather, the candidates' gathers and the partials' all-to-all. The caches themselves stay split (decode and
/// the MTP layer unchanged); the scratch holds one layer's prefix at a time.
pub const PromptGather = struct {
    gk: cuda.DeviceBuffer, // bf16 [G, ceil(cap / G), latent width]: every rank's local rows as gathered
    fk: cuda.DeviceBuffer, // bf16 [cap, latent width]: position order
    gi: cuda.DeviceBuffer, // bf16 [G, ceil(cap / G), index_dim]
    fi: cuda.DeviceBuffer, // bf16 [cap, index_dim]
    sc: cuda.DeviceBuffer, // int64 [128, cap]: DCP 1's score rows (Buffers.sc holds a rank's share only)
    cap: usize,

    pub fn init(d: *const cuda.Driver, w: *const W.Weights, capacity: usize, dcp: usize) !PromptGather {
        const c = w.config;
        const lr = dcpm.cdiv(capacity, dcp);
        const lat = c.latentWidth() * 2;
        const idx = c.index_dim * 2;
        var gk = try cuda.DeviceBuffer.alloc(d, dcp * lr * lat);
        errdefer gk.free();
        var fk = try cuda.DeviceBuffer.alloc(d, dcp * lr * lat);
        errdefer fk.free();
        var gi = try cuda.DeviceBuffer.alloc(d, dcp * lr * idx);
        errdefer gi.free();
        var fi = try cuda.DeviceBuffer.alloc(d, dcp * lr * idx);
        errdefer fi.free();
        const sc = try cuda.DeviceBuffer.alloc(d, 128 * capacity * 8);
        return .{ .gk = gk, .fk = fk, .gi = gi, .fi = fi, .sc = sc, .cap = capacity };
    }

    pub fn bytes(pg: *const PromptGather) usize {
        return pg.gk.len + pg.fk.len + pg.gi.len + pg.fi.len + pg.sc.len;
    }

    pub fn deinit(pg: *PromptGather) void {
        pg.sc.free();
        pg.fi.free();
        pg.gi.free();
        pg.fk.free();
        pg.gk.free();
    }
};

/// The print-bisect's runtime toggle: off through the prewarm (the capture mode forbids syncs),
/// on from the first real prefill.
pub var wdebug_on: bool = false;

pub const Forward = struct {
    w: *const W.Weights,
    k: *const kern.Kernels,
    s: cuda.Stream,
    inv: u64, // fp32 [rope / 2]: fused.inv_freq (Phase 1: the oracle's bytes; Phase 2a: Kernels.invFreq's)
    /// the world's NCCL communicator (rank and world as the weights' cut); null: the identity exchange of Phase 1
    comm: ?*const cuda.nccl.Communicator = null,
    /// Phase 3a: the RoCE one-shot runtime (roce.RoceReduce, w.fast): decode windows (R <= b.small) reduce over it,
    /// and the head's records and sampled candidates gather over it; null: NCCL
    fast: ?*const roce.Roce = null,
    /// Phase 3a: TF_GLM53_PREFILL_REDUCE=ring - windows wider than b.small (prompt chunks) reduce with NCCL's bf16
    /// ring all-reduce (fused.gather's ring branch); false: the all-gather + rank-order sum of Phase 2
    ring: bool = false,
    /// Phase 3a: what windows over max_rows rows need (the prompt GEMM's workspace on this stream, cuBLAS for the
    /// batched absorb / expand); null: no window over max_rows rows
    wide: ?*const prompt.Wide = null,
    /// The K4 hybrid pack's dense linears (findings-k4-loader.md): the engine's cuBLAS handle for
    /// exl3.Linear.dense GEMMs (the smalls dequantized to bf16 at load); null: no dense weight
    blas: ?*const cuda.cublas.Blas = null,
    /// Â§125.37 the dense-lane A/B arm (TF_GLM53_DENSE_LANE): the qmmf bf16 lane for the dense
    /// smalls - the DECODE windows only (R <= 64; the prompt windows keep cuBLAS); null: cuBLAS.
    qlane: ?*const cuda.qmmf.Lane = null,
    qpart: u64 = 0,
    /// 125.43(d) the mode-conditional dispatch cell: the verify window's pick (argmax: greedy ->
    /// the lane; local: sampled -> cuBLAS). Set by the VerifyCtx at the CAPTURE (the pick is a graph
    /// key - each pick's captures freeze its branch; the replays never read it). Null: the legacy
    /// boot-scoped behavior (the lane at every R<=64 dense window).
    lane_pick: ?*bool = null,
    /// weights.EXPERTS_IMPL == "shared": windows over b.xs_rows rows take prompt_experts (SharedExperts.prefill)
    shared_experts: bool = false,
    /// TF_EXL3_PROMPT_DET=slots16 (reproducible prompt experts); false: fp32 red.add in arrival order (the default)
    pe_det16: bool = true,
    /// Phase 3a speed. TF_GLM53_SIDE: the decode windows' second stream (cuda_decode.Side); null: one stream
    side: ?*dec.Side = null,
    /// TF_GLM53_L2PF: the L2 prefetch of the target's decode windows (cuda_decode.L2pf); null: off
    l2pf: ?*dec.L2pf = null,
    /// decode-sized selections (fewer than radix_min_rows rows) through the multi-block top-k; false: the
    /// one-program radix select for every block (the same columns)
    multi_select: bool = true,
    /// TF_GLM53_TOPK_STREAM: the PREFILL blocks' selection runs topk._topk_row - the same picks as the radix,
    /// bit-identical (the same tie rule; host-tested), in 2-3 streaming passes over the score row instead of the
    /// radix's 5 (findings-prefill-topk.md). Decode windows and the SP halves keep the radix (and the multi-block
    /// path) untouched; a score buffer too small to carve the collect scratch falls back to the radix too.
    topk_stream: bool = false,
    /// TF_GLM53_FP8_INDEX: the index keys live in an e4m3 plane (128 codes + an fp32 scale a token, 132 bytes -
    /// stock vLLM's cache ABI) carved inside the bf16 ic allocation; _ik_write8 writes it, _index_scores8 reads it
    /// with the dequant in-kernel (the kv8 v2 convention, the dot stays bf16 - increment 1). Every reader switches
    /// (prompt AND decode selections - the cache has one layout a boot); DCP 1 only (the engine refuses more).
    /// The scores change (e4m3 keys): NOT bit-identical - the gates are the numerics test + the quality bench.
    fp8_index: bool = false,
    fp8_stride: usize = 132, // inc 3: the plane's token row - 132 legacy, 144 the padded TMA-grade layout
    /// TF_GLM53_E1_FP8 (section 18): the DCP E1 query gather ships e4m3-packed rows (half the wire); the
    /// scores change q-side (~0.4% more) - the needle/coherence gates decide, NOT bit-identical
    e1_fp8: bool = false,
    e3_merge: bool = true, // 126(g): the decode window's two E3 gathers packed per group into ONE one-shot
    /// the prompt path's per-phase timers (cuda_prompt.Prof, --prompt-prof); null: none
    prof: ?*prompt.Prof = null,
    /// Phase 3a speed (--mtp-dense 1): the MTP layer's prompt rows (windows over max_rows rows) through cuBLAS -
    /// eh_proj and its routed experts as dense bf16 GEMMs (cuda_prompt.MtpDense); null: the router kernel and the
    /// decode expert kernel in blocks. Drafts only: the target verifies every token, so no output token moves.
    mtp_dense: ?*prompt.MtpDense = null,
    /// scoring only (RouteTap): fused.ffn_part's routed weights and picks to the host; null everywhere else
    route_tap: ?*RouteTap = null,
    /// Phase 3b: decode context parallelism (w.dcp: 1, or the world - every rank of the TP world is one of the DCP
    /// group, its rank w.rank). The State must be State.initDcp(.., dcp) and every Buffers BufOpts.dcp = dcp.
    dcp: usize = 1,
    /// E2 (findings §47): the hybrid FFN for the 1.1 pack's routed NVFP4 experts; null: r2's own branches.
    moe_hook: ?HybridFfn = null,

    /// Phase 4: the target layers whose output rows the drafters read (fused.Weights.tap_slot): every target window
    /// copies them into its buffers' taps (Buffers with BufOpts.taps = taps.n); n 0: none
    taps: Taps = .{},
    /// Phase 4b (--parallel N, fused.Rows): several streams in one window - the `pos` a layer is given is then an int32
    /// table of each row's own position and this an int32 table of each row's cache base row (its stream's slot
    /// start), every position-dependent kernel launched with ROWS (cuda_triton *Rows); 0: one stream, `pos` the
    /// window's first position. Under DCP (dcp > 1) the kernels are the *RowsDcp ones: a row's keys on the owner of its
    /// position at BASE + POS // dcp, its attention over each rank's share of its own stream's rows.
    base: u64 = 0,
    /// TF_GLM53_DCP_PROMPT_GATHER (PromptGather): the scratch, and the prompt chunk's end position (Runner.prefillChunk
    /// sets it on its copy; 0 everywhere else: no gather)
    pg: ?*const PromptGather = null,
    pg_end: usize = 0,
    /// TF_GLM53_KV8: the staged quantize / dequantize side buffers (cuda_kernels' kv8 pair); null: bf16 caches.
    /// The write path fills `Kv8.stage` through the unchanged `_kv_write` and quantizes into the plane; the read path
    /// dequantizes the needed rows into a bf16 scratch the unchanged attention kernels take as LC. DCP 1 only.
    kv8: ?*Kv8 = null,
    /// TF_GLM53_KV8_V2 (findings-kv8-v2.md): the WIDE windows read the e4m3 plane through the KV8 variants of
    /// _attn_chunks (in-kernel dequant) - no dequantPrompt pass, no bf16 mirror, no O(pos) per-chunk term. The
    /// staged path keeps the decode windows (and, with v2 off, the prompt path). New AOT variants (the recapture
    /// plan is findings-kv8-v2.md §5); v2 needs kv8.
    kv8_v2: bool = false,

    fn ops(f: *const Forward) kern.Ops {
        return .{ .k = f.k, .s = f.s };
    }

    pub fn t(f: *const Forward) tri.Tri { // pub: the SP split choreography (cuda_prompt.zig) reads it
        return .{ .set = &f.k.triton, .s = f.s };
    }

    pub fn mla(f: *const Forward) tri.Mla { // pub: the SP split choreography (cuda_prompt.zig) reads it
        const c = f.w.config;
        return .{ .heads = f.w.heads, .nope = c.nope, .rope = c.rope, .lw = c.kv_lora, .vd = c.v_dim, .topk = c.index_topk };
    }

    /// The side stream while a decode window runs on it (fused.Buffers.side_on), else null.
    fn sideOn(f: *const Forward) ?*dec.Side {
        const sd = f.side orelse return null;
        return if (sd.on) sd else null;
    }

    /// fused._fork: this Forward on the side stream, queued after everything f.s has queued so far.
    fn forkSide(f: *const Forward, sd: *dec.Side) !Forward {
        try sd.fork(f.s);
        var g = f.*;
        g.s = sd.s;
        g.side = null;
        g.l2pf = null;
        g.prof = null;
        return g;
    }

    /// `b` with the side stream's own EXL3 and router scratch (what torch's per-stream allocator gives fused).
    fn sideBufs(b: *const Buffers) Buffers {
        var s = b.*;
        s.lin = b.slin;
        s.rpart = b.srpart;
        return s;
    }

    /// TF_GLM53_L2PF: prefetch site `name` of layer L (nothing outside the target's decode windows).
    fn site(f: *const Forward, L: *const W.Layer, name: dec.SiteName) !void {
        const pf = f.l2pf orelse return;
        try pf.site(f.k, f.s, L.index, name);
    }

    /// --prompt-prof: a timing mark on this stream (null when the timers are off or suspended).
    pub fn pbegin(f: *const Forward) !?usize {
        const p = f.prof orelse return null;
        return p.begin(f.s);
    }

    /// --prompt-prof: the end of the region `m` opened, counted under `ph`.
    pub fn pend(f: *const Forward, m: ?usize, ph: prompt.Phase) !void {
        const p = f.prof orelse return;
        const i = m orelse return;
        try p.end(f.s, i, ph);
    }

    /// torch.index_select(embed, 0, ids): the window's token rows into b.x (row copies: exact).
    pub fn embed(f: *const Forward, b: *const Buffers, ids: []const u32) !void {
        const D = f.w.config.hidden;
        for (ids, 0..) |id, r| try f.ops().copy(b.x + r * D * 2, f.w.embed + @as(u64, id) * D * 2, D * 2);
    }

    /// st.pos.fill_(a): the window's first position.
    pub fn setPos(f: *const Forward, st: *const State, a: usize) !void {
        const v: i32 = @intCast(a);
        try f.ops().upload(st.pos, std.mem.asBytes(&v));
    }

    /// fused.layer for layer `li` over rows b.x[:R] at st.pos ..; `T`: the index range (null: no row selects).


    pub fn layer(f: *const Forward, b: *const Buffers, st: *const State, li: usize, R: usize, T: ?usize) !void {
        return f.layerWith(b, &f.w.layers[li], st.kc[li], st.ic[li], st.pos, R, T, 0);
    }

    /// fused.layer for any layer (the MTP layer too): its caches kc / ic (0: no indexer), positions from `pos`;
    /// `reuse` (TF_GLM53_MTP_REUSE, one-row MTP draft steps): attend b.tok[0] instead of selecting.
    pub fn layerWith(f: *const Forward, b: *const Buffers, L: *const W.Layer, kc: u64, ic: u64, pos: u64, R: usize, T: ?usize, reuse: u32) !void {
        const c = f.w.config;
        const D = c.hidden;
        const t_ = f.t();
        try t_.rmsnorm(b.x, D, L.input_norm, b.normed, D, R, D, c.eps);
        const wdbg = wdebug_on;
        if (wdbg) {
            try f.s.synchronize();
            std.log.info("wdebug: ph-norm", .{});
        }
        try f.attentionPart(b, L, kc, ic, pos, R, T, reuse);
        if (wdbg) {
            try f.s.synchronize();
            std.log.info("wdebug: ph-attn", .{});
        }
        const late = if (f.l2pf) |p| p.set.after else false; // TF_GLM53_L2PF_AFTER: the sites after their exchange
        if (!late) try f.site(L, .a); // the FFN's first weights, during the all-reduce
        const pg1 = try f.pbegin();
        const g1 = try f.gather(b, R);
        if (wdbg) {
            try f.s.synchronize();
            std.log.info("wdebug: ph-ag1", .{});
        }
        if (late) try f.site(L, .a);
        try t_.residualAdd(b.x, g1.ptr, R, D, g1.world);
        try f.pend(pg1, .ag);
        try t_.rmsnorm(b.x, D, L.post_attn_norm, b.normed, D, R, D, c.eps);
        try f.ffnPart(b, L, R);
        if (wdbg) {
            try f.s.synchronize();
            std.log.info("wdebug: ph-ffn", .{});
        }
        if (!late) try f.site(L, .f); // the next layer's input projections, during the all-reduce
        const pg2 = try f.pbegin();
        const g2 = try f.gather(b, R);
        if (late) try f.site(L, .f);
        try t_.residualAdd(b.x, g2.ptr, R, D, g2.world);
        try f.pend(pg2, .ag);
    }

    /// What residual_add sums: `world` fp32 slots of [R, D] in rank order.
    pub const Gathered = struct { ptr: u64, world: usize };

    /// fused.gather. World 1: b.part itself. RoCE (decode windows, R <= b.small): the one-shot rank-order sum into
    /// b.red ([1, R, D]). Ring (Phase 3a, wider windows): b.part to bf16, NCCL's bf16 ring all-reduce, widened into
    /// b.red ([1, R, D]). Otherwise NCCL comm.all_gather(b.part[:R], b.gath[:world R D]) on the compute stream ([world,
    /// R, D], summed in rank order by residual_add). Phase 1 (other ranks absent, LocalComm): zeros in every slot but
    /// this rank's, which holds b.part[:R].
    pub fn gather(f: *const Forward, b: *const Buffers, R: usize) !Gathered {
        const world = f.w.world;
        if (world == 1) return .{ .ptr = b.part, .world = 1 };
        const D = f.w.config.hidden;
        const o = f.ops();
        if (f.fast) |r| {
            if (R <= b.small) {
                if (r.bf16_wire) try r.allReduceBf16(f.s, b.part, b.red, R * D) else try r.allReduce(f.s, b.part, b.red, R * D * 4);
                return .{ .ptr = b.red, .world = 1 };
            }
        }
        if (f.comm) |c| {
            if (f.ring and R > b.small) {
                if (b.bpart == 0) return error.NoRingBuffers;
                try o.f32ToBf16(b.part, b.bpart, R * D);
                try c.allReduce(b.bpart, b.bred, R * D, .bf16, .sum, f.s.handle);
                try o.bf16ToF32(b.bred, b.red, R * D);
                return .{ .ptr = b.red, .world = 1 };
            }
            try c.allGather(b.part, b.gath, R * D, .f32, f.s.handle);
            return .{ .ptr = b.gath, .world = world };
        }
        try o.fill32(b.gath, 0, world * R * D);
        try o.copy(b.gath + f.w.rank * R * D * 4, b.part, R * D * 4);
        return .{ .ptr = b.gath, .world = world };
    }

    /// The K4 hybrid's dense linear (findings-k4-loader.md): out [R, N] (dtype y) = x [R, K] @ W with
    /// W the plain bf16 [N, K] row-major in l.words - cuBLAS, any R (decode or prompt): C col-major
    /// [N, R] = W^T (ta 't', lda K) x (tb 'n', ldb K), ldc N - the eh_proj convention. The f32 output
    /// stores the accumulate unrounded (the exl3 path's .f32 dtype); bf16 rounds on store as ever.
    fn denseGemm(f: *const Forward, l: *const exl3.Linear, x: u64, R: usize, out: u64, y: exl3.DType) !void {
        // Â§125.37 the dense-lane branch: the qmmf bf16 lane on the decode windows. A NEW
        // NUMERIC LANE (the lane's accumulation order differs from cuBLAS) - the fresh-anchor
        // re-roll gates apply; the cuBLAS default stands until the A/B wins.
        if (f.qlane) |q| {
            if (l.qwords != 0 and R <= 64 and (if (f.lane_pick) |p| p.* else true)) { // 125.43(d): the sampled windows take cuBLAS
                const w = if (l.qfp8) // 126.6(at) the fp8 canary: the original e4m3 codes at .fp8g
                    cuda.qmmf.Weight{ .mode = .fp8g, .codes = l.qwords, .scales = l.qscales,
                        .n = @intCast(l.n), .k = @intCast(l.k), .npad = @intCast((l.n + 127) / 128 * 128) }
                else
                    cuda.qmmf.Weight{ .mode = .bf16, .codes = l.qwords, .scales = 0,
                        .n = @intCast(l.n), .k = @intCast(l.k), .npad = @intCast((l.n + 127) / 128 * 128) };
                return switch (y) {
                    .bf16 => q.matmulOut(f.s, x, l.k, R, w, out, .bf16, f.qpart),
                    .f32 => q.matmulOut(f.s, x, l.k, R, w, out, .f32, f.qpart),
                    else => error.Invalid,
                };
            }
        }
        const blas = f.blas orelse return error.NoCublas;
        switch (y) {
            .bf16 => try blas.bgemmBf16(f.s.handle, .t, .n, l.n, R, l.k, l.words, l.k, 0, x, l.k, 0, out, l.n, 0, 1),
            .f32 => try blas.gemmBf16F32(f.s.handle, .t, .n, l.n, R, l.k, l.words, l.k, x, l.k, out, l.n),
            else => return error.Invalid,
        }
    }

    /// fused.lin: an EXL3 linear of x [R, K] (bf16, contiguous rows) into out [R, N] of dtype y - the row-invariant
    /// kernel up to max_rows rows, the prompt GEMM (x3prefill.matmul, weights decoded once) above.
    pub fn lin(f: *const Forward, b: *const Buffers, l: *const exl3.Linear, x: u64, R: usize, out: u64, y: exl3.DType) !void {
        if (l.dense) return f.denseGemm(l, x, R, out, y);
        if (R <= max_rows) return f.ops().linear(l, x, .bf16, out, y, R, b.lin);
        const wd = f.wide orelse return error.WindowTooWide;
        return wd.matmul(f, l, x, R, out, y);
    }

    /// fused.lins: linears of one input in one launch when groupable (up to max_rows rows), else one `lin` each.
    pub fn lins(f: *const Forward, b: *const Buffers, ls: []const *const exl3.Linear, x: u64, R: usize, outs: []const u64, ys: []const exl3.DType) !void {
        // the K4 hybrid: dense members never join an exl3 group (no strips, no scales) - per-matrix
        var any_dense = false;
        for (ls) |l| {
            if (l.dense) any_dense = true;
        }
        if (any_dense) {
            for (ls, outs, ys) |l, out, y| try f.lin(b, l, x, R, out, y);
            return;
        }
        if (R <= max_rows) return f.ops().linears(ls, x, .bf16, outs, ys, R, b.lin);
        for (ls, outs, ys) |l, out, y| try f.lin(b, l, x, R, out, y);
    }

    /// fused.attention_part and attention_core: this rank's fp32 partial in b.part[:R]. `reuse` 1 / 2 (an MTP draft
    /// step after its chain's first, one row): no index key written, no selection; the row attends b.tok[0] (mode 1
    /// puts its own position in the list's last slot). TF_GLM53_SIDE "a" / "w" inside a decode window: the key path
    /// (keys) on the side stream beside the query path, joined before select or the attention kernels.
    pub fn attentionPart(f: *const Forward, b: *const Buffers, L: *const W.Layer, kc: u64, ic: u64, pos: u64, R: usize, T: ?usize, reuse: u32) !void {
        const c = f.w.config;
        const t_ = f.t();
        if (reuse != 0 and (L.indexer == null or R != 1)) return error.Invalid;
        if (f.base != 0 and reuse != 0) return error.Invalid; // several streams: no MTP index reuse
        const sd = f.sideOn();
        const side_a = if (sd) |x| x.letters.a else false;
        const side_w = if (sd) |x| x.letters.w else false;
        // --prompt-prof: only without the side fork - the SP path times its chains on the comm stream (sp_chunk's
        // own marks); a main-stream bracket around side-stream launches would time idle
        const prof_main = sd == null;
        const p0 = if (prof_main) try f.pbegin() else null;
        if (side_a) {
            const g = try f.forkSide(sd.?);
            const bs = sideBufs(b);
            try g.lin(&bs, &L.kv_a, b.normed, R, b.kva, .bf16); // the bits of its share of the grouped launch
            try g.keys(&bs, L, kc, ic, pos, R, reuse);
            try f.lin(b, &L.q_a, b.normed, R, b.qa, .bf16);
        } else {
            try f.lins(b, &.{ &L.q_a, &L.kv_a }, b.normed, R, &.{ b.qa, b.kva }, &.{ .bf16, .bf16 });
            if (side_w) {
                const g = try f.forkSide(sd.?);
                const bs = sideBufs(b);
                try g.keys(&bs, L, kc, ic, pos, R, reuse);
            }
        }
        try t_.rmsnorm(b.qa, c.q_lora, L.q_a_norm, b.qn, c.q_lora, R, c.q_lora, c.eps);
        if (prof_main) try f.pend(p0, .front);
        const pk = if (prof_main) try f.pbegin() else null;
        if (!side_a and !side_w) try f.keys(b, L, kc, ic, pos, R, reuse);
        if (wdebug_on) {
            try f.s.synchronize();
            std.log.info("wdebug: attn-keys", .{});
        }
        if (prof_main) try f.pend(pk, .kv_write);
        // PromptGather: the chunk's keys are in this rank's share; the layer's prefix gathered, then DCP 1 over it
        if (f.pg) |pg| if (f.dcp > 1 and f.base == 0 and f.pg_end > 0 and R > b.small and T != null and reuse == 0 and sd == null) {
            if (f.pg_end > pg.cap) return error.WindowTooWide;
            try f.gatherPrefix(kc, pg.gk.ptr, pg.fk.ptr, c.latentWidth() * 2, f.pg_end);
            if (ic != 0) try f.gatherPrefix(ic, pg.gi.ptr, pg.fi.ptr, c.index_dim * 2, f.pg_end);
            var g = f.*;
            g.dcp = 1;
            g.pg = null;
            var bg = b.*;
            bg.sc = pg.sc.ptr;
            bg.dcp = 1;
            return g.attentionRest(&bg, L, pg.fk.ptr, if (ic != 0) pg.fi.ptr else 0, pos, R, T, reuse, null);
        };
        return f.attentionRest(b, L, kc, ic, pos, R, T, reuse, sd);
    }

    /// PromptGather: rows [0, end) of a split cache (this rank's local rows at `src`, position p on rank p % dcp at
    /// row p / dcp) into `full` in position order - every rank's ceil(end / dcp) rows gathered into `g`, then each
    /// rank's rows strided back.
    fn gatherPrefix(f: *const Forward, src: u64, g: u64, full: u64, row_bytes: usize, end: usize) !void {
        const G = f.dcp;
        const lr = dcpm.cdiv(end, G);
        const cm = f.comm orelse return error.NoCommunicator;
        try cm.allGather(src, g, lr * row_bytes / 2, .bf16, f.s.handle);
        for (0..G) |r| {
            const n = if (end > r) dcpm.cdiv(end - r, G) else 0;
            try f.ops().copyRows(full + r * row_bytes, G * row_bytes, g + r * lr * row_bytes, row_bytes, row_bytes, n);
        }
    }

    /// attention_part past the keys: the index queries, the selection and attention_core.
    fn attentionRest(f: *const Forward, b: *const Buffers, L: *const W.Layer, kc: u64, ic: u64, pos: u64, R: usize, T: ?usize, reuse: u32, sd: ?*dec.Side) !void {
        const c = f.w.config;
        const o = f.ops();
        const t_ = f.t();
        const D = c.hidden;
        const prof_main = sd == null;
        // b.tok[0, K-1:] = pos (DCP 1 only: under DCP the row's own key is in its owner's share already)
        if (reuse == 1 and T != null and f.dcp == 1) try o.copy(b.tok + (c.index_topk - 1) * 4, pos, 4);
        var q_done = false;
        if (L.indexer) |*ix| {
            if (T) |tt| {
                if (reuse == 0) {
                    const p1 = if (prof_main) try f.pbegin() else null;
                    if (wdebug_on) std.log.info("wdebug: fargs normed={x} wproj={x} iw={x} rpart={x} R={d} dims {d}->{d} base={x} pos={x}", .{ b.normed, @as(u64, @bitCast(ix.weights_proj)), b.iw, b.rpart, R, D, c.index_heads, f.base, pos });
                    try t_.router(b.normed, D, ix.weights_proj, b.iw, b.rpart, R, D, c.index_heads);
                    if (wdebug_on) {
                        try f.s.synchronize();
                        std.log.info("wdebug: idx-router", .{});
                    }
                    try f.lins(b, &.{ &ix.wq_b, &L.q_b }, b.qn, R, &.{ b.iq, b.q }, &.{ .bf16, .bf16 });
                    if (wdebug_on) {
                        try f.s.synchronize();
                        std.log.info("wdebug: idx-lins", .{});
                    }
                    q_done = true;
                    if (f.base != 0) {
                        try t_.iqRopeRows(b.iq, f.inv, pos, R, c.index_heads, c.index_dim, c.rope);
                    } else {
                        try t_.iqRope(b.iq, f.inv, pos, R, c.index_heads, c.index_dim, c.rope);
                    }
                    if (wdebug_on) {
                        try f.s.synchronize();
                        std.log.info("wdebug: idx-rope", .{});
                    }
                    if (sd) |x| try x.join(f.s); // the window's index keys are in
                    if (prof_main) try f.pend(p1, .front);
                    if (wdebug_on) {
                        try f.s.synchronize();
                        std.log.info("wdebug: attn-idxfront", .{});
                    }
                    try f.select(b, ic, pos, 0, R, tt); // its scoring and top-k mark themselves (index_scores / select)
                    if (wdebug_on) {
                        try f.s.synchronize();
                        std.log.info("wdebug: attn-select", .{});
                    }
                    if (std.c.getenv("TF_GLM53_KV8_DEBUG") != null and (L.index == 0 or L.index == 6) and R > 3000) {
                        // kv8debug: the non-SP reference select inputs - ic rows 1,2 and the iq at row 2048
                        const o3 = kern.Ops{ .k = f.k, .s = f.s };
                        const mic_row: usize = c.index_dim * 2;
                        var hb: [512]u8 = undefined;
                        for ([_]usize{ 1, 2 }) |ti| {
                            var s3: u32 = 0;
                            o3.download(std.mem.sliceAsBytes(hb[0..mic_row]), ic + ti * mic_row) catch continue;
                            f.s.synchronize() catch {};
                            for (hb[0..mic_row]) |bb| s3 +%= bb;
                            std.log.info("kv8debug: NSP L0 ic row {d} sum={x}", .{ ti, s3 });
                        }
                        var iqbuf: [16]u8 = undefined;
                        o3.download(std.mem.sliceAsBytes(iqbuf[0..]), b.iq + 2048 * c.index_heads * c.index_dim * 2) catch {};
                        f.s.synchronize() catch {};
                        std.log.info("kv8debug: NSP L0 iq row 2048 head: {x} {x} {x} {x}", .{ std.mem.bytesToValue(u32, iqbuf[0..4]), std.mem.bytesToValue(u32, iqbuf[4..8]), std.mem.bytesToValue(u32, iqbuf[8..12]), std.mem.bytesToValue(u32, iqbuf[12..16]) });
                    }
                    if (std.c.getenv("TF_GLM53_KV8_DEBUG") != null and (L.index <= 1 or L.index == 6) and R > 3000) {
                        // kv8debug: layer 1's selection at the first selection-regime rows - the first
                        // divergence between slots lives in layer 1's attention output (section 11 bug E)
                        const o2 = f.ops();
                        var tbuf: [16]u8 = undefined;
                        var cbuf: [8]u8 = undefined;
                        for ([_]usize{ 2048, 2052 }) |rr| {
                            if (rr >= R) continue;
                            o2.download(std.mem.sliceAsBytes(tbuf[0..]), b.tok + rr * c.index_topk * 4) catch continue;
                            o2.download(std.mem.sliceAsBytes(cbuf[0..]), b.cnt + rr * 4) catch {};
                            f.s.synchronize() catch {};
                            std.log.info("kv8debug: L1 sel row {d} cnt={d} tok0..3={d},{d},{d},{d}", .{
                                rr, std.mem.bytesToValue(i32, cbuf[0..4]),
                                std.mem.bytesToValue(i32, tbuf[0..4]),  std.mem.bytesToValue(i32, tbuf[4..8]),
                                std.mem.bytesToValue(i32, tbuf[8..12]), std.mem.bytesToValue(i32, tbuf[12..16]),
                            });
                        }
                    }
                }
            }
        }
        try f.attentionCore(b, L, kc, pos, R, q_done);
    }

    /// attention_part's keys(): the window's latent and rope key and, on indexer layers (not under reuse), the index
    /// key into the caches (kv_a already in b.kva).
    fn keys(f: *const Forward, b: *const Buffers, L: *const W.Layer, kc: u64, ic: u64, pos: u64, R: usize, reuse: u32) !void {
        const c = f.w.config;
        const t_ = f.t();
        const rank = f.dcpRank();
        if (f.kv8) |v| {
            // TF_GLM53_KV8: _kv_write fills the staging rows through kv8_stage's tables (the rope at the TRUE
            // position - §11), the quantizer takes the true rows
            try v.stageWrite(f, b, L, kc, pos, f.base, R, f.base != 0);
            if (wdebug_on) {
                try f.s.synchronize();
                std.log.info("wdebug: keys-stage", .{});
            }
        } else if (f.base != 0) {
            try t_.kvWriteRowsDcp(b.kva, b.kva_n, L.kv_a_norm, kc, pos, f.base, f.inv, c.eps, R, f.mla(), f.dcp, rank);
        } else {
            try t_.kvWriteDcp(b.kva, b.kva_n, L.kv_a_norm, kc, pos, f.inv, c.eps, R, f.mla(), f.dcp, rank);
        }
        if (L.indexer) |*ix| {
            if (reuse == 0) {
                if (wdebug_on) std.log.info("wdebug: kargs normed={x} wk={x} ik={x} rpart={x} R={d} dims {d}->{d}", .{ b.normed, @as(u64, @bitCast(ix.wk)), b.ik, b.rpart, R, c.hidden, c.index_dim });
                try t_.router(b.normed, c.hidden, ix.wk, b.ik, b.rpart, R, c.hidden, c.index_dim);
                if (wdebug_on) {
                    try f.s.synchronize();
                    std.log.info("wdebug: keys-irouter", .{});
                }
                if (f.fp8_index) {
                    // TF_GLM53_FP8_INDEX: the same plane pointer twice - IC the codes (*fp8e4nv), ICS the scales
                    // (*fp32, the same storage; a row's scale at word 32 of its 33)
                    if (f.base != 0) {
                        try t_.ikWriteRows8(b.ik, ix.k_norm_w, ix.k_norm_b, f.inv, ic, ic, pos, f.base, R, c.index_dim, c.rope, f.fp8_stride);
                        if (wdebug_on) {
                            try f.s.synchronize();
                            std.log.info("wdebug: keys-ikrows8", .{});
                        }
                    } else {
                        try t_.ikWrite8(b.ik, ix.k_norm_w, ix.k_norm_b, f.inv, ic, ic, pos, R, c.index_dim, c.rope, f.fp8_stride);
                    }
                } else if (f.base != 0) {
                    try t_.ikWriteRowsDcp(b.ik, ix.k_norm_w, ix.k_norm_b, f.inv, ic, pos, f.base, R, c.index_dim, c.rope, f.dcp, rank);
                    if (wdebug_on) {
                        try f.s.synchronize();
                        std.log.info("wdebug: keys-ikrowsdcp", .{});
                    }
                } else {
                    try t_.ikWriteDcp(b.ik, ix.k_norm_w, ix.k_norm_b, f.inv, ic, pos, R, c.index_dim, c.rope, f.dcp, rank);
                }
            }
        }
    }

    /// fused.attention_core for rows b.qn[:R] (selection in b.tok): q_b (unless attention_part ran it grouped with
    /// wq_b), absorb, attention over the cache, expand, o_proj -> this rank's fp32 partial b.part[:R]. Windows over
    /// max_rows rows (prompt chunks): q_b / o_proj through the prompt GEMM, absorb and expand as torch.bmm (cuBLAS,
    /// tensor cores: not row-exact, as the Python engine's) with _qrope beside.
    pub fn attentionCore(f: *const Forward, b: *const Buffers, L: *const W.Layer, kc: u64, pos: u64, R: usize, q_done: bool) !void {
        const t_ = f.t();
        const m = f.mla();
        if (wdebug_on) std.log.info("wdebug: acore kc={x} pos={d} base={x} R={d} small={d} dcp={d} kv8={} wide={}", .{ kc, pos, f.base, R, b.small, f.dcp, f.kv8 != null, R > max_rows });
        if (!q_done) {
            const p0 = try f.pbegin();
            try f.lin(b, &L.q_b, b.qn, R, b.q, .bf16);
            try f.pend(p0, .attn_proj);
        }
        const p1 = try f.pbegin();
        const wide = R > max_rows;
        if (wide) {
            if (f.base != 0) return error.WindowTooWide; // several streams: decode windows only
            const wd = f.wide orelse return error.WindowTooWide;
            try wd.absorb(f, b, L, pos, R);
        } else {
            if (f.base != 0) {
                try t_.absorbRows(b.q, L.wk, f.inv, b.qlat, b.qrot, pos, R, m);
            } else {
                try t_.absorb(b.q, L.wk, f.inv, b.qlat, b.qrot, pos, R, m);
            }
            try f.site(L, .o); // expand's and o_proj's weights, during the attention core
        }
        const tiling = if (R <= b.small) tri.attn_decode else tri.attn_prompt;
        if (f.sideOn()) |sd| try sd.join(f.s); // TF_GLM53_SIDE: the window's latents are in the cache
        if (f.kv8) |v| {
            // TF_GLM53_KV8: the unchanged attention kernels read the dequantized bf16 scratch, not the plane. Wide
            // windows take the direct mirror (their keys are rows 0..p, selections included); decode windows take the
            // block scratch with the static ramp tables (findings-kv8.md §2.3). TF_GLM53_KV8_V2: the wide windows
            // instead read the plane itself - the KV8 variants of _attn_chunks dequantize in-kernel (findings-kv8-v2.md).
            if (f.dcp > 1) {
                // The DECODE windows (R <= small) take the mode-1/2 BLOCK STAGING - the DCP1-parallel machinery,
                // depth-flat: the dequant's src = BASE[r] + TOK (the slots-plane row; the tok table holds the
                // slot-local shares), the blocks at r*K+i, the launches read them through the static ramp tables.
                // This replaces the O(valid-rows) mode-3/4 mirror PER LAYER PER ROUND that cost +51 ms/round at
                // 150K depth (findings-dcp2-build.md §C.7) and would cost ~+200 at 563K. The PROMPT windows keep
                // the mode-3 shard mirror (their reads are the dense shard prefix), now PROF-marked.
                if (R <= b.small) {
                    const sd = try v.dequantDecode(f, kc, pos, f.base, b.tok, R);
                    if (wdebug_on) std.log.info("wdebug: dqd lc={x} tok={x} base={x} kc={x} pos={d} fbase={x} R={d}", .{ sd.lc, sd.tok, sd.base, kc, pos, f.base, R });
                    try f.attentionDcp(b, sd.lc, pos, R, tiling, sd.tok, sd.base);
                } else {
                    const pdq = try f.pbegin();
                    const lc = try v.dequantShard(f, kc, pos, R);
                    try f.pend(pdq, .kv8_dequant);
                    if (wdebug_on and L.index == 0 and R >= 8) {
                        const o3 = kern.Ops{ .k = f.k, .s = f.s };
                        var hb: [1200]u8 = undefined;
                        const wbytes = v.w * 2;
                        const n = @min(wbytes, hb.len);
                        const rows_chk = [5]usize{ 0, R / 4, R / 2, (3 * R) / 4, R - 1 };
                        for (rows_chk) |row| {
                            o3.download(hb[0..n], lc + row * wbytes) catch continue;
                            f.s.synchronize() catch {};
                            var sum: u32 = 0;
                            for (hb[0..n]) |bb| sum +%= bb;
                            std.log.info("wdebug: pscratch-check kc={x} base={x} R={d} row={d} sum={x}", .{ kc, f.base, R, row, sum });
                        }
                    }
                    try f.attentionDcp(b, lc, pos, R, tiling, b.tok, f.base);
                }
            } else if (wide) {
                if (f.kv8_v2) {
                    try t_.attentionKv8(b.qlat, b.qrot, kc, b.tok, pos, b.po, b.pm, b.pl, b.ol, b.dummy, R, m);
                } else {
                    // --prompt-prof: the mirror re-dequantization timed apart from the attention read (the O(pos)
                    // per-chunk term; zero under v2)
                    const pd = try f.pbegin();
                    const lc = try v.dequantPrompt(f, kc, pos, R);
                    try f.pend(pd, .kv8_dequant);
                    try t_.attention(b.qlat, b.qrot, lc, b.tok, pos, b.po, b.pm, b.pl, b.ol, b.dummy, R, m, tiling);
                }
            } else if (f.base != 0) {
                const s = try v.dequantDecode(f, kc, pos, f.base, b.tok, R);
                if (wdebug_on) std.log.info("wdebug: dqd lc={x} tok={x} base={x} kc={x} pos={d} fbase={x} R={d}", .{ s.lc, s.tok, s.base, kc, pos, f.base, R });
                try t_.attentionRows(b.qlat, b.qrot, s.lc, s.tok, pos, s.base, b.po, b.pm, b.pl, b.ol, b.dummy, R, m, tiling);
            } else {
                const s = try v.dequantDecode(f, kc, pos, 0, b.tok, R);
                try t_.attention(b.qlat, b.qrot, s.lc, s.tok, pos, b.po, b.pm, b.pl, b.ol, b.dummy, R, m, tiling);
            }
        } else if (f.dcp > 1) {
            try f.attentionDcp(b, kc, pos, R, tiling, b.tok, f.base);
        } else if (f.base != 0) {
            try t_.attentionRows(b.qlat, b.qrot, kc, b.tok, pos, f.base, b.po, b.pm, b.pl, b.ol, b.dummy, R, m, tiling);
        } else {
            try t_.attention(b.qlat, b.qrot, kc, b.tok, pos, b.po, b.pm, b.pl, b.ol, b.dummy, R, m, tiling);
        }
        if (wide) {
            try f.wide.?.expand(f, b, L, R);
        } else {
            try t_.expand(b.ol, L.wv, b.o, R, m);
        }
        try f.pend(p1, .attn_core);
        const p2 = try f.pbegin();
        try f.lin(b, &L.o_proj, b.o, R, b.part, .f32);
        try f.pend(p2, .attn_proj);
    }

    // ------------------------------------------------------------------------------- Phase 2b: device windows ---

    /// fused.compute for rows b.ids[:R] (device ids) at st.pos ..: the target's caches written, its final hidden in
    /// b.hidden[:R], then `head` over every row (`last`: only row R - 1, as logits="last"). No host step: capturable.
    pub fn compute(f: *const Forward, b: *const Buffers, st: *const State, R: usize, T: ?usize, hm: HeadMode, last: bool) !void {
        // fused.compute's _SideWindow (TF_GLM53_SIDE: windows up to min(b.small, MAX_ROWS) rows) and _compute_pf
        // (TF_GLM53_L2PF: up to min(b.small, L2PF_ROWS) rows); both streams rejoin before the window ends
        const sd = f.sideWindow(b, R);
        var pf: ?*dec.L2pf = null;
        if (f.l2pf) |p| {
            if (!p.active and R <= @min(b.small, p.set.rows)) pf = p;
        }
        if (sd) |x| x.on = true;
        if (pf) |p| p.active = true;
        defer {
            if (sd) |x| x.on = false;
            if (pf) |p| p.active = false;
        }
        try f.computeWindow(b, st, R, T, hm, last);
        if (pf) |p| try p.join(f.s);
        if (sd) |x| try x.join(f.s);
    }

    /// The side stream a window of R rows of `b` runs beside, if any (_SideWindow: not already inside one).
    fn sideWindow(f: *const Forward, b: *const Buffers, R: usize) ?*dec.Side {
        const sd = f.side orelse return null;
        if (sd.on or !sd.letters.any() or R > @min(b.small, max_rows)) return null;
        return sd;
    }

    fn computeWindow(f: *const Forward, b: *const Buffers, st: *const State, R: usize, T: ?usize, hm: HeadMode, last: bool) !void {
        const c = f.w.config;
        const D = c.hidden;
        const o = f.ops();
        // THE PRINT-BISECT (sanitizer+NCCL unusable on this stack — the gated print+sync form): the
        // RUNTIME toggle (set at the first real prefill) — the prewarm's scratch windows run in
        // CAPTURE mode where a sync is itself illegal (the env-gated form crashed the boot on
        // instrument, not on defect); the first failing sync names the phase.
        const wdbg = wdebug_on;
        try o.embedRows(f.w.embed, b.ids, D, b.x, R, false);
        if (wdbg) {
            try f.s.synchronize();
            std.log.info("wdebug: embed R={d}", .{R});
        }
        for (0..f.w.layers.len) |li| {
            try f.layer(b, st, li, R, T);
            if (wdbg) {
                try f.s.synchronize();
                std.log.info("wdebug: layer {d}", .{li});
                if (li == 0 and R >= 2) {
                    var hb: [12288]u8 = undefined;
                    const D2 = D * 2;
                    const n = @min(D2, hb.len);
                    for ([_]usize{ 0, R - 1 }) |row| {
                        o.download(hb[0..n], b.x + row * D2) catch continue;
                        f.s.synchronize() catch {};
                        var sum: u32 = 0;
                        for (hb[0..n]) |bb| sum +%= bb;
                        std.log.info("wdebug: hidden-check R={d} row={d} sum={x} x={x} ids={x}", .{ R, row, sum, b.x, b.ids });
                    }
                }
            }
            try f.tapRows(b, li, b.x, R);
        }
        try o.copy(b.hidden, b.x, R * D * 2);
        if (hm == .none) return;
        if (last) return f.headDevice(b, b.x + (R - 1) * D * 2, f.w.final_norm, 1, hm);
        return f.headDevice(b, b.x, f.w.final_norm, R, hm);
    }

    /// Phase 4 (fused._compute's tap): layer li's output rows x [R, D] into b.taps' column block of its slot, when a
    /// drafter reads that layer (and `b` holds taps).
    pub fn tapRows(f: *const Forward, b: *const Buffers, li: usize, x: u64, R: usize) !void {
        if (f.taps.n == 0 or b.taps == 0 or li >= f.taps.slot.len) return;
        const slot = f.taps.slot[li];
        if (slot < 0) return;
        const D = f.w.config.hidden;
        const sl: usize = @intCast(slot);
        try f.ops().copyRows(b.taps + sl * D * 2, b.tap_n * D * 2, x, D * 2, D * 2, R);
    }

    /// fused.head on n rows from `x` (bf16, row stride D) normed with `norm`: "local" leaves this rank's logits in
    /// b.lg[:n] ([n, vocab_part], contiguous); "argmax" / "draft" leave the token in b.argmax[:n] (int64) after exchanging
    /// each rank's first maximum ([value, id, -, -] records, b.amax_all) and taking the lowest rank among the equal
    /// maxima. "draft": the 4-bit draft head over the reduced vocabulary (headq q4), ids through its table.
    pub fn headDevice(f: *const Forward, b: *const Buffers, x: u64, norm: u64, n: usize, mode: HeadMode) !void {
        if (std.c.getenv("TF_GLM53_ROUTER_TRACE")) |tv| if (std.mem.span(tv).len > 0 and std.mem.span(tv)[0] != 48) std.debug.print("HEAD n={d} mode={s}\n", .{ n, @tagName(mode) }); // §100 bisect
        const c = f.w.config;
        const D = c.hidden;
        const o = f.ops();
        const t_ = f.t();
        const world = f.w.world;
        if (!f.w.hasHead() or b.lg == 0) return error.NoHead;
        if (n > b.head_rows) return error.WindowTooWide;
        try t_.rmsnorm(x, D, norm, b.fnormed, D, n, D, c.eps);
        const V = f.w.vocab_part;
        switch (mode) {
            .none => return,
            .local => return headLogits(f.w, t_, o, b.fnormed, b.lg, b.hpart, b.hxh, n),
            .argmax => {
                try headLogits(f.w, t_, o, b.fnormed, b.lg, b.hpart, b.hxh, n);
                try o.argmaxRec(b.lg, V, V, 0, f.w.vocab_off, b.amax, n);
            },
            .draft => {
                const dh = f.w.draft orelse return error.NoDraftHead;
                try t_.groupSums(b.fnormed, D, b.gsum, n, D);
                try o.q4Head(b.fnormed, b.gsum, dh.w, dh.scales, dh.biases, dh.n, dh.npad, dh.k, dh.sk, b.lg, n);
                try o.argmaxRec(b.lg, dh.n, dh.n, dh.ids, 0, b.amax, n);
            },
        }
        if (f.fast != null and world > 1) {
            try f.fast.?.allGather(f.s, b.amax, b.amax_all, n * 16); // fused.head: w.fast.all_gather(a, g)
        } else if (f.comm) |cm| {
            try cm.allGather(b.amax, b.amax_all, n * 4, .f32, f.s.handle);
        } else if (world > 1) {
            try o.fill32(b.amax_all, 0, world * n * 4);
            try o.copy(b.amax_all + f.w.rank * n * 16, b.amax, n * 16);
        } else {
            try o.copy(b.amax_all, b.amax, n * 16);
        }
        try o.pickRank(b.amax_all, world, n, b.argmax);
    }

    /// fused.small_gather: every rank's fp32 words x [n] -> out [world, n_padded] in rank order over the RoCE gather
    /// when it is up and the whole result fits its buffers (n padded to whole 16-byte packs; the pad words are this
    /// rank's zeros), else NCCL ([world, n]). Returns the row stride in words. A gather only moves words.
    pub fn smallGather(f: *const Forward, x: u64, out: u64, n: usize) !usize {
        const world = f.w.world;
        const np = std.mem.alignForward(usize, n, 4);
        if (f.fast) |r| {
            if (world > 1 and world * np * 4 <= roce.max_bytes) {
                if (np > n) try f.ops().fill32(x + n * 4, 0, np - n);
                try r.allGather(f.s, x, out, np * 4);
                return np;
            }
        }
        if (f.comm) |cm| {
            try cm.allGather(x, out, n, .f32, f.s.handle);
        } else {
            try f.ops().copy(out, x, n * 4);
        }
        return n;
    }

    /// The MTP head's mode: the reduced 4-bit draft vocabulary when loaded, else the full share's argmax.
    pub fn draftMode(f: *const Forward) HeadMode {
        return if (f.w.draft != null) .draft else .argmax;
    }

    pub const MtpOpts = struct {
        /// head on the last row (logits="last"); false: logits="none" (prompt rows)
        head: bool = true,
        /// position 0's embedding zeroed (the first prompt chunk)
        zero_first: bool = false,
        /// TF_GLM53_MTP "x/normed": the chain passes on shared_head-normed rows
        chain_normed: bool = true,
        /// a chain's first step keeps its last row's selection in b.tok[0]
        keep_sel: bool = false,
        /// later one-row steps attend b.tok[0] (TF_GLM53_MTP_REUSE 1 / 2)
        reuse: u32 = 0,
        /// Phase 4b (mtp_compute's `last`, several streams): the head over `last_n` rows of the layer's output picked
        /// by the int64 device table `last` (each drafting stream's last row) into b.me first; 0: the window's last row
        last: u64 = 0,
        last_n: usize = 0,
    };

    /// fused.mtp_compute: rows (b.hin[:n] = the previous position's hidden, b.ids[:n] = the token) through the MTP
    /// layer at st.mpos ..; b.hidden[:n] the chain's rows; with `head` the last row's draft in b.argmax[0].
    pub fn mtpCompute(f: *const Forward, b: *const Buffers, st: *const State, n: usize, T: ?usize, opts: MtpOpts) !void {
        const c = f.w.config;
        const D = c.hidden;
        const o = f.ops();
        const t_ = f.t();
        const m = f.w.mtp orelse return error.NoMtpLayer;
        if (b.mcat == 0) return error.NoMtpScratch;
        try o.embedRows(f.w.embed, b.ids, D, b.me, n, opts.zero_first);
        try t_.rmsnorm(b.me, D, m.enorm, b.mh, D, n, D, c.eps);
        try t_.rmsnorm(b.hin, D, m.hnorm, b.normed, D, n, D, c.eps);
        try t_.cat2(b.mh, b.normed, b.mcat, n, D);
        if (f.mtp_dense != null and n > max_rows) {
            // prompt rows: one cuBLAS bf16 GEMM into b.x (bf16 of fp32 sums, as the router's split-K partials + cast)
            try f.mtp_dense.?.ehProj(f, b.mcat, m.eh_proj, b.x, n);
        } else {
            if (n > b.erows) return error.NoMtpScratch;
            try t_.router(b.mcat, 2 * D, m.eh_proj, b.mx32, b.epart, n, 2 * D, D);
            try o.f32ToBf16(b.mx32, b.x, n * D);
        }
        {
            // TF_GLM53_SIDE: draft and MTP refresh windows too (no L2 prefetch: l2pf skips the MTP passes)
            const sd = f.sideWindow(b, n);
            if (sd) |x| x.on = true;
            defer if (sd) |x| {
                x.on = false;
            };
            try f.layerWith(b, &f.w.mtp.?.layer, st.mkc, st.mic, st.mpos, n, T, opts.reuse);
            if (sd) |x| try x.join(f.s);
        }
        if (opts.keep_sel and T != null and n > 1) {
            try o.copy(b.tok, b.tok + (n - 1) * c.index_topk * 4, c.index_topk * 4);
            if (f.dcp > 1) try o.copy(b.cnt, b.cnt + (n - 1) * 4, 4); // b.cnt[0:1].copy_(b.cnt[n - 1:n])
        }
        if (opts.chain_normed) {
            try t_.rmsnorm(b.x, D, m.head_norm, b.hidden, D, n, D, c.eps);
        } else {
            try o.copy(b.hidden, b.x, n * D * 2);
        }
        if (!opts.head) return;
        if (opts.last != 0) {
            // several streams: index_select(x, 0, last, out=b.me[:S]) (the embeddings are spent by now), then the
            // head over those S rows
            if (opts.last_n == 0 or opts.last_n > n) return error.Invalid;
            try o.rowsGather(b.me, b.x, opts.last, opts.last_n, D * 2);
            return f.headDevice(b, b.me, m.head_norm, opts.last_n, f.draftMode());
        }
        try f.headDevice(b, b.x + (n - 1) * D * 2, m.head_norm, 1, f.draftMode());
    }

    /// fused.target_hidden_for_mtp (normed): rmsnorm(src.hidden[row0 .. row0 + n], final_norm) into `out` rows.
    pub fn targetHidden(f: *const Forward, src: *const Buffers, row0: usize, n: usize, out: u64) !void {
        const c = f.w.config;
        const D = c.hidden;
        try f.t().rmsnorm(src.hidden + row0 * D * 2, D, f.w.final_norm, out, D, n, D, c.eps);
    }

    /// fused.select (DCP 1) through the radix select for every window size: b.tok[row0 : row0 + R] ascending, window
    /// rows row0 .. (row0 > 0: a sequence-parallel prompt half's own rows) in blocks of SEL_ROWS = 128 from row0.
    pub fn select(f: *const Forward, b: *const Buffers, ic: u64, pos: u64, row0: usize, R: usize, T: usize) !void {
        const c = f.w.config;
        const t_ = f.t();
        if (T > b.score_cols) return error.ScoreBufferTooSmall;
        if (T < c.index_topk) return error.Invalid;
        if (f.dcp > 1) return f.selectDcp(b, ic, pos, row0, R, T);
        var r0: usize = row0;
        while (r0 < row0 + R) : (r0 += 128) {
            const n = @min(128, row0 + R - r0);
            const iq = b.iq + r0 * c.index_heads * c.index_dim * 2;
            const iw = b.iw + r0 * c.index_heads * 4;
            // --prompt-prof: the scoring over all prior keys apart from the top-k over its scores
            const pi = try f.pbegin();
            if (f.fp8_index) {
                // TF_GLM53_FP8_INDEX, increment 2 (checkpoint b): the native fp8-tensor-core scoring -
                // glm53_fp8_scores.cu on the fatbin path, off the AOT pack (findings-prefill-fp8-cuda.md).
                // The plane pointer twice (CODES the e4m3 bytes, SCALES the same storage as floats), the
                // QSCALE / WSCALE of the Triton constexprs as runtime args. EVERY selection switches - the
                // cache has one layout a boot, so the decode windows' selections read the plane too. The
                // Triton _index_scores8 stays in the tree as the documented fallback (the revert: this branch
                // back to t_.indexScores8 / indexScoresRows8).
                const qs: f32 = @floatCast(std.math.pow(f64, @floatFromInt(c.index_dim), -0.5));
                const ws: f32 = @floatCast(std.math.pow(f64, @floatFromInt(c.index_heads), -0.5));
                try f.ops().fp8Scores(iq, iw, ic, ic, pos, f.base, b.sc, T, r0, n, f.base != 0, qs, ws, f.fp8_stride);
            } else if (f.base != 0) {
                try t_.indexScoresRows(iq, iw, ic, pos, f.base, b.sc, T, r0, n, c.index_heads, c.index_dim);
            } else {
                try t_.indexScores(iq, iw, ic, pos, b.sc, T, r0, n, c.index_heads, c.index_dim);
            }
            try f.pend(pi, .index_scores);
            const ps = try f.pbegin();
            const out = b.tok + r0 * c.index_topk * 4;
            const blocks = (T + kern.topk_chunk - 1) / kern.topk_chunk;
            if (f.multi_select and n < radix_min_rows and n <= b.tsel.rows and blocks <= b.tsel.blocks) {
                // fused.select below RADIX_MIN_ROWS (torch.topk + sort there): the same K columns, ascending
                try f.ops().topkColumns(b.sc, T, T, n, c.index_topk, false, b.tsel, out, c.index_topk);
            } else if (f.topk_stream and R > max_rows and @as(usize, n) * T * 4 + 15 + @as(usize, n) * 1024 * 8 <= @min(b.rows, @as(usize, 128)) * b.score_cols * 8) {
                // TF_GLM53_TOPK_STREAM: the prefill blocks (R > max_rows; decode windows and the SP halves stay
                // radix). SCRATCH: this block's score words end at n * T int32 - the int64-packed-DCP sizing leaves
                // the buffer's second half spare at DCP 1, so the [n, 2 x 1024] int32 collect scratch (the score
                // words and the reversed columns, §10) carves from its tail (16-byte aligned: the sweep's warmup
                // pointers are 16-aligned and the variant specializes on it — an 8-aligned carve mismatches the
                // pack's variant; found at the first TOPK_STREAM boot)
                const scratch = b.sc + ((n * T * 4 + 15) & ~@as(usize, 15));
                try t_.topkRow(b.sc, out, scratch, T, n, c.index_topk);
            } else {
                try t_.selectKeys(b.sc, out, T, n, c.index_topk);
            }
            try f.pend(ps, .select);
        }
    }

    // ---------------------------------------------------------------- Phase 3b: decode context parallelism ---

    /// This rank in the DCP group (fused: `w.rank if w.dcp > 1 else 0`).
    pub fn dcpRank(f: *const Forward) usize { // pub: the SP split choreography (cuda_prompt.zig) reads it
        // The PARITY-MIRROR (findings-dcp2-divisor.md §4.2): at dcp < world every rank writes and scores its
        // PARITY CLASS's shard (p % dcp == rank % dcp); at dcp == world the parity is the rank itself.
        return if (f.dcp > 1) dcpm.parity(f.dcp, f.w.rank) else 0;
    }

    /// TF_GLM53_SMALL_P2P (round 2, findings-dcp2-build.md §F): the small decode windows at dcp < world take
    /// the subgroup p2p exchanges - DEFAULT ON (the optimized path; an upper-bound -17 ms/round vs the NCCL
    /// world gather at R 18). The calibrated pool cost is +8.0 MB a graph = 26.5 KB a captured nccl call,
    /// flat to 480 live graphs (the dcp-bench serve-scale probe) = ~3.8 GiB at 471 graphs, inside the §F.4
    /// prewarm preflight's accounting. Set =0 ONLY to bisect against the world-gather path.
    var small_p2p_cache: ?bool = null;
    fn smallP2p() bool {
        if (small_p2p_cache) |v| return v;
        small_p2p_cache = blk: {
            const e = std.c.getenv("TF_GLM53_SMALL_P2P") orelse break :blk true; // default: the optimized path
            break :blk !std.mem.eql(u8, std.mem.sliceTo(e, 0), "0");
        };
        return small_p2p_cache.?;
    }

    /// fused.dcp_gather: every rank's `bytes` at x into out [world, bytes] in rank order - the RoCE one-shot gather for
    /// decode-window exchanges (`small`) when it is up and the whole result fits its buffers, else NCCL (`count`
    /// elements of `dt`). A gather only moves words: both give the same bits.
    fn dcpGather(f: *const Forward, x: u64, out: u64, bytes: usize, count: usize, dt: cuda.nccl.DataType, small: bool) !void {
        const world = f.w.world;
        if (small) if (f.fast) |r| {
            if (bytes % 16 == 0 and world * bytes <= roce.max_bytes) return r.allGather(f.s, x, out, bytes);
        };
        const cm = f.comm orelse return error.NoCommunicator;
        try cm.allGather(x, out, count, dt, f.s.handle);
    }

    /// fused.select's DCP branch for window rows row0 .. row0 + R - 1 (blocks of SEL_ROWS = 128): this rank's slots
    /// t < Tl = ceil(T / dcp) scored as packed int64 keys (_index_scores DCP, RANK, PACK), its top kk = min(K, Tl) of
    /// them padded to K (int64 min + 1), every rank's candidates gathered (RoCE for decode windows), then the K best
    /// of the dcp * K - by the whole key (torch.topk) below RADIX_MIN_ROWS rows, by the high word with ties to the
    /// lower column at or above it (topk.top_keys: the Python radix branch's own tie rule) - and this rank's share as
    /// local slots ascending in b.tok, its count in b.cnt. The per-rank top kk is one set however it is found
    /// (scores tie-free by position): the multi-block top-k over the keys' high words (glm53_topk_*, mode 2) where its
    /// scratch fits, else topk.top_keys.
    fn selectDcp(f: *const Forward, b: *const Buffers, ic: u64, pos: u64, row0: usize, R: usize, T: usize) !void {
        const c = f.w.config;
        const t_ = f.t();
        const o = f.ops();
        const G = f.dcp;
        const rank = f.dcpRank();
        const wr = f.w.rank; // the group producer (the parity-mirror's block member)
        const mine = wr - dcpm.scoredBase(G, wr);
        const sub = G != f.w.world;
        const K = c.index_topk;
        if (b.dcp != G or b.cand == 0) return error.NoDcpBuffers;
        const Tl = dcpm.localCols(T, G);
        const kk = @min(K, Tl);
        var r0: usize = row0;
        while (r0 < row0 + R) : (r0 += 128) {
            const n = @min(128, row0 + R - r0);
            const iq = b.iq + r0 * c.index_heads * c.index_dim * 2;
            const iw = b.iw + r0 * c.index_heads * 4;
            // fused.RADIX and n >= RADIX_MIN_ROWS (dcp > 1). Several streams (a base table): never - a 32-row window of
            // four streams must merge the candidates as each stream's own narrower window does (by the whole key: equal
            // scores go to the lower position), or a tie at the cut would pick other keys than the stream picks alone
            const radix = n >= radix_min_rows and f.base == 0;
            if (f.base != 0) {
                // several streams: each row's own position and its stream's index-key rows (the candidates, their
                // gather and the merge below are a row's own: nothing in them knows the position)
                try t_.indexScoresRowsDcp(iq, iw, ic, pos, f.base, b.sc, Tl, r0, n, c.index_heads, c.index_dim, G, rank);
            } else {
                try t_.indexScoresDcp(iq, iw, ic, pos, b.sc, Tl, r0, n, c.index_heads, c.index_dim, G, rank);
            }
            const tok = b.tok + r0 * K * 4;
            const blocks = (Tl + kern.topk_chunk - 1) / kern.topk_chunk;
            // ROUND 3 (findings-dcp2-build.md §I): the radix columns pass loses to top_keys past ~32K local
            // columns (the select-bench ladder: 1903 vs 1448 us at 64K, 4713 vs 3951 at 176K - both arms exact,
            // the same set and tie rule) - route the deep prompt chunks to top_keys. TF_GLM53_TOPK_COL_MAX
            // (default 32768) is the crossover; 0 restores the old routing everywhere it fits.
            const topk_col_max: usize = blk: {
                const e = std.c.getenv("TF_GLM53_TOPK_COL_MAX") orelse break :blk 49152;
                break :blk std.fmt.parseInt(usize, std.mem.sliceTo(e, 0), 10) catch 49152;
            };
            if (Tl > topk_col_max and kk < Tl and n <= b.tsel.rows and blocks <= b.tsel.blocks and b.tsel.cap > 0 and kk <= b.tsel.cap) {
                // the compact radix (§I): beats the shipping arm (top_keys - the prompt select's own,
                // tsel.rows gated the columns pass out) by 13-20% past ~48K local columns; the guards
                // keep top_keys as the exact fallback whenever the compact scratch cannot serve
                try o.topkColumnsCompact(b.sc, Tl, Tl, n, kk, b.tsel, b.dsel, kk);
                try o.dcpCands(b.sc, Tl, b.dsel, kk, kk, K, true, b.dmine, n);
            } else if (n <= b.tsel.rows and blocks <= b.tsel.blocks) {
                // this rank's kk best columns ascending (b.tok's rows as scratch), then their keys
                try o.topkColumnsMode(b.sc, Tl, Tl, n, kk, 2, b.tsel, tok, K);
                try o.dcpCands(b.sc, Tl, tok, K, kk, K, false, b.dmine, n);
            } else if (kk == Tl) {
                // topk.top_keys(sc, kk) with kk == Tl (Tl <= K: short prompt chunks under DCP) keeps every key in
                // column order - the score rows themselves, bit for bit. Copied rather than launched: _select_keys'
                // K constexpr would be a different variant for each such Tl (none of them in a finite AOT set).
                try o.copy(b.dsel, b.sc, n * Tl * 8);
                try o.dcpCands(b.sc, Tl, b.dsel, kk, kk, K, true, b.dmine, n);
            } else {
                try t_.topKeys(b.sc, b.dsel, Tl, n, kk);
                try o.dcpCands(b.sc, Tl, b.dsel, kk, kk, K, true, b.dmine, n);
            }
            if (sub) {
                const cm = f.comm orelse return error.NoCommunicator;
                try o.copy(b.cand + mine * n * K * 8, b.dmine, n * K * 8);
                try dcpm.gatherFrom(cm, b.dmine, b.cand, n * K, .i64, 8, G, wr, f.s.handle);
            } else {
                try f.dcpGather(b.dmine, b.cand, n * K * 8, n * K, .i64, n <= b.small);
            }
            if (std.c.getenv("TF_GLM53_KV8_DEBUG") != null and ((r0 == 0 and row0 + R < 300 and n == 128) or r0 == 2048)) {
                // kv8debug: the select's stages for the row at position 2048 - the SP first block (r0 0,
                // local row 90) vs the non-SP block [2048, 2176) (local row 0). sc/dmine/cand int64 words.
                const dbg_row: usize = if (r0 == 2048) 0 else 90;
                const colw: usize = (b.score_cols + G - 1) / G;
                var db: [32]u8 = undefined;
                o.download(std.mem.sliceAsBytes(db[0..]), b.sc + dbg_row * colw * 8) catch {};
                f.s.synchronize() catch {};
                std.log.info("kv8debug: sel r0={d} sc row {d}: {x} {x} {x} {x}", .{ r0, dbg_row, std.mem.bytesToValue(u32, db[0..4]), std.mem.bytesToValue(u32, db[8..12]), std.mem.bytesToValue(u32, db[16..20]), std.mem.bytesToValue(u32, db[24..28]) });
                o.download(std.mem.sliceAsBytes(db[0..]), b.dmine + dbg_row * K * 8) catch {};
                f.s.synchronize() catch {};
                std.log.info("kv8debug: sel r0={d} dmine row {d}: {x} {x} {x} {x}", .{ r0, dbg_row, std.mem.bytesToValue(u32, db[0..4]), std.mem.bytesToValue(u32, db[8..12]), std.mem.bytesToValue(u32, db[16..20]), std.mem.bytesToValue(u32, db[24..28]) });
                for ([_]usize{ 0, 4 }) |s2| {
                    o.download(std.mem.sliceAsBytes(db[0..]), b.cand + (s2 * n + dbg_row) * K * 8) catch continue;
                    f.s.synchronize() catch {};
                    std.log.info("kv8debug: sel r0={d} cand[{d}] row {d}: {x} {x} {x} {x}", .{ r0, s2, dbg_row, std.mem.bytesToValue(u32, db[0..4]), std.mem.bytesToValue(u32, db[8..12]), std.mem.bytesToValue(u32, db[16..20]), std.mem.bytesToValue(u32, db[24..28]) });
                }
                {
                    // every rank's slice: cols 0..3 (the real candidates) and col 500 (the min+1 pad)
                    var c0w: [8]u32 = undefined;
                    var c1w: [8]u32 = undefined;
                    for (0..G) |s2| {
                        o.download(std.mem.sliceAsBytes(db[0..8]), b.cand + (s2 * n + dbg_row) * K * 8) catch continue;
                        f.s.synchronize() catch {};
                        c0w[s2] = std.mem.bytesToValue(u32, db[0..4]);
                        c1w[s2] = std.mem.bytesToValue(u32, db[8..12]);
                    }
                    std.log.info("kv8debug: sel r0={d} cand col0/col1 lo words per rank: {x}/{x} {x}/{x} {x}/{x} {x}/{x} {x}/{x} {x}/{x} {x}/{x} {x}/{x}", .{ r0, c0w[0], c1w[0], c0w[1], c1w[1], c0w[2], c1w[2], c0w[3], c1w[3], c0w[4], c1w[4], c0w[5], c1w[5], c0w[6], c1w[6], c0w[7], c1w[7] });
                    var all8: [8]u32 = undefined;
                    for (0..G) |s2| {
                        o.download(std.mem.sliceAsBytes(db[0..8]), b.cand + ((s2 * n + dbg_row) * K + 500) * 8) catch continue;
                        f.s.synchronize() catch {};
                        all8[s2] = std.mem.bytesToValue(u32, db[0..4]);
                    }
                    std.log.info("kv8debug: sel r0={d} cand col500 lo words all ranks: {x} {x} {x} {x} {x} {x} {x} {x}", .{ r0, all8[0], all8[1], all8[2], all8[3], all8[4], all8[5], all8[6], all8[7] });
                }
            }
            try o.dcpMerge(b.cand, n, K, G, rank, radix, tok, b.cnt + r0 * 4);
            if (std.c.getenv("TF_GLM53_KV8_DEBUG") != null and ((r0 == 0 and row0 + R < 300 and n == 128) or r0 == 2048)) {
                const dr: usize = if (r0 == 2048) 0 else 90;
                var tb2: [16]u8 = undefined;
                var cb2: [8]u8 = undefined;
                o.download(std.mem.sliceAsBytes(tb2[0..]), tok + dr * K * 4) catch {};
                o.download(std.mem.sliceAsBytes(cb2[0..]), b.cnt + (r0 + dr) * 4) catch {};
                f.s.synchronize() catch {};
                std.log.info("kv8debug: POSTMERGE r0={d} row {d} cnt={d} tok0..3={d},{d},{d},{d}", .{ r0, dr, std.mem.bytesToValue(i32, cb2[0..4]), std.mem.bytesToValue(i32, tb2[0..4]), std.mem.bytesToValue(i32, tb2[4..8]), std.mem.bytesToValue(i32, tb2[8..12]), std.mem.bytesToValue(i32, tb2[12..16]) });
            }
        }
    }

    /// fused._attention_dcp: this rank's absorbed queries packed ([R, H, lw + rd]) and gathered from every rank, all
    /// G * H heads attended over this rank's keys (_attn_dcp: chunk partials), merged per head into a normalized
    /// output and log-sum-exp by destination rank (_merge_lse), exchanged (decode windows: all-gather of everything,
    /// read our block; prompt chunks: NCCL all-to-all of head blocks) and combined in rank order (_dcp_combine) ->
    /// b.ol.
    fn attentionDcp(f: *const Forward, b: *const Buffers, kc: u64, pos: u64, R: usize, tl: tri.AttnTiling, tok: u64, base: u64) !void {
        const c = f.w.config;
        const t_ = f.t();
        const o = f.ops();
        const m = f.mla();
        const G = f.dcp;
        const rank = f.dcpRank();
        const H = m.heads;
        const lw = m.lw;
        const rd = m.rope;
        if (b.dcp != G or b.qall == 0) return error.NoDcpBuffers;
        const small = R <= b.small;
        const nch = @max(1, c.index_topk / tl.chk);
        // The parity-mirror (findings-dcp2-divisor.md §4.3/§4.4): at dcp < world the exchanges stay inside the
        // BLOCK (ranks scoredBase..+dcp-1 - the pair at DCP2, the quad at DCP4): qall block b = group
        // scoredBase+b's queries (the group's owner produces them), the E3 partials go to the group owners with
        // the contributor's PARITY as the slot. At dcp == world the block is the whole world and the plain
        // all-gather / all-to-all below is the same pattern - those paths are kept as-is (proper collectives).
        const wr = f.w.rank; // the head-group owner (TP8: rank r owns head-group r) and the group producer
        const mine = wr - dcpm.scoredBase(G, wr); // this rank's group within the block (== wr at dcp == world)
        const sub = G != f.w.world;
        // section 18 (TF_GLM53_E1_FP8): the E1 wire as e4m3 on the wide direct chunks - kv8Quant packs
        // qpack's rows (identity: DCP 1, POS[0] = b.dummy's 0), the gather moves half the bytes (.i8),
        // the scorer dequantizes in-register (the q-side twin of the KV8 key branch).
        const qfp8 = f.e1_fp8 and b.qall8 != 0 and nch == 1 and f.base == 0 and !small;
        try o.dcpQpack(b.qlat, b.qrot, b.qpack, R * H, lw, rd);
        // The SMALL windows at dcp < world gather the WHOLE world by default (the in-graph-proven all-gather /
        // RoCE one-shot) and the kernels read their scored groups' blocks at the base offset into the
        // world-gathered qall. TF_GLM53_SMALL_P2P (§F) swaps in the subgroup p2p - the [G] layout at 0,
        // an upper-bound -17 ms/round vs NCCL, at ~35 MB/graph of capture pool: off by default.
        const sp2p = small and smallP2p();
        const sbase = dcpm.scoredBase(G, wr);
        const qoff: u64 = if (sub and small and !sp2p) @intCast(sbase * R * H * (lw + rd) * 2) else 0;
        if (qfp8) {
            try o.kv8Quant(b.qpack, b.qpack8, b.dummy, b.dummy, R * m.heads, false, 1, 0, m.lw, m.lw + m.rope, 0, false); // 119.1: the query pack is not the latent plane; 119.2: the E1 wire stays v2 until the fp8 path is next taken
            if (sub) {
                const cm = f.comm orelse return error.NoCommunicator;
                const qb = R * m.heads * (m.lw + m.rope + 8);
                try o.copy(b.qall8 + mine * qb, b.qpack8, qb);
                try dcpm.gatherFrom(cm, b.qpack8, b.qall8, qb, .i8, 1, G, wr, f.s.handle);
            } else {
                try f.dcpGather(b.qpack8, b.qall8, R * m.heads * (m.lw + m.rope + 8), R * m.heads * (m.lw + m.rope + 8), .i8, small);
            }
        } else if (sub and (!small or sp2p)) {
            // the pair gather (49.3 us vs 167.7 world at R 18, dcp-bench §F)
            const cm = f.comm orelse return error.NoCommunicator;
            const qb = R * H * (lw + rd);
            try o.copy(b.qall + mine * qb * 2, b.qpack, qb * 2);
            try dcpm.gatherFrom(cm, b.qpack, b.qall, qb, .bf16, 2, G, wr, f.s.handle);
        } else {
            try f.dcpGather(b.qpack, b.qall, R * H * (lw + rd) * 2, R * H * (lw + rd), .bf16, small);
        }
        const e3m = f.e3_merge and base == 0 and nch == 1 and small and sub and !sp2p; // 126(g): the merged-E3 path (single-stream decode only)
        if (base != 0) {
            try t_.attnDcpRows(b.qall + qoff, kc, tok, b.cnt, pos, base, b.po, b.pm, b.pl, R, m, G, rank, tl);
        } else if (nch == 1) {
            // section 17: nch == 1 (the prompt chunks' 2048-key tiling) - the scorer writes the exchange
            // layout directly (osend/lsend), _merge_lse's single-chunk pass and the fp32 partial plane gone
            try t_.attnDcp(if (qfp8) b.qall8 else b.qall + qoff, kc, tok, b.cnt, pos, if (e3m) b.e3send else b.osend, if (e3m) b.e3send + R * H * lw * 2 else b.lsend, b.pl, R, m, G, rank, tl, true, qfp8, e3m);
        } else {
            try t_.attnDcp(b.qall + qoff, kc, tok, b.cnt, pos, b.po, b.pm, b.pl, R, m, G, rank, tl, false, false, false);
        }
        if (nch != 1) try t_.mergeLse(b.po, b.pm, b.pl, b.osend, b.lsend, R, G, m, nch);
        if (sub) {
            const cm = f.comm orelse return error.NoCommunicator;
            if (small and !sp2p) {
                if (e3m) {
                    // 126(g): ONE one-shot for both E3 planes (234 -> 156 collectives a round); the same
                    // bytes, the same member order - value-transparent, the DCP2 anchors must hold exactly
                    try f.dcpGather(b.e3send, b.e3recv, G * R * H * (lw * 2 + 4), G * R * H * (lw + 2), .bf16, true);
                } else {
                // the receipt path: the world all-gather (the small orecv/lrecv are [world][G] for it;
                // exchange's small o0 carries the base shift)
                try f.dcpGather(b.osend, b.orecv, G * R * H * lw * 2, G * R * H * lw, .bf16, true);
                try f.dcpGather(b.lsend, b.lrecv, G * R * H * 4, G * R * H, .f32, true);
                }
            } else {
                // the pair exchange (67 us vs ~168 world at R 18, dcp-bench §F) - TF_GLM53_SMALL_P2P,
                // and the prompt path's own; the exchange params are the !small ([G]-layout) ones
                try o.copy(b.orecv + mine * R * H * lw * 2, b.osend + mine * R * H * lw * 2, R * H * lw * 2);
                try dcpm.pairExchange(cm, b.osend, b.orecv, R * H * lw, .bf16, 2, G, wr, R * H * lw, f.s.handle);
                try o.copy(b.lrecv + mine * R * H * 4, b.lsend + mine * R * H * 4, R * H * 4);
                try dcpm.pairExchange(cm, b.lsend, b.lrecv, R * H, .f32, 4, G, wr, R * H, f.s.handle);
            }
        } else if (small) {
            try f.dcpGather(b.osend, b.orecv, G * R * H * lw * 2, G * R * H * lw, .bf16, true);
            try f.dcpGather(b.lsend, b.lrecv, G * R * H * 4, G * R * H, .f32, true);
        } else {
            const cm = f.comm orelse return error.NoCommunicator;
            try dcpm.allToAll(cm, b.osend, b.orecv, R * H * lw, .bf16, 2, f.s.handle);
            try dcpm.allToAll(cm, b.lsend, b.lrecv, R * H, .f32, 4, f.s.handle);
        }
        if (e3m) {
            // 126(g): the merged strides - dcp_combine's SS/SSL are runtime params: the (member,group)
            // block at R*H*(LW+2) bf16, its lse region R*H*(LW+2)/2 fp32 past the shifted base
            try t_.dcpCombine(b.e3recv, b.e3recv + R * H * lw * 2, b.ol, R, R * H * (lw + 2), R * H * ((lw + 2) / 2), m, G);
        } else {
            const ex = dcpm.exchange(small and (!sub or !sp2p), G, wr, R, H, lw);
            try t_.dcpCombine(b.orecv + ex.o0 * 2, b.lrecv + ex.l0 * 4, b.ol, R, ex.ss, ex.ssl, m, G);
        }
    }

    /// fused.head(x, final_norm, R, slice(R - 1, R), mode="argmax") for the window's last row: the final RMSNorm,
    /// this rank's logits over its vocabulary share (glue.router into b.lg), its first maximum (on the host, from
    /// b.lg: torch.argmax's rule), the [value, global id, 0, 0] record all-gathered (b.amax_all) and the lowest rank
    /// among the equal maxima. `host`: at least vocab_part floats. Synchronizes the stream (the pick goes to the host).
    pub fn head(f: *const Forward, b: *const Buffers, R: usize, host: []f32) !Pick {
        const c = f.w.config;
        const D = c.hidden;
        const V = f.w.vocab_part;
        const world = f.w.world;
        if (!f.w.hasHead() or b.lg == 0) return error.NoHead;
        if (host.len < V or world > max_world) return error.Invalid;
        const t_ = f.t();
        const o = f.ops();
        try t_.rmsnorm(b.x + (R - 1) * D * 2, D, f.w.final_norm, b.fnormed, D, 1, D, c.eps);
        try headLogits(f.w, t_, o, b.fnormed, b.lg, b.hpart, b.hxh, 1);
        try o.download(std.mem.sliceAsBytes(host[0..V]), b.lg);
        try f.s.synchronize();
        const i = argmaxFirst(host[0..V]);
        const rec: [4]f32 = .{ host[i], @floatFromInt(i + f.w.vocab_off), 0, 0 };
        try o.upload(b.amax, std.mem.asBytes(&rec));
        if (f.comm) |cm| {
            try cm.allGather(b.amax, b.amax_all, 4, .f32, f.s.handle);
        } else {
            try o.fill32(b.amax_all, 0, world * 4);
            try o.copy(b.amax_all + f.w.rank * 16, b.amax, 16);
        }
        var p: Pick = .{ .world = world };
        try o.download(std.mem.sliceAsBytes(p.gathered[0 .. world * 4]), b.amax_all);
        try f.s.synchronize();
        var vals: [max_world]f32 = undefined;
        for (0..world) |r| vals[r] = p.gathered[r * 4];
        const best = argmaxFirst(vals[0..world]);
        const id = p.gathered[best * 4 + 1];
        if (!(id >= 0 and id < @as(f32, @floatFromInt(c.vocab)))) return error.BadPick;
        p.token = @intFromFloat(id);
        p.local = i;
        return p;
    }

    /// fused.ffn_part: the dense MLP, or the router (fused.route) and experts_part.
    /// TF_GLM53_SIDE "f" inside a decode window: the shared expert on the side stream beside the router and the
    /// routed experts (fused.ffn_part), joined right before the routed down + combine launch reads b.sy.
    pub fn ffnPart(f: *const Forward, b: *const Buffers, L: *const W.Layer, R: usize) !void {
        // E2: the hybrid runs the port's MoE block in this same slot (the FFN half of the layer, on the compute
        // stream, before the gather) - the side stream, L2 prefetch and RoCE reduce around it are unchanged.
        if (f.moe_hook) |h| {
            if (h.has(h.ctx, L.index)) {
                try h.call(h.ctx, L.index, f.s, b.normed, b.part, R, R >= 64);
                return;
            }
        }
        if (L.experts != null) {
            var sy_ev: ?cuda.Event = null;
            var sd_used: ?*dec.Side = null;
            if (f.sideOn()) |sd| {
                if (sd.letters.f and (!f.shared_experts or R <= b.xs_rows)) {
                    const g = try f.forkSide(sd);
                    const bs = sideBufs(b);
                    try g.mlp(&bs, L, R, b.sy);
                    sy_ev = try sd.mark();
                    sd_used = sd;
                }
            }
            try f.route(b, L, 0, R);
            if (f.route_tap) |tap| try tap.take(f, b, L, R);
            try f.expertsPartWith(b, L, R, sy_ev);
            if (sd_used) |sd| sd.forked = false; // joined inside the routed chain (experts.routed's sy_ready)
            return;
        }
        const p0 = try f.pbegin();
        try f.mlp(b, L, R, b.part);
        try f.pend(p0, .dense);
    }

    /// fused.route: router logits and top-k of rows b.normed[r0 : r0 + n] -> b.pick / b.wts rows r0 ..
    pub fn route(f: *const Forward, b: *const Buffers, L: *const W.Layer, r0: usize, n: usize) !void {
        const c = f.w.config;
        const t_ = f.t();
        const D = c.hidden;
        try t_.router(b.normed + r0 * D * 2, D, L.router_w, b.mlog + r0 * c.experts * 4, b.rpart, n, D, c.experts);
        try t_.topk(b.mlog + r0 * c.experts * 4, L.router_bias, b.pick + r0 * c.top_k * 4, b.wts + r0 * c.top_k * 4, n, c.experts, c.top_k, c.routed_scaling, c.norm_topk);
    }

    /// fused.experts_part (shared expert not on a side stream): routed experts (picks in b.pick) + the shared expert
    /// of rows b.normed[:R] -> fp32 partial b.part[:R]. FOLD_SHARED with the decode kernel (the tf impl at every width,
    /// the shared impl up to its scratch's rows): the shared expert into b.sy first, the routed combine adds it.
    /// Wider windows of the shared impl (prompt chunks): SharedExperts.prefill (prompt_experts.cu) into b.part, then
    /// the shared expert into b.sy and b.part += b.sy.
    pub fn expertsPart(f: *const Forward, b: *const Buffers, L: *const W.Layer, R: usize) !void {
        return f.expertsPartWith(b, L, R, null);
    }

    /// expertsPart; `sy_ev` (TF_GLM53_SIDE "f"): the shared expert is already queued on the side stream into b.sy,
    /// this event marks its end (the routed chain waits on it right before its down + combine launch).
    pub fn expertsPartWith(f: *const Forward, b: *const Buffers, L: *const W.Layer, R: usize, sy_ev: ?cuda.Event) !void {
        const D = f.w.config.hidden;
        const ex = if (L.experts) |*e| e else return error.Invalid;
        if (!f.shared_experts or R <= b.xs_rows) {
            const pm0 = try f.pbegin(); // 128.27fz THE MOE MARK: the decode's fast path never marked - the experts ~40ms/round named by arithmetic only
            if (sy_ev == null) try f.mlp(b, L, R, b.sy); // first: its weights may still be in L2 (TF_GLM53_L2PF)
            const xs = b.xs orelse return error.NoExpertScratch;
            try f.ops().routed(b.normed, D, b.pick, b.wts, ex, xs, b.part, b.sy, R, sy_ev);
            try f.pend(pm0, .routed);
            return;
        }
        if (b.pe != null and kern.Ops.promptSupported(ex)) {
            if (f.k.pe.ipm != 112) return error.PromptExpertsUnsupported; // Buffers sized PromptScratch for IPM 112
            const p0 = try f.pbegin();
            try f.ops().promptRouted(b.normed, D, b.pick, b.wts, ex, b.pe.?, b.part, R, f.pe_det16);
            try f.pend(p0, .routed);
            const p1 = try f.pbegin();
            try f.mlp(b, L, R, b.sy);
            try f.ops().addF32(b.part, b.sy, R * D);
            try f.pend(p1, .shared);
            return;
        }
        // widths prompt_experts does not take (the MTP layer's 8-bit experts): the Python engine runs cuda-exl3's
        // grouped GEMM (not ported; its split-K atomics make it irreproducible run to run anyway). Here: the decode
        // kernel in blocks of its scratch's rows, every row its one-row bits. Only MTP prompt rows take this: drafts
        // only propose and the target verifies each, so no token depends on it (acceptance may move a little).
        // --mtp-dense 1 (the MTP layer): the shared expert into b.sy, then the dense bf16 experts through cuBLAS, the
        // combine adding b.sy (cuda_prompt.MtpDense.routed)
        if (f.mtp_dense) |md| {
            if (L.index == f.w.config.layers) {
                try f.mlp(b, L, R, b.sy);
                return md.routed(f, b, R);
            }
        }
        // blocks of xs_big.rows rows when the prompt buffers carved one (Phase 3a speed: each block re-reads every
        // routed expert it touches, so 10x fewer blocks read ~3x fewer bytes), else of the decode scratch's rows
        const xs = b.xs_big orelse (b.xs orelse return error.NoExpertScratch);
        const K = f.w.config.top_k;
        try f.mlp(b, L, R, b.sy);
        var r0: usize = 0;
        while (r0 < R) : (r0 += xs.rows) {
            const n = @min(xs.rows, R - r0);
            try f.ops().routed(b.normed + r0 * D * 2, D, b.pick + r0 * K * 4, b.wts + r0 * K * 4, ex, xs, b.part + r0 * D * 4, b.sy + r0 * D * 4, n, null);
        }
    }

    /// fused.mlp: gate and up (one launch up to max_rows rows, else one prompt GEMM each), SwiGLU, down into `out`.
    pub fn mlp(f: *const Forward, b: *const Buffers, L: *const W.Layer, R: usize, out: u64) !void {
        const width = L.mlp.gate.n;
        try f.lins(b, &.{ &L.mlp.gate, &L.mlp.up }, b.normed, R, &.{ b.g, b.u }, &.{ .bf16, .bf16 });
        try f.t().swiglu2(b.g, b.u, b.act, R, width);
        try f.lin(b, &L.mlp.down, b.act, R, out, .f32);
    }
};

/// This rank's logits [n, vocab_part] fp32 at `lg` of n normed bf16 rows `x` (row stride D): glue.router over the bf16
/// lm_head share, or (TF_GLM53_HEAD_Q, headq.zig) the EXL3 head's launch over the blocks that cover the share - rot_in
/// and linear_kernel, as every dense EXL3 linear - and the share copied out of its result. `scratch`: the router's
/// partials, or the launch's result fp32 [n, blocks * 128]; `xh`: fp16 [n, D] (a quantized head alone). Each row is
/// computed alone on both paths.
pub fn headLogits(w: *const W.Weights, t_: tri.Tri, o: kern.Ops, x: u64, lg: u64, scratch: u64, xh: u64, n: usize) !void {
    const D = w.config.hidden;
    const V = w.vocab_part;
    if (w.head_q) |*hq| {
        if (xh == 0) return error.NoHead;
        const sc: kern.LinearScratch = .{ .xh = .{ xh, 0, 0 }, .z = .{ 0, 0, 0 }, .xh_bytes = n * D * 2, .z_bytes = 0 };
        try o.linear(&hq.lin, x, .bf16, scratch, .f32, n, sc);
        return o.copyRows(lg, V * 4, scratch + hq.off * 4, hq.lin.n * 4, V * 4, n);
    }
    return t_.router(x, D, w.lm_head, lg, scratch, n, D, V);
}

/// fused.inv_freq on the host: a diagnostic only. torch's bytes come from libdevice's powf on the GPU, which this
/// does not reproduce exactly; Kernels.invFreq runs the same powf (glm53_rope.cu) and is what the engine uses.
pub fn invFreq(out: []f32, dim: usize, theta: f64) void {
    for (out, 0..) |*v, i| {
        const x: f32 = @as(f32, @floatFromInt(2 * i)) * (@as(f32, 1.0) / @as(f32, @floatFromInt(dim)));
        const p: f32 = @floatCast(std.math.pow(f64, @as(f64, @as(f32, @floatCast(theta))), @as(f64, x)));
        v.* = 1.0 / p;
    }
}

/// torch.argmax's choice over one row: the first maximum, NaN above every number (the first NaN wins).
pub fn argmaxFirst(v: []const f32) usize {
    var best: usize = 0;
    for (v, 0..) |x, i| {
        if (i == 0) continue;
        if (std.math.isNan(v[best])) break;
        if (std.math.isNan(x) or x > v[best]) best = i;
    }
    return best;
}

test "argmax: first maximum, NaN first" {
    try std.testing.expectEqual(@as(usize, 1), argmaxFirst(&.{ 1, 3, 3, 2 }));
    try std.testing.expectEqual(@as(usize, 2), argmaxFirst(&.{ 1, 3, std.math.nan(f32), std.math.nan(f32) }));
    try std.testing.expectEqual(@as(usize, 0), argmaxFirst(&.{ 0.0, -0.0 }));
    try std.testing.expectEqual(@as(usize, 0), argmaxFirst(&.{5}));
}

test "inv_freq's first entries" {
    var v: [32]f32 = undefined;
    invFreq(&v, 64, 8000000.0);
    try std.testing.expectEqual(@as(f32, 1.0), v[0]);
    try std.testing.expect(v[31] < v[30] and v[31] > 0);
}
