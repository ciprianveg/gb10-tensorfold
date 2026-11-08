//! GLM-5.3's forward on CUDA: one layer (MLA + DSA + dense/MoE), the prompt (prefill) chunk loop, the
//! decode step, the embedding read and the head. The blocks are 1.1's shape-agnostic entries where they
//! fit (see ops.zig) and the family's own where 1.1 is monomorphized to Flash's widths.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const wmod = @import("weights.zig");
const ops_mod = @import("ops.zig");
const indexer = @import("indexer.zig");
const glue = @import("glue/cuda_glue.zig");
const b16 = @import("glue/cuda_b16.zig");
const latent = @import("glue/cuda_latent.zig");
const mlp = @import("glue/cuda_mlp.zig");
const moe = @import("glue/cuda_moe.zig");
const mm_mod = @import("glue/cuda_mm.zig");

/// The kernels a forward runs. `ops` carries the glue set and our width-specific entries; mlp/moe are
/// the upstream blocks' own bundles (they load their fatbins through the family engine).
pub const Blocks = struct {
    ops: ops_mod.Ops,
    index: indexer.Kernels,
    mlp: mlp.Kernels,
    moe: moe.Kernels,
    ops_module: cuda.Module, // tf_glm_add_i32: the attention pass's position offset
    tp: ?*const cuda.tp_group.Group = null, // the world's group; null at world 1
    rank: usize = 0,
    world: usize = 1,
};

/// The slot this rank's fp32 partial goes to in the gather buffer (`Scratch.res`): the GLM engine's
/// layout, one world-sized slot a rank.
pub fn partialSlot(s: *const Scratch, rows: usize, d: usize, rank: usize) u64 {
    return s.res + rank * rows * d * 4;
}

/// TF_GLM53_SPLIT=2: a per-stage stopwatch inside `attn` (§39's phase-1 attribution: which kernel of the
/// attention half actually costs the 0.9 s a chunk). Nine stages, a sync after each - diagnostic only.
pub var stage_ms: [9]i64 = [_]i64{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };
pub var stage_on: bool = false;
var stage_ev: [2]?cuda.Event = .{ null, null };
var stage_which: usize = 0;
/// A CUDA-event stopwatch: the GPU time between two marks, which is the number that matters here.
fn stageMark(stream: cuda.Stream, i: usize) void {
    if (!stage_on) return;
    const cur = stage_which;
    const ev = stage_ev[cur] orelse blk: {
        const e = cuda.Event.init(stream.d, true) catch return;
        stage_ev[cur] = e;
        break :blk e;
    };
    ev.record(stream) catch return;
    if (stage_ev[1 - cur]) |prev| {
        stream.synchronize() catch return;
        stage_ms[i] += @intFromFloat(prev.elapsedMs(ev) catch 0.0);
    }
    stage_which = 1 - cur;
}
pub const stage_names = [9][]const u8{ "norm+proj", "q_norm+kv_norm", "q_b", "indexer", "absorbQ", "qpack+kvwrite", "attention", "expandV", "o_proj+gather" };

/// TF_GLM53_PROBE: read a stage's buffer back and print its sum/absmax/non-finite count, so a NaN is
/// attributed to the exact stage instead of to "the layer". Off unless the variable is set.
fn probe(stream: cuda.Stream, ptr: u64, bytes: usize, comptime width: usize, label: []const u8) !void {
    if (std.c.getenv("TF_GLM53_PROBE") == null) return;
    try stream.synchronize();
    const host = try std.heap.page_allocator.alloc(u8, bytes);
    defer std.heap.page_allocator.free(host);
    const b: cuda.DeviceBuffer = .{ .d = stream.d, .ptr = ptr, .len = bytes, .borrowed = true };
    try b.download(0, host);
    var sum: f64 = 0;
    var mx: f64 = 0;
    var bad: usize = 0;
    var i: usize = 0;
    while (i < bytes) : (i += width) {
        const f: f64 = switch (width) {
            2 => @floatCast(@as(f32, @bitCast(@as(u32, std.mem.readInt(u16, host[i..][0..2], .little)) << 16))),
            4 => @floatCast(std.mem.bytesToValue(f32, host[i..][0..4])),
            else => unreachable,
        };
        if (!std.math.isFinite(f)) bad += 1;
        sum += f;
        mx = @max(mx, @abs(f));
    }
    std.debug.print("    PROBE {s} sum={e} absmax={e} nonfinite={d}\n", .{ label, sum, mx, bad });
}

/// The tensor-parallel residual: every rank's partial (written at its slot) all-gathered, then summed
/// rank 0 first into `x` - the GLM engine's `gather` + `residualAdd(world)`.
pub fn gatherResidual(bl: Blocks, stream: cuda.Stream, x: u64, s: *const Scratch, rows: usize, d: usize) !void {
    if (bl.world > 1) {
        const g = bl.tp orelse return error.NoCommunicator;
        // TF_GLM53_FENCE: a stream sync between the partial's write and the collective. The diagnostic
        // that tells "the kernel -> NCCL ordering" apart from "the compute itself" (findings §37).
        if (std.c.getenv("TF_GLM53_FENCE") != null) try stream.synchronize();
        // The destination is a buffer of its own, never the source's: an in-place allGather (source inside
        // the receive buffer) left a ~4e-4 run-to-run split between ranks in the world-8 digest table
        // (findings §36.4/§37). NCCL documents in-place AllGather, but on this ring a rank's own slice is
        // already the region a peer's chunk would land in, and the ordered read is not preserved.
        try g.allGather(stream, partialSlot(s, rows, d, bl.rank), s.res2, rows * d * 4);
    } else {
        // world 1: the gather is the identity - the local partial into its own slot
        const dst = s.res2 + bl.rank * rows * d * 4;
        try stream.d.check(stream.d.api.cuMemcpyDtoDAsync_v2(dst, partialSlot(s, rows, d, bl.rank), rows * d * 4, stream.handle), "cuMemcpyDtoDAsync");
    }
    if (std.c.getenv("TF_GLM53_FENCE") != null) try stream.synchronize();
    try probe(stream, s.res2, rows * d * 4 * bl.world, 4, "residual.slots");
    try glue.residualAdd(bl.ops.glue, stream, x, x, s.res2, rows, d, bl.world);
    try probe(stream, x, rows * d * 2, 2, "residual.x");
}

/// A rank's caches: the latent cache (keys+values absorbed, [capacity][lw + rope] bf16) and the indexer's
/// key/gate rings ([capacity][i_dim] and [capacity][i_heads], bf16 as index_write writes them).
///
/// ONE PER LAYER, as the canonical state has it (`fused.State.kc` / `.ic` are a ring a layer): a layer's
/// historical rows are its own keys, so a shared cache makes a decode step read the *last* layer's latents
/// for every earlier position. Prefill hides it - each call rewrites every row it reads - and decode does
/// not: this was the "coherent for two tokens, then degenerate" fault (§36).
pub const Caches = struct {
    lc: u64,
    ik: u64, // the index-key plane: e4m3 codes + an fp32 scale a row (132 or 144 bytes), not bf16 keys
    ig: u64,
    lw: usize,
    capacity: usize,
    ik_stride: usize = indexer.plane_stride, // TF_GLM53_FP8_STRIDE=144 selects the padded plane
    layer_lc: usize = 0, // bytes a layer's latent cache takes (0: every layer shares one - single-layer runs)
    layer_ik: usize = 0,
    layer_ig: usize = 0,

    /// This layer's slice of the ring.
    pub fn ofLayer(c: Caches, layer: u32) Caches {
        return .{
            .lc = c.lc + layer * c.layer_lc,
            .ik = c.ik + layer * c.layer_ik,
            .ig = c.ig + layer * c.layer_ig,
            .lw = c.lw,
            .capacity = c.capacity,
            .ik_stride = c.ik_stride,
            .layer_lc = 0,
            .layer_ik = 0,
            .layer_ig = 0,
        };
    }

    /// The bytes ONE layer's caches take; the caller multiplies by the layers it runs.
    /// Per-layer bytes of the three rings; `layers` of them are what a run allocates.
    pub fn ringBytes(c: *const cfg.Config, capacity: usize, ik_stride: usize, layers: usize) struct { lc: usize, ik: usize, ig: usize } {
        const one = bytes(c, capacity, ik_stride);
        return .{ .lc = one.lc * layers, .ik = one.ik * layers, .ig = one.ig * layers };
    }

    pub fn bytes(c: *const cfg.Config, capacity: usize, ik_stride: usize) struct { lc: usize, ik: usize, ig: usize } {
        return .{
            // the latent row is [absorbed latent (kv_lora) | rope key (rope)] - r2's latentWidth()
            .lc = capacity * (c.kv_lora + c.rope) * 2,
            .ik = capacity * ik_stride,
            .ig = capacity * c.i_heads * 2,
        };
    }
};

/// Rows-sized scratch. The MLP and MoE blocks keep their own scratch (they own the grouped plan).
pub const Scratch = struct {
    normed: u64, // bf16 [rows][dims]
    res: u64, // fp32 [world][rows][dims]: this rank's partial at its slot (the gather's source)
    res2: u64, // fp32 [world][rows][dims]: the gather's destination, NOT aliased with the source
    dp: u64, // bf16 [rows][q_lora + lw + rope]
    qr: u64, // bf16 [rows][q_lora]
    lat: u64, // bf16 [rows][lw]
    q: u64, // bf16 [rows][heads qk]
    ikr: u64, // fp32 [rows][i_dim]: the plane writer's input (LayerNorm+RoPE in kernel)
    igr: u64, // fp32 [rows][i_heads]
    qa: u64, // bf16 [rows][heads lw]: the absorbed latent half (absorbQ's output)
    qp: u64, // bf16 [rows][heads (lw + rope)]: [qlat | rot(q's rope half)] - the attention's query row
    ol: u64, // bf16 [rows][heads lw]
    vn: u64, // bf16 [rows][heads dv]
    qi: u64, // bf16 [rows][i_heads i_dim]
    partials: latent.Partials,
    dummy: u64, // an i32: the merge's count
    tokens: u64, // i32 [rows][topk] (the selection's output; not ported yet)
    counts: u64, // i32 [rows]
    scores: u64, // [rows][T] keys: int32 order words (fp8 default) or int64 packed (bf16 fallback)
    select: indexer.Scratch = .{ .hist = 0, .state = 0, .cnt = 0, .rows = 0, .blocks = 0 },
    mm_part: u64, // fp32 split-K partials
    at: u64, // an i32: a row block's first position
    xs_qr: u64, // fp32 [rows][q_lora / 64]
    xs_lat: u64, // fp32 [rows][lw / 64]
};

/// The forward's BF16 matmuls go through exactly one entry, `b16.matmul` (glue/cuda_b16.zig), whose
/// split-K partials size this rank's `mm_part`: `b16.partBytes(rows, n, k)` per call, the max over the
/// calls below. Pinned here because the scratch table must not guess it: a new matmul in `attn`/`ffn`/
/// `mtpStep`/the head must be added to this list (and `mmPartCovers` refuses a bound below a call).
pub fn mmPartBytes(c: *const cfg.Config, share: shard.Share, rows: usize) usize {
    const d = c.hidden;
    const heads = share.mla.len();
    const sites = [_][2]usize{
        .{ c.q_lora + c.kv_lora + c.rope, d }, // [q_a | kv_a] (kv_a carries the rope key)
        .{ heads * c.qHead(), c.q_lora }, // q_b
        .{ c.i_dim, d }, // indexer wk
        .{ c.i_heads, d }, // indexer wproj
        .{ c.i_heads * c.i_dim, c.q_lora }, // indexer wq_b
        .{ d, heads * c.v_dim }, // o_proj
        .{ d, 2 * d }, // mtp eh_proj
        .{ c.vocab, d }, // head
    };
    var total: usize = 0;
    for (sites) |site| total = @max(total, partBytesEither(rows, site[0], site[1]));
    // the dense layers' and the shared expert's linears, when they take the bf16 path
    total = @max(total, partBytesEither(rows, 2 * c.dense_inter, d));
    total = @max(total, partBytesEither(rows, d, c.dense_inter));
    total = @max(total, partBytesEither(rows, 2 * c.shared_inter, d));
    total = @max(total, partBytesEither(rows, d, c.shared_inter));
    return total;
}

/// The split-K partials a site needs, whichever entry runs it: the raw rows (b16) or the tiled entry
/// (§38 - `mm.partBytes` caps the rows at 64 and asks for `n` columns a slice).
fn partBytesEither(rows: usize, n: usize, k: usize) usize {
    return @max(b16.partBytes(rows, n, k), mm_mod.partBytes(rows, .{ .w = 0, .n = n, .k = k }));
}

/// True when a scratch `bytes` bound covers every matmul site at `rows` (the too-small refusal's input).
pub fn mmPartCovers(c: *const cfg.Config, share: shard.Share, rows: usize, bytes: usize) bool {
    return bytes >= mmPartBytes(c, share, rows);
}

/// The bytes a forward's scratch takes for `rows` rows: one row set (the blocks reuse it per chunk).
pub fn scratchBytes(c: *const cfg.Config, share: shard.Share, rows: usize) struct {
    common: usize,
    partials: usize,
    tokens: usize,
} {
    const d = c.hidden;
    const heads = share.mla.len();
    const common = rows * 2 * (d + d + (c.q_lora + c.kv_lora + c.rope) + c.q_lora + c.kv_lora + heads * c.qHead() +
        c.i_dim + heads * c.kv_lora + heads * c.kv_lora + heads * c.v_dim + c.i_heads * c.i_dim) +
        rows * 4 * (c.i_heads + c.q_lora / 64 + c.kv_lora / 64);
    // latent.Partials: (heads x rows x 128) fp32 partials for the output merge, and the p/o/m buffers
    const partials = 3 * rows * heads * latent.headBlock(rows) * 4 + rows * 4;
    const tokens = rows * c.i_topk * 4 + rows * 4;
    return .{ .common = common, .partials = partials, .tokens = tokens };
}

/// The forward scratch's sizes, one row set a chunk, derived from the same formulas the GLM family's
/// state uses (cuda_state.zig's arena) for the fields this family's forward reads. `scores_t` is the
/// keys buffer's width (the caller's window); `blocks` is the radix's block count for `scores_t`.
pub const Layout = struct {
    normed: usize,
    res: usize,
    res2: usize,
    dp: usize,
    qr: usize,
    lat: usize,
    q: usize,
    ikr: usize,
    igr: usize,
    qa: usize,
    qp: usize,
    ol: usize,
    vn: usize,
    qi: usize,
    po: usize,
    pm: usize,
    pl: usize,
    dummy: usize,
    tokens: usize,
    counts: usize,
    scores: usize,
    sel_hist: usize,
    sel_state: usize,
    sel_cnt: usize,
    sel_list: usize,
    sel_lcnt: usize,
    sel_lflg: usize,
    mm_part: usize,
    at: usize,
    xs_qr: usize,
    xs_lat: usize,
    // the two blocks' own scratch, from their shared constructors (cuda_mlp/cuda_moe/grouped.planSizes)
    mlp_tmp: usize,
    mlp_gu: usize,
    mlp_act: usize,
    mlp_xs_act: usize,
    moe_mlog: usize,
    moe_router_part: usize,
    moe_pick: usize,
    moe_wts: usize,
    moe_members: usize,
    moe_items: usize,
    moe_counts: usize,
    moe_rank: usize,
    moe_hist: usize,
    moe_eact: usize,
    moe_ey: usize,
    moe_sgu: usize,
    moe_sact: usize,
    moe_sxs: usize,
    moe_sy: usize,
    total: usize,
};

/// `nch_max`: the widest attention a call takes, in 512-key chunks. The dense path reads
/// `chunksFor(pos + rows)` and the sparse path `chunksFor(i_topk)`, so a LONG prompt needs the caller's
/// window (a chunk at 3 584 rows reads 8 chunks) - sizing it at `chunksFor(i_topk)` alone overflows the
/// chunk partials and faults (the chunk=512 crash, §39).
pub fn scratchLayout(c: *const cfg.Config, share: shard.Share, rows: usize, scores_t: usize, blocks: usize) Layout {
    return scratchLayoutNch(c, share, rows, scores_t, blocks, 0);
}

pub fn scratchLayoutNch(c: *const cfg.Config, share: shard.Share, rows: usize, scores_t: usize, blocks: usize, nch_caller: usize) Layout {
    const d = c.hidden;
    const heads = share.mla.len();
    const lw = c.kv_lora;
    const prows = rows; // one window: the merge partials cover the rows the call renders
    // The sparse path takes chunksFor(topk) 512-key chunks a call, and the chunk partials are one a chunk:
    // po [nch][rows][heads][lw], pm/pl [nch][rows][heads]. Sized for the widest call (the dense path uses
    // one chunk, the sparse path up to chunksFor(i_topk)).
    const nch_max = @max(@max(1, latent.chunksFor(c.i_topk)), nch_caller);
    // buffers stay head-sized (the GLM state's own rule): the head-tile probe that enlarged them to the
    // 16-head kernel tile did not move the fault, so it is not the layout.
    const mls = mlp.scratchSizes(rows, share.dense.len());
    // pairs = rows * (the ROUTED top-k + 1) - the MoE block's slot count, not the indexer's top-k (the
    // canonical state's `pairs = rows * slots`, slots = c.topk + 1; the indexer's i_topk only sizes `tokens`)
    const ms = moe.scratchSizes(rows, rows * (c.topk + 1), c.experts, @max(c.experts, c.i_dim), 16, share.moe.len(), c.hidden, share.shared.len());
    var l: Layout = undefined;
    l = .{
        .normed = rows * d * 2,
        .res = rows * d * 4 * share.world, // this rank's fp32 partial (the gather's source)
        .res2 = rows * d * 4 * share.world, // the gather's destination: a separate buffer, see gatherResidual
        .dp = rows * (c.q_lora + lw + c.rope) * 2,
        .qr = rows * c.q_lora * 2,
        .lat = rows * lw * 2,
        .q = rows * heads * c.qHead() * 2,
        .ikr = rows * c.i_dim * 4,
        .igr = rows * c.i_heads * 4,
        .qa = rows * heads * lw * 2,
        .qp = rows * heads * (lw + c.rope) * 2,
        .ol = rows * heads * lw * 2,
        .vn = rows * heads * c.v_dim * 2,
        .qi = rows * c.i_heads * c.i_dim * 2,
        .po = nch_max * prows * heads * lw * 4,
        .pm = nch_max * prows * heads * 4,
        .pl = nch_max * prows * heads * 4,
        .dummy = 4,
        .tokens = rows * c.i_topk * 4,
        .counts = rows * 4,
        .scores = rows * scores_t * 4,
        .sel_hist = rows * indexer.sel_bins * 4,
        .sel_state = rows * 4 * 4,
        .sel_cnt = rows * blocks * 2 * 4,
        .sel_list = rows * indexer.topk_compact_cap * 4,
        .sel_lcnt = rows * 4,
        .sel_lflg = rows * 4,
        .mm_part = mmPartBytes(c, share, rows),
        .at = 4,
        .xs_qr = rows * (c.q_lora / 64) * 4,
        .xs_lat = rows * (lw / 64) * 4,
        .mlp_tmp = mls.tmp,
        .mlp_gu = mls.gu,
        .mlp_act = mls.act,
        .mlp_xs_act = mls.xs_act,
        .moe_mlog = ms.mlog,
        .moe_router_part = ms.router_part,
        .moe_pick = ms.pick,
        .moe_wts = ms.wts,
        .moe_members = ms.plan.members,
        .moe_items = ms.plan.items,
        .moe_counts = ms.plan.counts,
        .moe_rank = ms.plan.rank,
        .moe_hist = ms.plan.hist,
        .moe_eact = ms.eact,
        .moe_ey = ms.ey,
        .moe_sgu = ms.sgu,
        .moe_sact = ms.sact,
        .moe_sxs = ms.sxs,
        .moe_sy = ms.sy,
        .total = 0,
    };
    l.total = arenaTotal(&l);
    return l;
}

/// The arena's total: every field 256-byte aligned, as the GLM state's arena does.
pub fn arenaTotal(l: anytype) usize {
    var total: usize = 0;
    inline for (@typeInfo(@TypeOf(l.*)).@"struct".field_names) |name| {
        if (comptime std.mem.eql(u8, name, "total")) continue;
        total += std.mem.alignForward(usize, @field(l.*, name), 256);
    }
    return @max(total, 256);
}

/// A checked bump allocator over one device buffer: every `take` rounds to 256 bytes and refuses past the
/// end, so a short scratch is an error, never an overlap.
pub const Arena = struct {
    base: u64,
    size: usize,
    used: usize = 0,

    pub fn take(a: *Arena, n: usize) !u64 {
        const at = std.mem.alignForward(usize, a.used, 256);
        const end = at + std.mem.alignForward(usize, n, 256);
        if (end > a.size) return error.ArenaTooSmall;
        a.used = end;
        return a.base + at;
    }
};

/// A Scratch laid out in `arena`'s buffer: the field order is the layout's order, the bounds checked.
pub fn place(a: *Arena, l: anytype) !Scratch {
    var s: Scratch = .{ .normed = 0, .res = 0, .res2 = 0, .dp = 0, .qr = 0, .lat = 0, .q = 0, .ikr = 0, .igr = 0, .qa = 0, .qp = 0, .ol = 0, .vn = 0, .qi = 0, .partials = undefined, .dummy = 0, .tokens = 0, .counts = 0, .scores = 0, .mm_part = 0, .at = 0, .xs_qr = 0, .xs_lat = 0 };
    s.normed = try a.take(l.normed);
    s.res = try a.take(l.res);
    s.res2 = try a.take(l.res2);
    s.dp = try a.take(l.dp);
    s.qr = try a.take(l.qr);
    s.lat = try a.take(l.lat);
    s.q = try a.take(l.q);
    s.ikr = try a.take(l.ikr);
    s.igr = try a.take(l.igr);
    s.qa = try a.take(l.qa);
    s.qp = try a.take(l.qp);
    s.ol = try a.take(l.ol);
    s.vn = try a.take(l.vn);
    s.qi = try a.take(l.qi);
    s.partials = .{ .po = try a.take(l.po), .pm = try a.take(l.pm), .pl = try a.take(l.pl) };
    s.dummy = try a.take(l.dummy);
    s.tokens = try a.take(l.tokens);
    s.counts = try a.take(l.counts);
    s.scores = try a.take(l.scores);
    s.select = .{
        .hist = try a.take(l.sel_hist),
        .state = try a.take(l.sel_state),
        .cnt = try a.take(l.sel_cnt),
        .list = try a.take(l.sel_list),
        .lcnt = try a.take(l.sel_lcnt),
        .lflg = try a.take(l.sel_lflg),
        .rows = l.sel_state / 16,
        .blocks = if (l.sel_cnt == 0) 0 else l.sel_cnt / (l.sel_state / 16) / 8,
        .cap = if (l.sel_state == 0) 0 else indexer.topk_compact_cap,
    };
    s.mm_part = try a.take(l.mm_part);
    s.at = try a.take(l.at);
    s.xs_qr = try a.take(l.xs_qr);
    s.xs_lat = try a.take(l.xs_lat);
    return s;
}

/// The two blocks' scratch placed after the forward's own: one arena, the same checked allocator.
pub fn placeBlocks(a: *Arena, l: Layout) !struct { scratch: Scratch, mlp: mlp.Scratch, moe: moe.Scratch } {
    const sc = try place(a, l);
    return .{
        .scratch = sc,
        .mlp = .{
            .tmp = try a.take(l.mlp_tmp),
            .gu = try a.take(l.mlp_gu),
            .act = try a.take(l.mlp_act),
            .xs_act = try a.take(l.mlp_xs_act),
            .mm_part = sc.mm_part, // the forward's split-K partials (one buffer, shared)
        },
        .moe = .{
            .mlog = try a.take(l.moe_mlog),
            .router_part = try a.take(l.moe_router_part),
            .pick = try a.take(l.moe_pick),
            .wts = try a.take(l.moe_wts),
            .plan = .{
                .members = try a.take(l.moe_members),
                .items = try a.take(l.moe_items),
                .counts = try a.take(l.moe_counts),
                .rank = try a.take(l.moe_rank),
                .hist = try a.take(l.moe_hist),
            },
            .eact = try a.take(l.moe_eact),
            .ey = try a.take(l.moe_ey),
            .sgu = try a.take(l.moe_sgu),
            .sact = try a.take(l.moe_sact),
            .sxs = try a.take(l.moe_sxs),
            .sy = try a.take(l.moe_sy),
            .mm_part = sc.mm_part,
        },
    };
}

/// Embedding rows: `ids` i32 [rows] -> bf16 [rows][dims] from the rank's embed table.
pub fn embed(bl: Blocks, stream: cuda.Stream, w: *const wmod.Weights, ids: u64, rows: usize, out: u64) !void {
    try glue.embedB16(bl.ops.glue, stream, ids, w.embed, out, rows, w.config.hidden, 1);
}

/// The index key and gate cache write: upstream's tf_glm_index_write, exactly our 128-dim LayerNorm.
fn indexCacheWrite(bl: Blocks, stream: cuda.Stream, kr: u64, k_stride: usize, gr: u64, ln_w: u64, ln_b: u64, ik: u64, ig: u64, pos: u64, rows: usize) !void {
    var a: cuda.Args = .{};
    a.add(kr);
    a.add(@as(i64, @intCast(k_stride)));
    a.add(gr);
    a.add(ln_w);
    a.add(ln_b);
    a.add(ik);
    a.add(ig);
    a.add(pos);
    a.add(@as(f32, 1e-6));
    try cuda.launch.launch(bl.ops.glue.glue.index_write, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, stream, &a);
}

/// The attention half of one layer over `rows` rows: out fp32 [rows][dims] is this rank's partial.
/// The caller adds it as a residual before the post norm and the FFN half.
pub fn attn(bl: Blocks, stream: cuda.Stream, w: *const wmod.Weights, l: *const wmod.Layer, caches: Caches, s: *const Scratch, x: u64, x_stride: usize, rows: usize, pos: u64, pos_host: usize, nch: usize, out: u64) !void {
    const c = &w.config;
    const a = l.attn;
    const d = c.hidden;
    const heads = a.heads;

    const trace = std.c.getenv("TF_GLM53_TRACE") != null;
    // this layer's own slice of the per-layer rings (§36: a shared cache hands a decode step the last
    // layer's historical keys)
    const cc = caches.ofLayer(l.index);
    // 1) input norm, then [q_a | kv_a]; q_a and kv_a norms
    try ops_mod.norm(bl.ops, stream, d, x, x_stride, l.in_norm, s.normed, d, c.eps, rows);
    try probe(stream, s.normed, rows * d * 2, 2, "attn.normed");
    if (trace) std.debug.print("  attn: norm ok\n", .{});
    stageMark(stream, 0);
    // the projection is [q_a | kv_a] and kv_a carries the rope key: kv_lora + rope wide (r2's kv_row)
    const row = c.q_lora + c.kv_lora + c.rope;
    try b16.matmul(bl.ops.glue, stream, s.normed, d, a.proj, s.dp, s.mm_part, rows, row, d, false);
    try probe(stream, s.dp, rows * (c.q_lora + c.kv_lora) * 2, 2, "attn.proj");
    if (trace) std.debug.print("  attn: proj ok\n", .{});
    try ops_mod.norm(bl.ops, stream, c.q_lora, s.dp, row, a.q_norm, s.qr, c.q_lora, c.eps, rows);
    if (trace) std.debug.print("  attn: q_norm ok\n", .{});
    try glue.rmsnorm(bl.ops.glue, stream, s.dp + @as(u64, c.q_lora) * 2, row, a.kv_norm, s.lat, c.kv_lora, null, c.eps, rows, c.kv_lora);
    try probe(stream, s.lat, rows * c.kv_lora * 2, 2, "attn.lat");
    if (trace) std.debug.print("  attn: kv_norm ok\n", .{});
    stageMark(stream, 1);

    // 2) q_b by heads; the indexer's keys (wk x), gates (weights_proj x) and queries (wq_b on q_a)
    try b16.matmul(bl.ops.glue, stream, s.qr, c.q_lora, a.q_b, s.q, s.mm_part, rows, heads * a.qk, c.q_lora, false);
    try probe(stream, s.q, rows * heads * a.qk * 2, 2, "attn.q_b");
    if (trace) std.debug.print("  attn: q_b ok\n", .{});
    stageMark(stream, 2);
    // The indexer exists only on the pack's "full" layers (config.full): a shared layer reuses the last
    // full layer's top-k and carries NO indexer tensors, so a.index.* are zero there - calling the
    // matmuls unconditionally read a null weight (the first MoE layer's illegal address).
    if (c.full.isSet(l.index)) {
        try b16.matmul(bl.ops.glue, stream, s.normed, d, a.index.kw, s.ikr, s.mm_part, rows, c.i_dim, d, true); // fp32: the plane writer
        try b16.matmul(bl.ops.glue, stream, s.normed, d, a.index.wproj, s.igr, s.mm_part, rows, c.i_heads, d, true);
        try b16.matmul(bl.ops.glue, stream, s.qr, c.q_lora, a.index.qb, s.qi, s.mm_part, rows, c.i_heads * c.i_dim, c.q_lora, false);
        try indexer.writePlane(bl.index, stream, s.ikr, a.index.ln_w, a.index.ln_b, w.inv, cc.ik, pos, rows, c.i_dim, c.rope, 1, 0, 1e-6);
        if (trace) std.debug.print("  attn: plane ok\n", .{});
    }
    stageMark(stream, 3);
    try latent.absorbQ(bl.ops.glue, stream, s.q, a.wk, s.qa, rows, heads, a.qk, a.lw);
    try probe(stream, s.qa, rows * heads * a.lw * 2, 2, "attn.absorbQ");
    stageMark(stream, 4);
    if (trace) std.debug.print("  attn: absorbQ ok\n", .{});

    // 2.5) THE ROPE HALF (findings §36): the query row becomes [qlat | rot(q's rope dims at pos)] and the
    // cache row [normed latent | rot(kv_a's rope dims at pos)], so the attention's dot over the row is the
    // whole MLA score (r2: `qpack [rows, H, lw + rd]`, `_kv_write` rotating "the rope half of its row at
    // the row's true position"). Without it the attention has no positional signal beyond the causal mask.
    const rd = c.rope;
    try ops_mod.qpack(bl.ops.extras, stream, s.qa, s.q, w.inv, pos, s.qp, rows, heads, a.qk - rd, a.lw, rd);
    try probe(stream, s.qp, rows * heads * (a.lw + rd) * 2, 2, "attn.qpack");
    try ops_mod.kvWrite(bl.ops.extras, stream, s.lat, a.lw, s.dp + @as(u64, c.q_lora + c.kv_lora) * 2, row, cc.lc, pos, rows, a.lw, rd, w.inv);
    try probe(stream, cc.lc, cc.capacity * (a.lw + rd) * 2, 2, "attn.cache");
    stageMark(stream, 5);
    if (trace) std.debug.print("  attn: rope + cache ok\n", .{});
    const scale: f32 = @floatCast(std.math.pow(f64, @floatFromInt(a.qk), -0.5));
    if (indexer.denseMode(c, pos_host, rows)) {
        try latent.attentionWide(bl.ops.extras.chunks, bl.ops.glue, stream, s.qp, cc.lc, pos, s.partials, s.ol, s.dummy, rows, heads, a.lw, rd, nch, scale);
        try probe(stream, s.ol, rows * heads * a.lw * 2, 2, "attn.ol");
    } else {
        const T = pos_host + rows;
        // the fp8 plane's scorer (int32 order words) and the radix in mode 0; the bf16 pair is the fallback
        try indexer.scoreFp8(bl.index, stream, s.qi, s.igr, cc.ik, s.scores, T, 0, pos, rows, c.i_heads, c.i_dim);
        try indexer.select(bl.index, stream, s.scores, s.tokens, T, rows, c.i_topk, 0, s.select);
        try latent.sparseAttentionWide(bl.ops.extras.chunks, bl.ops.glue, stream, s.qp, cc.lc, s.tokens, s.counts, c.i_topk, s.partials, s.ol, rows, heads, a.lw, rd, scale);
    }
    stageMark(stream, 6);
    try latent.expandV(bl.ops.glue, stream, s.ol, a.wv, s.vn, rows, heads, a.dv, a.lw);
    try probe(stream, s.vn, rows * heads * a.dv * 2, 2, "attn.vn");
    stageMark(stream, 7);

    // 4) o_proj into this rank's fp32 partial
    try b16.matmul(bl.ops.glue, stream, s.vn, heads * a.dv, a.o, out, s.mm_part, rows, d, heads * a.dv, true);
    try probe(stream, out, rows * d * 4, 4, "attn.out");
    stageMark(stream, 8);
}

/// The FFN half of one layer from the residual's post norm: out fp32 [rows][dims], this rank's partial.
pub fn ffn(bl: Blocks, stream: cuda.Stream, w: *const wmod.Weights, l: *const wmod.Layer, s: *const Scratch, ms: *const mlp.Scratch, es: *const moe.Scratch, x: u64, rows: usize, prompt: bool, out: u64) !void {
    const c = &w.config;
    const d = c.hidden;
    try ops_mod.norm(bl.ops, stream, d, x, d, l.post_norm, s.normed, d, c.eps, rows);
    try probe(stream, s.normed, rows * d * 2, 2, "ffn.normed");
    switch (l.ffn) {
        .dense => |m| {
            try mlp.block(bl.mlp, stream, .{
                .w = switch (m.w) {
                    .bf16 => |b| .{ .bf16 = .{ .gu = b.gu, .down = b.down } },
                    .nvfp4 => |q| .{ .nvfp4 = .{ .gate = q.gate, .up = q.up, .down = q.down } },
                },
                .width = m.width,
                .dims = m.dims,
                .limit = m.limit,
            }, ms.*, s.normed, d, rows, prompt, out);
            try probe(stream, ms.gu, rows * 2 * m.width * 2, 2, "ffn.gu");
            try probe(stream, ms.act, rows * m.width * 2, 2, "ffn.act");
            try probe(stream, out, rows * d * 4, 4, "ffn.out");
        },
        .moe => |m| try moe.block(bl.moe, stream, .{
            .router = m.router,
            .bias = m.bias,
            .routed = m.routed,
            .exl3 = null,
            .count = m.count,
            .shared_gu = m.shared_gu,
            .shared_down = m.shared_down,
            .shared = m.shared,
            .top_k = m.top_k,
            .scale = m.scale,
            .norm = m.norm,
        }, es.*, s.normed, d, rows, prompt, out),
    }
}

/// The stack: embed, every layer with its two residual adds, the final norm. `out` bf16 [rows][dims].
pub fn stack(bl: Blocks, stream: cuda.Stream, w: *const wmod.Weights, caches: Caches, s: *const Scratch, ms: *const mlp.Scratch, es: *const moe.Scratch, ids: u64, x: u64, rows: usize, pos: u64, pos_host: usize, nch: usize, prompt: bool) !void {
    const d = w.config.hidden;
    try embed(bl, stream, w, ids, rows, x);
    for (w.layers[0..w.config.run]) |*l| {
        try attn(bl, stream, w, l, caches, s, x, d, rows, pos, pos_host, nch, partialSlot(s, rows, d, bl.rank));
        try gatherResidual(bl, stream, x, s, rows, d);
        try ffn(bl, stream, w, l, s, ms, es, x, rows, prompt, partialSlot(s, rows, d, bl.rank));
        try gatherResidual(bl, stream, x, s, rows, d);
    }
    try ops_mod.norm(bl.ops, stream, d, x, d, w.norm, x, d, w.config.eps, rows);
}

/// The output projection onto this rank's vocabulary share, fp32 wide for the sampler.
pub fn head(bl: Blocks, stream: cuda.Stream, w: *const wmod.Weights, s: *const Scratch, x: u64, rows: usize, out: u64) !void {
    try b16.matmul(bl.ops.glue, stream, x, w.config.hidden, w.head, out, s.mm_part, rows, w.share.vocab.len(), w.config.hidden, true);
}

/// A prompt's chunk plan: rows per call (the attention block's block size), each at its own position.
pub fn promptChunks(total: usize, per: usize) usize {
    return (total + per - 1) / per;
}

test "the prompt chunk plan covers every row; the cache sizes are per token and dtype" {
    try std.testing.expectEqual(@as(usize, 1), promptChunks(100, 128));
    try std.testing.expectEqual(@as(usize, 4), promptChunks(512, 128));
    try std.testing.expectEqual(@as(usize, 5), promptChunks(513, 128));
    var c: cfg.Config = .{};
    c.kv_lora = 512;
    c.i_dim = 128;
    c.i_heads = 32;
    const b = Caches.bytes(&c, 4096, indexer.plane_stride);
    // a latent row is kv_lora + rope wide (§36), and the rings are one a layer
    try std.testing.expectEqual(@as(usize, 4096 * (512 + 64) * 2), b.lc);
    try std.testing.expectEqual(@as(usize, 4096 * 132), b.ik);
    try std.testing.expectEqual(@as(usize, 4096 * 32 * 2), b.ig);
}
/// One decode step: `rows` streams (one row each, their positions in `pos`), the stack at rows=1..
pub fn decodeStep(bl: Blocks, stream: cuda.Stream, w: *const wmod.Weights, caches: Caches, s: *const Scratch, ms: *const mlp.Scratch, es: *const moe.Scratch, ids: u64, x: u64, rows: usize, pos: u64, pos_host: usize, nch: usize, logits: u64) !void {
    try stack(bl, stream, w, caches, s, ms, es, ids, x, rows, pos, pos_host, nch, false);
    try head(bl, stream, w, s, x, rows, logits);
}

/// The MTP head's projection scratch: [enorm(x) | hnorm(embedding)] [rows][2 dims] bf16.
pub fn mtpProjectBytes(c: *const cfg.Config, rows: usize) usize {
    return rows * 2 * c.hidden * 2;
}

/// The MTP head: eh_proj of [enorm(x) | hnorm(e)] into the MTP layer (its own caches and position),
/// then shared_head.norm and this rank's head rows. `proj` is `mtpProjectBytes` scratch.
pub fn mtpStep(bl: Blocks, stream: cuda.Stream, w: *const wmod.Weights, s: *const Scratch, ms: *const mlp.Scratch, es: *const moe.Scratch, caches: Caches, x: u64, ids: u64, rows: usize, pos: u64, pos_host: usize, nch: usize, proj: u64, logits: u64) !void {
    const mtp = w.mtp orelse return error.NoMtp;
    const c = &w.config;
    const d = c.hidden;
    // the embedding row beside the hidden state, both normed into the projection buffer's halves
    try glue.embedB16(bl.ops.glue, stream, ids, w.embed, s.res, rows, d, 1);
    try ops_mod.norm(bl.ops, stream, d, x, d, mtp.enorm, proj, 2 * d, c.eps, rows);
    try ops_mod.norm(bl.ops, stream, d, s.res, d, mtp.hnorm, proj + @as(u64, d) * 2, 2 * d, c.eps, rows);
    try b16.matmul(bl.ops.glue, stream, proj, 2 * d, mtp.eh, s.normed, s.mm_part, rows, d, 2 * d, false);

    // the MTP layer itself (BF16 experts quantized at load), then shared_head.norm and the head
    try attn(bl, stream, w, &mtp.layer, caches, s, s.normed, d, rows, pos, pos_host, nch, s.res);
    try glue.residualAdd(bl.ops.glue, stream, s.normed, s.normed, s.res, rows, d, w.share.world);
    try ffn(bl, stream, w, &mtp.layer, s, ms, es, s.normed, rows, false, s.res);
    try glue.residualAdd(bl.ops.glue, stream, s.normed, s.normed, s.res, rows, d, w.share.world);
    try ops_mod.norm(bl.ops, stream, d, s.normed, d, mtp.norm, s.normed, d, c.eps, rows);
    try head(bl, stream, w, s, s.normed, rows, logits);
}

test "the decode and MTP entry points are the stack's: one row per stream, the MTP projection is 2 dims wide" {
    const c: cfg.Config = .{};
    try std.testing.expectEqual(@as(usize, 512 * 2 * 6144 * 2), mtpProjectBytes(&c, 512));
    _ = &decodeStep;
    _ = &mtpStep;
}

test "the pinned matmul entry's split-K partials: the bound covers every site, a short bound is refused" {
    const c: cfg.Config = .{};
    const sh = try shard.Share.of(&c, 0, 8);
    const need = mmPartBytes(&c, sh, 1);
    try std.testing.expect(need > 0); // a decode row takes split-K partials
    try std.testing.expect(mmPartCovers(&c, sh, 1, need));
    try std.testing.expect(!mmPartCovers(&c, sh, 1, need - 1));
    // a 128-row prompt chunk sums split-K in registers on the RAW entry (b16), but the TILED entry (§38)
    // takes slices a 64-row bucket at a time, so the bound that covers *every* site is non-zero there too
    try std.testing.expect(mmPartBytes(&c, sh, 128) > 0);
}

test "the scratch arena: the layout is its total, placement fits, a short arena refuses" {
    const c: cfg.Config = .{};
    const sh = try shard.Share.of(&c, 0, 8);
    const l = scratchLayout(&c, sh, 128, 8192, 4);
    try std.testing.expect(l.total >= 256);
    // the bound covers both entries (§38): non-zero at 128 rows (the tiled entry slices there, 64 rows a
    // bucket), and SMALLER at a decode row (one row a slice against a 64-row bucket)
    try std.testing.expect(l.mm_part > 0);
    try std.testing.expect(scratchLayout(&c, sh, 1, 8192, 4).mm_part < l.mm_part);
    var a: Arena = .{ .base = 0x1000, .size = l.total };
    const blocks_ = try placeBlocks(&a, l);
    const s = blocks_.scratch;
    try std.testing.expect(a.used <= l.total);
    try std.testing.expect(blocks_.moe.plan.items != 0 and blocks_.mlp.gu != 0);
    try std.testing.expect(s.scores != 0 and s.select.hist != 0 and s.partials.po != 0);
    // every address is 256-aligned, and no two fields overlap: the arena is a bump allocator
    try std.testing.expect(s.normed % 256 == 0 and s.scores % 256 == 0);
    try std.testing.expect(s.res > s.normed and s.dp > s.res);
    // a short arena refuses with the named error rather than overlapping
    var short: Arena = .{ .base = 0x1000, .size = l.total / 2 };
    try std.testing.expectError(error.ArenaTooSmall, placeBlocks(&short, l));
}
