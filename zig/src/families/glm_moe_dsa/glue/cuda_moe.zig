//! GLM-5.3-Flash's MoE on CUDA, a rank's share of the Python family's moe_block: top-8 plus shared, grouped NVFP4.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const experts = @import("cuda_experts.zig");
const b16 = @import("cuda_b16.zig");
const exl3 = @import("cuda_exl3.zig");
const prof_mod = @import("cuda_prof.zig");

/// One MoE layer on a rank: router (bf16 [count][dims]) and fp32 bias, routed experts, shared [gate | up] and down.
pub const Layer = struct {
    router: u64,
    bias: u64,
    routed: cuda.experts.Layer, // NVFP4 experts, or only the shape (width, dims, limit) when `exl3` holds them
    exl3: ?exl3.Experts = null,
    count: usize,
    shared_gu: u64,
    shared_down: u64,
    shared: usize,
    top_k: usize,
    scale: f32,
    norm: bool,
};

/// The block's scratch for up to `rows` rows (forward.Buffers' MoE part).
pub const Scratch = struct {
    mlog: u64, // fp32 [rows][count]
    router_part: u64, // fp32 [router_ks][rows][count]
    pick: u64, // i32 [rows][top_k + 1]
    wts: u64, // fp32 [rows][top_k + 1]
    plan: cuda.grouped.Plan,
    eact: u64, // bf16 [rows (top_k + 1)][width]
    ey: u64, // [rows][top_k + 1][dims]: fp32, bf16 in a prompt chunk
    sgu: u64, // bf16 [rows][2 shared]
    sact: u64, // bf16 [rows][shared]
    sxs: u64, // fp32 [rows][shared / 64]
    sy: u64, // fp32 [rows][dims]
    mm_part: u64, // fp32 split-K partials of the shared expert's matmuls
    x3: exl3.Scratch = .{ .picks = 0, .xg = 0, .xu = 0, .z = 0 }, // EXL3 experts (eact holds their down inputs)
};

/// The routed block's scratch bytes: its own buffers plus the grouped plan's (`grouped.planSizes`).
pub const ScratchSizes = struct {
    mlog: usize,
    router_part: usize,
    pick: usize,
    wts: usize,
    plan: @import("cuda").grouped.PlanSizes,
    eact: usize,
    ey: usize,
    sgu: usize,
    sact: usize,
    sxs: usize,
    sy: usize,
};

/// `rows` rows, `pairs` picks over `count` experts of width `ew`, hidden `d`, shared width `sh`; the
/// router's partials span `@max(count, i_dim)` columns (the wider of the two callers' shapes).
pub fn scratchSizes(rows: usize, pairs: usize, count: usize, router_wide: usize, tile: usize, ew: usize, d: usize, sh: usize) ScratchSizes {
    return .{
        .mlog = rows * count * 4,
        .router_part = glue.router_ks * rows * router_wide * 4,
        .pick = pairs * 4,
        .wts = pairs * 4,
        .plan = @import("cuda").grouped.planSizes(pairs, count, tile),
        .eact = pairs * ew * 2,
        .ey = pairs * d * 4,
        .sgu = rows * 2 * sh * 2,
        .sact = rows * sh * 2,
        .sxs = rows * (sh / 64) * 4,
        .sy = rows * d * 4,
    };
}

/// The kernels the block runs: the captured Triton set, the grouped NVFP4 experts, the plan, and glm_ops.
pub const Kernels = struct { set: glue.Set, experts: *const experts.Kernels, exl3: *const exl3.Kernels, put_slot: cuda.Function, prof: ?*prof_mod.Prof = null };

/// forward.moe_block: out [rows][dims] fp32 partial; `prompt`: 64-pair items, the prompt kernels, bf16 slot rows.
pub fn block(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, prompt: bool, out: u64) !void {
    try body(k, stream, l, s, x, x_stride, rows, prompt);
    try fill(k, stream, l, s, x, x_stride, 0, rows, prompt, out);
}

/// The block up to its fill: the router, the top-k pick, the plan and the routed experts into the slot rows.
pub fn body(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, prompt: bool) !void {
    const d = l.routed.dims;
    const slots = l.top_k + 1;
    const shared_id: i32 = @intCast(l.count);
    try glue.router(k.set, stream, x, x_stride, l.router, s.mlog, s.router_part, rows, d, l.count);
    try glue.select(k.set, stream, s.mlog, l.bias, s.pick, s.wts, rows, l.top_k, l.count, l.scale, l.norm);
    if (k.prof) |p| p.mark(stream, .router);
    if (l.exl3) |w| {
        defer if (k.prof) |p| p.mark(stream, .routed);
        return exl3.run(k.exl3, k.experts, stream, w, s.x3, s.plan, x, x_stride, s.pick, slots, rows, l.count, d, l.routed.width, l.routed.limit, prompt, s.ey, s.eact);
    }
    const plan: experts.Plan = .{ .route = s.plan, .slots = slots, .experts = l.count + 1, .prompt = prompt };
    try experts.route(k.experts, stream, s.pick, rows * slots, plan);
    if (k.prof) |p| p.mark(stream, .plan); // the §87 decode profile: the plan build split out (print-only)
    try experts.run(k.experts, stream, .gate_up, l.routed, plan, x, x_stride, rows, s.eact, shared_id);
    if (k.prof) |p| p.mark(stream, .gate_up);
    try experts.run(k.experts, stream, if (prompt) .down else .down_f32, l.routed, plan, s.eact, l.routed.width, rows, s.ey, shared_id);
    if (k.prof) |p| p.mark(stream, .down);
}

/// moe_block's fill of rows lo..hi: shared expert into the last slot, slot sum into out fp32 (any range, same bits).
pub fn fill(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, lo: usize, hi: usize, prompt: bool, out: u64) !void {
    const d = l.routed.dims;
    const slots = l.top_k + 1;
    const n = hi - lo;
    const bf16_slots = prompt and l.exl3 == null; // the slot rows: bf16 in an NVFP4 prompt chunk, else fp32
    const eb: usize = if (bf16_slots) 2 else 4;
    try b16.matmul(k.set, stream, x + lo * x_stride * 2, x_stride, l.shared_gu, s.sgu + lo * 2 * l.shared * 2, s.mm_part, n, 2 * l.shared, d, false);
    try glue.swiglu(k.set, stream, s.sgu + lo * 2 * l.shared * 2, s.sact + lo * l.shared * 2, s.sxs + lo * (l.shared / 64) * 4, l.routed.limit, n, l.shared);
    try b16.matmul(k.set, stream, s.sact + lo * l.shared * 2, l.shared, l.shared_down, s.sy + lo * d * 4, s.mm_part, n, d, l.shared, true);
    if (k.prof) |p| p.mark(stream, .shared); // the §87 decode profile (print-only)
    var a: cuda.Args = .{};
    a.add(s.sy + lo * d * 4);
    a.add(s.ey + lo * slots * d * eb);
    for ([_]usize{ n, d, slots, l.top_k, @intFromBool(bf16_slots) }) |v| a.add(@as(c_int, @intCast(v)));
    try cuda.launch.launch(k.put_slot, .{ .grid = .{ .x = @intCast((n * d + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
    try glue.combine(k.set, stream, s.ey + lo * slots * d * eb, !bf16_slots, s.wts + lo * slots * 4, out, n, d, slots);
    if (k.prof) |p| p.mark(stream, .combine); // the §87 decode profile (print-only)
}

test "the extracted routed scratch sizes equal the state's inline formulas" {
    const g = @import("cuda").grouped;
    const cases = [_]struct { rows: usize, pairs: usize, count: usize, wide: usize, ew: usize, d: usize, sh: usize }{
        .{ .rows = 128, .pairs = 128 * 9, .count = 256, .wide = 256, .ew = 256, .d = 6144, .sh = 2048 },
        .{ .rows = 1, .pairs = 9, .count = 256, .wide = 256, .ew = 256, .d = 6144, .sh = 2048 },
        .{ .rows = 4096, .pairs = 4096 * 9, .count = 256, .wide = 256, .ew = 256, .d = 6144, .sh = 2048 },
    };
    for (cases) |t| {
        const s = scratchSizes(t.rows, t.pairs, t.count, t.wide, 16, t.ew, t.d, t.sh);
        try std.testing.expectEqual(t.rows * t.count * 4, s.mlog);
        try std.testing.expectEqual(8 * t.rows * t.wide * 4, s.router_part); // glue.router_ks = 8
        try std.testing.expectEqual(t.pairs * 4, s.pick);
        try std.testing.expectEqual(t.pairs * 4, s.wts);
        const hist_len = if (t.pairs > g.small) (t.pairs + g.small - 1) / g.small * (t.count + 1) else 1;
        try std.testing.expectEqual(t.pairs * 4, s.plan.members);
        try std.testing.expectEqual(g.maxItems(t.pairs, t.count + 1, 16) * 3 * 4, s.plan.items);
        try std.testing.expectEqual(@as(usize, 8), s.plan.counts);
        try std.testing.expectEqual(@max(1, t.pairs) * 4, s.plan.rank);
        try std.testing.expectEqual(hist_len * 4, s.plan.hist);
        try std.testing.expectEqual(t.pairs * t.ew * 2, s.eact);
        try std.testing.expectEqual(t.pairs * t.d * 4, s.ey);
        try std.testing.expectEqual(t.rows * 2 * t.sh * 2, s.sgu);
        try std.testing.expectEqual(t.rows * t.sh * 2, s.sact);
        try std.testing.expectEqual(t.rows * (t.sh / 64) * 4, s.sxs);
        try std.testing.expectEqual(t.rows * t.d * 4, s.sy);
    }
}
