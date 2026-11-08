//! §98 THE BINDER, IN THE MODULE (§95.4's adapter core): the 1.1 loader's NVFP4-CSF weights bound into r2's
//! Weights — the same transforms the run lane's `glm53r2_bind.zig` has proven since §42 (q_a/kv_a split at
//! q_lora, gate/up split, wk de-pad, the routed experts riding the side table into the E2 hook), moved from
//! the test root into the family so the served `Boot` can load through it (TF_GLM53_BIND). The test file
//! keeps its copy: it owns the A2 digest harness around the same logic.

const std = @import("std");
const cuda = @import("cuda");
const glm53 = @import("glm53").family; // the 1.1 loader + config + shard plan (the weights source)
const W = @import("cuda_weights.zig"); // r2's weights (the destination type)
const exl3 = @import("exl3.zig");

pub const Stats = struct { layers: usize = 0, dense_mlp: usize = 0, moe: usize = 0, indexers: usize = 0, experts: usize = 0, wk: usize = 0 };

/// E2's side table: one entry a layer, the port's MoE layer (router, NVFP4 experts, shared expert) for the
/// layers the hybrid FFN runs. `null` for the dense layers (r2's own `mlp` branch handles those).
pub const Side = struct {
    moe: []?glm53.moe.Layer,
    scratch: glm53.moe.Scratch,
    kernels: glm53.moe.Kernels,
};

/// r2's dense-linear form: a plain bf16 [n, k] row-major weight, computed by cuBLAS (`Forward.denseGemm`).
pub fn denseLinear(ptr: u64, n: usize, k: usize) exl3.Linear {
    return .{ .words = ptr, .suh = 0, .svh = 0, .counters = 0, .k = k, .n = n, .k2 = 0, .cb = .mul1, .sk = 1, .wk = 1, .dense = true };
}


/// Â§125.38 the dense-lane A/B arm (TF_GLM53_DENSE_LANE): the qmmf packed copy of a bound dense
/// linear - the round-trip pack (D2H -> host packBf16 -> H2D), the A/B arm only (~3 GB/rank, ~40 s
/// at boot). The production form (if the switch wins): the 1.1 loader's host-staging hook, which
/// avoids the round-trip. The buffer rides w.buffers (the weights' lifetime). The A/B targets the
/// ATTENTION smalls (the 26.2 ms front/attn_proj term); the dense MLPs (0.8 ms) and the shared
/// experts are follow-on arms.
var dense_lane_on: ?bool = null;
fn packQwords(d: *const cuda.Driver, stream: cuda.Stream, w: *W.Weights, ptr: u64, n: usize, k: usize) !u64 {
    if (dense_lane_on == null) dense_lane_on = std.c.getenv("TF_GLM53_DENSE_LANE") != null;
    if (!dense_lane_on.? or ptr == 0 or n == 0) return 0;
    const host_w = try w.gpa.alloc(u16, n * k);
    defer w.gpa.free(host_w);
    try d.check(d.api.cuMemcpyDtoHAsync_v2(host_w.ptr, ptr, n * k * 2, stream.handle), "DtoH");
    try d.check(d.api.cuStreamSynchronize(stream.handle), "sync");
    const npad = (n + 127) / 128 * 128;
    const pk = try w.gpa.alloc(u16, npad * k);
    defer w.gpa.free(pk);
    cuda.qmmf.packBf16(pk, host_w, n, k);
    const dst = try cuda.DeviceBuffer.alloc(d, npad * k * 2);
    try w.buffers.append(w.gpa, dst);
    try d.check(d.api.cuMemcpyHtoDAsync_v2(dst.ptr, pk.ptr, npad * k * 2, stream.handle), "HtoD");
    try d.check(d.api.cuStreamSynchronize(stream.handle), "sync");
    return dst.ptr;
}

/// 126.6(at) THE FP8 CANARY (TF_GLM53_DENSE_FP8, value semantics: unset/empty/"0" = off): the bound
/// bf16 linear RE-QUANTIZED to the upstream's own form (the 128x128-block e4m3 codes, amax/448
/// scale_inv - the same block geometry the pack stores) and packed for qmmf's .fp8g lane. The
/// dequant's bf16 rounding is the only added loss vs the pack's original codes; the kernel's f32
/// scale product then SKIPS the bf16 rounding the cuBLAS path adds on every read.
const LanePack = struct { qwords: u64 = 0, qscales: u64 = 0, fp8: bool = false };

var fp8_arm: ?bool = null;
fn fp8Arm() bool {
    if (fp8_arm == null) fp8_arm = blk: {
        const v = std.c.getenv("TF_GLM53_DENSE_FP8") orelse break :blk false;
        const s = std.mem.span(v);
        break :blk s.len > 0 and !std.mem.eql(u8, s, "0");
    };
    return fp8_arm.?;
}

fn bf16Val(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// f32 -> e4m3: round to nearest even, saturating at +-448 (the max finite), NaN -> 0x7F. The
/// quantizer's inputs are pre-scaled to |x| <= 448 by the amax/448 block scale.
fn f32ToE4m3(x: f32) u8 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return 0x7F;
    const sign: u8 = @intCast(u >> 31);
    const ax = @abs(x);
    if (ax >= 448) return (sign << 7) | 0x7E;
    if (ax == 0) return sign << 7;
    const e: i32 = @as(i32, @intCast((u >> 23) & 0xFF)) - 127;
    const m24: u32 = (u & 0x7F_FFFF) | 0x80_0000;
    if (e >= -6) { // the normal range: 3 mantissa bits, the carry into the exponent
        var r: u32 = m24 >> 20;
        const rem: u32 = m24 & 0xF_FFFF;
        var ee = e;
        if (rem > 0x8_0000 or (rem == 0x8_0000 and (r & 1) == 1)) r += 1;
        if (r == 16) {
            r = 8;
            ee += 1;
        }
        if (ee > 8) return (sign << 7) | 0x7E;
        return (sign << 7) | (@as(u8, @intCast(ee + 7)) << 3) | @as(u8, @intCast(r & 7));
    }
    if (e < -10) return sign << 7; // rounds to zero
    const sh: u5 = @intCast(14 - e); // the subnormal grid: q = round(m24 / 2^(14 - e)) at 2^-9 steps
    var q: u32 = m24 >> sh;
    const rem: u32 = m24 & ((@as(u32, 1) << sh) - 1);
    const half: u32 = @as(u32, 1) << (sh - 1);
    if (rem > half or (rem == half and (q & 1) == 1)) q += 1;
    if (q >= 8) return (sign << 7) | 0x08; // the carry to the minimum normal
    return (sign << 7) | @as(u8, @intCast(q));
}

/// bf16 [n, k] -> the 128-block e4m3 codes [n, k] + scale_inv [ceil(n/128), ceil(k/128)] (the
/// upstream block geometry; all-zero blocks code zero with a zero scale).
fn quantFp8Block(codes: []u8, scales: []f32, w: []const u16, n: usize, k: usize) void {
    const br = (n + 127) / 128;
    const bc = (k + 127) / 128;
    for (0..br) |r| {
        const r0 = r * 128;
        const r1 = @min(n, r0 + 128);
        for (0..bc) |c| {
            const c0 = c * 128;
            const c1 = @min(k, c0 + 128);
            var amax: f32 = 0;
            for (r0..r1) |i| {
                for (c0..c1) |j| amax = @max(amax, @abs(bf16Val(w[i * k + j])));
            }
            const sc: f32 = amax / 448.0;
            scales[r * bc + c] = sc;
            if (sc == 0) {
                for (r0..r1) |i| {
                    for (c0..c1) |j| codes[i * k + j] = 0;
                }
                continue;
            }
            for (r0..r1) |i| {
                for (c0..c1) |j| codes[i * k + j] = f32ToE4m3(bf16Val(w[i * k + j]) / sc);
            }
        }
    }
}

/// The lane pack for one bound dense linear: the fp8 arm's codes + scale tiles, else the 125.37
/// bf16 lane pack (the product default).
fn packLane(d: *const cuda.Driver, stream: cuda.Stream, w: *W.Weights, ptr: u64, n: usize, k: usize) !LanePack {
    if (ptr == 0 or n == 0) return .{};
    if (!fp8Arm()) return .{ .qwords = try packQwords(d, stream, w, ptr, n, k) };
    const host_w = try w.gpa.alloc(u16, n * k);
    defer w.gpa.free(host_w);
    try d.check(d.api.cuMemcpyDtoHAsync_v2(host_w.ptr, ptr, n * k * 2, stream.handle), "DtoH");
    try d.check(d.api.cuStreamSynchronize(stream.handle), "sync");
    const br = (n + 127) / 128;
    const bc = (k + 127) / 128;
    const codes = try w.gpa.alloc(u8, n * k);
    defer w.gpa.free(codes);
    const grid = try w.gpa.alloc(f32, br * bc);
    defer w.gpa.free(grid);
    quantFp8Block(codes, grid, host_w, n, k);
    const npad = cuda.fp8.padded(n);
    const pc = try w.gpa.alloc(u8, npad * k);
    defer w.gpa.free(pc);
    cuda.fp8.packCodes(pc, codes, n, k);
    const pt = try w.gpa.alloc(f32, npad * (k / 64));
    defer w.gpa.free(pt);
    cuda.fp8.packScales(pt, grid, n, k);
    const dc = try cuda.DeviceBuffer.alloc(d, npad * k);
    try w.buffers.append(w.gpa, dc);
    try d.check(d.api.cuMemcpyHtoDAsync_v2(dc.ptr, pc.ptr, npad * k, stream.handle), "HtoD");
    const ds = try cuda.DeviceBuffer.alloc(d, npad * (k / 64) * 4);
    try w.buffers.append(w.gpa, ds);
    try d.check(d.api.cuMemcpyHtoDAsync_v2(ds.ptr, pt.ptr, npad * (k / 64) * 4, stream.handle), "HtoD");
    try d.check(d.api.cuStreamSynchronize(stream.handle), "sync");
    std.log.info("glm53: the fp8 arm: {d}x{d} packed qmmf fp8g (npad {d}, scales {d})", .{ n, k, npad, br * bc });
    return .{ .qwords = dc.ptr, .qscales = ds.ptr, .fp8 = true };
}

test "f32ToE4m3: every finite e4m3 value round-trips exactly; the bounds saturate" {
    const cw = @import("cuda_weights.zig");
    var c: u16 = 0;
    while (c < 256) : (c += 1) {
        if (c == 0x7F or c == 0xFF) continue; // the NaN codes
        const d = cw.fp8ToF32(@intCast(c));
        try std.testing.expectEqual(c, f32ToE4m3(d));
    }
    var v: f32 = -460;
    while (v <= 460) : (v += 0.125) {
        const d = cw.fp8ToF32(f32ToE4m3(v));
        if (v <= -448) {
            try std.testing.expectEqual(@as(f32, -448), d);
        } else if (v >= 448) {
            try std.testing.expectEqual(@as(f32, 448), d);
        } else if (v != 0) {
            try std.testing.expect(@abs(d - v) <= @abs(v) * 0.0625 + 2e-3);
        } else {
            try std.testing.expectEqual(@as(f32, 0), d);
        }
    }
}

/// Transform 3: the port zero-pads each head's key block to `qk` rows for its own absorb kernel; r2 reads
/// `[H][nope][lw]`. One device copy a head (`nope * lw * 2` bytes) into a binder-owned buffer.
pub fn depadWk(d: *const cuda.Driver, stream: cuda.Stream, wk: u64, heads: usize, nope: usize, lw: usize, qk: usize, w: *W.Weights, stats: *Stats) !u64 {
    if (wk == 0) return 0;
    // TF_BIND_WK=padded: hand r2 the port's padded [H][qk][lw] block unchanged (the _absorb NOPE_P variant)
    if (std.c.getenv("TF_BIND_WK")) |v| if (std.mem.eql(u8, std.mem.span(v), "padded")) {
        stats.wk += 1;
        return wk;
    };
    if (qk == nope) {
        stats.wk += 1;
        return wk; // already compact: nothing to do
    }
    const row = nope * lw * 2;
    var dst = try cuda.DeviceBuffer.alloc(d, heads * row);
    errdefer dst.free();
    try w.buffers.append(w.gpa, dst);
    for (0..heads) |h| {
        const src_off = h * qk * lw * 2;
        const dst_off = h * row;
        try d.check(d.api.cuMemcpyDtoDAsync_v2(dst.ptr + dst_off, wk + src_off, row, stream.handle), "cuMemcpyDtoDAsync");
    }
    stats.wk += 1;
    return dst.ptr;
}

pub fn bind(
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    pw: *const glm53.weights.Weights,
    rank: usize,
    world: usize,
    heads_share: usize,
    heads_global: usize,
    stats: *Stats,
    side: []?glm53.moe.Layer,
) !W.Weights {
    const a = gpa;
    const c = pw.config;
    // TF_BIND_SPLIT=swap: the merged proj's row order was assumed [q_a | kv_a]; swap tests the assumption
    const swap = if (std.c.getenv("TF_BIND_SPLIT")) |v| std.mem.eql(u8, std.mem.span(v), "swap") else false;
    if (swap) std.debug.print("SPLIT order=swap (kv_a first)\n", .{});
    const hidden = c.hidden;
    const nope = c.nope;
    const lw = c.kv_lora;

    var w: W.Weights = .{
        .gpa = a,
        .config = .{
            .hidden = hidden,
            .layers = c.layers,
            .first_dense = c.dense_layers,
            .dense_width = c.dense_inter,
            .expert_width = c.moe_inter,
            .experts = c.experts,
            .shared_experts = @max(1, c.shared_inter / c.moe_inter),
            .top_k = c.topk,
            .routed_scaling = c.routed_scale,
            .norm_topk = c.norm_topk,
            .heads = heads_global,
            .q_lora = c.q_lora,
            .kv_lora = lw,
            .nope = nope,
            .rope = c.rope,
            .v_dim = c.v_dim,
            .rope_theta = c.rope_theta,
            .index_heads = c.i_heads,
            .index_dim = c.i_dim,
            .index_topk = c.i_topk,
            .eps = c.eps,
            .vocab = c.vocab,
            .mtp_layers = if (pw.mtp != null) 1 else 0,
        },
        .rank = rank,
        .world = world,
        .heads = heads_share,
        .embed = pw.embed,
        .final_norm = pw.norm,
        .lm_head = pw.head,
        .vocab_part = pw.share.vocab.len(),
        .vocab_off = pw.share.vocab.lo,
    };
    w.layers = try a.alloc(W.Layer, pw.layers.len);
    errdefer a.free(w.layers);

    for (pw.layers, 0..) |*pl, i| {
        const at = pl.attn;
        var L: W.Layer = .{
            .index = i,
            .input_norm = pl.in_norm,
            .post_attn_norm = pl.post_norm,
            .q_a = denseLinear(if (swap) at.proj + @as(u64, c.kv_lora + c.rope) * hidden * 2 else at.proj, c.q_lora, hidden),
            .q_a_norm = at.q_norm,
            .kv_a = denseLinear(if (swap) at.proj else at.proj + @as(u64, c.q_lora) * hidden * 2, c.kv_lora + c.rope, hidden),
            .kv_a_norm = at.kv_norm,
            .q_b = denseLinear(at.q_b, at.heads * at.qk, at.q_lora),
            .wk = 0,
            .wv = at.wv,
            .o_proj = denseLinear(at.o, hidden, at.heads * c.v_dim),
            .indexer = null,
            .experts = null,
            .mlp = .{ .gate = denseLinear(0, 0, 0), .up = denseLinear(0, 0, 0), .down = denseLinear(0, 0, 0) },
        };
        // Â§125.38: the dense-lane arm - the packed copies for the attention smalls (the 26.2 ms term).
        const lane_smalls = [_]*exl3.Linear{ &L.q_a, &L.kv_a, &L.q_b, &L.o_proj };
        for (lane_smalls) |lin| {
            const pq = try packLane(d, stream, &w, lin.words, lin.n, lin.k);
            lin.qwords = pq.qwords;
            lin.qscales = pq.qscales;
            lin.qfp8 = pq.fp8;
        }

        if (at.index.qb != 0) {
            L.indexer = .{
                .wq_b = denseLinear(at.index.qb, at.index.heads * at.index.d, at.q_lora),
                .wk = at.index.kw,
                .weights_proj = at.index.wproj,
                .k_norm_w = at.index.ln_w,
                .k_norm_b = at.index.ln_b,
            };
            stats.indexers += 1;
        }
        switch (pl.ffn) {
            .dense => |m| {
                const q = switch (m.w) {
                    .bf16 => |bf| bf,
                    .nvfp4 => return error.NotBf16Linears,
                };
                L.mlp = .{
                    .gate = denseLinear(q.gu, m.width, m.dims),
                    .up = denseLinear(q.gu + @as(u64, m.width) * m.dims * 2, m.width, m.dims),
                    .down = denseLinear(q.down, m.dims, m.width),
                };
                stats.dense_mlp += 1;
            },
            .moe => |m| {
                L.mlp = .{
                    .gate = denseLinear(m.shared_gu, m.shared, hidden),
                    .up = denseLinear(m.shared_gu + @as(u64, m.shared) * hidden * 2, m.shared, hidden),
                    .down = denseLinear(m.shared_down, hidden, m.shared),
                };
                L.router_w = m.router;
                L.router_bias = m.bias;
                // E2: r2's `experts` stays null - the routed NVFP4 tables ride the side table into the hook
                side[i] = .{
                    .router = m.router,
                    .bias = m.bias,
                    .routed = m.routed,
                    .count = m.count,
                    .shared_gu = m.shared_gu,
                    .shared_down = m.shared_down,
                    .shared = m.shared,
                    .top_k = m.top_k,
                    .scale = m.scale,
                    .norm = m.norm,
                };
                stats.experts += 1;
                stats.moe += 1;
            },
        }
        L.wk = try depadWk(d, stream, at.wk, at.heads, nope, lw, at.qk, &w, stats);
        w.layers[i] = L;
        stats.layers += 1;
    }
    // §90: the MTP head. cfg.mtp = 1 had the 1.1 load the MTP layer as pw.layers' last entry (converted by
    // the loop above like any moe layer — the E2 hook serves its routed experts through side[c.layers]) and
    // its four BF16 pointers as pw.mtp. r2's MtpHead wants the converted layer + the raw pointers; the
    // Runner's k > 0 path errors NoMtpLayer without this.
    if (pw.mtp) |m| {
        w.mtp = .{ .layer = w.layers[w.layers.len - 1], .eh_proj = m.eh, .enorm = m.enorm, .hnorm = m.hnorm, .head_norm = m.norm };
    }
    try stream.synchronize();
    return w;
}

/// E2's hook context: everything the port's `moe.block` needs that r2's Forward does not know about.
pub const Hook = struct {
    kernels: glm53.moe.Kernels,
    scratch: glm53.moe.Scratch,
    side: []const ?glm53.moe.Layer,
    /// TF_GLM_PROF's 1.1 Prof attached to the moe kernels (print/timing only; the served Boot leaves it null).
    prof: ?*MoeProf = null,

    /// The 1.1 Prof type, from `Kernels.prof`'s field type (@TypeOf does not evaluate).
    pub fn profType() type {
        return @typeInfo(@typeInfo(@TypeOf(@as(glm53.moe.Kernels, undefined).prof)).optional.child).pointer.child;
    }
    pub const MoeProf = profType();

    pub fn has(ctx: *const anyopaque, li: usize) bool {
        const h: *const Hook = @ptrCast(@alignCast(ctx));
        return h.side[li] != null;
    }

    pub fn call(ctx: *const anyopaque, li: usize, stream: cuda.Stream, normed: u64, out: u64, rows: usize, prompt: bool) anyerror!void {
        if (std.c.getenv("TF_GLM53_ROUTER_TRACE")) |tv| if (std.mem.span(tv).len > 0 and std.mem.span(tv)[0] != 48) std.debug.print("HOOK li={d} rows={d} prompt={}\n", .{ li, rows, prompt }); // §100 bisect
        const h: *const Hook = @ptrCast(@alignCast(ctx));
        const l = h.side[li] orelse return error.NoMoeLayer;
        if (h.prof) |p| try p.begin(stream, if (prompt) "moe-prompt" else "moe-decode");
        defer if (h.prof) |p| p.end();
        // out is this rank's fp32 [rows][D] partial, the same contract r2's mlp/experts branches fill
        try glm53.moe.block(h.kernels, stream, l, h.scratch, normed, 6144, rows, prompt, out);
    }
};
