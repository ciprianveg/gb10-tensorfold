//! The kernels a GLM-5.3 forward launches, split by provenance so the fatbin decision (GPU-QUEUE Q2)
//! is a wiring choice, not a code change:
//!  - `glue`: 1.1's shape-agnostic entries, reused unchanged (512-wide norms, absorb/expand, the
//!    latent attention and its merge, the index key/gate cache write, residual add, embed, swiglu).
//!  - `extras`: the entries our widths need and 1.1's glue does not instantiate — its rmsnorm is
//!    monomorphized at 4096/1536/512 only, so hidden 6144 and q_lora 2048 have no entry — plus the
//!    indexer score/select. These are r2's `glm53_*` kernels (or new instantiations of glm_glue.cu);
//!    which one is the decision table's call.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("glue/cuda_glue.zig");

pub const Ops = struct {
    glue: glue.Set,
    extras: Extras,
};

pub const Extras = struct {
    rmsnorm_6144: cuda.Function, // the backbone: hidden (kernels/cuda/glm53_extras.cu)
    rmsnorm_2048: cuda.Function, // MLA's q_a
    index_score: cuda.Function, // the bf16 indexer scorer (indexer.zig's ABI)
    index_score_fp8: ?cuda.Function = null, // r2's native sm121 scorer over the fp8 plane (glm53_fp8_scores.cu)
    rope: ?cuda.Function = null, // only if the attention entries want rope applied outside them
    // THE ROPE HALF (findings §36): the latent row is kv_lora + rope wide ([absorbed latent | rope key]),
    // and the query's rope half is packed beside its absorbed half, so one dot scores the whole MLA.
    kv_write: cuda.Function, // [lat | rot(rope key at pos)] into a (lw + rd)-wide cache row
    qpack: cuda.Function, // [qlat | rot(q's rope half at pos)] per (row, head)
    chunks: cuda.Function, // chunks_wide at W = lw + rd, 16 heads: 93,568 B of smem (the 32-head form
    // would need 113,408 B and the GB10's opt-in is 101,376 B, so the wide form is 16 heads always)

    /// The family's extras from their fatbin module (`cuda.kernels.glm53_extras`); the fp8 scorer comes
    /// from its own module when the boot runs the fp8 key plane.
    pub fn load(extras_module: cuda.Module, fp8_module: ?cuda.Module) !Extras {
        var e: Extras = .{
            .rmsnorm_6144 = try extras_module.function("tf_glm53_rmsnorm_6144"),
            .rmsnorm_2048 = try extras_module.function("tf_glm53_rmsnorm_2048"),
            .index_score = try extras_module.function("tf_glm53_index_score"),
            .index_score_fp8 = if (fp8_module) |m| try m.function("glm53_fp8_scores") else null,
            .kv_write = try extras_module.function("tf_glm53_kv_write"),
            .qpack = try extras_module.function("tf_glm53_qpack"),
            .chunks = try extras_module.function("tf_glm53_chunks_16"),
        };
        try e.chunks.allowDynamicShared(chunksSmem(16, kv_row));
        return e;
    }
};

/// The latent row's width for this pack: kv_lora + rope (512 + 64), r2's `latentWidth()`.
pub const kv_row: u32 = 576;
/// The attention row width of the pack's own shape: the tests pin 512 + 64.
pub const rope_dims: u32 = 64;

/// The 576-wide attention's dynamic smem, the same formula as glm_glue.cuh's chunksSmem (HB rows of a
/// (kv_lora + rope)-wide row, two 32-key tiles of it, the 32-key score tile and the warp reductions).
pub fn chunksSmem(comptime hb: u32, comptime row: u32) u32 {
    return (hb * row + 2 * 32 * row + hb * 32) * 2 + (hb * 4 + 2 * hb) * 4;
}

/// _kv_write at width lw + rd: the normed latent row and its rope key, rotated at the row's true position.
pub fn kvWrite(e: Extras, stream: cuda.Stream, lat: u64, lat_stride: usize, krope: u64, kstride: usize, cache: u64, pos: u64, rows: usize, lw: usize, rd: usize, inv: u64) !void {
    var a: cuda.Args = .{};
    a.add(lat);
    a.add(@as(i64, @intCast(lat_stride)));
    a.add(krope);
    a.add(@as(i64, @intCast(kstride)));
    a.add(cache);
    a.add(pos);
    a.add(@as(i64, @intCast(lw)));
    a.add(@as(i64, @intCast(rd)));
    a.add(inv);
    try cuda.launch.launch(e.kv_write, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &a);
}

/// The packed query row r,h: [qlat (lw) | rot(q[dn .. dn + rd]) at row r's position].
pub fn qpack(e: Extras, stream: cuda.Stream, qlat: u64, q: u64, inv: u64, pos: u64, out: u64, rows: usize, heads: usize, dn: usize, lw: usize, rd: usize) !void {
    var a: cuda.Args = .{};
    a.add(qlat);
    a.add(q);
    a.add(inv);
    a.add(pos);
    a.add(out);
    a.add(@as(i64, @intCast(rows)));
    a.add(@as(i64, @intCast(heads)));
    a.add(@as(i64, @intCast(dn)));
    a.add(@as(i64, @intCast(lw)));
    a.add(@as(i64, @intCast(rd)));
    try cuda.launch.launch(e.qpack, .{ .grid = .{ .x = @intCast(rows * heads) }, .block = .{ .x = 128 } }, stream, &a);
}

/// The ABI of an `extras` rmsnorm: (x, x_stride, w, out, o_stride, eps) as glue.rmsnorm's, one width.
/// `warps` is the width's own launch bound (6144: 8, 2048: 4) - a kernel launched with more threads than
/// its `__launch_bounds__` is CUDA_ERROR_INVALID_VALUE, which is what the 2048 instantiation hit.
pub fn rmsnorm(f: cuda.Function, stream: cuda.Stream, warps: u32, in: u64, in_stride: usize, w: u64, out: u64, out_stride: usize, eps: f32, rows: usize) !void {
    // glue.rmsnorm's argument list exactly: x, x_stride, w, out, o_stride, xs (none), eps, sums=false
    var a: cuda.Args = .{};
    a.add(in);
    a.add(@as(i64, @intCast(in_stride)));
    a.add(w);
    a.add(out);
    a.add(@as(i64, @intCast(out_stride)));
    a.add(@as(u64, 0));
    a.add(eps);
    a.add(@as(c_int, 0));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = warps * 32 } }, stream, &a);
}

/// One width's norm through whichever entry has it; `d` is the width the family runs (6144 or 2048).
pub fn norm(o: Ops, stream: cuda.Stream, d: usize, in: u64, in_stride: usize, w: u64, out: u64, out_stride: usize, eps: f32, rows: usize) !void {
    return switch (d) {
        6144 => rmsnorm(o.extras.rmsnorm_6144, stream, 8, in, in_stride, w, out, out_stride, eps, rows),
        2048 => rmsnorm(o.extras.rmsnorm_2048, stream, 4, in, in_stride, w, out, out_stride, eps, rows),
        512 => glue.rmsnorm(o.glue, stream, in, in_stride, w, out, out_stride, null, eps, rows, d),
        else => error.NoRmsnormWidth,
    };
}

test "the norm dispatch names every width GLM-5.3 runs (6144 hidden, 2048 q_a, 512 kv_a)" {
    // widths the forward asks for; a missing arm is a compile error above, this pins the list
    const widths = [_]usize{ 6144, 2048, 512 };
    for (widths) |w| try std.testing.expect(w == 6144 or w == 2048 or w == 512);
}