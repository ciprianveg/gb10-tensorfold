//! The 8-bit KV cache machinery, carried from r2 (glm53_kv8.cu, byte-identical: md5
//! 944b52833a5d851659a2aa764869dc6b) so the NVFP4 family's cache is the EXL3 line's at the same dcp:
//! the staging tables (the rope fix), the latent plane's staged quantize (e4m3 codes + two fp32 scales a
//! row) and the dequantize the attention kernels read, in the modes the forward selects.
const std = @import("std");
const cuda = @import("cuda");

pub const Kernels = struct {
    stage: cuda.Function,
    quant: cuda.Function,
    dequant: cuda.Function,
    mod: cuda.Module,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var mod = try cuda.Module.load(d, cuda.kernels.glm53_kv8);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .stage = try mod.function("glm53_kv8_stage"),
            .quant = try mod.function("glm53_kv8_quant"),
            .dequant = try mod.function("glm53_kv8_dequant"),
        };
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }
};

/// The per-row bytes of the kv8 latent plane: lw e4m3 codes + two fp32 scales.
pub fn rowBytes(lw: usize) usize {
    return lw + 8;
}

/// glm53_kv8_stage: the window's positions into the staging table (the rope fix's prefix rows).
pub fn stage(k: *const Kernels, stream: cuda.Stream, spos: u64, sbase: u64, pos: u64, rows: usize, rows_mode: bool, dcp: usize) !void {
    if (rows == 0) return;
    var a: cuda.Args = .{};
    a.add(pos);
    a.add(spos);
    a.add(sbase);
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(c_int, @intFromBool(rows_mode)));
    a.add(@as(c_int, @intCast(dcp)));
    try cuda.launch.launch(k.stage, .{ .grid = .{ .x = @intCast((rows + 127) / 128) }, .block = .{ .x = 128 } }, stream, &a);
}

/// glm53_kv8_quant: staging row r (bf16, `w` values) -> the plane's row for r (e4m3 codes + two scales).
pub fn quant(k: *const Kernels, stream: cuda.Stream, stage_buf: u64, plane: u64, pos: u64, base: u64, rows: usize, rows_mode: bool, dcp: usize, rank: usize, lw: usize, w: usize) !void {
    if (rows == 0) return;
    var a: cuda.Args = .{};
    a.add(stage_buf);
    a.add(plane);
    a.add(pos);
    a.add(base);
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(c_int, @intFromBool(rows_mode)));
    a.add(@as(c_int, @intCast(dcp)));
    a.add(@as(c_int, @intCast(rank)));
    a.add(@as(c_int, @intCast(lw)));
    a.add(@as(c_int, @intCast(w)));
    try cuda.launch.launch(k.quant, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &a);
}

/// glm53_kv8_dequant: the bf16 rows the attention kernels read, out of the e4m3 plane.
/// mode 0: prompt rows [0, pos + rows) over n_max; 1: one stream, rows x K from the selection;
/// 2: a base table; 3/4: the streaming/multi-window shapes (a fixed grid for graph replay).
pub fn dequant(k: *const Kernels, stream: cuda.Stream, plane: u64, out: u64, pos: u64, base: u64, tok: u64, rows: usize, kw: usize, lw: usize, w: usize, mode: c_int, dcp: usize, n_max: usize) !void {
    if (rows == 0 and mode != 0 and mode != 4) return;
    var a: cuda.Args = .{};
    a.add(plane);
    a.add(out);
    a.add(pos);
    a.add(base);
    a.add(tok);
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(c_int, @intCast(kw)));
    a.add(@as(c_int, @intCast(lw)));
    a.add(@as(c_int, @intCast(w)));
    a.add(mode);
    a.add(@as(c_int, @intCast(dcp)));
    a.add(@as(i64, @intCast(n_max)));
    const grid: usize = if (mode == 4) @min(n_max, 4096) else if (mode == 0 or mode == 3) n_max else rows * kw;
    try cuda.launch.launch(k.dequant, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = 128 } }, stream, &a);
}

test "the kv8 plane's row arithmetic: lw codes plus two fp32 scales" {
    try std.testing.expectEqual(@as(usize, 520), rowBytes(512));
}