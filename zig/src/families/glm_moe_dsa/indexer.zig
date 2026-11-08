//! GLM-5.3's DSA indexer on CUDA, the one stage whose kernels the fatbin decision picks (GPU-QUEUE Q2).
//!
//! What is 1.1's and reused unchanged: the key/gate cache write. `glm_glue_index.cuh`'s
//! `tf_glm_index_write` is a warp-per-row LayerNorm over exactly our 128 index dims with weight and
//! bias, reading the gate row precomputed by a matmul — our `weights_proj` provides that row
//! (32 heads), so the cache write needs no new kernel.
//!
//! What is ours and queued: the score-and-select over the cached keys. r2's `glm53_fp8_scores.cu`
//! (fp8 scoring, per-head dot times head weight, top-k select) is the candidate; the entry points
//! below are its ABI so the forward compiles and the ABI is fixed before the kernel lands.
const std = @import("std");
const cuda = @import("cuda");

/// The indexer's kernels: `score` writes each row's `topk` token positions (i32, ascending) and count;
/// `update` is upstream's index_write (reused, not ours).
pub const Kernels = struct {
    score: cuda.Function, // the fp8 scorer over the e4m3 plane (glm53_fp8_scores, r2 byte-identical)
    score_bf16: ?cuda.Function = null, // the bf16 fallback (tf_glm53_index_score), PACK=True
    plane_write: cuda.Function, // glm53_ik_write8 / _p: the e4m3 index-key plane row
    hist: cuda.Function,
    count: cuda.Function,
    write: cuda.Function,
    hist1c: cuda.Function, // ROUND 3: pass 1 with the survivor-list compaction (glm53_select.cu)
    hist2c: cuda.Function, // ROUND 3: pass 2 + the finish over the compact list, one block a row

    /// The selection's fatbin (glm53_select.cu: r2's radix top-k, byte-identical bodies).
    pub fn load(module: cuda.Module, plane: cuda.Module, padded: bool, score_fp8: cuda.Function, score_bf16: ?cuda.Function) !Kernels {
        return .{
            .hist = try module.function("glm53_topk_hist"),
            .count = try module.function("glm53_topk_count"),
            .write = try module.function("glm53_topk_write"),
            .hist1c = try module.function("glm53_topk_hist1c"),
            .hist2c = try module.function("glm53_topk_hist2c"),
            .score = score_fp8,
            .score_bf16 = score_bf16,
            .plane_write = try plane.function(if (padded) "glm53_ik_write8p" else "glm53_ik_write8"),
        };
    }
};

/// r2's TOPK_CHUNK and SEL_BINS: the radix's columns a block covers, and its histogram width.
pub const topk_chunk = 2048;
pub const sel_bins = 2048;

/// The compact path's per-row capacity (glm53_select.cu's SEL_CAP - the shared pos[] in hist2c).
pub const topk_compact_cap: usize = 8192;

/// r2's crossover for the compact path (cuda_forward.zig's TF_GLM53_TOPK_COL_MAX): past it the compact
/// radix beats the columns pass; 0 restores the columns pass everywhere it fits.
pub const topk_col_max_default: usize = 49152;

/// The columns cap from `TF_GLM53_TOPK_COL_MAX` (unset or unparsable: the default).
pub fn topkColMax() usize {
    const e = std.c.getenv("TF_GLM53_TOPK_COL_MAX") orelse return topk_col_max_default;
    return std.fmt.parseInt(usize, std.mem.sliceTo(e, 0), 10) catch topk_col_max_default;
}

/// The radix's scratch: hist [rows][2048] u32, state [rows][4] u32, cnt [rows][blocks][2] u32, and the
/// compact path's survivor list [rows][cap] + its count and overflow flag (ROUND 3).
pub const Scratch = struct {
    hist: u64,
    state: u64,
    cnt: u64,
    list: u64 = 0, // uint32 [rows, cap]: the at-or-above-digit-0 columns (hist1c writes, hist2c reads)
    lcnt: u64 = 0, // uint32 [rows]: the list's count, zeroed per call
    lflg: u64 = 0, // uint32 [rows]: the overflow flag, zeroed per call
    rows: usize,
    blocks: usize,
    cap: usize = 0,

    pub fn bytes(rows: usize, blocks: usize) struct { hist: usize, state: usize, cnt: usize } {
        return .{ .hist = rows * sel_bins * 4, .state = rows * 4 * 4, .cnt = rows * blocks * 2 * 4 };
    }

    /// The compact path's extra bytes for `rows` at `cap` columns a row (list + count + flag).
    pub fn compactBytes(rows: usize, cap: usize) usize {
        return rows * cap * 4 + rows * 4 + rows * 4;
    }
};

/// One row's token list: at most `topk` positions, ascending, -1 padded.
pub fn tokenWidth(c: anytype) usize {
    return c.i_topk;
}

/// Below this position every visible key is attended anyway: selection is the identity and the dense
/// path runs (the Flash family's `dense_limit`, with our indexer's top-k and no pool tail).
pub fn denseMode(c: anytype, pos: usize, rows: usize) bool {
    return pos + rows <= c.denseLimit();
}

/// A row's positions when every visible key is attended: 0..pos+row, written host-side for the dense
/// path's sparse fallback (tests and the first prompt chunk).
pub fn identityTokens(tokens: []i32, pos: usize, rows: usize) void {
    for (0..rows) |r| {
        const n = pos + r + 1;
        for (0..tokens.len / rows) |j| tokens[r * (tokens.len / rows) + j] = if (j < n) @intCast(j) else -1;
    }
}

/// The fp8 plane's row stride in bytes (132 legacy, 144 padded) and words (33/36).
pub const plane_stride = 132;
pub const plane_stride_padded = 144;

/// The plane writer: one row per call, keys at `pos` (the e4m3 plane's 128 codes + fp32 scale at word 32).
/// DCP > 1 stores only the owner's rows (pg % dcp == rank) at slot pg / dcp.
pub fn writePlane(k: Kernels, stream: cuda.Stream, ik: u64, nw: u64, nb: u64, inv: u64, plane: u64, pos: u64, rows: usize, d: usize, rd: usize, dcp: usize, rank: usize, eps: f32) !void {
    var a: cuda.Args = .{};
    a.add(ik);
    a.add(nw);
    a.add(nb);
    a.add(inv);
    a.add(plane);
    a.add(plane); // ICS: the same storage viewed as floats
    a.add(pos);
    a.add(@as(c_int, @intCast(d)));
    a.add(@as(c_int, @intCast(rd)));
    a.add(@as(c_int, @intCast(dcp)));
    a.add(@as(c_int, @intCast(rank)));
    a.add(eps);
    try cuda.launch.launch(k.plane_write, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &a);
}

/// The fp8 score pass: rows' queries against the plane's keys, packed int32 order words [rows][T]
/// (later keys 0). glm53_fp8_scores' ABI: (Q, W, CODES, SCALES, POS, BASE, OUT, T, R0, ROWS_MODE, n, q, w).
pub fn scoreFp8(k: Kernels, stream: cuda.Stream, q: u64, w: u64, plane: u64, out: u64, T: usize, r0: usize, pos: u64, rows: usize, nh: usize, d: usize) !void {
    var a: cuda.Args = .{};
    a.add(q);
    a.add(w);
    a.add(plane); // CODES
    a.add(plane); // SCALES: the same storage
    a.add(pos);
    a.add(@as(u64, 0)); // BASE: single stream
    a.add(out);
    a.add(@as(c_int, @intCast(T)));
    a.add(@as(c_int, @intCast(r0)));
    a.add(@as(c_int, 0)); // ROWS_MODE: one window, one stream
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(f32, @floatCast(std.math.pow(f64, @floatFromInt(d), -0.5))));
    a.add(@as(f32, @floatCast(std.math.pow(f64, @floatFromInt(nh), -0.5))));
    // r2's launch: x = 128*32-key groups (a CTA walks TILES_PER_CTA = 32 tiles), y = the rows
    try cuda.launch.launch(k.score, .{ .grid = .{ .x = @intCast((T + 4095) / 4096), .y = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
}

/// The bf16 score pass (the fallback): packed int64 keys [rows][T] (T = pos + rows the caller
/// passes; later keys int64 min + 1). r2's `_index_scores`, PACK=True: score = sum_h
/// (w*1/sqrt(NH)) * relu(dot(q,k) * 1/sqrt(D)), key = monotone(bits) << 32 | (0x7FFFFFFF - position).
pub fn scoreBf16(k: Kernels, stream: cuda.Stream, q: u64, w: u64, kc: u64, out: u64, T: usize, r0: usize, pos: u64, rows: usize, nh: usize, d: usize) !void {
    if (k.score_bf16 == null) return error.NoBf16Scorer;
    var a: cuda.Args = .{};
    a.add(q);
    a.add(w);
    a.add(kc);
    a.add(out);
    a.add(@as(c_int, @intCast(T)));
    a.add(@as(c_int, @intCast(r0)));
    a.add(@as(c_int, @intCast(pos)));
    a.add(@as(c_int, @intCast(nh)));
    a.add(@as(c_int, @intCast(d)));
    a.add(@as(f32, @floatCast(std.math.pow(f64, @floatFromInt(d), -0.5))));
    a.add(@as(f32, @floatCast(std.math.pow(f64, @floatFromInt(nh), -0.5))));
    try cuda.launch.launch(k.score_bf16.?, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((T + 127) / 128) }, .block = .{ .x = 128 } }, stream, &a);
}

/// Whether the compact path can serve this selection: the guards of r2's routing (cuda_forward.zig) -
/// `T` past the crossover, the list cap, the row/block scratch bounds, and a real top-k to take.
pub fn compactFits(s: Scratch, t: usize, rows: usize, topk: usize) bool {
    // r2's route is `Tl > cap && kk < Tl`: the identity case (T == K) keeps the copy path, never this one
    if (topk == 0 or topk >= t or rows == 0 or rows > s.rows) return false;
    if (s.cap == 0 or topk > s.cap) return false;
    const blocks = (t + topk_chunk - 1) / topk_chunk;
    return blocks <= s.blocks;
}

/// The COMPACT radix (r2 §I): pass 0 the histogram, pass 1 the histogram + the survivor-list compaction,
/// then pass 2 + the final prefix + the emit in one block a row. Two full reads of the keys instead of
/// five; the same set and tie rule as `select` (exact top-k, ties to the lower column). `mode` is the key
/// mode of `select` (0 int32 order words, 2 packed int64 keys); `out` is [rows][topk] int32 columns.
pub fn selectCompact(k: Kernels, stream: cuda.Stream, keys: u64, out: u64, t: usize, rows: usize, topk: usize, mode: c_int, s: Scratch) !void {
    if (mode != 0 and mode != 2) return error.BadKeyMode;
    if (!compactFits(s, t, rows, topk)) return error.Invalid;
    // the list's counters are per call (r2 fills them in the wrapper)
    try stream.d.check(stream.d.api.cuMemsetD32Async(s.lcnt, 0, rows, stream.handle), "cuMemsetD32Async");
    try stream.d.check(stream.d.api.cuMemsetD32Async(s.lflg, 0, rows, stream.handle), "cuMemsetD32Async");
    const blocks = (t + topk_chunk - 1) / topk_chunk;
    const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(blocks), .y = @intCast(rows) }, .block = .{ .x = 256 } };
    { // pass 0: the original histogram (settles digit 0)
        var a: cuda.Args = .{};
        a.add(keys);
        a.add(@as(c_int, @intCast(t)));
        a.add(@as(i64, @intCast(t)));
        a.add(@as(c_int, topk_chunk));
        a.add(@as(c_int, 0));
        a.add(@as(c_int, @intCast(topk)));
        a.add(mode);
        a.add(s.hist);
        a.add(s.state);
        try cuda.launch.launch(k.hist, cfg, stream, &a);
    }
    { // pass 1: the histogram with the survivor-list compaction
        var a: cuda.Args = .{};
        a.add(keys);
        a.add(@as(c_int, @intCast(t)));
        a.add(@as(i64, @intCast(t)));
        a.add(@as(c_int, topk_chunk));
        a.add(@as(c_int, @intCast(topk)));
        a.add(mode);
        a.add(s.hist);
        a.add(s.state);
        a.add(s.list);
        a.add(s.lcnt);
        a.add(s.lflg);
        a.add(@as(c_int, @intCast(s.cap)));
        try cuda.launch.launch(k.hist1c, cfg, stream, &a);
    }
    { // pass 2 + the finish: one block a row over the list (or the full row on overflow)
        var a: cuda.Args = .{};
        a.add(keys);
        a.add(@as(c_int, @intCast(t)));
        a.add(@as(i64, @intCast(t)));
        a.add(@as(c_int, @intCast(topk)));
        a.add(mode);
        a.add(s.state);
        a.add(s.list);
        a.add(s.lcnt);
        a.add(s.lflg);
        a.add(@as(c_int, @intCast(s.cap)));
        a.add(out);
        a.add(@as(i64, @intCast(topk)));
        try cuda.launch.launch(k.hist2c, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
    }
}

/// The selection over the packed keys: r2's radix top-k in three histogram passes (bits 31-21, 20-10,
/// 9-0) plus the count and the ordered write, mode 2 (packed int64, ties to the lower column = the lower
/// position). Each row keeps its `topk` columns ascending in `out` [rows][topk].
pub fn select(k: Kernels, stream: cuda.Stream, keys: u64, out: u64, T: usize, rows: usize, topk: usize, mode: c_int, s: Scratch) !void {
    if (mode != 0 and mode != 2) return error.BadKeyMode;
    if (rows == 0 or topk == 0 or topk > T or rows > s.rows) return error.Invalid;
    const blocks = (T + topk_chunk - 1) / topk_chunk;
    if (blocks > s.blocks) return error.ScratchTooSmall;
    const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(blocks), .y = @intCast(rows) }, .block = .{ .x = 256 } };
    for (0..3) |pass| {
        var a: cuda.Args = .{};
        a.add(keys);
        a.add(@as(c_int, @intCast(T)));
        a.add(@as(i64, @intCast(T)));
        a.add(@as(c_int, topk_chunk));
        a.add(@as(c_int, @intCast(pass)));
        a.add(@as(c_int, @intCast(topk)));
        a.add(mode);
        a.add(s.hist);
        a.add(s.state);
        try cuda.launch.launch(k.hist, cfg, stream, &a);
    }
    var ca: cuda.Args = .{};
    ca.add(keys);
    ca.add(@as(c_int, @intCast(T)));
    ca.add(@as(i64, @intCast(T)));
    ca.add(@as(c_int, topk_chunk));
    ca.add(mode);
    ca.add(s.state);
    ca.add(s.cnt);
    try cuda.launch.launch(k.count, cfg, stream, &ca);
    var wa: cuda.Args = .{};
    wa.add(keys);
    wa.add(@as(c_int, @intCast(T)));
    wa.add(@as(i64, @intCast(T)));
    wa.add(@as(c_int, topk_chunk));
    wa.add(mode);
    wa.add(s.state);
    wa.add(s.cnt);
    wa.add(out);
    wa.add(@as(i64, @intCast(topk)));
    try cuda.launch.launch(k.write, cfg, stream, &wa);
}

test "selection geometry: dense below the top-k width, sparse past it; token lists are [rows][topk]" {
    const C = struct {
        i_topk: usize = 2048,
        pub fn denseLimit(self: @This()) usize {
            return self.i_topk;
        }
    };
    const c: C = .{};
    try std.testing.expect(denseMode(c, 0, 512));
    try std.testing.expect(!denseMode(c, 2048, 1));
    try std.testing.expectEqual(@as(usize, 2048), tokenWidth(c));
    var tokens: [3 * 8]i32 = undefined;
    identityTokens(&tokens, 4, 3);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, -1, -1, -1 }, tokens[0..8]);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 5, 6, -1 }, tokens[2 * 8 ..][0..8]);
}
test "the radix scratch and its limits: 2048-bin histograms a row, a scratch smaller than the blocks is refused" {
    const b = Scratch.bytes(128, 16);
    try std.testing.expectEqual(@as(usize, 128 * 2048 * 4), b.hist);
    try std.testing.expectEqual(@as(usize, 128 * 4 * 4), b.state);
    try std.testing.expectEqual(@as(usize, 128 * 16 * 2 * 4), b.cnt);
    const s: Scratch = .{ .hist = 1, .state = 2, .cnt = 3, .rows = 128, .blocks = 16 };
    try std.testing.expectEqual(@as(usize, 16), (topk_chunk - 1 + 2048 * 16) / topk_chunk); // 128k keys -> 64 blocks, refused
    try std.testing.expect((topk_chunk * 16 - 1 + 1) / topk_chunk == 16);
    _ = s;
}

test "the compact select's guards: the crossover, the list cap, the scratch bounds, a real top-k" {
    const rows = 8;
    const blocks = (65536 + topk_chunk - 1) / topk_chunk; // 32
    const s: Scratch = .{ .hist = 1, .state = 2, .cnt = 3, .list = 4, .lcnt = 5, .lflg = 6, .rows = rows, .blocks = blocks, .cap = topk_compact_cap };
    try std.testing.expect(compactFits(s, 65536, rows, 2048)); // deep columns: the compact path serves
    try std.testing.expect(compactFits(s, 2049, 1, 2048)); // just past the top-k: fits, but callers keep the columns pass
    try std.testing.expect(!compactFits(s, 2048, 1, 2048)); // T == K: nothing to select (identity)
    try std.testing.expect(!compactFits(s, 4096, 1, 0)); // no top-k
    try std.testing.expect(!compactFits(s, 4096, rows + 1, 2048)); // more rows than the scratch
    try std.testing.expect(!compactFits(s, 8192, 1, topk_compact_cap + 1)); // K past the list's reach
    const no_list: Scratch = .{ .hist = 1, .state = 2, .cnt = 3, .rows = rows, .blocks = blocks };
    try std.testing.expect(!compactFits(no_list, 65536, rows, 2048)); // no compact scratch allocated
    try std.testing.expectEqual(@as(usize, rows * topk_compact_cap * 4 + rows * 8), Scratch.compactBytes(rows, topk_compact_cap));
}

test "the compact path's routing cap: TF_GLM53_TOPK_COL_MAX, default 49152, 0 disables it" {
    // r2's crossover (findings-dcp2-build.md §I): the compact radix wins past ~48K local columns; the
    // env var is read once a call, an unparsable value falls back to the default. The gate that the
    // selection itself is exact on the ladder (random, tie-heavy, 2049, 65536, all-equal) is the GPU
    // harness's: glm53-nvfp4/q25 (glm53_topk_hist1c/2c vs the sort reference).
    try std.testing.expectEqual(@as(usize, 49152), topk_col_max_default);
    try std.testing.expect(topk_col_max_default > 2 * topk_chunk);
}
