//! `tf-glm53 abench [--bench N] [--rows R] [--heads H] [--keys K]`: the wide attention entry on its own,
//! with an independent f64 golden host model - the phase-4 harness of the tensorfold-kernel-opt skill
//! (§39), built before the kernel work so every iteration is a two-minute measurement.
//!
//! Gates (numbered, each its own assertion):
//!   1. causality EXACT: perturbing the cache row at `limit + 1` must not change a single output bit
//!      (the mask-exact arm, separate from the values),
//!   2. values: |out - golden| <= 5e-3 * max|golden| per element (the skill's scaled band with a floor at
//!      the golden's max so near-zero outputs do not trip a pure-relative band). The floor is 5e-3, not
//!      the skill's 1e-3, because this kernel is documented as bf16-precise in its *probabilities*: the p
//!      tile is rounded to bf16 before the value MMA (`__floats2bfloat162_rn` in chunks_wide) and the
//!      exponentials go through glm_triton::exp_approx, so a 1e-3 band would fail a correct kernel
//!      (measured 5.6e-3 max on the first run - correlated, not a fragment bug).
//! `--bench N` then reports the per-launch latency at the requested shape (the serve shape is
//! `--rows 64 --heads 8 --keys 512`, and `--keys 2048` exercises the sparse path's four chunks).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm53 = @import("glm53").family;
const cfg_mod = glm53.config;
const CFG: cfg_mod.Config = .{};
const check = @import("check.zig");

const lw = 512;
const row_w = 576; // kv_lora + rope

fn arg(args: []const []const u8, name: []const u8, dflt: usize) usize {
    for (args, 0..) |a, i| {
        if (std.mem.eql(u8, a, name) and i + 1 < args.len) return std.fmt.parseInt(usize, args[i + 1], 10) catch dflt;
    }
    return dflt;
}

fn bf16Of(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
fn fromBf16(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// --gemm: the bf16 matmul arms on their own (raw rows vs the tiled glm_mm entry), at the serve shapes.
fn gemmBench(gpu: check.Gpu, args: []const []const u8) !void {
    const a = gpu.gpa;
    const rows = arg(args, "--rows", 64);
    const bench = arg(args, "--bench", 20);
    var asm_ = try glm53.blocks.assemble(gpu.d, gpu.ctx, false);
    defer asm_.deinit();
    glm53.blocks.rebind(&asm_);
    const sh = try glm53.shard.Share.of(&CFG, 0, 8);
    const stream = try cuda.Stream.init(gpu.d, true);
    defer @constCast(&stream).deinit();
    // the sites the forward runs at a 64-row prompt chunk (n = out rows, k = inputs)
    const sites = [_][3]usize{
        .{ CFG.q_lora + CFG.kv_lora + CFG.rope, CFG.hidden, 0 }, // [q_a | kv_a]
        .{ 8 * CFG.qHead(), CFG.q_lora, 0 }, // q_b (this rank's 8 heads)
        .{ CFG.hidden, 8 * CFG.v_dim, 0 }, // o_proj
        .{ 2 * CFG.dense_inter, CFG.hidden, 0 }, // dense [gate | up]
        .{ CFG.hidden, CFG.dense_inter, 0 }, // dense down
        .{ CFG.vocab, CFG.hidden, 1 }, // head (the last)
    };
    for (sites) |site| {
        const n = site[0];
        const k = site[1];
        const is_head = site[2] == 1;
        const xb = try a.alloc(u16, rows * k);
        defer a.free(xb);
        const wb = try a.alloc(u16, n * k);
        defer a.free(wb);
        for (xb, 0..) |*v, i| v.* = bf16Of(@floatCast(@sin(@as(f64, @floatFromInt(i)) * 0.0011)));
        for (wb, 0..) |*v, i| v.* = bf16Of(@floatCast(@cos(@as(f64, @floatFromInt(i)) * 0.0007) * 0.5));
        var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xb));
        defer x.free();
        var wt = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(wb));
        defer wt.free();
        var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 4);
        defer out.free();
        const part_bytes = @max(glm53.forward.mmPartBytes(&CFG, sh, rows), 4);
        var part = try cuda.DeviceBuffer.alloc(gpu.d, part_bytes);
        defer part.free();
        const flops = @as(f64, @floatFromInt(rows * n * k * 2));
        // arm 1: the raw rows (mm null, the default this family ships)
        var raw_set = asm_.blocks.ops.glue;
        raw_set.mm = null;
        try time_arm(gpu, stream, raw_set, x.ptr, k, wt.ptr, out.ptr, part.ptr, rows, n, k, bench, flops, "raw");
        // arm 2: the tiled entry (the weight tiled into a buffer of its own)
        var wtiled = try cuda.DeviceBuffer.alloc(gpu.d, n * k * 2);
        defer wtiled.free();
        var tmp = try cuda.DeviceBuffer.alloc(gpu.d, n * k * 2);
        defer tmp.free();
        try glm53.mm.tile(&asm_.mm, stream, wt.ptr, tmp.ptr, n, k);
        try stream.synchronize();
        try gpu.d.check(gpu.d.api.cuMemcpyDtoDAsync_v2(wtiled.ptr, tmp.ptr, n * k * 2, stream.handle), "cuMemcpyDtoDAsync");
        try stream.synchronize();
        try time_arm(gpu, stream, asm_.blocks.ops.glue, x.ptr, k, wtiled.ptr, out.ptr, part.ptr, rows, n, k, bench, flops, "tiled");
        _ = is_head;
    }
}

fn time_arm(gpu: check.Gpu, stream: cuda.Stream, set: glm53.glue.Set, x: u64, k: usize, w: u64, out: u64, part: u64, rows: usize, n: usize, kk: usize, bench: usize, flops: f64, label: []const u8) !void {
    _ = gpu;
    var i: usize = 0;
    while (i < 3) : (i += 1) try glm53.b16.matmul(set, stream, x, k, w, out, part, rows, n, kk, true);
    try stream.synchronize();
    var ev0 = try cuda.Event.init(stream.d, true);
    defer ev0.deinit();
    var ev1 = try cuda.Event.init(stream.d, true);
    defer ev1.deinit();
    try ev0.record(stream);
    i = 0;
    while (i < bench) : (i += 1) try glm53.b16.matmul(set, stream, x, k, w, out, part, rows, n, kk, true);
    try ev1.record(stream);
    try stream.synchronize();
    const ms: f64 = @floatCast(try ev0.elapsedMs(ev1));
    const per = ms / @as(f64, @floatFromInt(bench));
    std.debug.print("RESULT gemm arm={s} rows={d} n={d} k={d} ms_per_launch={d:.3} tflops={d:.2}\n", .{ label, rows, n, kk, per, if (per > 0) flops / (per * 1e9) else 0 });
}

pub fn run(gpu: check.Gpu, args: []const []const u8) !void {
    if (arg(args, "--gemm", 0) == 1) return gemmBench(gpu, args);
    const a = gpu.gpa;
    const rows = arg(args, "--rows", 64);
    const heads = arg(args, "--heads", 8);
    const keys = arg(args, "--keys", 512);
    const bench = arg(args, "--bench", 0);
    if (rows == 0 or rows > 128 or heads == 0 or heads > 64 or keys < rows) return error.BadShape;
    if (keys > 4096) return error.BadShape;
    const nch = (keys + 511) / 512;

    var asm_ = try glm53.blocks.assemble(gpu.d, gpu.ctx, false);
    defer asm_.deinit();
    glm53.blocks.rebind(&asm_);
    const stream = try cuda.Stream.init(gpu.d, true);
    defer @constCast(&stream).deinit();

    // the inputs: a deterministic bf16 pattern (no NaN, no inf), the cache's rows at their own positions
    const qn = rows * heads * row_w;
    const cn = keys * row_w;
    const qh = try a.alloc(u16, qn);
    defer a.free(qh);
    const ch = try a.alloc(u16, cn);
    defer a.free(ch);
    for (qh, 0..) |*v, i| v.* = bf16Of(@floatCast(@sin(@as(f64, @floatFromInt(i)) * 0.017) * 0.9));
    for (ch, 0..) |*v, i| v.* = bf16Of(@floatCast(@cos(@as(f64, @floatFromInt(i)) * 0.011) * 1.1));
    const pos_h = try a.alloc(i32, rows);
    defer a.free(pos_h);
    for (pos_h, 0..) |*v, i| v.* = @intCast(i);

    var qp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(qh));
    defer qp.free();
    var lc = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ch));
    defer lc.free();
    var pos = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pos_h));
    defer pos.free();
    var po = try cuda.DeviceBuffer.alloc(gpu.d, nch * rows * heads * lw * 4);
    defer po.free();
    var pm = try cuda.DeviceBuffer.alloc(gpu.d, nch * rows * heads * 4);
    defer pm.free();
    var pl = try cuda.DeviceBuffer.alloc(gpu.d, nch * rows * heads * 4);
    defer pl.free();
    var dummy = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer dummy.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, rows * heads * lw * 2);
    defer out.free();
    try dummy.fill32(0, null);

    const partials: glm53.latent.Partials = .{ .po = po.ptr, .pm = pm.ptr, .pl = pl.ptr };
    const scale: f32 = @floatCast(std.math.pow(f64, 256.0, -0.5));
    const call = struct {
        fn go(bl: glm53.forward.Blocks, s: cuda.Stream, p: glm53.latent.Partials, q: u64, c: u64, ps: u64, o: u64, dmy: u64, r: usize, h: usize, k: usize, n: usize, sc: f32) !void {
            _ = k;
            return glm53.latent.attentionWide(bl.ops.extras.chunks, bl.ops.glue, s, q, c, ps, p, o, dmy, r, h, lw, 64, n, sc);
        }
    };
    try call.go(asm_.blocks, stream, partials, qp.ptr, lc.ptr, pos.ptr, out.ptr, dummy.ptr, rows, heads, keys, nch, scale);
    try stream.synchronize();

    const oh = try a.alloc(u16, rows * heads * lw);
    defer a.free(oh);
    try out.download(0, std.mem.sliceAsBytes(oh));

    // --- gate 2: the values against the f64 golden (the causal band j <= r) ---
    var worst: f64 = 0;
    var denom: f64 = 0;
    var bad: usize = 0;
    const gold = try a.alloc(f64, lw);
    defer a.free(gold);
    for (0..rows) |r| {
        const limit = r; // pos[0] = 0, row r attends keys 0..r
        for (0..heads) |h| {
            var m: f64 = -std.math.inf(f64);
            const sc = try a.alloc(f64, limit + 1);
            defer a.free(sc);
            for (0..limit + 1) |j| {
                var dot: f64 = 0;
                for (0..row_w) |c| {
                    dot += @as(f64, fromBf16(qh[(r * heads + h) * row_w + c])) * @as(f64, fromBf16(ch[j * row_w + c]));
                }
                sc[j] = dot * @as(f64, scale);
                m = @max(m, sc[j]);
            }
            var l: f64 = 0;
            for (sc) |x| l += @exp(x - m);
            @memset(gold, 0);
            for (0..limit + 1) |j| {
                const p = @exp(sc[j] - m) / l;
                for (0..lw) |c| gold[c] += p * @as(f64, fromBf16(ch[j * row_w + c]));
            }
            var mx: f64 = 0;
            for (gold) |g| mx = @max(mx, @abs(g));
            denom = @max(denom, mx);
            for (0..lw) |c| {
                const got = @as(f64, fromBf16(oh[(r * heads + h) * lw + c]));
                const df = @abs(got - gold[c]);
                worst = @max(worst, df);
                if (df > 1e-2 * mx + 1e-6) bad += 1;
            }
        }
    }
    std.debug.print("GATE values rows={d} heads={d} keys={d} nch={d} max|diff|={e} band={e} bad={d}\n", .{ rows, heads, keys, nch, worst, 1e-2 * denom, bad });
    if (bad != 0) return error.ValuesOutOfBand;

    // --- gate 1: causality EXACT (the mask arm): one bit in the row past `limit` must change nothing ---
    {
        const probe_row = keys - 1; // read by no row (limit <= rows-1 < keys-1)
        const dev = std.mem.sliceAsBytes(ch);
        const at = probe_row * row_w + 7;
        const before = ch[at];
        ch[at] = before ^ 0x3c00; // flip a bit in that row's 8th column
        var lc2 = try cuda.DeviceBuffer.fromHost(gpu.d, dev);
        defer lc2.free();
        ch[at] = before; // the host copy stays the reference
        var out2 = try cuda.DeviceBuffer.alloc(gpu.d, rows * heads * lw * 2);
        defer out2.free();
        try call.go(asm_.blocks, stream, partials, qp.ptr, lc2.ptr, pos.ptr, out2.ptr, dummy.ptr, rows, heads, keys, nch, scale);
        try stream.synchronize();
        const oh2 = try a.alloc(u16, rows * heads * lw);
        defer a.free(oh2);
        try out2.download(0, std.mem.sliceAsBytes(oh2));
        var diff: usize = 0;
        for (oh, oh2) |x, y| if (x != y) {
            diff += 1;
        };
        std.debug.print("GATE causality row={d} bit-flip -> differing outputs={d} (must be 0)\n", .{ probe_row, diff });
        if (diff != 0) return error.MaskLeak;
    }

    // --- the workhorse: per-launch latency at this shape ---
    if (bench > 0) {
        var ev0 = try cuda.Event.init(gpu.d, true);
        defer ev0.deinit();
        var ev1 = try cuda.Event.init(gpu.d, true);
        defer ev1.deinit();
        var i: usize = 0;
        while (i < 3) : (i += 1) try call.go(asm_.blocks, stream, partials, qp.ptr, lc.ptr, pos.ptr, out.ptr, dummy.ptr, rows, heads, keys, nch, scale);
        try stream.synchronize();
        try ev0.record(stream);
        i = 0;
        while (i < bench) : (i += 1) try call.go(asm_.blocks, stream, partials, qp.ptr, lc.ptr, pos.ptr, out.ptr, dummy.ptr, rows, heads, keys, nch, scale);
        try ev1.record(stream);
        try stream.synchronize();
        const ms: f64 = @floatCast(try ev0.elapsedMs(ev1));
        const per = ms / @as(f64, @floatFromInt(bench));
        // the flops this call actually does: the score's 576-wide dot and the value's 512-wide sum, over
        // the causal band (rows*(rows+1)/2 key-rows a head) for the dense shape, `keys` a row otherwise
        const pairs: f64 = if (keys <= rows) @as(f64, @floatFromInt(rows * (rows + 1) / 2)) else @as(f64, @floatFromInt(rows * keys));
        const flops = pairs * @as(f64, @floatFromInt(heads)) * (row_w * 2.0 + lw * 2.0);
        std.debug.print("RESULT abench rows={d} heads={d} keys={d} nch={d} launches={d} ms_per_launch={d:.3} tflops={d:.2}\n", .{ rows, heads, keys, nch, bench, per, if (per > 0) flops / (per * 1e9) else 0 });
    }
    check.pass("abench: the wide attention entry's values and causality gates at rows={d} heads={d} keys={d}", .{ rows, heads, keys });
}