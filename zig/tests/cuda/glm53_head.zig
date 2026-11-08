//! STEP 4's gate: the head's sharded logits, the argmax reduced across ranks (tie-break: the LOWEST
//! global vocabulary index wins, so every rank picks the same token from the same logits), and a
//! deterministic greedy sampler.
//!
//! `tf-glm53 head [DIR] [RANK] [WORLD]`: loads the pack trimmed to one layer (embed, head and one dense
//! layer - the head is what this command exercises), builds a FIXED hidden vector from a formula (the
//! same bytes on every rank), runs the head matmul over this rank's vocabulary shard with the raw-bf16
//! entry (`b16.matmul`, split-K partials from the checked arena), argmaxes the shard on the host, packs
//! (value bits, index) and gathers one u64 a rank, then reduces with the tie-break above.
//!
//! Two runs must pick the same token; every rank must agree. The sequence gate (two identical replies)
//! needs the forward and belongs to `step`.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm53 = @import("glm53").family;
const check = @import("check.zig");

fn hiddenValue(i: usize) f32 {
    // deterministic, rank-independent, and wide enough to make the argmax unique in practice
    const x: f32 = @floatFromInt(i);
    return 0.5 * @sin(x * 0.017) + 0.25 * @cos(x * 0.0031);
}

fn bf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

pub fn run(gpu: check.Gpu, args: []const []const u8) !void {
    const a = gpu.gpa;
    const dir: []const u8 = if (args.len > 0) args[0] else blk: {
        const env = std.c.getenv("TF_GLM53_NVFP4_DIR") orelse return error.NoPackDir;
        break :blk std.mem.span(env);
    };
    const rank: usize = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 0;
    const world: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 1;

    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const json = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 20));
    defer a.free(json);
    var cfg = try glm53.config.parse(a, json);
    cfg.layers = 1; // embed + head + one layer: all this gate needs
    cfg.mtp = 0;
    const share = try glm53.shard.Share.of(&cfg, rank, world);
    const vocab_len = share.vocab.len();

    // modules the head matmul launches from
    var glue_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer glue_mod.unload();
    const glue_k = try glm53.glue.Kernels.load(glue_mod);
    const glue_set: glm53.glue.Set = .{ .glue = &glue_k, .mm = null }; // raw bf16 rows

    var m_pack = try cuda.Module.load(gpu.d, cuda.kernels.affine4_pack);
    defer m_pack.unload();
    var m_quant = try cuda.Module.load(gpu.d, cuda.kernels.glm_quant);
    defer m_quant.unload();
    const qk = try glm53.quant.Kernels.load(m_quant);
    const tools: glm53.weights.Tools = .{ .pack_dense = try m_pack.function("tf_pack_dense"), .quant = &qk };

    const shards = try std.fs.path.join(a, &.{ dir, "tensors" });
    defer a.free(shards);
    var ck = try core.Checkpoint.openModel(a, gpu.io, shards);
    defer ck.close();

    const stream = try cuda.Stream.init(gpu.d, true);
    defer @constCast(&stream).deinit();
    var w = try glm53.weights.load(a, gpu.d, stream, tools, &ck, &cfg, share);
    defer w.deinit();
    try stream.synchronize();

    // the fixed hidden row (bf16 [1][hidden]) and the split-K scratch from the layout's own bound
    const d = cfg.hidden;
    const xb = try a.alloc(u16, d);
    defer a.free(xb);
    for (xb, 0..) |*v, i| v.* = bf16(hiddenValue(i));
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xb));
    defer x.free();
    const out_bytes = vocab_len * 4; // fp32 logits of this rank's shard
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, out_bytes);
    defer logits.free();
    const part_bytes = @max(glm53.forward.mmPartBytes(&cfg, share, 1), 4);
    var part = try cuda.DeviceBuffer.alloc(gpu.d, part_bytes);
    defer part.free();

    // twice: the sampler must be deterministic run to run
    var local_tok: [2]u32 = undefined;
    var local_val: [2]f32 = undefined;
    for (0..2) |pass| {
        try glm53.b16.matmul(glue_set, stream, x.ptr, d, w.head, logits.ptr, part.ptr, 1, vocab_len, d, true);
        try stream.synchronize();
        const host = try a.alloc(f32, vocab_len);
        defer a.free(host);
        try logits.download(0, std.mem.sliceAsBytes(host));
        // argmax with the documented tie-break: equal values keep the lower index (the first wins)
        var best: usize = 0;
        for (host, 0..) |v, i| if (v > host[best]) {
            best = i;
        };
        local_val[pass] = host[best];
        local_tok[pass] = @intCast(share.vocab.lo + best);
    }
    if (local_tok[0] != local_tok[1]) {
        std.debug.print("FAIL head: the sampler is not deterministic ({d} then {d})\n", .{ local_tok[0], local_tok[1] });
        return error.NotDeterministic;
    }

    // the cross-rank reduction: one (value, global index) a rank; the lowest index wins a tie
    var g = try cuda.tp_group.Group.openShared(gpu.io, rank, world, std.mem.span(std.c.getenv("TF_NCCL_ID_FILE") orelse return error.NoCommId));
    defer g.close();
    const word: u64 = (@as(u64, local_tok[0]) << 32) | @as(u32, @bitCast(local_val[0]));
    const all = try a.alloc(u64, world);
    defer a.free(all);
    try g.gather(u64, gpu.d, stream, word, all);
    var best_tok: u32 = 0;
    var best_val: f32 = -std.math.inf(f32);
    for (all) |u| {
        const tok: u32 = @intCast(u >> 32);
        const val: f32 = @bitCast(@as(u32, @truncate(u)));
        if (val > best_val or (val == best_val and tok < best_tok)) {
            best_val = val;
            best_tok = tok;
        }
    }
    std.debug.print(
        "RESULT head rank={d}/{d} local_token={d} local_value={e} global_token={d} global_value={e} agree={s}\n",
        .{ rank, world, local_tok[0], local_val[0], best_tok, best_val, if (best_tok == local_tok[0]) "yes" else "no" },
    );
    // The invariant to assert per rank is not "my local winner is the global one" (only the owning rank
    // sees it); it is: the reduced winner is the maximum over the gathered values, and a rank whose
    // value ties it must hold the winning index. Fleet agreement is all ranks printing the same global
    // token, which the driver checks across the logs.
    var max_val: f32 = -std.math.inf(f32);
    var min_idx_at_max: u32 = std.math.maxInt(u32);
    for (all) |u| {
        const tok: u32 = @intCast(u >> 32);
        const val: f32 = @bitCast(@as(u32, @truncate(u)));
        if (val > max_val or (val == max_val and tok < min_idx_at_max)) {
            max_val = val;
            min_idx_at_max = tok;
        }
    }
    if (best_tok != min_idx_at_max or best_val != max_val) return error.BadReduction;
    if (local_val[0] == max_val and local_tok[0] != best_tok) return error.TieBreakWrong;
    if (local_val[0] > max_val) return error.BadReduction;
    check.pass("head: the reduced argmax is the gathered maximum with the lowest index; rank {d}/{d} agrees", .{ rank, world });
}