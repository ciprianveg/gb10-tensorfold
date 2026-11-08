//! Q1/Q3 — the GLM-5.3 NVFP4-CSF load smoke on one rank (the engine bridge the queue was blocked on).
//!
//! `tf-cuda-test glm53-load [DIR] [RANK] [WORLD] [LAYERS]`: reads the pack's config, takes this rank's
//! shard, opens the checkpoint, loads the kernels the loader launches (`tf_pack_dense` from
//! affine4_pack, the NVFP4 quantizers from glm_quant) and runs `weights.load` for real on the device.
//! It uploads nothing synthetic: every tensor is read out of the 84-shard checkpoint, the same path the
//! serve will use.
//!
//! LAYERS > 0 trims the model to that many layers and drops the MTP head, which is Q1's three-dense-layer
//! smoke (layer 0-2) and its first MoE layer (LAYERS=4); LAYERS=0 is the whole pack (Q3's TP8 load).
//!
//! Output: one RESULT line with the device bytes, the load time, the checkpoint bytes and the per-share
//! spans, then PASS. Nothing is inferred: a missing tensor, a refused shape or an empty GPU image fails
//! the command.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm53 = @import("glm53").family;
const check = @import("check.zig");

/// STEP 1's gate: assemble the block set from the pack's images and print its census.
pub fn blocks(gpu: check.Gpu, args: []const []const u8) !void {
    _ = args;
    var a = try glm53.blocks.assemble(gpu.d, gpu.ctx, false);
    defer a.deinit();
    glm53.blocks.rebind(&a);
    glm53.blocks.census(&a);
    check.pass("glm53-blocks: the block set assembled", .{});
}

pub fn run(gpu: check.Gpu, args: []const []const u8) !void {
    const a = gpu.gpa;
    const dir: []const u8 = if (args.len > 0) args[0] else blk: {
        const env = std.c.getenv("TF_GLM53_NVFP4_DIR") orelse return error.NoPackDir;
        break :blk std.mem.span(env);
    };
    const rank: usize = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 0;
    const world: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 8;
    const trim: u32 = if (args.len > 3) try std.fmt.parseInt(u32, args[3], 10) else 0;

    // -- the config, exactly as the serve reads it
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const json = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 20));
    defer a.free(json);
    var cfg = try glm53.config.parse(a, json);
    const full_layers = cfg.layers;
    if (trim != 0) {
        if (trim > cfg.layers) return error.TooFewLayers;
        cfg.layers = trim;
        cfg.mtp = 0; // the MTP head loads with the whole model (Q4's serve), not with the trim
    }

    const share = try glm53.shard.Share.of(&cfg, rank, world);

    // -- the kernels the load launches
    const m_pack = try cuda.Module.load(gpu.d, cuda.kernels.affine4_pack);
    const m_quant = try cuda.Module.load(gpu.d, cuda.kernels.glm_quant);
    const qk = try glm53.quant.Kernels.load(m_quant);
    const tools: glm53.weights.Tools = .{ .pack_dense = try m_pack.function("tf_pack_dense"), .quant = &qk };

    const before = try gpu.ctx.memInfo();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();

    // the shards and the input-scale sidecar live in <pack>/tensors (config.json at the pack root)
    const shards = try std.fs.path.join(a, &.{ dir, "tensors" });
    defer a.free(shards);
    var ck = try core.Checkpoint.openModel(a, gpu.io, shards);
    defer ck.close();

    // PR-1050 (b): the process-local allocator receipt at load begin (TF_CUDA_MEMORY_RECEIPTS=1)
    cuda.memory_receipt.emit(@intCast(rank), .load_begin, true);
    const t0 = check.now(gpu.io);
    var w = try glm53.weights.load(a, gpu.d, stream, tools, &ck, &cfg, share);
    defer w.deinit();
    try stream.synchronize();
    const dt_ns = check.now(gpu.io) - t0;
    cuda.memory_receipt.emit(@intCast(rank), .load_end, false);

    const after = try gpu.ctx.memInfo();
    var bytes: usize = 0;
    var largest: usize = 0;
    for (w.buffers.items) |b| {
        bytes += b.len;
        if (b.len > largest) largest = b.len;
    }

    // STEP 1's join: every module plan.walk names resolved to a loaded tensor (or its packed .words)
    const cen = try glm53.weights.Census.join(&w, &cfg);

    std.debug.print(
        "RESULT glm53-load dir={s} rank={d}/{d} layers={d}{s} tensors={d} device_bytes={d} largest={d} " ++
            "load_ms={d} mem_free_before={d} mem_free_after={d} vocab=[{d},{d}) moe=[{d},{d}) mla=[{d},{d})\n",
        .{
            dir, rank, world, cfg.layers, if (trim == 0) " (full)" else " (trimmed)",
            w.buffers.items.len, bytes, largest, @divTrunc(dt_ns, 1_000_000),
            before.free, after.free,
            share.vocab.lo, share.vocab.hi, share.moe.lo, share.moe.hi, share.mla.lo, share.mla.hi,
        },
    );
    std.debug.print(
        "RESULT glm53-census expected={d} found={d} missing={d} first_missing={s}\n",
        .{ cen.expected, cen.found, cen.missing, std.mem.sliceTo(&cen.first_missing, 0) },
    );
    if (cen.missing != 0) return error.CensusMissing;
    if (trim == 0 and w.layers.len != full_layers + 1) return error.LayerCountMismatch;
    check.pass("glm53-load: {d} tensors, {d} device bytes, {d} ms", .{ w.buffers.items.len, bytes, @divTrunc(dt_ns, 1_000_000) });
}