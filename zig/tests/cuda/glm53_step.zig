//! STEP 5 (world 1 first): the forward loop on real weights - prefill chunks, one decode step, and a
//! digest per stage, so a later divergence can be bisected.
//!
//! `tf-glm53 step [DIR] [LAYERS] [IDS]`: loads the pack trimmed to LAYERS (a trimmed model is not the
//! served one; this command proves the loop runs the real weights, the real kernels and the real cache),
//! assembles the step-1 blocks, lays the step-2 arena, allocates the latent/index caches and runs
//! `forward.stack` (embed -> per-layer attn (latent attention + indexer selection) -> residual -> ffn ->
//! residual -> final norm), then `forward.head` for logits and a greedy token, then one `decodeStep` on
//! that token. Digests: the hidden state after the stack (sum/absmax over this rank's rows) and the
//! logits' top-k, printed so two runs and later stages can be compared.
//!
//! World 8 (the whole model) needs the TP peers wiring inside attn/ffn and the kv8 cache path; both are
//! named gaps, not silent ones (findings §28).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm53 = @import("glm53").family;
const check = @import("check.zig");

fn digest(host: []const u8, comptime T: type) struct { sum: f64, absmax: f64 } {
    const v = std.mem.bytesAsSlice(T, host);
    var sum: f64 = 0;
    var mx: f64 = 0;
    for (v) |x| {
        const f: f64 = if (T == u16)
            @as(f64, @floatCast(@as(f32, @bitCast(@as(u32, x) << 16)))) // bf16 bits
        else
            @floatCast(x);
        sum += f;
        mx = @max(mx, @abs(f));
    }
    return .{ .sum = sum, .absmax = mx };
}

/// The same read-out plus the non-finite count: a NaN weight and a NaN activation are different bugs, and
/// "nan" alone does not say which it is.
fn digestFin(host: []const u8, comptime T: type) struct { sum: f64, absmax: f64, bad: usize } {
    const v = std.mem.bytesAsSlice(T, host);
    var sum: f64 = 0;
    var mx: f64 = 0;
    var bad: usize = 0;
    for (v) |x| {
        const f: f64 = if (T == u16)
            @as(f64, @floatCast(@as(f32, @bitCast(@as(u32, x) << 16))))
        else
            @floatCast(x);
        if (!std.math.isFinite(f)) bad += 1;
        sum += f;
        mx = @max(mx, @abs(f));
    }
    return .{ .sum = sum, .absmax = mx, .bad = bad };
}

pub fn run(gpu: check.Gpu, args: []const []const u8) !void {
    const a = gpu.gpa;
    const dir: []const u8 = if (args.len > 0) args[0] else blk: {
        const env = std.c.getenv("TF_GLM53_NVFP4_DIR") orelse return error.NoPackDir;
        break :blk std.mem.span(env);
    };
    const layers: u32 = if (args.len > 1) try std.fmt.parseInt(u32, args[1], 10) else 3;
    const ids_text: []const u8 = if (args.len > 2) args[2] else "1,2,3,4";
    const rank: usize = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 0;
    const world: usize = if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else 1;

    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const json = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 20));
    defer a.free(json);
    var cfg = try glm53.config.parse(a, json);
    if (layers == 0 or layers > cfg.layers) return error.BadLayers;
    cfg.layers = layers;
    cfg.mtp = 0;
    cfg.run = layers; // the number of layers the loop walks
    // load N layers but walk only R: separates "the loaded weights are corrupt" from "layer k's math is"
    if (std.c.getenv("TF_GLM53_RUN")) |rz| {
        const r = try std.fmt.parseInt(u32, std.mem.span(rz), 10);
        if (r == 0 or r > cfg.run) return error.BadLayers;
        cfg.run = r;
    // the clamp is a model-family default in this port (10.0), not a pack field: a switch to isolate it
    if (std.c.getenv("TF_GLM53_SWIGLU_LIMIT")) |v| cfg.swiglu_limit = std.fmt.parseFloat(f32, std.mem.span(v)) catch cfg.swiglu_limit;
    }
    const share = try glm53.shard.Share.of(&cfg, rank, world);

    // prompt ids
    var ids_buf: [64]i32 = undefined;
    var n_ids: usize = 0;
    var it = std.mem.splitScalar(u8, ids_text, ',');
    while (it.next()) |tok| {
        if (n_ids == ids_buf.len) break;
        ids_buf[n_ids] = try std.fmt.parseInt(i32, std.mem.trim(u8, tok, " "), 10);
        n_ids += 1;
    }
    if (n_ids == 0) return error.NoIds;

    var asm_ = try glm53.blocks.assemble(gpu.d, gpu.ctx, false);
    defer asm_.deinit();
    glm53.blocks.rebind(&asm_);
    // the world's NCCL group (TF_NCCL_ID_FILE, §22 recipe); world 1 needs none
    var group: ?cuda.tp_group.Group = null;
    defer if (group) |*g| g.close();
    if (world > 1) {
        const idf = std.c.getenv("TF_NCCL_ID_FILE") orelse return error.NoCommId;
        group = try cuda.tp_group.Group.openShared(gpu.io, rank, world, std.mem.sliceTo(idf, 0));
    }
    asm_.blocks.tp = if (group) |*g| g else null;
    asm_.blocks.rank = rank;
    asm_.blocks.world = world;

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

    const t0 = check.now(gpu.io);
    var w = try glm53.weights.load(a, gpu.d, stream, tools, &ck, &cfg, share);
    defer w.deinit();
    try stream.synchronize();
    const load_ms = @divTrunc(check.now(gpu.io) - t0, 1_000_000);
    // THE TILED PATH (§38): the GLM engine's own glm_mm_tile16 over every linear b16.matmul reads, then
    // the tiled entry. Opt-in (TF_GLM53_TILED=1): it is the canonical layout and it is correct, but it
    // measured NO gain (the linears are not the bottleneck - the attention is) and it is not bit-identical
    // (the K-slice order differs), so the raw path stays the default the §36/§37 numbers were taken on.
    if (std.c.getenv("TF_GLM53_TILED") != null) {
        const t_tile = check.now(gpu.io);
        const n_tiled = try glm53.tile.tileLinears(&w, &asm_.mm, stream, gpu.d);
        try stream.synchronize();
        glm53.blocks.useTiled(&asm_);
        std.debug.print("TRACE tiled {d} linears in {d} ms\n", .{ n_tiled, @divTrunc(check.now(gpu.io) - t_tile, 1_000_000) });
    }
    std.debug.print("TRACE loaded (layers={d} run={d})\n", .{ cfg.layers, cfg.run });

    // TF_GLM53_WEIGHTS: every named weight buffer of the first layers, read back from the device, so a
    // corrupted upload (a NaN in a weight, an overrun from a later layer's load) is visible directly.
    if (std.c.getenv("TF_GLM53_WEIGHTS") != null) {
        for (w.named.items) |n| {
            const head: []const u8 = n.name;
            if (std.mem.startsWith(u8, head, "layers.")) {
                const dot = std.mem.indexOfScalarPos(u8, head, "layers.".len, '.') orelse continue;
                const li = std.fmt.parseInt(u32, head["layers.".len..dot], 10) catch continue;
                if (li > 2) continue;
            }
            std.debug.print("  WEIGHT {s} len={d}", .{ head, n.len });
            const b: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = n.ptr, .len = n.len, .borrowed = true };
            const host = try a.alloc(u8, n.len);
            defer a.free(host);
            try b.download(0, host);
            const t = if (n.len % 4 == 0 and std.mem.endsWith(u8, n.name, ".bias"))
                digestFin(host, f32)
            else
                digestFin(host, u16);
            std.debug.print(" sum={e} absmax={e} nonfinite={d}\n", .{ t.sum, t.absmax, t.bad });
        }
    }

    // the arena: the forward's scratch, the dense and routed blocks', one checked buffer
    const scores_t = n_ids + 1;
    const blocks_n = (scores_t + glm53.indexer.topk_chunk - 1) / glm53.indexer.topk_chunk;
    const layout = glm53.forward.scratchLayout(&cfg, share, n_ids, scores_t, blocks_n);
    var arena_buf = try cuda.DeviceBuffer.alloc(gpu.d, layout.total);
    defer arena_buf.free();
    var arena: glm53.forward.Arena = .{ .base = arena_buf.ptr, .size = layout.total };
    const placed = try glm53.forward.placeBlocks(&arena, layout);
    std.debug.print("TRACE arena total={d} used={d}\n", .{ layout.total, arena.used });

    // the caches: the bf16 latent cache and the fp8 index plane (the plain path at DCP1). The attention
    // reads whole 512-key chunks, so the capacity is rounded up to one: a shorter cache is an
    // out-of-bounds read (the illegal address this run first hit).
    const capacity = std.mem.alignForward(usize, n_ids + 8, 512);
    // one ring set a layer (a layer's history is its own keys, §36): the latent row is kv_lora + rope wide
    const cb = glm53.forward.Caches.ringBytes(&cfg, capacity, glm53.indexer.plane_stride, cfg.layers);
    var lc = try cuda.DeviceBuffer.alloc(gpu.d, cb.lc);
    defer lc.free();
    var ik = try cuda.DeviceBuffer.alloc(gpu.d, cb.ik);
    defer ik.free();
    var ig = try cuda.DeviceBuffer.alloc(gpu.d, cb.ig);
    defer ig.free();
    // The canonical GLM state allocates its arena ZEROED (`cuda_state.zig`: `a.buf.fill8(0, null)`) and its
    // caches likewise; this family's did not, and the difference is visible: without it the layer-0 digest
    // differs per rank (un-owned scratch read through a split-K partial). Conformance, not politeness.
    try arena_buf.fill8(0, null);
    try lc.fill8(0, null);
    try ik.fill8(0, null);
    try ig.fill8(0, null);
    const caches: glm53.forward.Caches = .{
        .lc = lc.ptr,
        .ik = ik.ptr,
        .ig = ig.ptr,
        .lw = cfg.kv_lora,
        .capacity = capacity,
        .layer_lc = capacity * (cfg.kv_lora + cfg.rope) * 2,
        .layer_ik = capacity * glm53.indexer.plane_stride,
        .layer_ig = capacity * cfg.i_heads * 2,
    };

    // the positions each row writes/reads: a DEVICE i32 array (the cache write and the attention take a
    // pointer, not a scalar) - passing 0 here was a null dereference
    var pos_host_buf: [64]i32 = undefined;
    for (pos_host_buf[0..n_ids], 0..) |*v, i| v.* = @intCast(i);
    var pos_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pos_host_buf[0..n_ids]));
    defer pos_dev.free();

    var ids = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ids_buf[0..n_ids]));
    defer ids.free();
    var x = try cuda.DeviceBuffer.alloc(gpu.d, n_ids * cfg.hidden * 2);
    defer x.free();
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, n_ids * share.vocab.len() * 4);
    defer logits.free();

    std.debug.print("TRACE caches lc={d} ik={d} ig={d}\n", .{ cb.lc, cb.ik, cb.ig });
    // the layer loop, spelled out here so every stage is a trace line and a digest
    try glm53.forward.embed(asm_.blocks, stream, &w, ids.ptr, n_ids, x.ptr);
    try stream.synchronize();
    std.debug.print("TRACE embed ok\n", .{});
    // TF_GLM53_STAGES: the coarse stage chain (embed / post-attn residual / post-ffn residual) for the
    // cross-path bisect against r2's driver (findings §51)
    const stages = std.c.getenv("TF_GLM53_STAGES") != null;
    var stage_buf: []u8 = &[_]u8{};
    if (stages) stage_buf = try a.alloc(u8, n_ids * cfg.hidden * 4);
    const stage_host: []u8 = stage_buf;
    defer if (stages) a.free(stage_host);
    const stageAt = struct {
        fn at(g: check.Gpu, s: cuda.Stream, buf: []u8, ptr: u64, len: usize, name: []const u8) !void {
            try s.synchronize();
            try g.d.check(g.d.api.cuCtxSynchronize(), "cuCtxSynchronize");
            try (cuda.DeviceBuffer{ .d = g.d, .ptr = ptr, .len = len, .borrowed = true }).download(0, buf[0..len]);
            const d = digest(buf[0..len], u16);
            std.debug.print("  STAGE {s} sum={e} absmax={e}\n", .{ name, d.sum, d.absmax });
        }
    };
    if (stages) try stageAt.at(gpu, stream, stage_host, x.ptr, n_ids * cfg.hidden * 2, "embed");
    // §71 identical-input isolation (harness-only): TF_GLM53_START=<layer> runs [start, layers) and
    // TF_GLM53_INJECT=<file> replaces the hidden state with another run's dump of that same width, so both
    // engines can be fed the SAME input at a deep layer and compared under the §59 band.
    var start: usize = 0;
    if (std.c.getenv("TF_GLM53_START")) |v| start = try std.fmt.parseInt(usize, std.mem.span(v), 10);
    if (start > cfg.run) return error.BadStart;
    if (std.c.getenv("TF_GLM53_INJECT")) |p| {
        const want = n_ids * cfg.hidden * 2;
        const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, std.mem.span(p), a, .unlimited);
        if (bytes.len != want) return error.BadInject;
        try x.upload(0, bytes);
        std.debug.print("INJECT {s} rows={d} hidden={d}\n", .{ std.mem.span(p), n_ids, cfg.hidden });
    }
    for (w.layers[start..cfg.run]) |*l| {
        const li = l.index;
        std.debug.print("TRACE layer {d} start (moe={s}) in_norm={x} post_norm={x} q_b={x} wk={x} wv={x} inv={x}\n", .{
            li, if (l.ffn == .moe) "yes" else "no", l.in_norm, l.post_norm, l.attn.q_b, l.attn.wk, l.attn.wv, w.inv,
        });
        const slot = glm53.forward.partialSlot(&placed.scratch, n_ids, cfg.hidden, rank);
        try glm53.forward.attn(asm_.blocks, stream, &w, l, caches, &placed.scratch, x.ptr, cfg.hidden, n_ids, pos_dev.ptr, 0, 1, slot);
        try glm53.forward.gatherResidual(asm_.blocks, stream, x.ptr, &placed.scratch, n_ids, cfg.hidden);
        try stream.synchronize();
        if (stages) {
            // the port's equivalents of r2's cache row / ol: its latent ring row 0 and the absorbed output
            const lc0 = glm53.forward.Caches.ofLayer(caches, @intCast(li)).lc;
            const lat_row = (cfg.kv_lora + cfg.rope) * 2;
            try stageAt.at(gpu, stream, stage_host, lc0, lat_row, "L0.cache_row0");
            try stageAt.at(gpu, stream, stage_host, placed.scratch.ol, n_ids * share.mla.len() * cfg.kv_lora * 2, "L0.ol");
            try stageAt.at(gpu, stream, stage_host, placed.scratch.qr, n_ids * cfg.q_lora * 2, "L0.qn");
            try stageAt.at(gpu, stream, stage_host, placed.scratch.q, n_ids * share.mla.len() * (cfg.nope + cfg.rope) * 2, "L0.q_postqb");
            try stageAt.at(gpu, stream, stage_host, placed.scratch.qa, n_ids * share.mla.len() * cfg.kv_lora * 2, "L0.qlat");
            try stageAt.at(gpu, stream, stage_host, x.ptr, n_ids * cfg.hidden * 2, "L0.attn_res");
        }
        std.debug.print("TRACE layer {d} attn ok\n", .{li});
        // `prompt` selects the MoE block's 64-pair prompt tile; a short window takes the decode path
        // (the canonical engine's own split), else the prompt experts read past a 4-row chunk
        const prompt_rows = n_ids >= 64;
        try glm53.forward.ffn(asm_.blocks, stream, &w, l, &placed.scratch, &placed.mlp, &placed.moe, x.ptr, n_ids, prompt_rows, slot);
        try glm53.forward.gatherResidual(asm_.blocks, stream, x.ptr, &placed.scratch, n_ids, cfg.hidden);
        try stream.synchronize();
        if (stages) try stageAt.at(gpu, stream, stage_host, x.ptr, n_ids * cfg.hidden * 2, "L0.ffn_res");
        std.debug.print("TRACE layer {d} ffn ok\n", .{li});
        // every layer when tracing: the gate is a full 78-row table, not a sample
        if (std.c.getenv("TF_GLM53_TRACE") != null) {
            // a context-wide sync before the digest: a stream sync alone left the read racy against the
            // NCCL proxy's completion (the cross-rank table splits this round chased)
            try gpu.d.check(gpu.d.api.cuCtxSynchronize(), "cuCtxSynchronize");
            const probe = try a.alloc(u8, n_ids * cfg.hidden * 2);
            defer a.free(probe);
            try x.download(0, probe);
            const dgl = digest(probe, u16);
            std.debug.print("  layer {d} digest sum={e} absmax={e}\n", .{ li, dgl.sum, dgl.absmax });
            // TF_GLM53_DUMP=<dir>: the raw rows for the formal rel-L2 gate (§58.3)
            if (std.c.getenv("TF_GLM53_DUMP")) |dd| {
                const dpath = try std.fmt.allocPrint(a, "{s}/v11-L{d}.bin", .{ std.mem.span(dd), li });
                defer a.free(dpath);
                const f = try std.Io.Dir.cwd().createFile(gpu.io, dpath, .{});
                defer f.close(gpu.io);
                try f.writeStreamingAll(gpu.io, probe);
            }
        }
    }
    try glm53.ops.norm(asm_.blocks.ops, stream, cfg.hidden, x.ptr, cfg.hidden, w.norm, x.ptr, cfg.hidden, cfg.eps, n_ids);
    try stream.synchronize();
    const xh = try a.alloc(u8, n_ids * cfg.hidden * 2);
    defer a.free(xh);
    try x.download(0, xh);
    const dg = digest(xh, u16);

    std.debug.print("TRACE stack done\n", .{});
    try glm53.forward.head(asm_.blocks, stream, &w, &placed.scratch, x.ptr, n_ids, logits.ptr);
    try stream.synchronize();
    const lh = try a.alloc(f32, n_ids * share.vocab.len());
    defer a.free(lh);
    try logits.download(0, std.mem.sliceAsBytes(lh));
    var last_best: usize = 0;
    const last = lh[(n_ids - 1) * share.vocab.len() ..][0..share.vocab.len()];
    for (last, 0..) |v, i| {
        if (v > last[last_best]) last_best = i;
    }
    const token: u32 = @intCast(last_best);

    std.debug.print("TRACE head done\n", .{});
    std.debug.print(
        "RESULT step layers={d} rows={d} rank={d}/{d} load_ms={d} hidden_sum={e} hidden_absmax={e} last_token={d} logit={e}\n",
        .{ cfg.layers, n_ids, rank, world, load_ms, dg.sum, dg.absmax, token, last[last_best] },
    );
    check.pass("step: {d} layers ran on real weights, hidden digest and a token", .{cfg.layers});
}