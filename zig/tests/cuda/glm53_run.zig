//! `tf-glm53 run [DIR] [TEXT] [NGEN] [RANK] [WORLD]`: the pack's own tokenizer, a prompt in prefill
//! chunks, then greedy decode one token a round - the first end-to-end text out of this engine, and the
//! input the read-only vLLM comparison takes (findings §36).
//!
//! The prompt is tokenized from `<DIR>`'s sibling `glm53-nvfp4-vision/tokenizer.json` (the served
//! checkpoint's own tokenizer; `TF_GLM53_TOKENIZER` names another), encoded without post-processor
//! tokens - exactly what vLLM's completions endpoint does to the same raw text - and prefilled in chunks
//! of `chunk` rows, each at its absolute positions, with the caches growing. Decode is one row a round at
//! the next position; the token is the gathered argmax over the sharded vocabulary (lowest index wins a
//! tie), so world 1 and world 8 take the same path.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm53 = @import("glm53").family;
const check = @import("check.zig");

/// Prompt rows a prefill call takes (the canonical engine's prompt_att_rows is 512; this keeps the
/// scratch small and the chunk boundary well inside one 512-key chunk for short prompts).
pub const chunk: usize = 512; // the weight stream is the bound at small M (§39): a bigger chunk amortizes it

pub fn run(gpu: check.Gpu, args: []const []const u8) !void {
    const a = gpu.gpa;
    const dir: []const u8 = if (args.len > 0) args[0] else blk: {
        const env = std.c.getenv("TF_GLM53_NVFP4_DIR") orelse return error.NoPackDir;
        break :blk std.mem.span(env);
    };
    var file_text: ?[]u8 = null;
    defer if (file_text) |t| a.free(t);
    var text: []const u8 = if (args.len > 1) args[1] else "The capital of France is";
    if (std.c.getenv("TF_GLM53_PROMPT_FILE")) |pf| {
        const b = try std.Io.Dir.cwd().readFileAlloc(gpu.io, std.mem.span(pf), a, .limited(1 << 26));
        file_text = b;
        text = b;
    }
    const ngen: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 16;
    const rank: usize = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 0;
    const world: usize = if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else 1;

    // --- the tokenizer (the pack carries none; the served checkpoint beside it does) ---
    var tok_path_buf: [1024]u8 = undefined;
    const tok_dir: []const u8 = if (std.c.getenv("TF_GLM53_TOKENIZER")) |p| std.mem.span(p) else blk: {
        break :blk std.fmt.bufPrint(&tok_path_buf, "{s}/../glm53-nvfp4-vision", .{dir}) catch return error.BadPath;
    };
    var tok = try core.tokenizer.loadTokenizer(gpu.io, a, tok_dir);
    defer tok.deinit();
    const encoded = try tok.encode(a, text);
    defer a.free(encoded);
    var n_prompt = encoded.len;
    if (std.c.getenv("TF_GLM53_MAX_PROMPT")) |mp| {
        const cap = try std.fmt.parseInt(usize, std.mem.span(mp), 10);
        if (cap < n_prompt) n_prompt = cap;
    }
    if (n_prompt == 0) return error.BadPrompt;
    const prompt_ids_u32 = encoded[0..n_prompt];

    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const json = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 20));
    defer a.free(json);
    var cfg = try glm53.config.parse(a, json);
    if (std.c.getenv("TF_GLM53_LAYERS")) |lz| cfg.layers = try std.fmt.parseInt(u32, std.mem.span(lz), 10);
    cfg.run = cfg.layers;
    cfg.mtp = 0;
    const share = try glm53.shard.Share.of(&cfg, rank, world);

    // the prompt as i32, and the positions every row of the run writes at (absolute, ascending)
    var ids = try a.alloc(i32, n_prompt + ngen + 1);
    defer a.free(ids);
    for (prompt_ids_u32, 0..) |v, i| ids[i] = @intCast(v);
    const capacity = std.mem.alignForward(usize, n_prompt + ngen + 8, 512);
    const pos_host = try a.alloc(i32, capacity);
    defer a.free(pos_host);
    for (pos_host, 0..) |*v, i| v.* = @intCast(i);

    var asm_ = try glm53.blocks.assemble(gpu.d, gpu.ctx, false);
    defer asm_.deinit();
    glm53.blocks.rebind(&asm_);
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
    std.debug.print("TRACE loaded layers={d} load_ms={d}\n", .{ cfg.layers, load_ms });

    // the arena over the largest chunk the run takes, and the caches over the whole run's positions
    const rows_max: usize = chunk; // one chunk's rows: the prefill path may re-run any prefix
    // the sparse path's key window is T = the row's absolute position + the rows: the widest call is the
    // last prompt chunk (or the last decode row), so the scores buffer and the radix's block count are
    // sized for the whole run, not for one chunk.
    const scores_t: usize = n_prompt + ngen + 1;
    const blocks_n: usize = (scores_t + glm53.indexer.topk_chunk - 1) / glm53.indexer.topk_chunk;
    const layout = glm53.forward.scratchLayoutNch(&cfg, share, rows_max, scores_t, blocks_n, glm53.latent.chunksFor(n_prompt + ngen + 1));
    var arena_buf = try cuda.DeviceBuffer.alloc(gpu.d, layout.total);
    defer arena_buf.free();
    var arena: glm53.forward.Arena = .{ .base = arena_buf.ptr, .size = layout.total };
    const placed = try glm53.forward.placeBlocks(&arena, layout);
    try arena_buf.fill8(0, null);

    // one ring set a layer (a layer's history is its own keys, §36): the latent row is kv_lora + rope wide
    const cb = glm53.forward.Caches.ringBytes(&cfg, capacity, glm53.indexer.plane_stride, cfg.layers);
    var lc = try cuda.DeviceBuffer.alloc(gpu.d, cb.lc);
    defer lc.free();
    var ik = try cuda.DeviceBuffer.alloc(gpu.d, cb.ik);
    defer ik.free();
    var ig = try cuda.DeviceBuffer.alloc(gpu.d, cb.ig);
    defer ig.free();
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

    var pos_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pos_host));
    defer pos_dev.free();
    var x = try cuda.DeviceBuffer.alloc(gpu.d, rows_max * cfg.hidden * 2);
    defer x.free();
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, share.vocab.len() * 4);
    defer logits.free();
    const vocab_host = try a.alloc(f32, share.vocab.len());
    defer a.free(vocab_host);

    // TF_GLM53_WARMUP: one untimed pass over the first chunk. The first attention pass of a process
    // costs ~181 s on its own (findings §38: the whole first chunk is attn, later chunks are 1.8 s), so a
    // bench that includes it measures the driver, not the engine.
    if (std.c.getenv("TF_GLM53_WARMUP") != null and n_prompt >= 1) {
        const nw = @min(chunk, n_prompt);
        var ids_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ids[0..nw]));
        defer ids_dev.free();
        const tw = check.now(gpu.io);
        try glm53.forward.embed(asm_.blocks, stream, &w, ids_dev.ptr, nw, x.ptr);
        for (w.layers[0..cfg.run]) |*l| {
            const slot = glm53.forward.partialSlot(&placed.scratch, nw, cfg.hidden, rank);
            try glm53.forward.attn(asm_.blocks, stream, &w, l, caches, &placed.scratch, x.ptr, cfg.hidden, nw, pos_dev.ptr, 0, glm53.latent.chunksFor(nw), slot);
            try glm53.forward.gatherResidual(asm_.blocks, stream, x.ptr, &placed.scratch, nw, cfg.hidden);
            try glm53.forward.ffn(asm_.blocks, stream, &w, l, &placed.scratch, &placed.mlp, &placed.moe, x.ptr, nw, nw >= 64, slot);
            try glm53.forward.gatherResidual(asm_.blocks, stream, x.ptr, &placed.scratch, nw, cfg.hidden);
        }
        try stream.synchronize();
        std.debug.print("TRACE warmup rows={d} ms={d}\n", .{ nw, @divTrunc(check.now(gpu.io) - tw, 1_000_000) });
    }
    // --- prefill in chunks ---
    const t_pre = check.now(gpu.io);
    var t_chunk = t_pre;
    var done: usize = 0;
    while (done < n_prompt) {
        const n = @min(chunk, n_prompt - done);
        var ids_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ids[done .. done + n]));
        defer ids_dev.free();
        const pos = pos_dev.ptr + 4 * done;
        // TF_GLM53_SPLIT: the attn/ffn split of the chunk, with a sync after each stage (diagnostic only)
        const split = std.c.getenv("TF_GLM53_SPLIT") != null;
        var attn_ms: i128 = 0;
        var ffn_ms: i128 = 0;
        const tc0 = check.now(gpu.io);
        try glm53.forward.embed(asm_.blocks, stream, &w, ids_dev.ptr, n, x.ptr);
        for (w.layers[0..cfg.run]) |*l| {
            const slot = glm53.forward.partialSlot(&placed.scratch, n, cfg.hidden, rank);
            const ta = check.now(gpu.io);
            try glm53.forward.attn(asm_.blocks, stream, &w, l, caches, &placed.scratch, x.ptr, cfg.hidden, n, pos, done, glm53.latent.chunksFor(done + n), slot);
            try glm53.forward.gatherResidual(asm_.blocks, stream, x.ptr, &placed.scratch, n, cfg.hidden);
            if (split) {
                try stream.synchronize();
                attn_ms += @divTrunc(check.now(gpu.io) - ta, 1_000_000);
            }
            const tf_ = check.now(gpu.io);
            try glm53.forward.ffn(asm_.blocks, stream, &w, l, &placed.scratch, &placed.mlp, &placed.moe, x.ptr, n, n >= 64, slot);
            try glm53.forward.gatherResidual(asm_.blocks, stream, x.ptr, &placed.scratch, n, cfg.hidden);
            if (split) {
                try stream.synchronize();
                ffn_ms += @divTrunc(check.now(gpu.io) - tf_, 1_000_000);
            }
        }
        if (split) std.debug.print("  split chunk at={d}: embed+all={d} ms attn={d} ffn={d} ms\n", .{ done, @divTrunc(check.now(gpu.io) - tc0, 1_000_000), attn_ms, ffn_ms });
        if (std.c.getenv("TF_GLM53_SPLIT")) |sv| {
            if (std.mem.eql(u8, std.mem.span(sv), "2")) {
                glm53.forward.stage_on = true;
                std.debug.print("  stages (per first layer, ms): ", .{});
                inline for (glm53.forward.stage_names, 0..) |sname, i| std.debug.print("{s}={d} ", .{ sname, glm53.forward.stage_ms[i] });
                std.debug.print("\n", .{});
                glm53.forward.stage_ms = [_]i64{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                glm53.forward.stage_on = false;
            }
        }
        try glm53.ops.norm(asm_.blocks.ops, stream, cfg.hidden, x.ptr, cfg.hidden, w.norm, x.ptr, cfg.hidden, cfg.eps, n);
        try stream.synchronize();
        done += n;
        if (std.c.getenv("TF_GLM53_TIME") != null) {
            const t = check.now(gpu.io);
            std.debug.print("  chunk rows={d} at={d} ms={d}\n", .{ n, done - n, @divTrunc(t - t_chunk, 1_000_000) });
        }
        t_chunk = check.now(gpu.io);
    }
    try stream.synchronize();
    const prefill_ms = @divTrunc(check.now(gpu.io) - t_pre, 1_000_000);
    std.debug.print("TRACE prefill ok rows={d} prefill_ms={d} prefill_tok_s={d:.1}\n", .{ n_prompt, prefill_ms, if (prefill_ms == 0) 0.0 else @as(f64, @floatFromInt(n_prompt)) * 1000.0 / @as(f64, @floatFromInt(prefill_ms)) });
    const t_dec = check.now(gpu.io);

    // --- greedy decode, one row a round (TF_GLM53_FORCE: teacher-force a reference continuation) ---
    var force_ids: []const u32 = &.{};
    var force_alloc: ?[]u32 = null;
    if (std.c.getenv("TF_GLM53_FORCE") orelse std.c.getenv("TF_GLM53_FORCE_FILE")) |fzp| {
        var fz: []const u8 = std.mem.span(fzp);
        var file_buf: ?[]u8 = null;
        defer if (file_buf) |b| a.free(b);
        if (std.c.getenv("TF_GLM53_FORCE") == null) {
            const b = try std.Io.Dir.cwd().readFileAlloc(gpu.io, fz, a, .limited(1 << 20));
            file_buf = b;
            fz = b; // the file is the exact text (no trailing newline expected)
        }
        const enc = try tok.encode(a, fz);
        force_alloc = enc;
        force_ids = enc;
    }
    defer if (force_alloc) |fa| a.free(fa);
    var agree: usize = 0;
    var compared: usize = 0;
    var out = std.ArrayList(u32).empty;
    defer out.deinit(a);
    var pos: usize = n_prompt;
    var last_row = x.ptr + (n_prompt - 1) * cfg.hidden * 2;
    var n_gen: usize = 0;
    while (n_gen < ngen) : (n_gen += 1) {
        try glm53.forward.head(asm_.blocks, stream, &w, &placed.scratch, last_row, 1, logits.ptr);
        try stream.synchronize();
        try logits.download(0, std.mem.sliceAsBytes(vocab_host));
        var next = try pick(gpu, &stream, a, group, world, share.vocab.lo, vocab_host);
        if (force_ids.len > 0) {
            const want = if (n_gen < force_ids.len) force_ids[n_gen] else 0;
            compared += 1;
            if (next == want) agree += 1;
            std.debug.print("  step {d}: ours={d} vllm={d} {s}\n", .{ n_gen, next, want, if (next == want) "agree" else "DIFFER" });
            if (n_gen < force_ids.len) next = want;
        }
        try out.append(a, next);
        if (isEos(&cfg, next) or n_gen + 1 == ngen) break;
        // the next round: this token at `pos`, one row
        ids[pos] = @intCast(next);
        var ids_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ids[pos .. pos + 1]));
        defer ids_dev.free();
        try glm53.forward.embed(asm_.blocks, stream, &w, ids_dev.ptr, 1, x.ptr);
        for (w.layers[0..cfg.run]) |*l| {
            const slot = glm53.forward.partialSlot(&placed.scratch, 1, cfg.hidden, rank);
            try glm53.forward.attn(asm_.blocks, stream, &w, l, caches, &placed.scratch, x.ptr, cfg.hidden, 1, pos_dev.ptr + 4 * pos, pos, glm53.latent.chunksFor(pos + 1), slot);
            try glm53.forward.gatherResidual(asm_.blocks, stream, x.ptr, &placed.scratch, 1, cfg.hidden);
            try glm53.forward.ffn(asm_.blocks, stream, &w, l, &placed.scratch, &placed.mlp, &placed.moe, x.ptr, 1, false, slot);
            try glm53.forward.gatherResidual(asm_.blocks, stream, x.ptr, &placed.scratch, 1, cfg.hidden);
        }
        try glm53.ops.norm(asm_.blocks.ops, stream, cfg.hidden, x.ptr, cfg.hidden, w.norm, x.ptr, cfg.hidden, cfg.eps, 1);
        try stream.synchronize();
        last_row = x.ptr;
        pos += 1;
    }

    if (force_ids.len > 0) std.debug.print("TRACE teacher agreement {d}/{d}\n", .{ agree, compared });

    // TF_GLM53_PREFILL_PATH: the same continuation computed by PREFILL only (each prefix re-prefilled, its
    // last row's argmax taken) - if this differs from the decode loop's tokens, the decode path is the
    // defect; if it matches, decode and prefill are consistent and the divergence is the forward itself.
    if (std.c.getenv("TF_GLM53_PREFILL_PATH") != null) {
        // the same continuation computed by PREFILL only, at several chunk widths: a width whose text
        // matches the decode loop's and the reference's isolates the call shape that breaks.
        for ([_]usize{ 1, 2, 3, 4 }) |cw| {
            var pre = std.ArrayList(u32).empty;
            defer pre.deinit(a);
            var seq = std.ArrayList(i32).empty;
            defer seq.deinit(a);
            for (prompt_ids_u32) |v| try seq.append(a, @intCast(v));
            var k: usize = 0;
            while (k < ngen) : (k += 1) {
                var pre_done: usize = 0;
                while (pre_done < seq.items.len) {
                    const n = @min(cw, seq.items.len - pre_done);
                    var ids_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(seq.items[pre_done .. pre_done + n]));
                    defer ids_dev.free();
                    const ppos = pos_dev.ptr + 4 * pre_done;
                    try glm53.forward.embed(asm_.blocks, stream, &w, ids_dev.ptr, n, x.ptr);
                    for (w.layers[0..cfg.run]) |*l| {
                        const slot = glm53.forward.partialSlot(&placed.scratch, n, cfg.hidden, rank);
                        try glm53.forward.attn(asm_.blocks, stream, &w, l, caches, &placed.scratch, x.ptr, cfg.hidden, n, ppos, pre_done, glm53.latent.chunksFor(pre_done + n), slot);
                        try glm53.forward.gatherResidual(asm_.blocks, stream, x.ptr, &placed.scratch, n, cfg.hidden);
                        try glm53.forward.ffn(asm_.blocks, stream, &w, l, &placed.scratch, &placed.mlp, &placed.moe, x.ptr, n, n >= 64, slot);
                        try glm53.forward.gatherResidual(asm_.blocks, stream, x.ptr, &placed.scratch, n, cfg.hidden);
                    }
                    try glm53.ops.norm(asm_.blocks.ops, stream, cfg.hidden, x.ptr, cfg.hidden, w.norm, x.ptr, cfg.hidden, cfg.eps, n);
                    try stream.synchronize();
                    pre_done += n;
                }
                try glm53.forward.head(asm_.blocks, stream, &w, &placed.scratch, x.ptr + (seq.items.len - 1) * cfg.hidden * 2, 1, logits.ptr);
                try stream.synchronize();
                try logits.download(0, std.mem.sliceAsBytes(vocab_host));
                const t = try pick(gpu, &stream, a, group, world, share.vocab.lo, vocab_host);
                try pre.append(a, t);
                try seq.append(a, @intCast(t));
            }
            const pre_text = try tok.decode(a, pre.items, true);
            defer a.free(pre_text);
            var same: usize = 0;
            for (pre.items, 0..) |t, i| if (i < out.items.len and t == out.items[i]) {
                same += 1;
            };
            std.debug.print("RESULT prefill-path chunk={d} agree_decode={d}/{d} ids={any} text=\"{s}\"\n", .{ cw, same, @min(pre.items.len, out.items.len), pre.items, pre_text });
        }
    }

    try stream.synchronize();
    const decode_ms = @divTrunc(check.now(gpu.io) - t_dec, 1_000_000);
    const text_out = try tok.decode(a, out.items, true);
    defer a.free(text_out);
    std.debug.print("RESULT bench prefill_tokens={d} prefill_ms={d} prefill_tok_s={d:.2} decode_tokens={d} decode_ms={d} decode_tok_s={d:.3}\n", .{ n_prompt, prefill_ms, if (prefill_ms == 0) 0.0 else @as(f64, @floatFromInt(n_prompt)) * 1000.0 / @as(f64, @floatFromInt(prefill_ms)), out.items.len, decode_ms, if (decode_ms == 0) 0.0 else @as(f64, @floatFromInt(out.items.len)) * 1000.0 / @as(f64, @floatFromInt(decode_ms)) });
    std.debug.print("RESULT run rank={d}/{d} prompt_tokens={d} generated={d} ids={any} text=\"{s}\"\n", .{ rank, world, n_prompt, out.items.len, out.items, text_out });
    check.pass("run: {d} prompt tokens prefilled, {d} greedy tokens decoded", .{ n_prompt, out.items.len });
}

fn isEos(c: *const glm53.config.Config, id: u32) bool {
    for (c.eos[0..c.eos_n]) |e| if (id == e) return true;
    return false;
}

/// The gathered greedy token: this rank's shard argmax (lowest index on a tie), then the world's maximum
/// with the lowest global index - the head command's reduction, per decode round.
fn pick(gpu: check.Gpu, stream: *const cuda.Stream, a: std.mem.Allocator, group: ?cuda.tp_group.Group, world: usize, lo: usize, host: []const f32) !u32 {
    var best: usize = 0;
    for (host, 0..) |v, i| if (v > host[best]) {
        best = i;
    };
    var tok: u32 = @intCast(lo + best);
    var val: f32 = host[best];
    if (world > 1) {
        const g = group orelse return error.NoCommunicator;
        const word: u64 = (@as(u64, tok) << 32) | @as(u32, @bitCast(val));
        const all = try a.alloc(u64, world);
        defer a.free(all);
        try g.gather(u64, gpu.d, stream.*, word, all);
        val = -std.math.inf(f32);
        var best_tok: u32 = 0;
        for (all) |u| {
            const t: u32 = @intCast(u >> 32);
            const v: f32 = @bitCast(@as(u32, @truncate(u)));
            if (v > val or (v == val and t < best_tok)) {
                val = v;
                best_tok = t;
            }
        }
        tok = best_tok;
    }
    std.debug.print("TRACE token {d} value {e}\n", .{ tok, val });
    return tok;
}