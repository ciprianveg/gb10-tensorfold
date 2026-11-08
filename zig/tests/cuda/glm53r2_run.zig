//! `tf-glm53r2 run <pack> <text> <ngen> <rank> <world>`: GENERATION on r2's engine over the 1.1-loaded weights
//! (the A4a behavioral gate's driver). The weights and the E2 hook are built exactly as `glm53r2 bind` builds
//! them (same binder, same alias table — proven through L77), and the decode loop is r2's own `Runner`
//! (`cuda_runner.zig`): chunked prefill, kv8 decode windows, greedy picks, optional CUDA graphs.
//!
//! Env: TF_GLM53_TOKENIZER (default `<pack>/../glm53-nvfp4-vision`), TF_GLM53_CAPACITY (default prompt+ngen+8),
//! TF_GLM53_KV8 (1), TF_GLM53_DCP (1), TF_GLM53_GRAPHS (1), TF_GLM53_PROMPT_ROWS (window, 128),
//! TF_GLM53_AOT (the staged pack), TF_NCCL_ID_FILE (world > 1).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const glm53 = @import("glm53").family;
const r2 = @import("glm53r2");
const binder = @import("glm53r2_bind.zig");

fn argNum(args: []const []const u8, i: usize, dflt: usize) usize {
    if (i >= args.len) return dflt;
    return std.fmt.parseInt(usize, args[i], 10) catch dflt;
}

pub fn run(gpu: check.Gpu, args: []const []const u8) !void {
    const a = gpu.gpa;
    const dir = if (args.len > 0) args[0] else return error.NoPack;
    // A deep-context prompt (100K+ tokens, ~0.5 MB of text) cannot ride argv (MAX_ARG_STRLEN is 128 KiB), so when
    // the argument names an existing file the prompt is read from it (harness-only, same class as START/INJECT).
    var text_owned: ?[]u8 = null;
    defer if (text_owned) |t| a.free(t);
    const text: []const u8 = blk: {
        const arg: []const u8 = if (args.len > 1) args[1] else break :blk "The capital of France is";
        if (std.Io.Dir.cwd().readFileAlloc(gpu.io, arg, a, .limited(8 << 20))) |t| {
            text_owned = t;
            break :blk t;
        } else |_| {}
        break :blk arg;
    };
    const ngen = argNum(args, 2, 32);
    const rank = argNum(args, 3, 0);
    const world = argNum(args, 4, 1);

    // --- the tokenizer (the pack carries none; the served checkpoint beside it does) -------------------------
    var tok_path_buf: [1024]u8 = undefined;
    const tok_dir: []const u8 = if (std.c.getenv("TF_GLM53_TOKENIZER")) |p| std.mem.span(p) else blk: {
        break :blk std.fmt.bufPrint(&tok_path_buf, "{s}/../glm53-nvfp4-vision", .{dir}) catch return error.BadPath;
    };
    var tok = try core.tokenizer.loadTokenizer(gpu.io, a, tok_dir);
    defer tok.deinit();
    const encoded = try tok.encode(a, text);
    defer a.free(encoded);
    if (encoded.len == 0) return error.BadPrompt;
    // The whole prompt is NOT printed: a file prompt would put ~0.5 MB x 8 ranks into the logs (the §84 lesson is
    // about reading them; this is about not writing them). Short prompts print in full, as before.
    const q_show: []const u8 = if (text.len > 200) text[0..200] else text;
    std.debug.print("PROMPT tokens={d} q={s}{s}\n", .{ encoded.len, q_show, if (text.len > 200) " [truncated]" else "" });

    // --- the 1.1 load + the binder's alias table (identical to the proven bind path) -------------------------
    const json = try std.Io.Dir.cwd().readFileAlloc(gpu.io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 20));
    defer a.free(json);
    var cfg = try glm53.config.parse(a, json);
    if (std.c.getenv("TF_GLM53_LAYERS")) |lz| {
        const s = std.mem.span(lz);
        if (s.len > 0) cfg.layers = try std.fmt.parseInt(u32, s, 10); // "": unset (run8.sh's ${X:-} passthrough)
    }
    cfg.run = cfg.layers;
    // TF_GLM53_K (the §90 MTP wiring, the end-state spec mode): the draft depth k. k > 0 loads the MTP layer
    // (cfg.mtp = 1: the 1.1 reads layers.{78}.eh_proj/enorm/hnorm/shared_head.norm + the layer itself, all
    // BF16 in the CSF pack's shard 84) and runs the Runner's draft chain (the verify windows R = k + 1);
    // 0: serial only (the k=0 lanes, every bench before §90).
    const mtp_k: usize = if (std.c.getenv("TF_GLM53_K")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0 else 0;
    cfg.mtp = if (mtp_k > 0) 1 else 0;
    const share = try glm53.shard.Share.of(&cfg, rank, world);
    _ = share.mla.len(); // §91: the share mla len is the GLOBAL count — see the heads fix below
    // §91: r2's W.heads is "this rank's attention heads" (cuda_weights.zig:101) and the wide path reads it
    // directly (cuda_prompt.zig:157/174). The 1.1 SHARDS the MLA heads (q_b loads rows [mla.lo, mla.hi) of
    // the global heads — glm_moe_dsa/weights.zig:311), so the loaded truth is Attn.heads (8 at world 8) —
    // while share.mla.len() returns the GLOBAL 64. The binder's run() default fed r2 the global (a world-1-era
    // footgun), dormant through the run lane (the non-wide paths read the per-layer Attn.heads), but the wide
    // path at H=64 launches forms the pack never carried AND its tile configs exceed the GB10's shared budget
    // (_attn_chunks warps 8 wants 155,648 B). loaded_heads is sourced after the load, below.

    var asm_ = try glm53.blocks.assemble(gpu.d, gpu.ctx, false);
    defer asm_.deinit();
    glm53.blocks.rebind(&asm_);
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
    var pw = try glm53.weights.load(a, gpu.d, stream, tools, &ck, &cfg, share);
    defer pw.deinit();
    try stream.synchronize();
    std.debug.print("TRACE loaded layers={d} load_ms={d}\n", .{ cfg.layers, @divTrunc(check.now(gpu.io) - t0, 1_000_000) });

    var bstats: binder.Stats = .{};
    // §91 (the heads fix): the loaded per-rank MLA head count — the truth the 1.1's shard actually loaded.
    const loaded_heads = pw.layers[0].attn.heads;
    const heads_for_r2 = loaded_heads;
    const side = try a.alloc(?glm53.moe.Layer, pw.layers.len);
    defer a.free(side);
    @memset(side, null);
    var w = try binder.bind(gpu, stream, &pw, rank, world, loaded_heads, heads_for_r2, &bstats, side);
    defer a.free(w.layers);
    std.debug.print("BIND layers={d} moe={d} experts={d}\n", .{ bstats.layers, bstats.moe, bstats.experts });
    std.debug.print("HEADS w.heads={d} config.heads={d} layer0.attn.heads={d} (the wide path reads f.w.heads; §91 debug)\n", .{ w.heads, w.config.heads, pw.layers[0].attn.heads });

    const aot = if (std.c.getenv("TF_GLM53_AOT")) |v| std.mem.span(v) else "/home/ciprian/aot-world8-dcp2-mirror";
    var k = try r2.kernels.Kernels.load(a, gpu.io, gpu.ctx, aot);
    defer k.deinit();

    const kv8 = if (std.c.getenv("TF_GLM53_KV8")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    const dcp: usize = if (std.c.getenv("TF_GLM53_DCP")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 1 else 1;

    // the communicator (the port's proven rendezvous), r2's view over the same handle
    var group: ?cuda.tp_group.Group = null;
    defer if (group) |*g| g.close();
    var comm: ?cuda.nccl.Communicator = null;
    if (world > 1) {
        const idpath = std.c.getenv("TF_NCCL_ID_FILE") orelse return error.NoCommId;
        group = try cuda.tp_group.Group.openShared(gpu.io, rank, world, std.mem.sliceTo(idpath, 0));
        comm = .{ .lib = &group.?.lib, .comm = group.?.comm, .rank = rank, .world = world };
        std.debug.print("COMM rank={d}/{d} up\n", .{ rank, world });
    }

    var inv = try cuda.DeviceBuffer.alloc(gpu.d, cfg.rope * 4);
    defer inv.free();
    if (pw.inv != 0) {
        try gpu.d.check(gpu.d.api.cuMemcpyDtoDAsync_v2(inv.ptr, pw.inv, (cfg.rope / 2) * 4, stream.handle), "cuMemcpyDtoDAsync");
    } else {
        try k.invFreq(stream, inv.ptr, cfg.rope, cfg.rope_theta);
    }

    // the E2 hook: the port's MoE block for the routed layers (same scratch layout the binder uses).
    // §92 THE WIDE-PATH FIX: the serve-shape chunks run R up to serve_rows through the hook — rows_max=128 was
    // the pre-§91 128-row prompt-chunk width and overflowed the moe planes at the serve shape (the §91 illegal
    // access, named by the §92 print-bisect: the whole front/attention span probe-clean, the fault in the first
    // routed layer's hook call — glm53.moe.block writes `rows` rows straight into the scratch, no blocking).
    // Sized as the 1.1 engine sizes its own arena (its served chunk rows); only the moe block + the shared
    // mm_part are placed — the attention/residual planes stay r2's own pb.
    const hook_served = blk: {
        const v = std.c.getenv("TF_GLM53_SERVED") orelse break :blk false;
        const s = std.mem.span(v);
        break :blk s.len > 0 and !std.mem.eql(u8, s, "0");
    };
    const hook_serve_rows: usize = if (std.c.getenv("TF_GLM53_SERVE_ROWS")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 4096 else 4096;
    const rows_max = @max(128, if (hook_served) hook_serve_rows else 0);
    const scores_t = rows_max + 8;
    const blocks_n = (scores_t + glm53.indexer.topk_chunk - 1) / glm53.indexer.topk_chunk;
    var layout = glm53.forward.scratchLayoutNch(&cfg, share, rows_max, scores_t, blocks_n, glm53.latent.chunksFor(scores_t));
    inline for (@typeInfo(@TypeOf(layout)).@"struct".field_names) |fname| {
        if (comptime (std.mem.startsWith(u8, fname, "moe_") or std.mem.eql(u8, fname, "mm_part") or std.mem.eql(u8, fname, "dummy") or std.mem.eql(u8, fname, "at") or std.mem.eql(u8, fname, "total"))) continue;
        @field(layout, fname) = 0;
    }
    layout.total = glm53.forward.arenaTotal(&layout);
    var arena_buf = try cuda.DeviceBuffer.alloc(gpu.d, layout.total);
    defer arena_buf.free();
    var arena: glm53.forward.Arena = .{ .base = arena_buf.ptr, .size = layout.total };
    const placed = try glm53.forward.placeBlocks(&arena, layout);
    try arena_buf.fill8(0, null);
    var hook = binder.Hook{ .kernels = asm_.blocks.moe, .scratch = placed.moe, .side = side };

    // TF_GLM_PROF=1 (the §87 decode profile): the 1.1 Prof on the hook's moe kernels - per-section GPU ms
    // (router/plan/gate_up/down/shared/combine), a report every 150 hook calls (~2 eager decode steps of 75
    // routed layers). Profile runs want GRAPHS=0: graph replays skip the hook, only the capture is visible.
    if (binder.Hook.MoeProf.wanted()) {
        const prof = try binder.Hook.MoeProf.init(gpu.d, gpu.io);
        prof.every = 150;
        hook.kernels.prof = prof;
        hook.prof = prof;
    }

    var f: r2.forward.Forward = .{ .w = &w, .k = &k, .s = stream, .inv = inv.ptr, .dcp = dcp, .moe_hook = .{ .ctx = &hook, .has = binder.Hook.has, .call = binder.Hook.call } };
    f.comm = if (comm) |*c| c else null;

    // TF_GLM53_ROCE=1 (the §87 decode fix): the decode profile named the per-layer NCCL allGather as 86% of
    // the 424 ms decode step (ag 354-376 ms = 78 layers x ~4.6 ms of small-message latency on the compute
    // stream; attn_proj 16 + front 16 + attn_core 4 + kv_write 2 + dense 1 + moe 24 made up the rest). The
    // engine's own decode path takes the RoCE one-shot instead (gather()'s `f.fast and R <= b.small` branch,
    // microsecond-class); this wires it exactly as cuda_engine.zig does: TCP rendezvous -> Roce.start ->
    // verify against the NCCL communicator. Prompt chunks (R=128 > small) get the bf16 ring path (f.ring).
    // TF_GLM53_RDV: rank 0's address (default .111:29601); TF_GLM53_HCAS: the rails; TF_GLM53_GID: default 3.
    var fast: ?*r2.roce.Roce = null;
    defer if (fast) |rc| rc.deinit(a);
    var rdv: cuda.rendezvous.Rendezvous = undefined;
    var rdv_open = false;
    defer if (rdv_open) rdv.close();
    const roce_on = world > 1 and blk: {
        const v = std.c.getenv("TF_GLM53_ROCE") orelse break :blk false;
        break :blk !std.mem.eql(u8, std.mem.span(v), "0");
    };
    if (roce_on) {
        const addr = std.mem.span(std.c.getenv("TF_GLM53_RDV") orelse "192.168.178.111:29601");
        const master = try cuda.rendezvous.parseMaster(addr);
        rdv = try cuda.rendezvous.Rendezvous.open(gpu.io, master.ip, master.port, rank, world, 300);
        rdv_open = true;
        var hb: [2][]const u8 = undefined;
        var n_hca: usize = 0;
        const hcas_s = std.mem.span(std.c.getenv("TF_GLM53_HCAS") orelse "rocep1s0f1");
        var it = std.mem.tokenizeScalar(u8, hcas_s, ',');
        while (it.next()) |raw| {
            if (n_hca == hb.len) break;
            var name = std.mem.trim(u8, raw, " =^");
            if (std.mem.indexOfScalar(u8, name, ':')) |cx| name = name[0..cx];
            if (name.len == 0) continue;
            hb[n_hca] = name;
            n_hca += 1;
        }
        var gid: u32 = 3;
        if (std.c.getenv("TF_GLM53_GID")) |v| gid = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 3;
        fast = try r2.roce.Roce.start(a, gpu.d, &rdv, rank, world, .{ .hcas = hb[0..n_hca], .gid_index = gid });
        if (comm) |*c| {
            if (!try fast.?.verify(a, c, stream, w.config.hidden, r2.forward.decode_rows)) return error.RoceCheckFailed;
        }
        f.fast = fast;
        f.ring = true;
        std.debug.print("ROCE rank {d}/{d}: one-shot up, hcas={s} (decode: RoCE one-shot, prompt: bf16 ring)\n", .{ rank, world, hcas_s });
    }

    // TF_GLM53_PAD=<KiB> (the §88 layout probe): a dummy device allocation between the RoCE block and the
    // Runner's arena - the §87.5/§88 corruption is layout-sensitive (prose 3/3 corrupt on one build, 2/2
    // clean on a 10-line rebuild), so the pad shifts the post-region placement per boot without a rebuild,
    // making the sensitivity testable and the corrupt layout reproducible on demand.
    var roce_pad: ?cuda.DeviceBuffer = null;
    defer if (roce_pad) |*p| p.free();
    if (std.c.getenv("TF_GLM53_PAD")) |v| {
        const kb = std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0;
        if (kb > 0) {
            roce_pad = try cuda.DeviceBuffer.alloc(gpu.d, kb * 1024);
            try roce_pad.?.fill8(0, null);
            std.debug.print("ROCE pad: {d} KiB\n", .{kb});
        }
    }

    // TF_GLM53_DECODE_PROF=1 (the §87 decode profile): r2's own per-phase Prof (attn_proj/attn_core/routed/
    // shared/dense/head/rs/front/select/ag/kv_write/kv8_dequant/index_scores, main vs comm streams) through the
    // prefill AND the decode rounds. Use GRAPHS=0: the capture suspends marks and replays record none.
    const R2Prof = @typeInfo(@typeInfo(@TypeOf(@as(r2.forward.Forward, undefined).prof)).optional.child).pointer.child;
    var r2prof: R2Prof = undefined;
    var r2prof_on = false;
    if (std.c.getenv("TF_GLM53_DECODE_PROF") != null) {
        r2prof = R2Prof.init(a, gpu.d, stream);
        r2prof.on = true;
        r2prof_on = true;
        f.prof = &r2prof;
    }
    defer if (r2prof_on) r2prof.deinit();
    const cublas_lib = try cuda.cublas.Library.open();
    defer @constCast(&cublas_lib).close();
    var ws = try cuda.DeviceBuffer.alloc(gpu.d, 32 << 20);
    defer ws.free();
    var blas = try cuda.cublas.Blas.init(&cublas_lib, ws.ptr, ws.len, cuda.cublas.default_math);
    defer blas.deinit();
    f.blas = &blas;

    const capacity = if (std.c.getenv("TF_GLM53_CAPACITY")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch (encoded.len + ngen + 8) else encoded.len + ngen + 8;
    const graphs = if (std.c.getenv("TF_GLM53_GRAPHS")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else true;
    const window: usize = if (std.c.getenv("TF_GLM53_PROMPT_ROWS")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 128 else 128;
    // §91: the SERVE SHAPE — served Settings (the Runner's pb/pb1 at prompt_rows, the engine's chunking) +
    // the engine's Sp overlap (§89.2: the machinery was built, gated on served; the wide chunks clear the
    // 258-row engagement threshold). SERVE_SP builds the side stream + the Wides + the Sp exactly as
    // cuda_engine.zig does. The E2 hook is SP-safe: both halves' FFNs chain on the one side stream (the same
    // stream-order argument as the kv8 staging, §89.2). NOTE the flags parse "" and "0" as OFF — run8.sh's
    // ${X:-} passthrough sets the empty string (the §91 boot caught sp=true on the "before" arm).
    const served = blk: {
        const v = std.c.getenv("TF_GLM53_SERVED") orelse break :blk false;
        const s = std.mem.span(v);
        break :blk s.len > 0 and !std.mem.eql(u8, s, "0");
    };
    const serve_sp = blk: {
        const v = std.c.getenv("TF_GLM53_SERVE_SP") orelse break :blk false;
        const s = std.mem.span(v);
        break :blk s.len > 0 and !std.mem.eql(u8, s, "0");
    };
    const serve_rows: usize = if (std.c.getenv("TF_GLM53_SERVE_ROWS")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 4096 else 4096;
    const serve_rows_short: usize = if (std.c.getenv("TF_GLM53_SERVE_ROWS_SHORT")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 2048 else 2048;
    const stage_rows = @max(@max(window, r2.forward.max_rows), if (served) serve_rows else 0);
    std.debug.print("GEN capacity={d} kv8={} dcp={d} graphs={} window={d} served={} serve_rows={d} sp={}\n", .{ capacity, kv8, dcp, graphs, window, served, serve_rows, served and serve_sp });

    // TF_GLM53_KV8 (the §85 crash attribution, 2026-10-21): the env var was read for the GEN line but never
    // threaded - every high-capacity run booted the BF16 latent planes (95,232 B/token vs kv8's 50,928: 1.87x)
    // and the cache allocation alone inside Runner.init killed 600K/800K/900K/1M (four glm-oom-guard kills,
    // all eight nodes together, anon flat at ~0.9 GiB - the growth is device-side cuMemAlloc). The fix threads
    // it exactly as the engine does (cuda_engine.zig's Kv8.init): the side buffers MUST exist with the e4m3
    // planes - keys() stages every cache write through them and attention reads the dequantized scratch;
    // e4m3 planes without them would be read as bf16 (garbage). Single stream at DCP 1: kv_plane = 0 (the
    // engine's multi-slot DCP case does not apply), stage_rows covers the `window`-row prompt chunks (the
    // engine stages its 8192-row served chunks), rows_max = fw.max_rows, prompt_cap = dcpm.localCapacity(cap, 1).
    var kv8_side: ?r2.forward.Kv8 = null;
    defer if (kv8_side) |*v| v.deinit();
    if (kv8) {
        kv8_side = try r2.forward.Kv8.init(a, gpu.d, w.config, stage_rows, r2.forward.max_rows, capacity + 1);
        f.kv8 = &kv8_side.?;
    }
    // §94: the L2 prefetch (the EXL line's 0046, "its kernel used unchanged"; the ENGINE defaults it ON —
    // our lane never wired it). The sites already fire every layer (layerWith's f.site(.a/.f/.o)); this only
    // builds the object and sets f.l2pf. TF_GLM53_L2PF: ""/0/off = OFF (the ${X:-} rule), 1/on/bulk, lines,
    // touch; TF_GLM53_L2PF_MB per site (default 8, max 64); TF_GLM53_L2PF_AFTER moves a site past its exchange.
    var l2pf_side: ?r2.decode.L2pf = null;
    defer if (l2pf_side) |*v| v.deinit();
    const pf_mode = if (std.c.getenv("TF_GLM53_L2PF")) |v| r2.engine.l2pfMode(std.mem.span(v)) catch -1 else -1;
    if (pf_mode >= 0) {
        const pf_mb: f64 = if (std.c.getenv("TF_GLM53_L2PF_MB")) |v| std.fmt.parseFloat(f64, std.mem.span(v)) catch 8.0 else 8.0;
        if (!(pf_mb > 0 and pf_mb <= 64)) return error.BadL2pfBudget;
        const pf_after = blk: {
            const v = std.c.getenv("TF_GLM53_L2PF_AFTER") orelse break :blk false;
            const s = std.mem.span(v);
            break :blk s.len > 0 and !std.mem.eql(u8, s, "0");
        };
        l2pf_side = try r2.decode.L2pf.init(a, gpu.d, &w, .{ .mode = pf_mode, .mb = pf_mb, .after = pf_after });
        f.l2pf = &l2pf_side.?;
        std.debug.print("L2PF up: mode={d} mb={d:.1} after={} (0046's class, the decode windows' weight prefetch)\n", .{ pf_mode, pf_mb, pf_after });
    }
    // the cache arithmetic beside the GEN line (the §71 cross-check: kv8 = 50,928 B/token; bf16 = 95,232)
    std.debug.print("CACHE capacity={d} dcp={d} kv8={} lat_row={d} bytes={d} ({d:.2} GiB)\n", .{
        capacity, dcp, kv8, r2.forward.State.latentRow(kv8, w.config),
        r2.forward.State.bytesForDcpKv8(&w, capacity, dcp, kv8),
        @as(f64, @floatFromInt(r2.forward.State.bytesForDcpKv8(&w, capacity, dcp, kv8))) / (1 << 30),
    });

    // §90: the MTP draft chain — k from TF_GLM53_K (the Runner builds the mtp buffers/graphs for it),
    // reuse from TF_GLM53_MTP_REUSE (0/1/2: draft steps 2.. attend the first step's selection; needs the
    // MTP layer's indexer — default 0, the plain chain).
    const mtp_reuse: u32 = if (std.c.getenv("TF_GLM53_MTP_REUSE")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0 else 0;
    // §91: the Sp build — AFTER f's final state (the roce block's fast/ring, the kv8 assignment: Sp.init
    // copies main.* for the side Forward and builds the side kv8 staging from main.kv8). The holders live at
    // run()-scope so the Runner's set.sp pointer stays valid. The MAIN Wide is REQUIRED (cuda_prompt.zig:437:
    // the SP chunk's frontFfn errors WindowTooWide without f.wide) — the engine's main_wide pattern, blas-less
    // (the side's own pattern: the wide GEMMs run the AOT/triton path).
    var sp_holder: ?r2.prompt.Sp = null;
    defer if (sp_holder) |*s| s.deinit();
    var side_stream_h: ?cuda.Stream = null;
    defer if (side_stream_h) |*s| s.deinit();
    var main_wide_h: ?r2.prompt.Wide = null;
    defer if (main_wide_h) |*x| x.deinit();
    var side_wide_h: ?r2.prompt.Wide = null;
    defer if (side_wide_h) |*x| x.deinit();
    var blas_lib: ?cuda.cublas.Library = null;
    defer if (blas_lib) |*l| l.close();
    var blas_ws: ?cuda.DeviceBuffer = null;
    defer if (blas_ws) |*b_| b_.free();
    var blas_h: ?cuda.cublas.Blas = null;
    defer if (blas_h) |*b_| b_.deinit();
    if (served) {
        // The MAIN Wide + cuBLAS are REQUIRED for the served shape itself (cuda_forward.zig:1110/1289: the
        // wide-class attention - rows past the fast-class cutoff - needs f.blas (the absorbed GEMMs) and
        // f.wide; the 128-row run-lane chunks never took the wide branch, the 4096-row serve chunks always
        // do). The engine's pattern (cuda_engine.zig:548-555): the torch-class workspace + the math default.
        blas_lib = try cuda.cublas.Library.openPath("libcublas.so.13");
        const major: c_int = @intCast((try gpu.ctx.capability()) / 10);
        const ws_bytes = cuda.cublas.torchWorkspaceBytes(major, null);
        blas_ws = try cuda.DeviceBuffer.alloc(gpu.d, @max(ws_bytes, 16));
        blas_h = try cuda.cublas.Blas.init(&blas_lib.?, blas_ws.?.ptr, ws_bytes, 0);
        f.blas = &blas_h.?;
        main_wide_h = try r2.prompt.Wide.init(gpu.d, &k, stream, &w, serve_rows, null, &blas_h.?);
        f.wide = &main_wide_h.?;
    }
    if (served and serve_sp and world > 1) {
        side_stream_h = try cuda.Stream.init(gpu.d, true);
        side_wide_h = try r2.prompt.Wide.init(gpu.d, &k, side_stream_h.?, &w, serve_rows / 2 + 1, null, null);
        sp_holder = try r2.prompt.Sp.init(a, gpu.d, &f, side_stream_h.?, &side_wide_h.?);
        std.debug.print("SP up: side stream + wides (main rows {d}, side rows {d}), the halves' frontAttn/frontFfn chain on it\n", .{ serve_rows, serve_rows / 2 + 1 });
    }
    var r = try r2.runner.Runner.init(a, gpu.io, gpu.d, f, capacity, .{
        .k = mtp_k, .graphs = graphs, .window = window, .reuse = mtp_reuse, .dcp = dcp, .kv8 = kv8,
        .served = served, .prompt_rows = if (served) serve_rows else 0, .prompt_rows_short = if (served) serve_rows_short else 0,
        .sp = if (sp_holder != null) &sp_holder.? else null,
    });
    defer r.deinit();

    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    // §90: k_req = the built depth (generate's opts.k: 0 would mean serial-only even at set.k > 0).
    // 118.3: the sampled debug surface (TF_GLM53_TEMP / _TOP_K / _TOP_P) - the server lane sampled
    // path (the Gumbel debut) reproduced in the 3-min run lane; greedy (null) when TEMP is unset or 0.
    // The seed matches the server form: seedFor(prompt ids), the salt 0 arm.
    const temp: f64 = if (std.c.getenv("TF_GLM53_TEMP")) |v| std.fmt.parseFloat(f64, std.mem.span(v)) catch 0 else 0;
    const sampling: ?r2.sampling.Sampling = if (temp > 0) .{
        .seed = r2.sampling.seedFor(encoded, 0),
        .temperature = temp,
        .top_k = if (std.c.getenv("TF_GLM53_TOP_K")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 20 else 20,
        .top_p = if (std.c.getenv("TF_GLM53_TOP_P")) |v| std.fmt.parseFloat(f64, std.mem.span(v)) catch 0.95 else 0.95,
        .min_p = 0.0,
    } else null;
    const st = try r.generate(encoded, ngen, sampling, mtp_k, &out);
    const text_out = try tok.decode(a, out.items, true);
    defer a.free(text_out);
    std.debug.print("RESULT r2 prompt={d} gen={d} prefill_s={d:.3} decode_s={d:.3} tok_s={d:.2} graphs={d} eager={d}\n", .{
        st.prompt_len, out.items.len, st.prefill_s, st.decode_s, st.tokS(out.items.len), r.graph_runs, r.eager_runs,
    });
    if (mtp_k > 0) std.debug.print("MTP k={d} rounds={d} drafted={d} accepted={d} accept_rate={d:.3}\n", .{ mtp_k, st.rounds, st.drafted, st.accepted, if (st.drafted > 0) @as(f64, @floatFromInt(st.accepted)) / @as(f64, @floatFromInt(st.drafted)) else 0 });
    std.debug.print("TOKENS", .{});
    for (out.items) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\nOUT {s}\n", .{text_out});
    check.pass("run: {d} tokens generated on r2's engine over the 1.1-loaded weights", .{out.items.len});
    // §88: the one-shot's proxy stats at exit (posted / completed / last_seq) - the corrupt runs' frozen
    // epoch and 3-44 s flag cascades need the proxy's own view (a dead proxy: last_seq frozen, completed
    // << posted; a backlog: completed lagging then catching up).
    if (fast) |rc| {
        const s = rc.stats();
        std.debug.print("ROCE stats rank {d}: posted {d} completed {d} last_seq {d} poisoned {d}\n", .{ rank, s[0], s[1], s[2], @intFromBool(rc.poisoned()) });
    }
}