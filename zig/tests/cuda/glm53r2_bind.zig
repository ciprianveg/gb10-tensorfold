//! STATUS: BLOCKED, not wired (findings §46). r2's `W.Layer.experts` is `cuda_kernels.Experts` — the EXL3
//! expert layout (`gate_ptr`/`up_ptr`, `k2`, `suh_g`/`svh_g`) — not `cuda.experts.Layer`, and r2's `ffnPart`
//! (`cuda_forward.zig:1890`, `expertsPartWith`) consumes only that. §42.2/§45's "experts pass straight through"
//! was wrong at the type level: the pass-through holds for the shared `cuda` module, not for r2's forward. See
//! §46 for the two ways to close it and their costs.
//!
//! A2 — `tf-glm53r2 bind <pack> <rank> <world> <rows>`: the 1.1 loader's weights driven through r2's forward,
//! one layer at a time, with the per-layer digest in `glm53_step.zig`'s exact format so the two tables diff.
//!
//! This file is ours (the A2 seam), and it holds the binder inline on purpose: the binder needs BOTH modules
//! (`glm53`, the 1.1 loader, and `glm53r2`, r2's compute), and a file can belong to one module only — the
//! executable's root module is the only place that owns both.
//!
//! The four transforms (§42.2/§45.1): q_a/kv_a split out of the merged proj at row `q_lora`; gate/up split out of
//! `[gate|up]` for the dense MLP and the shared expert; `wk` DE-PADDED from `[H][qk][lw]` to `[H][nope][lw]`; and
//! the routed experts passed straight through (both sides use `cuda.experts.Layer`, `.nvfp4`).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const glm53 = @import("glm53").family; // the 1.1 loader + config + shard plan (the weights source)
const r2 = @import("glm53r2");

fn argNum(args: []const []const u8, i: usize, dflt: usize) usize {
    if (i >= args.len) return dflt;
    return std.fmt.parseInt(usize, args[i], 10) catch dflt;
}

// --- THE RESIDENCY PROBE (§68.1/§69): harness-side only, no serve-path code -------------------------------------
// Which checkpoint grows ~1.1 GiB a layer? Prints device free/used (cuMemGetInfo: unified on GB10, so it tracks
// the whole pool), VmRSS and MemAvailable at each stage, plus a 2 s sampler while the load runs.
const memProbeTools = struct {
    /// std.Io.Dir readFileAlloc into a FixedBufferAllocator: the probe must not perturb the GPU allocator.
    fn read(_: check.Gpu, buf: []u8, path: [:0]const u8) []const u8 {
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return buf[0..0];
        defer _ = std.c.close(fd);
        const n = std.c.read(fd, buf.ptr, buf.len);
        if (n < 0) return buf[0..0];
        return buf[0..@intCast(n)];
    }
    fn kb(text: []const u8, key: []const u8) u64 {
        var it = std.mem.splitScalar(u8, text, '\n');
        while (it.next()) |line| {
            const t = std.mem.trim(u8, line, " \t\r");
            if (!std.mem.startsWith(u8, t, key)) continue;
            var j = key.len;
            while (j < t.len and (t[j] == ':' or t[j] == ' ' or t[j] == '\t')) : (j += 1) {}
            var k = j;
            while (k < t.len and t[k] >= '0' and t[k] <= '9') : (k += 1) {}
            return std.fmt.parseInt(u64, t[j..k], 10) catch 0;
        }
        return 0;
    }
};

fn memProbe(gpu: check.Gpu, tag: []const u8, tick: usize) void {
    const gib: f64 = 1 << 30;
    const mi = gpu.ctx.memInfo() catch cuda.Context.MemInfo{ .free = 0, .total = 0 };
    var sb: [16384]u8 = undefined;
    var hb: [16384]u8 = undefined;
    const st = memProbeTools.read(gpu, &sb, "/proc/self/status");
    const info = memProbeTools.read(gpu, &hb, "/proc/meminfo");
    const rss = memProbeTools.kb(st, "VmRSS");
    const avail = memProbeTools.kb(info, "MemAvailable");
    std.debug.print("MEM {s} t={d} dev_free={d:.2} dev_used={d:.2} rss={d:.2} host_avail={d:.2} GiB\n", .{
        tag,                                                         tick,
        @as(f64, @floatFromInt(mi.free)) / gib,                       @as(f64, @floatFromInt(mi.total - mi.free)) / gib,
        @as(f64, @floatFromInt(rss)) / (1024 * 1024),                 @as(f64, @floatFromInt(avail)) / (1024 * 1024),
    });
}

const MemSampler = struct {
    gpu: check.Gpu,
    stop: *std.atomic.Value(bool),
    fn run(self: MemSampler) void {
        var tick: usize = 0;
        while (!self.stop.load(.acquire)) {
            std.Io.sleep(self.gpu.io, .fromMilliseconds(2000), .awake) catch {};
            tick += 1;
            memProbe(self.gpu, "sample", tick);
        }
    }
};

/// `glm53_step.zig`'s digest: the bf16 hidden row's sum and absmax (f64 accumulation).
fn digest(host: []const u8) struct { sum: f64, absmax: f64 } {
    const v = std.mem.bytesAsSlice(u16, host);
    var sum: f64 = 0;
    var mx: f64 = 0;
    for (v) |x| {
        const f: f64 = @floatCast(@as(f32, @bitCast(@as(u32, x) << 16)));
        sum += f;
        mx = @max(mx, @abs(f));
    }
    return .{ .sum = sum, .absmax = mx };
}

// --- THE BINDER ------------------------------------------------------------------------------------------------
pub const Stats = struct { layers: usize = 0, dense_mlp: usize = 0, moe: usize = 0, indexers: usize = 0, experts: usize = 0, wk: usize = 0 };

/// E2's side table: one entry a layer, the port's MoE layer (router, NVFP4 experts, shared expert) for the
/// layers the hybrid FFN runs. `null` for the dense layers (r2's own `mlp` branch handles those).
pub const Side = struct {
    moe: []?glm53.moe.Layer,
    scratch: glm53.moe.Scratch,
    kernels: glm53.moe.Kernels,
};

/// r2's dense-linear form: a plain bf16 [n, k] row-major weight, computed by cuBLAS (`Forward.denseGemm`).
pub fn denseLinear(ptr: u64, n: usize, k: usize) r2.exl3.Linear {
    return .{ .words = ptr, .suh = 0, .svh = 0, .counters = 0, .k = k, .n = n, .k2 = 0, .cb = .mul1, .sk = 1, .wk = 1, .dense = true };
}

/// Transform 3: the port zero-pads each head's key block to `qk` rows for its own absorb kernel; r2 reads
/// `[H][nope][lw]`. One device copy a head (`nope * lw * 2` bytes) into a binder-owned buffer.
pub fn depadWk(gpu: check.Gpu, stream: cuda.Stream, wk: u64, heads: usize, nope: usize, lw: usize, qk: usize, w: *r2.weights.Weights, stats: *Stats) !u64 {
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
    var dst = try cuda.DeviceBuffer.alloc(gpu.d, heads * row);
    errdefer dst.free();
    try w.buffers.append(w.gpa, dst);
    for (0..heads) |h| {
        const src_off = h * qk * lw * 2;
        const dst_off = h * row;
        try gpu.d.check(gpu.d.api.cuMemcpyDtoDAsync_v2(dst.ptr + dst_off, wk + src_off, row, stream.handle), "cuMemcpyDtoDAsync");
    }
    stats.wk += 1;
    return dst.ptr;
}

pub fn bind(
    gpu: check.Gpu,
    stream: cuda.Stream,
    pw: *const glm53.weights.Weights,
    rank: usize,
    world: usize,
    heads_share: usize,
    heads_global: usize,
    stats: *Stats,
    side: []?glm53.moe.Layer,
) !r2.weights.Weights {
    const a = gpu.gpa;
    const c = pw.config;
    // TF_BIND_SPLIT=swap: the merged proj's row order was assumed [q_a | kv_a]; swap tests the assumption
    const swap = if (std.c.getenv("TF_BIND_SPLIT")) |v| std.mem.eql(u8, std.mem.span(v), "swap") else false;
    if (swap) std.debug.print("SPLIT order=swap (kv_a first)\n", .{});
    const hidden = c.hidden;
    const nope = c.nope;
    const lw = c.kv_lora;

    var w: r2.weights.Weights = .{
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
    w.layers = try a.alloc(r2.weights.Layer, pw.layers.len);
    errdefer a.free(w.layers);

    for (pw.layers, 0..) |*pl, i| {
        const at = pl.attn;
        var L: r2.weights.Layer = .{
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
        L.wk = try depadWk(gpu, stream, at.wk, at.heads, nope, lw, at.qk, &w, stats);
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
    /// TF_GLM_PROF=1 (the §87 decode profile): the 1.1 Prof attached to the moe kernels; the type comes
    /// off the Kernels field so the harness needs no family-module import. Print/timing only.
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
        const h: *const Hook = @ptrCast(@alignCast(ctx));
        const l = h.side[li] orelse return error.NoMoeLayer;
        if (h.prof) |p| try p.begin(stream, if (prompt) "moe-prompt" else "moe-decode");
        defer if (h.prof) |p| p.end();
        // out is this rank's fp32 [rows][D] partial, the same contract r2's mlp/experts branches fill
        try glm53.moe.block(h.kernels, stream, l, h.scratch, normed, 6144, rows, prompt, out);
    }
};

/// TF_GLM53_DUMP=<dir>: write the layer's hidden rows (bf16) after every layer, so the formal rel-L2 gate can be
/// computed from the two engines' buffers instead of from the cancellation-sensitive signed sum (§58.3).
fn dumpLayer(gpu: check.Gpu, stream: cuda.Stream, dir: []const u8, tag: []const u8, li: usize, ptr: u64, bytes: usize) !void {
    try stream.synchronize();
    const path = try std.fmt.allocPrint(gpu.gpa, "{s}/{s}-L{d}.bin", .{ dir, tag, li });
    defer gpu.gpa.free(path);
    const buf = try gpu.gpa.alloc(u8, bytes);
    defer gpu.gpa.free(buf);
    try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = ptr, .len = bytes, .borrowed = true }).download(0, buf);
    const f = try std.Io.Dir.cwd().createFile(gpu.io, path, .{});
    defer f.close(gpu.io);
    try f.writeStreamingAll(gpu.io, buf);
}

pub fn run(gpu: check.Gpu, args: []const []const u8) !void {
    const a = gpu.gpa;
    const dir = if (args.len > 0) args[0] else return error.NoPack;
    const rank = argNum(args, 1, 0);
    const world = argNum(args, 2, 1);
    const rows = argNum(args, 3, 1);
    if (rows == 0 or rows > 8) return error.BadRows;

    // --- the 1.1 load, verbatim (glm53_run.zig) ----------------------------------------------------------------
    const cfg_path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(cfg_path);
    const json = try std.Io.Dir.cwd().readFileAlloc(gpu.io, cfg_path, a, .limited(1 << 20));
    defer a.free(json);
    var cfg = try glm53.config.parse(a, json);
    if (std.c.getenv("TF_GLM53_LAYERS")) |lz| cfg.layers = try std.fmt.parseInt(u32, std.mem.span(lz), 10);
    cfg.run = cfg.layers;
    cfg.mtp = 0;
    const share = try glm53.shard.Share.of(&cfg, rank, world);
    // TF_BIND_HEADS=share: r2's config.heads gets the PER-RANK count (its EXL packs only ever ran world 1,
    // where global == per-rank, so the question never arose there); default: the global count.
    const heads_share_ = share.mla.len();
    const heads_global_ = if (std.c.getenv("TF_BIND_HEADS")) |v| blk: {
        if (std.mem.eql(u8, std.mem.span(v), "share")) break :blk heads_share_;
        break :blk heads_share_ * world;
    } else heads_share_ * world;

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

    var sampler_stop = std.atomic.Value(bool).init(false);
    var sampler: ?std.Thread = null;
    const sample_on = std.c.getenv("TF_BIND_MEMSAMPLE") != null;
    memProbe(gpu, "after-ckpt-open", 0);
    if (sample_on) sampler = try std.Thread.spawn(.{}, MemSampler.run, .{MemSampler{ .gpu = gpu, .stop = &sampler_stop }});
    const t0 = check.now(gpu.io);
    var pw = try glm53.weights.load(a, gpu.d, stream, tools, &ck, &cfg, share);
    defer pw.deinit();
    try stream.synchronize();
    sampler_stop.store(true, .release);
    if (sampler) |th| th.join();
    memProbe(gpu, "after-weights.load", 0);
    std.debug.print("TRACE loaded layers={d} load_ms={d}\n", .{ cfg.layers, @divTrunc(check.now(gpu.io) - t0, 1_000_000) });

    // the head-count diagnostic (§53.2): every count in play, and the span the digest reads
    std.debug.print("QBN layer0.attn.heads={d} attn.qk={d} attn.lw={d} attn.dv={d} proj.hidden={d} q_b.n_bound={d}\n", .{
        pw.layers[0].attn.heads, pw.layers[0].attn.qk, pw.layers[0].attn.lw, pw.layers[0].attn.dv, cfg.hidden, pw.layers[0].attn.heads * pw.layers[0].attn.qk,
    });
    std.debug.print("HEADS port.cfg={d} share.mla={d} (lo={d} hi={d}) world={d} | r2.w.heads={d} r2.cfg.heads={d} q_b.n={d} bytes_per_row={d}\n", .{
        cfg.heads, share.mla.len(), share.mla.lo, share.mla.hi, world, heads_share_, heads_global_, heads_share_ * (cfg.nope + cfg.rope), heads_share_ * (cfg.nope + cfg.rope) * 2,
    });
    var stats: Stats = .{};
    const side = try a.alloc(?glm53.moe.Layer, pw.layers.len);
    defer a.free(side);
    @memset(side, null);
    var w = try bind(gpu, stream, &pw, rank, world, heads_share_, heads_global_, &stats, side);
    defer a.free(w.layers);
    std.debug.print("BIND layers={d} dense_mlp={d} moe={d} indexers={d} experts={d} wk={d} heads={d}/{d}\n", .{
        stats.layers, stats.dense_mlp, stats.moe, stats.indexers, stats.experts, stats.wk, w.heads, w.config.heads,
    });
    memProbe(gpu, "after-bind()", 0);

    // --- r2's kernels and forward --------------------------------------------------------------------------------
    const aot = if (std.c.getenv("TF_GLM53_AOT")) |v| std.mem.span(v) else "/home/ciprian/aot-world8-dcp2-mirror";
    var k = try r2.kernels.Kernels.load(a, gpu.io, gpu.ctx, aot);
    defer k.deinit();
    memProbe(gpu, "after-kernels.load", 0);

    var b = try r2.forward.Buffers.init(a, gpu.d, &w, rows, 4096 + rows);
    defer b.deinit();
    memProbe(gpu, "after-buffers.init", 0);
    // the census arms (§43/§49): the capacity, kv8 and dcp come from the environment so the SAME driver boots
    // a4a/a4b/a4c/target. `TF_GLM53_CAPACITY` is the stream's context in tokens (default: the rows).
    const kv8 = if (std.c.getenv("TF_GLM53_KV8")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else false;
    const dcp: usize = if (std.c.getenv("TF_GLM53_DCP")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 1 else 1;
    const capacity: usize = if (std.c.getenv("TF_GLM53_CAPACITY")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch rows else rows;
    var st = try r2.forward.State.initSlotsKv8(a, gpu.d, &w, capacity, dcp, 1, kv8);
    memProbe(gpu, "after-state.init", 0);
    std.debug.print("CACHE capacity={d} dcp={d} kv8={} local={d} lat_row={d} bytes={d} ({d:.2} GiB)\n", .{
        capacity, dcp, kv8, st.local, st.lat_row, r2.forward.State.bytesForDcpKv8(&w, capacity, dcp, kv8),
        @as(f64, @floatFromInt(r2.forward.State.bytesForDcpKv8(&w, capacity, dcp, kv8))) / (1 << 30),
    });
    defer st.deinit();

    // TF_GLM53_INV: whose rope frequencies (the two conventions are the first suspect on layer 0).
    //   port    - the 1.1 loader's `w.inv` bytes verbatim (the reference the digest is compared against)
    //   compute - r2's Kernels.invFreq(dim, theta)
    // --- the TP8 communicator (§56.3): the port's proven rendezvous (`Group.openShared`) holds the raw nccl.Comm;
    //     r2's Forward wants its own Communicator view over the same handle (lib pointer included).
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
    const inv_src = if (std.c.getenv("TF_GLM53_INV")) |v| std.mem.span(v) else "port";
    if (!std.mem.eql(u8, inv_src, "compute") and pw.inv != 0) {
        try gpu.d.check(gpu.d.api.cuMemcpyDtoDAsync_v2(inv.ptr, pw.inv, (cfg.rope / 2) * 4, stream.handle), "cuMemcpyDtoDAsync");
        std.debug.print("INV source=port (the loader's bytes)\n", .{});
    } else {
        try k.invFreq(stream, inv.ptr, cfg.rope, cfg.rope_theta);
        std.debug.print("INV source=compute (Kernels.invFreq)\n", .{});
    }

    // the port's MoE block for E2: its kernels come from the assembled port block set, its scratch from the
    // port's own arena layout (the same sizing `glm53_run.zig` uses), so the hybrid runs the audited path.
    // §95 (§92.2's product follow-on): a served boot through this path runs chunks at serve_rows — the
    // binder's `rows` arg (the A4a harness's <=8) would overflow the moe planes exactly as the run lane
    // did (§92: glm53.moe.block writes `rows` rows unblocked). Sized like the driver's §92 fix: the serve
    // rows when TF_GLM53_SERVED, and only the moe block + the shared mm_part placed.
    const hook_served = blk: {
        const v = std.c.getenv("TF_GLM53_SERVED") orelse break :blk false;
        const s = std.mem.span(v);
        break :blk s.len > 0 and !std.mem.eql(u8, s, "0");
    };
    const hook_serve_rows: usize = if (std.c.getenv("TF_GLM53_SERVE_ROWS")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 4096 else 4096;
    const rows_max = @max(@max(rows, 64), if (hook_served) hook_serve_rows else 0);
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

    var hook = Hook{ .kernels = asm_.blocks.moe, .scratch = placed.moe, .side = side };

    var f: r2.forward.Forward = .{ .w = &w, .k = &k, .s = stream, .inv = inv.ptr, .dcp = dcp, .moe_hook = .{ .ctx = &hook, .has = Hook.has, .call = Hook.call } };
    f.comm = if (comm) |*c| c else null;
    const cublas_lib = try cuda.cublas.Library.open();
    defer @constCast(&cublas_lib).close();
    var ws = try cuda.DeviceBuffer.alloc(gpu.d, 32 << 20);
    defer ws.free();
    var blas = try cuda.cublas.Blas.init(&cublas_lib, ws.ptr, ws.len, cuda.cublas.default_math);
    defer blas.deinit();
    f.blas = &blas;

    // --- the ladder: one id row, a layer at a time, the digest after each ------------------------------------
    const ids = try a.alloc(u32, rows);
    defer a.free(ids);
    for (ids, 0..) |*v, i| v.* = 1000 + @as(u32, @intCast(i));
    try f.embed(&b, ids);
    try f.setPos(&st, 0);
    const host = try a.alloc(u8, rows * cfg.hidden * 4);
    defer a.free(host);
    const stages = if (std.c.getenv("TF_GLM53_STAGES")) |v| !(v[0] == 0 or std.mem.eql(u8, std.mem.span(v), "0")) else false;
    const dig = struct {
        fn f32at(g: check.Gpu, s: cuda.Stream, hostbuf: []u8, ptr: u64, len: usize, name: []const u8) !void {
            try s.synchronize();
            try g.d.check(g.d.api.cuCtxSynchronize(), "cuCtxSynchronize");
            try (cuda.DeviceBuffer{ .d = g.d, .ptr = ptr, .len = len, .borrowed = true }).download(0, hostbuf[0..len]);
            const v = std.mem.bytesAsSlice(f32, hostbuf[0..len]);
            var sum: f64 = 0;
            var mx: f64 = 0;
            for (v) |x| {
                sum += x;
                mx = @max(mx, @abs(x));
            }
            std.debug.print("  STAGE {s} sum={e} absmax={e}\n", .{ name, sum, mx });
        }

        fn at(g: check.Gpu, s: cuda.Stream, hostbuf: []u8, ptr: u64, len: usize, name: []const u8) !void {
            try s.synchronize();
            try g.d.check(g.d.api.cuCtxSynchronize(), "cuCtxSynchronize");
            try (cuda.DeviceBuffer{ .d = g.d, .ptr = ptr, .len = len, .borrowed = true }).download(0, hostbuf[0..len]);
            const d = digest(hostbuf[0..len]);
            std.debug.print("  STAGE {s} sum={e} absmax={e}\n", .{ name, d.sum, d.absmax });
        }
    };
    if (stages) try dig.at(gpu, stream, host, b.x, rows * cfg.hidden * 2, "embed");
    // §71 identical-input isolation: TF_GLM53_START / TF_GLM53_INJECT seed b.x from another run's dump.
    var start: usize = 0;
    if (std.c.getenv("TF_GLM53_START")) |v| start = try std.fmt.parseInt(usize, std.mem.span(v), 10);
    if (start > cfg.layers) return error.BadStart;
    if (std.c.getenv("TF_GLM53_INJECT")) |p| {
        const want = rows * cfg.hidden * 2;
        const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, std.mem.span(p), a, .unlimited);
        if (bytes.len != want) return error.BadInject;
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = b.x, .len = want, .borrowed = true }).upload(0, bytes);
        std.debug.print("INJECT {s} rows={d} hidden={d}\n", .{ std.mem.span(p), rows, cfg.hidden });
    }
    for (start..cfg.layers) |li| {
        const L = &w.layers[li];
        if (stages) {
            // layerWith's body, stage by stage (its parts are pub for this)
            try f.t().rmsnorm(b.x, cfg.hidden, L.input_norm, b.normed, cfg.hidden, rows, cfg.hidden, cfg.eps);
            try dig.at(gpu, stream, host, b.normed, rows * cfg.hidden * 2, "L0.in_norm");
            try f.attentionPart(&b, L, st.kc[li], st.ic[li], st.pos, rows, null, 0);
            // the query side, after the pack: q_b's output and the absorb's two halves
            try dig.at(gpu, stream, host, b.qn, rows * cfg.q_lora * 2, "L0.qn");
            try dig.at(gpu, stream, host, b.q, rows * w.heads * (cfg.nope + cfg.rope) * 2, "L0.q_postqb");
            try dig.at(gpu, stream, host, b.qlat, rows * w.heads * cfg.kv_lora * 2, "L0.qlat");
            try dig.at(gpu, stream, host, b.qrot, rows * w.heads * cfg.rope * 2, "L0.qrot");
            // the seam's two halves: the cache row as written, and the absorbed latent output before expandV
            try dig.at(gpu, stream, host, st.kc[li], st.lat_row, "L0.cache_row0");
            try dig.at(gpu, stream, host, b.ol, rows * w.heads * cfg.kv_lora * 2, "L0.ol");
            try dig.at(gpu, stream, host, b.o, rows * w.heads * cfg.v_dim * 2, "L0.attn_out");
            const g1 = try f.gather(&b, rows);
            try f.t().residualAdd(b.x, g1.ptr, rows, cfg.hidden, g1.world);
            try dig.at(gpu, stream, host, b.x, rows * cfg.hidden * 2, "L0.attn_res");
            try f.t().rmsnorm(b.x, cfg.hidden, L.post_attn_norm, b.normed, cfg.hidden, rows, cfg.hidden, cfg.eps);
            try f.ffnPart(&b, L, rows);
            try dig.f32at(gpu, stream, host, b.part, rows * cfg.hidden * 4, "L0.ffn_part");
            const g2 = try f.gather(&b, rows);
            try f.t().residualAdd(b.x, g2.ptr, rows, cfg.hidden, g2.world);
            try dig.at(gpu, stream, host, b.x, rows * cfg.hidden * 2, "L0.ffn_res");
        } else {
            try f.layer(&b, &st, li, rows, null);
            try stream.synchronize();
            try gpu.d.check(gpu.d.api.cuCtxSynchronize(), "cuCtxSynchronize");
            try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = b.x, .len = rows * cfg.hidden * 2, .borrowed = true }).download(0, host[0 .. rows * cfg.hidden * 2]);
            const dg = digest(host[0 .. rows * cfg.hidden * 2]);
            std.debug.print("  layer {d} digest sum={e} absmax={e}\n", .{ li, dg.sum, dg.absmax });
            if (std.c.getenv("TF_GLM53_DUMP")) |dd| try dumpLayer(gpu, stream, std.mem.span(dd), "r2", li, b.x, rows * cfg.hidden * 2);
        }
    }
    try stream.synchronize();
    check.pass("bind: {d} layers on r2's forward over the 1.1-loaded weights", .{cfg.layers});
}