//! A TP rank's GLM-5.3 weights on CUDA from the NVFP4-CSF pack: MLA (BF16), the five-tensor DSA
//! indexer, dense layers 0-2 (BF16), the routed experts (NVFP4 with plain or CSF scale planes,
//! decoded rank-locally), the BF16 shared expert and router, and the MTP layer (BF16 experts
//! quantized on the device, as the Flash family does).
//!
//! The host side is split from the uploads: `gather` produces exactly the bytes a rank holds, so the
//! pack can be validated — and the rank cuts digest-checked — on a CPU-only box.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const shard = @import("cuda_shard.zig");
const plan = @import("plan.zig");
const quant = @import("quant.zig");

const Tensor = core.checkpoint.Tensor;
const X = cuda.experts.Nvfp4Experts;

/// The loader looks up the pack's own names (`plan.Item.name` carries "model." where the pack has it);
/// the checkpoint is opened unfiltered. The Flash family's "model.language_model." does not apply.
pub const prefix = "";

/// The DSA indexer's weights, replicated on every rank (GLM-5.3's five tensors: no pool-compress gate/ape).
pub const Index = struct {
    kw: u64, // bf16 [i_dim][dims]
    wproj: u64, // bf16 [i_heads][dims]
    ln_w: u64, // bf16 [i_dim]
    ln_b: u64, // bf16 [i_dim]
    qb: u64, // bf16 [i_heads * i_dim][q_lora]
    heads: usize,
    d: usize,
};

/// One MLA layer on a rank: proj [q_a | kv_a], norms, q_b by heads, kv_b as wk/wv per head, o by columns.
pub const Attn = struct {
    proj: u64,
    q_norm: u64,
    kv_norm: u64,
    q_b: u64,
    /// [heads][qk][lw]: the checkpoint's nope (192) rows a head, then (qk - nope) = 64 ZERO rows, so the
    /// absorb entry (which is monomorphized at D = qk = 256) reads in bounds and adds exactly zero for the
    /// rope half of the query. See the kv_b arm of `Build.item`.
    wk: u64,
    wv: u64,
    o: u64,
    index: Index,
    heads: usize,
    qk: usize,
    dv: usize,
    lw: usize,
    q_lora: usize,
    dims: usize,
    eps: f32,
    dense_limit: usize,
};

pub const Moe = struct {
    router: u64,
    bias: u64,
    routed: cuda.experts.Layer,
    count: usize,
    shared_gu: u64,
    shared_down: u64,
    shared: usize,
    top_k: usize,
    scale: f32,
    norm: bool,
    /// W4A4 static input scales from the pack's sidecar: gate/up per expert (2 x count) and down
    /// (count). W4A16 math ignores them; a W4A4 path multiplies the incoming activation by 1/scale.
    in_up: ?[]f32 = null,
    in_down: ?[]f32 = null,
};

pub const DenseLayer = struct { w: Dense, width: usize, dims: usize, limit: f32 };

pub const Ffn = union(enum) { dense: DenseLayer, moe: Moe };

pub const Layer = struct { index: u32, in_norm: u64, post_norm: u64, attn: Attn, ffn: Ffn };

/// The plan-vs-loaded join: every module `plan.walk` names must be visible in `Weights.named` under
/// the loader's own short name for that role, so the loader contract is checked per layer instead of
/// implied by the pack test. Roles whose tensor is stacked into another buffer (q_a into dsa.proj, the
/// dense/shared gate into their .gu) are covered by the buffer they were merged into and are not counted
/// twice.
pub const Census = struct {
    expected: usize = 0,
    found: usize = 0,
    merged: usize = 0, // stacked with a later role: no name of its own
    missing: usize = 0,
    first_missing: [160]u8 = @splat(0),

    pub fn join(w: *const Weights, c: *const cfg.Config) !Census {
        var self: Census = .{};
        const Ctx = struct {
            w: *const Weights,
            s: *Census,
            pub fn item(ctx: *@This(), it: plan.Item) !void {
                ctx.s.*.expected += 1;
                var buf: [160]u8 = undefined;
                const short = switch (it.role) {
                    .embed => plan.prefix ++ "embed_tokens.weight",
                    .head => "lm_head.weight",
                    .final_norm => plan.prefix ++ "norm.weight",
                    // stacked: their bytes live in the buffer the later role uploads
                    .q_a, .dense_gate, .shared_gate => {
                        ctx.s.*.merged += 1;
                        return;
                    },
                    .in_norm => try std.fmt.bufPrint(&buf, "layers.{d}.input_norm", .{it.layer}),
                    .post_norm => try std.fmt.bufPrint(&buf, "layers.{d}.post_norm", .{it.layer}),
                    .kv_a => try std.fmt.bufPrint(&buf, "layers.{d}.dsa.proj", .{it.layer}),
                    .q_a_norm => try std.fmt.bufPrint(&buf, "layers.{d}.dsa.q_norm", .{it.layer}),
                    .kv_a_norm => try std.fmt.bufPrint(&buf, "layers.{d}.dsa.kv_norm", .{it.layer}),
                    .q_b => try std.fmt.bufPrint(&buf, "layers.{d}.dsa.q_b", .{it.layer}),
                    .kv_b => try std.fmt.bufPrint(&buf, "layers.{d}.dsa.absorb.wk", .{it.layer}),
                    .o_proj => try std.fmt.bufPrint(&buf, "layers.{d}.dsa.o", .{it.layer}),
                    .idx_wk => try std.fmt.bufPrint(&buf, "layers.{d}.dsa.index.kw", .{it.layer}),
                    .idx_wproj => try std.fmt.bufPrint(&buf, "layers.{d}.dsa.index.wproj", .{it.layer}),
                    .idx_wqb => try std.fmt.bufPrint(&buf, "layers.{d}.dsa.index.qb", .{it.layer}),
                    .idx_kn_w => try std.fmt.bufPrint(&buf, "layers.{d}.dsa.index.ln_w", .{it.layer}),
                    .idx_kn_b => try std.fmt.bufPrint(&buf, "layers.{d}.dsa.index.ln_b", .{it.layer}),
                    .dense_up => try std.fmt.bufPrint(&buf, "layers.{d}.mlp.gu", .{it.layer}),
                    .dense_down => try std.fmt.bufPrint(&buf, "layers.{d}.mlp.down", .{it.layer}),
                    .shared_up => try std.fmt.bufPrint(&buf, "layers.{d}.moe.shared.gu", .{it.layer}),
                    .shared_down => try std.fmt.bufPrint(&buf, "layers.{d}.moe.shared.down", .{it.layer}),
                    .router => try std.fmt.bufPrint(&buf, "layers.{d}.moe.router", .{it.layer}),
                    .router_bias => try std.fmt.bufPrint(&buf, "layers.{d}.moe.bias", .{it.layer}),
                    .eh_proj => try std.fmt.bufPrint(&buf, "layers.{d}.mtp.eh", .{it.layer}),
                    .enorm => try std.fmt.bufPrint(&buf, "layers.{d}.mtp.enorm", .{it.layer}),
                    .hnorm => try std.fmt.bufPrint(&buf, "layers.{d}.mtp.hnorm", .{it.layer}),
                    .shared_head_norm => try std.fmt.bufPrint(&buf, "layers.{d}.mtp.shared_head_norm", .{it.layer}),
                    // routed experts: the packed tables keep the plan name (fp4) or the bf16 quantized
                    // tables under layers.N.moe.experts.*
                    .expert_gate, .expert_up, .expert_down => blk: {
                        // the routed experts land as packed tables under layers.N.moe.experts.*
                        // (fp4 words/scales or the MTP layer's bf16 quantized tables), not under the
                        // checkpoint's tensor name
                        const short = try std.fmt.bufPrint(&buf, "layers.{d}.moe.experts.", .{it.layer});
                        if (ctx.w.hasPrefix(short)) {
                            ctx.s.*.found += 1;
                            return;
                        }
                        break :blk short;
                    },
                };
                if (ctx.w.find(short) != null) {
                    ctx.s.*.found += 1;
                } else {
                    ctx.s.*.missing += 1;
                    if (ctx.s.*.missing == 1) {
                        const n = @min(short.len, ctx.s.*.first_missing.len);
                        @memcpy(ctx.s.*.first_missing[0..n], short[0..n]);
                    }
                }
            }
        };
        var ctx: Ctx = .{ .w = w, .s = &self };
        try plan.walk(c, &ctx);
        return self;
    }
};

/// The MTP head: norms, eh_proj, the layer itself (BF16 experts quantized on the device), shared_head.norm.
pub const Mtp = struct { enorm: u64, hnorm: u64, eh: u64, norm: u64, layer: Layer };

/// The kernels a load runs: tf_pack_dense for the NVFP4 uploads, and the quantizer for the MTP layer's
/// BF16 experts (the Flash family's cuda_quant set).
pub const Tools = struct { pack_dense: cuda.Function, quant: ?*const @import("quant.zig").Kernels = null };

/// A loaded tensor by the pack's path without "model.", for digests against the checkpoint.
pub const Named = struct { name: []u8, ptr: u64, len: usize };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: cfg.Config,
    share: shard.Share,
    embed: u64 = 0,
    inv: u64 = 0, // the RoPE frequencies the index plane's writer rotates with: [qk_rope_head_dim / 2] f32
    layers: []Layer = &.{},
    norm: u64 = 0,
    head: u64 = 0,
    mtp: ?Mtp = null,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    named: std.ArrayList(Named) = .empty,
    scales: std.ArrayList([]f32) = .empty, // the sidecar's input scales, freed with the weights

    pub fn deinit(w: *Weights) void {
        for (w.scales.items) |s| w.gpa.free(s);
        w.scales.deinit(w.gpa);
        for (w.buffers.items) |*b| b.free();
        for (w.named.items) |n| w.gpa.free(n.name);
        w.buffers.deinit(w.gpa);
        w.named.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    pub fn hasPrefix(w: *const Weights, p: []const u8) bool {
        for (w.named.items) |n| if (std.mem.startsWith(u8, n.name, p)) return true;
        return false;
    }

    pub fn find(w: *const Weights, name: []const u8) ?Named {
        for (w.named.items) |n| if (std.mem.eql(u8, n.name, name)) return n;
        return null;
    }
};

pub const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    tools: Tools,
    w: *Weights,
    ck: *core.Checkpoint,
    host: std.ArrayList(u8) = .empty,
    name_buf: [192]u8 = undefined,

    pub fn get(L: *Loader, it: plan.Item) !Tensor {
        const name = try std.fmt.bufPrint(&L.name_buf, prefix ++ "{s}", .{it.name});
        return L.ck.get(name);
    }

    fn put(L: *Loader, name: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(L.d, L.host.items);
        try L.w.buffers.append(L.gpa, b);
        try L.w.named.append(L.gpa, .{ .name = try L.gpa.dupe(u8, name), .ptr = b.ptr, .len = b.len });
        L.host.clearRetainingCapacity();
        return b.ptr;
    }

    /// Elements [lo, hi) of `t` flat as `to` (bf16 or fp32; other dtypes only as stored).
    fn add(L: *Loader, t: Tensor, lo: usize, hi: usize, to: core.checkpoint.DType) !void {
        if (t.dtype == to) {
            const size = t.bytes.len / t.numel();
            return L.host.appendSlice(L.gpa, t.bytes[lo * size .. hi * size]);
        }
        for (lo..hi) |i| {
            const v: f32 = switch (t.dtype) {
                .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16),
                .f32 => std.mem.bytesToValue(f32, t.bytes[4 * i ..][0..4]),
                else => return error.UnexpectedTensor,
            };
            switch (to) {
                .f32 => try L.host.appendSlice(L.gpa, std.mem.asBytes(&v)),
                .bf16 => try L.host.appendSlice(L.gpa, std.mem.asBytes(&toBf16(v))),
                else => return error.UnexpectedTensor,
            }
        }
    }

    /// Rows [lo, hi) of `t`'s outermost axis (`total` long; a vector's elements): the row cut.
    fn rows(L: *Loader, t: Tensor, lo: usize, hi: usize, total: usize, to: core.checkpoint.DType) !void {
        const per = t.numel() / total;
        if (per * total != t.numel()) return error.UnexpectedTensor;
        try L.add(t, lo * per, hi * per, to);
    }

    fn whole(L: *Loader, t: Tensor, to: core.checkpoint.DType) !void {
        try L.add(t, 0, t.numel(), to);
    }

    /// Columns [lo, hi) of every row of a matrix: the column cut.
    fn cols(L: *Loader, t: Tensor, lo: usize, hi: usize, to: core.checkpoint.DType) !void {
        if (t.rank != 2) return error.UnexpectedTensor;
        const k = t.dim(1);
        for (0..t.dim(0)) |r| try L.add(t, r * k + lo, r * k + hi, to);
    }

    /// An NVFP4 linear's rows [lo, hi), inputs [klo, khi) as Fp4Linear.from_checkpoint (tf_pack_dense, packScales).
    /// Scale planes come from plain or CSF-compressed storage, decoded rank-locally (core.nvfp4_slice).
    fn fp4(L: *Loader, it: plan.Item, lo: usize, hi: usize, klo: usize, khi: usize) !cuda.qmmf.Weight {
        var l = try core.nvfp4_slice.read(L.ck, L.gpa, plan.module(it.name), .{ .from = lo, .to = hi }, .{ .from = klo, .to = khi });
        defer l.deinit(L.gpa);
        const n = hi - lo;
        const k = khi - klo;
        try L.host.appendSlice(L.gpa, l.codes);
        var src = try cuda.DeviceBuffer.fromHost(L.d, L.host.items);
        defer src.free();
        L.host.clearRetainingCapacity();
        const npad = cuda.nvfp4.padded(n);
        const kg = k / 64;
        const total: u64 = @as(u64, npad / 64) * kg * 512;
        const words = try cuda.DeviceBuffer.alloc(L.d, total * 4);
        try L.w.buffers.append(L.gpa, words);
        try L.w.named.append(L.gpa, .{ .name = try std.fmt.allocPrint(L.gpa, "{s}.words", .{it.name}), .ptr = words.ptr, .len = words.len });
        var args: cuda.Args = .{};
        args.add(src.ptr);
        for ([_]usize{ n, k / 8, kg }) |v| args.add(@as(c_int, @intCast(v)));
        args.add(words.ptr);
        args.add(@as(c_longlong, @intCast(total)));
        try cuda.launch.launch(L.tools.pack_dense, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, L.stream, &args);
        try L.stream.synchronize();
        try L.host.resize(L.gpa, npad * k / 16);
        cuda.nvfp4.packScales(L.host.items, l.scales, n, k);
        const bs_name = try std.fmt.allocPrint(L.gpa, "{s}.scales", .{it.name});
        defer L.gpa.free(bs_name);
        const bs = try L.put(bs_name);
        return cuda.nvfp4.weight(words.ptr, bs, l.global, n, k);
    }

    /// One module's rank-local host bytes, cut by the plan role and this rank's share. CPU-only.
    pub fn gather(L: *Loader, it: plan.Item) !void {
        const c = &L.w.config;
        const s = L.w.share;
        const t = try L.get(it);
        switch (it.role) {
            .embed, .head => try L.rows(t, s.vocab.lo, s.vocab.hi, c.vocab, .bf16),
            .final_norm, .in_norm, .post_norm, .q_a_norm, .kv_a_norm, .idx_kn_w, .idx_kn_b, .enorm, .hnorm, .shared_head_norm => try L.whole(t, .bf16),
            .q_a, .kv_a, .idx_wk, .idx_wproj, .idx_wqb, .router, .router_bias => try L.whole(t, if (it.role == .router_bias) .f32 else .bf16),
            .q_b => try L.rows(t, s.mla.lo * c.qHead(), s.mla.hi * c.qHead(), c.heads * c.qHead(), .bf16),
            .kv_b => { // the absorbed keys and values are cut from the full [heads (nope + v)] rows by the loader
                try L.whole(t, .bf16);
            },
            .o_proj => try L.cols(t, s.mla.lo * c.v_dim, s.mla.hi * c.v_dim, .bf16),
            .dense_gate, .dense_up => try L.rows(t, s.dense.lo, s.dense.hi, c.dense_inter, .bf16),
            .dense_down => try L.cols(t, s.dense.lo, s.dense.hi, .bf16),
            .shared_gate, .shared_up => try L.rows(t, s.shared.lo, s.shared.hi, c.shared_inter, .bf16),
            .shared_down => try L.cols(t, s.shared.lo, s.shared.hi, .bf16),
            .expert_gate, .expert_up, .expert_down => {
                if (it.flavor == .fp4) {
                    // the rank's slice through the production reader (plain or CSF scales):
                    // gate/up cut the intermediate rows, down cuts the input columns
                    var l = if (it.role == .expert_down)
                        try core.nvfp4_slice.read(L.ck, L.gpa, plan.module(it.name), .{ .from = 0, .to = c.hidden }, .{ .from = s.moe.lo, .to = s.moe.hi })
                    else
                        try core.nvfp4_slice.read(L.ck, L.gpa, plan.module(it.name), .{ .from = s.moe.lo, .to = s.moe.hi }, .{ .from = 0, .to = c.hidden });
                    defer l.deinit(L.gpa);
                    try L.host.appendSlice(L.gpa, l.codes);
                } else {
                    try L.rows(t, s.moe.lo, s.moe.hi, c.moe_inter, .bf16);
                }
            },
            .eh_proj => try L.whole(t, .bf16),
        }
    }
};

/// Host bf16 from fp32: round to nearest even (torch's float -> bfloat16), NaN kept quiet.
fn toBf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// A dense layer's weights: the pack keeps layers 0-2 BF16 (the quant ignore-list); the NVFP4 arm is
/// here for a checkpoint that quantizes them (the Flash family's shape).
pub const Dense = union(enum) {
    bf16: struct { gu: u64, down: u64 }, // [gate | up] rows [2 width][dims], down [dims][width]
    nvfp4: struct { gate: cuda.qmmf.Weight, up: cuda.qmmf.Weight, down: cuda.qmmf.Weight },
};

/// One layer's gathered-and-uploaded modules; the walk fills it, `finish` assembles the struct.
const Build = struct {
    in_norm: u64 = 0,
    post_norm: u64 = 0,
    q_norm: u64 = 0,
    kv_norm: u64 = 0,
    q_b: u64 = 0,
    wk: u64 = 0,
    wv: u64 = 0,
    o: u64 = 0,
    kw: u64 = 0,
    wproj: u64 = 0,
    ln_w: u64 = 0,
    ln_b: u64 = 0,
    idx_qb: u64 = 0,
    gu: u64 = 0,
    down: u64 = 0,
    router: u64 = 0,
    bias: u64 = 0,
    shared_gu: u64 = 0,
    shared_down: u64 = 0,
    proj: std.ArrayList(u8) = .empty, // [q_a | kv_a] stacked until the walk ends
    gu_host: std.ArrayList(u8) = .empty, // [gate | up] stacked until the walk ends

    fn item(self: *@This(), L: *Loader, it: plan.Item) !void {
        const c = &L.w.config;
        const s = L.w.share;
        var pre_buf: [64]u8 = undefined;
        const pre = try std.fmt.bufPrint(&pre_buf, "layers.{d}.", .{it.layer});
        var buf: [96]u8 = undefined;
        switch (it.role) {
            .in_norm => {
                try L.whole(try L.get(it), .bf16);
                self.in_norm = try L.put(try std.fmt.bufPrint(&buf, "{s}input_norm", .{pre}));
            },
            .post_norm => {
                try L.whole(try L.get(it), .bf16);
                self.post_norm = try L.put(try std.fmt.bufPrint(&buf, "{s}post_norm", .{pre}));
            },
            .q_a, .kv_a => { // stacked [q_a | kv_a]: the walk visits q_a first
                try L.whole(try L.get(it), .bf16);
                try self.proj.appendSlice(L.gpa, L.host.items);
                L.host.clearRetainingCapacity();
            },
            .q_a_norm => {
                try L.whole(try L.get(it), .bf16);
                self.q_norm = try L.put(try std.fmt.bufPrint(&buf, "{s}dsa.q_norm", .{pre}));
            },
            .kv_a_norm => {
                try L.whole(try L.get(it), .bf16);
                self.kv_norm = try L.put(try std.fmt.bufPrint(&buf, "{s}dsa.kv_norm", .{pre}));
            },
            .q_b => {
                try L.rows(try L.get(it), s.mla.lo * c.qHead(), s.mla.hi * c.qHead(), c.heads * c.qHead(), .bf16);
                self.q_b = try L.put(try std.fmt.bufPrint(&buf, "{s}dsa.q_b", .{pre}));
            },
            .kv_b => { // the absorbed keys then values, per head: wk's [nope) rows, wv's [v_dim) rows
                const t = try L.get(it);
                const per = c.nope + c.v_dim;
                const total = c.heads * per;
                // The absorb entry (`tf_glm_absorb_q`) is monomorphized for the Flash shape: D = 256 rows a
                // head and a q row 256 wide. This checkpoint's absorbed key is qk_nope = 192 rows a head and
                // its query is 256 wide (192 nope + 64 rope), so the kernel reads 64 rows a head past the
                // key block - for the last head, past the buffer, into whatever the allocator put there
                // (the layer-1 NaN). Padding each head's key block with (qk - nope) ZERO rows makes the
                // kernel's extra 64 products exactly zero: the same latent the canonical r2 engine's
                // specialized absorb (dn = 192, q stride = 256) computes, without touching the kernel.
                const qk = c.qHead();
                if (qk <= c.nope) return error.BadAbsorbShape;
                for (s.mla.lo..s.mla.hi) |head| {
                    try L.rows(t, head * per, head * per + c.nope, total, .bf16);
                    try L.host.appendNTimes(L.gpa, 0, (qk - c.nope) * c.kv_lora * 2);
                }
                self.wk = try L.put(try std.fmt.bufPrint(&buf, "{s}dsa.absorb.wk", .{pre}));
                for (s.mla.lo..s.mla.hi) |head| try L.rows(t, head * per + c.nope, (head + 1) * per, total, .bf16);
                self.wv = try L.put(try std.fmt.bufPrint(&buf, "{s}dsa.absorb.wv", .{pre}));
            },
            .o_proj => {
                try L.cols(try L.get(it), s.mla.lo * c.v_dim, s.mla.hi * c.v_dim, .bf16);
                self.o = try L.put(try std.fmt.bufPrint(&buf, "{s}dsa.o", .{pre}));
            },
            .idx_wk => {
                try L.whole(try L.get(it), .bf16);
                self.kw = try L.put(try std.fmt.bufPrint(&buf, "{s}dsa.index.kw", .{pre}));
            },
            .idx_wproj => {
                try L.whole(try L.get(it), .bf16);
                self.wproj = try L.put(try std.fmt.bufPrint(&buf, "{s}dsa.index.wproj", .{pre}));
            },
            .idx_wqb => {
                try L.whole(try L.get(it), .bf16);
                self.idx_qb = try L.put(try std.fmt.bufPrint(&buf, "{s}dsa.index.qb", .{pre}));
            },
            .idx_kn_w => {
                try L.whole(try L.get(it), .bf16);
                self.ln_w = try L.put(try std.fmt.bufPrint(&buf, "{s}dsa.index.ln_w", .{pre}));
            },
            .idx_kn_b => {
                try L.whole(try L.get(it), .bf16);
                self.ln_b = try L.put(try std.fmt.bufPrint(&buf, "{s}dsa.index.ln_b", .{pre}));
            },
            .dense_gate, .dense_up => { // stacked [gate | up]; the walk visits gate first
                try L.rows(try L.get(it), s.dense.lo, s.dense.hi, c.dense_inter, .bf16);
                try self.gu_host.appendSlice(L.gpa, L.host.items);
                L.host.clearRetainingCapacity();
            },
            .dense_down => {
                try L.cols(try L.get(it), s.dense.lo, s.dense.hi, .bf16);
                self.down = try L.put(try std.fmt.bufPrint(&buf, "{s}mlp.down", .{pre}));
            },
            .shared_gate, .shared_up => {
                try L.rows(try L.get(it), s.shared.lo, s.shared.hi, c.shared_inter, .bf16);
                try self.gu_host.appendSlice(L.gpa, L.host.items);
                L.host.clearRetainingCapacity();
            },
            .shared_down => {
                try L.cols(try L.get(it), s.shared.lo, s.shared.hi, .bf16);
                self.shared_down = try L.put(try std.fmt.bufPrint(&buf, "{s}moe.shared.down", .{pre}));
            },
            .router => {
                try L.whole(try L.get(it), .bf16);
                self.router = try L.put(try std.fmt.bufPrint(&buf, "{s}moe.router", .{pre}));
            },
            .router_bias => {
                try L.whole(try L.get(it), .f32);
                self.bias = try L.put(try std.fmt.bufPrint(&buf, "{s}moe.bias", .{pre}));
            },
            .expert_gate, .expert_up, .expert_down, .eh_proj, .enorm, .hnorm, .shared_head_norm => {},
            else => return error.UnexpectedRole,
        }
    }
};

/// The expert tables plus the W4A4 static input scales the pack's sidecar carries (host-side).
const ExpertTables = struct { layer: cuda.experts.Layer, in_up: ?[]f32 = null, in_down: ?[]f32 = null };

/// A layer's routed experts from the pack: the rank's slice through the production reader (plain or
/// CSF scales), packed as Experts4 host-side, one upload per table.
const empty: core.nvfp4_slice.Linear = .{ .n = 0, .k = 0, .codes = &.{}, .scales = &.{}, .global = 0 };

var nvfp4_logged = false;

fn packExperts(L: *Loader, i: u32) !ExpertTables {
    const c = &L.w.config;
    const s = L.w.share;
    const d: usize = c.hidden;
    const ni = s.moe.len();
    const count: usize = c.experts;
    const up_len = X.bytes(1, ni, d, 2) / 4;
    const down_len = X.bytes(1, d, ni, 1) / 4;
    const up = try L.gpa.alloc(u32, count * up_len);
    defer L.gpa.free(up);
    const down = try L.gpa.alloc(u32, count * down_len);
    defer L.gpa.free(down);
    const up_scale = try L.gpa.alloc(f32, count * 2);
    defer L.gpa.free(up_scale);
    const down_scale = try L.gpa.alloc(f32, count);
    defer L.gpa.free(down_scale);
    var name_buf: [96]u8 = undefined;
    for (0..count) |e| {
        // Safe-init: a failed read mid-loop must not defer-free an undefined tail (the 0xaa crash
        // class); an empty Linear's deinit frees nothing and the error propagates loudly instead.
        var ls: [3]core.nvfp4_slice.Linear = .{empty, empty, empty};
        defer for (&ls) |*l| l.deinit(L.gpa);
        inline for (.{ "gate_proj", "up_proj" }, 0..) |proj, m| {
            const name = try plan.expertName(&name_buf, c, i, e, proj);
            ls[m] = core.nvfp4_slice.read(L.ck, L.gpa, plan.module(name), .{ .from = s.moe.lo, .to = s.moe.hi }, .{ .from = 0, .to = d }) catch |err| {
                std.log.info("nvfp4fail layer={d} e={d} proj={s} name={s} err={s}", .{ i, e, proj, name, @errorName(err) });
                return err;
            };
        }
        {
            const name = try plan.expertName(&name_buf, c, i, e, "down_proj");
            ls[2] = try core.nvfp4_slice.read(L.ck, L.gpa, plan.module(name), .{ .from = 0, .to = d }, .{ .from = s.moe.lo, .to = s.moe.hi });
        }
        if (!nvfp4_logged) {
            nvfp4_logged = true;
            for (&ls, 0..) |*l, q| std.log.info("nvfp4dbg l={d} e={d} ls[{d}] codes=0x{x}+{d} scales=0x{x}+{d} g={d}", .{ i, e, q, @intFromPtr(l.codes.ptr), l.codes.len, @intFromPtr(l.scales.ptr), l.scales.len, l.global });
        }
        X.packGateUp(up[e * up_len ..][0..up_len], .{ ls[0].codes, ls[0].scales }, .{ ls[1].codes, ls[1].scales }, ni, d);
        X.packOne(down[e * down_len ..][0..down_len], ls[2].codes, ls[2].scales, d, ni);
        up_scale[e * 2] = ls[0].global;
        up_scale[e * 2 + 1] = ls[1].global;
        down_scale[e] = ls[2].global;
    }
    var ptrs: [4]u64 = undefined;
    var ptr_buf: [96]u8 = undefined;
    inline for (&ptrs, [_][]const u8{ std.mem.sliceAsBytes(up), std.mem.sliceAsBytes(down), std.mem.sliceAsBytes(up_scale), std.mem.sliceAsBytes(down_scale) }, [_][]const u8{ "up", "down", "up_scale", "down_scale" }) |*p, bytes, what| {
        try L.host.appendSlice(L.gpa, bytes);
        p.* = try L.put(try std.fmt.bufPrint(&ptr_buf, "layers.{d}.moe.experts.{s}", .{ i, what }));
    }
    // the W4A4 static input scales ride along (host-side; W4A16 ignores them, a W4A4 path uses them)
    const ins = try expertInputScales(L.gpa, L.ck, c, i);
    try L.w.scales.append(L.gpa, ins.up);
    try L.w.scales.append(L.gpa, ins.down);
    return .{ .layer = .{ .format = .nvfp4, .up = ptrs[0], .down = ptrs[1], .up_scale = ptrs[2], .down_scale = ptrs[3], .width = ni, .dims = d, .experts = count, .limit = c.swiglu_limit }, .in_up = ins.up, .in_down = ins.down };
}

/// The MTP layer's experts are BF16 in the pack: cut to the rank and quantized on the device, 8 at a
/// time, the scale as weight_scale_2 (the Flash family's quantizedExperts, same kernel).
fn quantizeExperts(L: *Loader, i: u32) !ExpertTables {
    const qk = L.tools.quant orelse return error.NoQuantizer;
    const c = &L.w.config;
    const s = L.w.share;
    const d: usize = c.hidden;
    const ni = s.moe.len();
    const count: usize = c.experts;
    const up_len = X.bytes(1, ni, d, 2) / 4;
    const down_len = X.bytes(1, d, ni, 1) / 4;
    const chunk = 8;
    var wb = try cuda.DeviceBuffer.alloc(L.d, chunk * ni * d * 2);
    defer wb.free();
    var cb = try cuda.DeviceBuffer.alloc(L.d, chunk * ni * d / 2);
    defer cb.free();
    var sb = try cuda.DeviceBuffer.alloc(L.d, chunk * ni * d / 16);
    defer sb.free();
    var gb = try cuda.DeviceBuffer.alloc(L.d, chunk * 4);
    defer gb.free();
    const codes = try L.gpa.alloc(u8, 3 * chunk * ni * d / 2);
    defer L.gpa.free(codes);
    const scales = try L.gpa.alloc(u8, 3 * chunk * ni * d / 16);
    defer L.gpa.free(scales);
    const up = try L.gpa.alloc(u32, count * up_len);
    defer L.gpa.free(up);
    const down = try L.gpa.alloc(u32, count * down_len);
    defer L.gpa.free(down);
    const up_scale = try L.gpa.alloc(f32, count * 2);
    defer L.gpa.free(up_scale);
    const down_scale = try L.gpa.alloc(f32, count);
    defer L.gpa.free(down_scale);
    var g: [3][chunk]f32 = undefined;
    var name_buf: [96]u8 = undefined;
    var e0: usize = 0;
    while (e0 < count) : (e0 += chunk) {
        const ec = @min(chunk, count - e0);
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
            const n = if (m < 2) ni else d;
            const k = if (m < 2) d else ni;
            for (e0..e0 + ec) |e| {
                const name = try plan.expertName(&name_buf, c, i, e, proj);
                const t = try L.ck.get(name);
                if (m < 2) try L.rows(t, s.moe.lo, s.moe.hi, c.moe_inter, .bf16) else try L.cols(t, s.moe.lo, s.moe.hi, .bf16);
            }
            try wb.upload(0, L.host.items);
            L.host.clearRetainingCapacity();
            try quant.nvfp4(qk, L.stream, wb.ptr, ec, n, k, cb.ptr, sb.ptr, gb.ptr);
            try L.stream.synchronize();
            try cb.download(0, codes[m * chunk * n * k / 2 ..][0 .. ec * n * k / 2]);
            try sb.download(0, scales[m * chunk * n * k / 16 ..][0 .. ec * n * k / 16]);
            try gb.download(0, std.mem.sliceAsBytes(g[m][0..ec]));
        }
        const mc = chunk * ni * d / 2;
        for (0..ec) |j| {
            const e = e0 + j;
            const one = ni * d;
            var cs: [3][]const u8 = undefined;
            var ss: [3][]const u8 = undefined;
            for (0..3) |m| {
                cs[m] = codes[m * mc + j * one / 2 ..][0 .. one / 2];
                ss[m] = scales[m * mc / 8 + j * one / 16 ..][0 .. one / 16];
            }
            X.packGateUp(up[e * up_len ..][0..up_len], .{ cs[0], ss[0] }, .{ cs[1], ss[1] }, ni, d);
            X.packOne(down[e * down_len ..][0..down_len], cs[2], ss[2], d, ni);
            up_scale[e * 2] = g[0][j];
            up_scale[e * 2 + 1] = g[1][j];
            down_scale[e] = g[2][j];
        }
    }
    var ptrs: [4]u64 = undefined;
    var ptr_buf: [96]u8 = undefined;
    inline for (&ptrs, [_][]const u8{ std.mem.sliceAsBytes(up), std.mem.sliceAsBytes(down), std.mem.sliceAsBytes(up_scale), std.mem.sliceAsBytes(down_scale) }, [_][]const u8{ "up", "down", "up_scale", "down_scale" }) |*p, bytes, what| {
        try L.host.appendSlice(L.gpa, bytes);
        p.* = try L.put(try std.fmt.bufPrint(&ptr_buf, "layers.{d}.moe.experts.{s}", .{ i, what }));
    }
    return .{ .layer = .{ .format = .nvfp4, .up = ptrs[0], .down = ptrs[1], .up_scale = ptrs[2], .down_scale = ptrs[3], .width = ni, .dims = d, .experts = count, .limit = c.swiglu_limit } };
}

/// Assemble one layer: the plan's modules, then the expert tables.
fn loadLayer(L: *Loader, i: u32) !Layer {
    const c = &L.w.config;
    const s = L.w.share;
    var b: Build = .{};
    defer b.proj.deinit(L.gpa);
    defer b.gu_host.deinit(L.gpa);
    var walker = struct {
        b: *Build,
        L: *Loader,
        pub fn item(self: *@This(), it: plan.Item) !void {
            return self.b.item(self.L, it);
        }
    }{ .b = &b, .L = L };
    try plan.walkLayer(c, i, &walker);

    var buf: [96]u8 = undefined;
    try L.host.appendSlice(L.gpa, b.proj.items);
    b.proj.clearRetainingCapacity();
    const proj = try L.put(try std.fmt.bufPrint(&buf, "layers.{d}.dsa.proj", .{i}));

    const attn: Attn = .{
        .proj = proj,
        .q_norm = b.q_norm,
        .kv_norm = b.kv_norm,
        .q_b = b.q_b,
        .wk = b.wk,
        .wv = b.wv,
        .o = b.o,
        .index = .{ .kw = b.kw, .wproj = b.wproj, .ln_w = b.ln_w, .ln_b = b.ln_b, .qb = b.idx_qb, .heads = c.i_heads, .d = c.i_dim },
        .heads = s.mla.len(),
        .qk = c.qHead(),
        .dv = c.v_dim,
        .lw = c.kv_lora,
        .q_lora = c.q_lora,
        .dims = c.hidden,
        .eps = c.eps,
        .dense_limit = c.denseLimit(),
    };

    const ffn: Ffn = if (i < c.dense_layers) dense: {
        try L.host.appendSlice(L.gpa, b.gu_host.items);
        b.gu_host.clearRetainingCapacity();
        const gu = try L.put(try std.fmt.bufPrint(&buf, "layers.{d}.mlp.gu", .{i}));
        break :dense .{ .dense = .{ .w = .{ .bf16 = .{ .gu = gu, .down = b.down } }, .width = s.dense.len(), .dims = c.hidden, .limit = c.swiglu_limit } };
    } else moe: {
        try L.host.appendSlice(L.gpa, b.gu_host.items);
        b.gu_host.clearRetainingCapacity();
        const shared_gu = try L.put(try std.fmt.bufPrint(&buf, "layers.{d}.moe.shared.gu", .{i}));
        const routed = if (i == c.layers) try quantizeExperts(L, i) else try packExperts(L, i);
        break :moe .{ .moe = .{
            .router = b.router,
            .bias = b.bias,
            .routed = routed.layer,
            .in_up = routed.in_up,
            .in_down = routed.in_down,
            .count = c.experts,
            .shared_gu = shared_gu,
            .shared_down = b.shared_down,
            .shared = s.shared.len(),
            .top_k = c.topk,
            .scale = c.routed_scale,
            .norm = c.norm_topk,
        } };
    };

    return .{ .index = i, .in_norm = b.in_norm, .post_norm = b.post_norm, .attn = attn, .ffn = ffn };
}

/// The RoPE frequencies (r2's fused.inv_freq: 1 / theta^(2i/dim), fp32, dim = qk_rope_head_dim).
pub fn invFreq(a: std.mem.Allocator, c: *const cfg.Config) ![]f32 {
    const half = c.rope / 2;
    const out = try a.alloc(f32, half);
    errdefer a.free(out);
    for (0..half) |i| {
        const e: f32 = @floatCast(@as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(c.rope)));
        out[i] = 1.0 / std.math.pow(f32, @floatCast(c.rope_theta), e);
    }
    return out;
}

/// Load the whole checkpoint for one rank: the top-level tensors, every layer, and the MTP head.
/// GPU work is confined to the uploads and the MTP layer's device quantization.

/// §70 per-layer load probe: PRINT-ONLY, gated by TF_GLM53_TRACE. Names the first layer where the two
/// binaries' residency curves diverge. No allocation, no behaviour change.
fn loadProbe(L: *const Loader, i: u32) void {
    if (std.c.getenv("TF_GLM53_TRACE") == null) return;
    var cum: u64 = 0;
    for (L.w.named.items) |n| cum += n.len;
    var fr: usize = 0;
    var tot: usize = 0;
    _ = L.d.api.cuMemGetInfo_v2(&fr, &tot);
    const gib: f64 = 1 << 30;
    var sb: [16384]u8 = undefined;
    var hb: [16384]u8 = undefined;
    const st = procRead(&sb, "/proc/self/status");
    const info = procRead(&hb, "/proc/meminfo");
    std.debug.print("LOADPROBE layer={d} cum_MiB={d:.1} dev_free={d:.2} dev_used={d:.2} rss={d:.2} avail={d:.2} anon={d:.2} cached={d:.2} shmem={d:.2} mapped={d:.2}\n", .{
        i, @as(f64, @floatFromInt(cum)) / (1024 * 1024),
        @as(f64, @floatFromInt(fr)) / gib, @as(f64, @floatFromInt(tot - fr)) / gib,
        @as(f64, @floatFromInt(procKb(st, "VmRSS"))) / (1024 * 1024),
        @as(f64, @floatFromInt(procKb(info, "MemAvailable"))) / (1024 * 1024),
        @as(f64, @floatFromInt(procKb(info, "AnonPages"))) / (1024 * 1024),
        @as(f64, @floatFromInt(procKb(info, "Cached"))) / (1024 * 1024),
        @as(f64, @floatFromInt(procKb(info, "Shmem"))) / (1024 * 1024),
        @as(f64, @floatFromInt(procKb(st, "VmRSS"))) / (1024 * 1024),
    });
}

fn procRead(buf: []u8, path: [:0]const u8) []const u8 {
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return buf[0..0];
    defer _ = std.c.close(fd);
    const n = std.c.read(fd, buf.ptr, buf.len);
    if (n < 0) return buf[0..0];
    return buf[0..@intCast(n)];
}

fn procKb(text: []const u8, key: []const u8) u64 {
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

/// §70 config parity: the loader's effective parameters, printed by BOTH binaries from the SAME call.
fn loadParams(c: *const cfg.Config, share: shard.Share) void {
    if (std.c.getenv("TF_GLM53_TRACE") == null) return;
    std.debug.print("LOADPARAMS layers={d} mtp={d} experts={d} moe_inter={d} hidden={d} vocab={d} | rank={d}/{d} moe=[{d},{d}) mla=[{d},{d}) vocab=[{d},{d}) scales={d}\n", .{
        c.layers, c.mtp, c.experts, c.moe_inter, c.hidden, c.vocab,
        share.rank, share.world, share.moe.lo, share.moe.hi, share.mla.lo, share.mla.hi, share.vocab.lo, share.vocab.hi,
        share.moe.len(),
    });
}

pub fn load(a: std.mem.Allocator, d: *const cuda.Driver, stream: cuda.Stream, tools: Tools, ck: *core.Checkpoint, c: *const cfg.Config, share: shard.Share) !Weights {
    var w: Weights = .{ .gpa = a, .config = c.*, .share = share };
    errdefer w.deinit();
    var L: Loader = .{ .gpa = a, .d = d, .stream = stream, .tools = tools, .w = &w, .ck = ck };
    defer L.host.deinit(a);

    // the index plane's RoPE frequencies (host-computed, uploaded once; r2 keeps them on the weights)
    {
        const inv = try invFreq(a, c);
        defer a.free(inv);
        L.host.clearRetainingCapacity();
        try L.host.appendSlice(a, std.mem.sliceAsBytes(inv));
        w.inv = try L.put("rope.inv");
    }

    // embed and lm_head: this rank's vocabulary rows; the final norm whole
    // the embedding is REPLICATED, as the GLM engine loads it (`L.whole`): a token's row must exist on
    // every rank, or each rank starts from a different hidden state. Only the head's rows are sharded.
    const embed = plan.Item{ .name = plan.prefix ++ "embed_tokens.weight", .role = .embed, .flavor = .bf16, .n = c.vocab, .k = c.hidden };
    try L.whole(try L.get(embed), .bf16);
    w.embed = try L.put(plan.prefix ++ "embed_tokens.weight");
    const norm = plan.Item{ .name = plan.prefix ++ "norm.weight", .role = .final_norm, .flavor = .bf16, .n = c.hidden, .k = 0 };
    try L.whole(try L.get(norm), .bf16);
    w.norm = try L.put(plan.prefix ++ "norm.weight");
    const head = plan.Item{ .name = "lm_head.weight", .role = .head, .flavor = .bf16, .n = c.vocab, .k = c.hidden };
    try L.rows(try L.get(head), share.vocab.lo, share.vocab.hi, c.vocab, .bf16);
    w.head = try L.put("lm_head.weight");

    loadParams(c, share);
    w.layers = try a.alloc(Layer, c.layers + c.mtp);
    for (0..c.layers + c.mtp) |i| {
        w.layers[i] = try loadLayer(&L, @intCast(i));
        loadProbe(&L, @intCast(i));
    }

    if (c.mtp > 0) {
        const m = c.layers;
        var buf: [96]u8 = undefined;
        _ = try std.fmt.bufPrint(&buf, "", .{});
        const eh = plan.Item{ .name = try std.fmt.allocPrint(a, plan.prefix ++ "layers.{d}.eh_proj.weight", .{m}), .role = .eh_proj, .flavor = .bf16, .n = c.hidden, .k = 2 * c.hidden };
        try L.whole(try L.get(eh), .bf16);
        const eh_ptr = try L.put(try std.fmt.bufPrint(&buf, "layers.{d}.mtp.eh", .{m}));
        const enorm = plan.Item{ .name = try std.fmt.allocPrint(a, plan.prefix ++ "layers.{d}.enorm.weight", .{m}), .role = .enorm, .flavor = .bf16, .n = c.hidden, .k = 0 };
        try L.whole(try L.get(enorm), .bf16);
        const enorm_ptr = try L.put(try std.fmt.bufPrint(&buf, "layers.{d}.mtp.enorm", .{m}));
        const hnorm = plan.Item{ .name = try std.fmt.allocPrint(a, plan.prefix ++ "layers.{d}.hnorm.weight", .{m}), .role = .hnorm, .flavor = .bf16, .n = c.hidden, .k = 0 };
        try L.whole(try L.get(hnorm), .bf16);
        const hnorm_ptr = try L.put(try std.fmt.bufPrint(&buf, "layers.{d}.mtp.hnorm", .{m}));
        const shn = plan.Item{ .name = try std.fmt.allocPrint(a, plan.prefix ++ "layers.{d}.shared_head.norm.weight", .{m}), .role = .shared_head_norm, .flavor = .bf16, .n = c.hidden, .k = 0 };
        try L.whole(try L.get(shn), .bf16);
        const norm_ptr = try L.put(try std.fmt.bufPrint(&buf, "layers.{d}.mtp.shared_head_norm", .{m}));
        w.mtp = .{ .enorm = enorm_ptr, .hnorm = hnorm_ptr, .eh = eh_ptr, .norm = norm_ptr, .layer = w.layers[m] };
    }
    return w;
}

test "the load assembly type-checks against the CUDA module (no GPU work here)" {
    _ = &load;
    _ = &packExperts;
    _ = &quantizeExperts;
    _ = &loadLayer;
}

/// A layer's W4A4 static input scales from the pack's sidecar (`model-inputscales.safetensors`): one
/// f32 per quantized projection, gate and up sharing the expert's input (2 x count) and down its own
/// (count). Read on the host; W4A16 math does not consume them, a W4A4 path scales activations by
/// 1/scale before quantization.
pub const InputScales = struct {
    up: []f32, // [count][2]: gate, up
    down: []f32, // [count]

    pub fn deinit(s: *InputScales, a: std.mem.Allocator) void {
        a.free(s.up);
        a.free(s.down);
        s.* = undefined;
    }
};

pub fn expertInputScales(a: std.mem.Allocator, ck: *core.Checkpoint, c: *const cfg.Config, i: u32) !InputScales {
    const count: usize = c.experts;
    const up = try a.alloc(f32, count * 2);
    errdefer a.free(up);
    const down = try a.alloc(f32, count);
    errdefer a.free(down);
    var buf: [96]u8 = undefined;
    var nb: [160]u8 = undefined;
    for (0..count) |e| {
        inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
            const name = try plan.expertName(&buf, c, i, e, proj);
            const key = try std.fmt.bufPrint(&nb, "{s}.input_scale", .{plan.module(name)});
            const t = try ck.get(key);
            if (t.dtype != .f32 or t.numel() != 1) return error.BadInputScale;
            const v = std.mem.bytesToValue(f32, t.bytes[0..4]);
            if (!std.math.isFinite(v) or v <= 0) return error.BadInputScale;
            switch (m) {
                0, 1 => up[e * 2 + m] = v,
                else => down[e] = v,
            }
        }
    }
    return .{ .up = up, .down = down };
}

test "the load assembly type-checks the W4A4 scale path too" {
    _ = &expertInputScales;
}

test "the census join: an empty weight set resolves none of the plan's modules (the walk is real)" {
    const c: cfg.Config = .{};
    const w: Weights = .{ .gpa = std.testing.allocator, .config = c, .share = undefined };
    const cen = try Census.join(&w, &c);
    try std.testing.expect(cen.expected > 0);
    // the stacked roles (q_a, dense_gate, shared_gate) have no name of their own
    try std.testing.expectEqual(cen.expected - cen.merged, cen.missing);
    try std.testing.expect(cen.merged > 0);
    try std.testing.expectEqual(@as(usize, 0), cen.found);
    try std.testing.expect(cen.first_missing[0] != 0);
}
