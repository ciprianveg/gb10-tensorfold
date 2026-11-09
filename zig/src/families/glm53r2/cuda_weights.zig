//! GLM-5.3 layer slices on the GPU in the Python engine's layouts (cuda/weights.py load_layer + fused.Weights):
//! EXL3 linears in the strips layout, kv_b cut into the absorb (wk) and expand (wv) blocks, the indexer's bf16 parts,
//! the router, and the routed experts as experts.prepare tables (TF_GLM53_EXPERTS=tf: one trellis a matrix).
//! Reads go through mapped safetensors (core.Checkpoint), each rank's part cut as split.rule says. Phase 2a: every
//! layer, the final norm and this rank's vocabulary share of lm_head, in O(1) host memory: after each layer the files
//! it read leave the page cache (madvise + posix_fadvise DONTNEED, as weights.RankReader.release does), so a 4-rank
//! load fits GB10's unified memory beside the caches. Device tensors come from 256 MiB arena chunks (512-byte
//! aligned like torch's caching allocator) instead of one cuMemAlloc each; each MoE layer's expert trellises are one
//! allocation per projection.
//! Phase 2b: `Options.fast` reads each layer's bytes with O_DIRECT through io_uring a layer ahead (cuda_source.zig)
//! instead of through the mapping; `Options.mtp` adds the MTP layer (index num_hidden_layers: a full decoder layer,
//! eh_proj, enorm, hnorm, shared_head.norm) and `Options.draft_vocab` the 4-bit draft head over the reduced draft
//! vocabulary (headq q4 + fused.DRAFT_VOCAB: lm_head rows rank * V/4 .. of the first V ids, rank world-1 also the
//! last 128; glm5_next qmm.quantize4, shared qmm.pack - computed on the host with torch's float32 rounding).
//! TODO(P3): the cuda-exl3 shared stacks (experts_cx) for prompt chunks.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Config = @import("config.zig").Config;
const exl3 = @import("exl3.zig");
const split = @import("split.zig");
const kern = @import("cuda_kernels.zig");
const source = @import("cuda_source.zig");
const headq = @import("headq.zig");

const Tensor = core.checkpoint.Tensor;
const config_max_layers = @import("config.zig").max_layers;

/// One-time log latches for the fp16 -> bf16 round casts (the 4.04bpw-h8 pack stores the
/// bf-named non-scale tensors as fp16; the loader is single-threaded per rank process).
const RoundCastLogged = struct {
    var kv_b: bool = false;
    var gate: bool = false;
    var idx_wk: bool = false;
    var idx_weights_proj: bool = false;
    var idx_k_norm_w: bool = false;
    var idx_k_norm_b: bool = false;
};

pub const Experts = kern.Experts;

pub const Mlp = struct { gate: exl3.Linear, up: exl3.Linear, down: exl3.Linear };

pub const Indexer = struct {
    wq_b: exl3.Linear,
    wk: u64, // bf16 [index_dim, D]
    weights_proj: u64, // bf16 [index_heads, D]
    k_norm_w: u64, // bf16 [index_dim]
    k_norm_b: u64,
};

pub const Layer = struct {
    index: usize,
    input_norm: u64,
    post_attn_norm: u64,
    q_a: exl3.Linear,
    q_a_norm: u64,
    kv_a: exl3.Linear, // its stored 640 outputs (the cache keeps 576)
    kv_a_norm: u64,
    q_b: exl3.Linear,
    wk: u64, // bf16 [H, nope, kv_lora]: kv_b's key block (absorb)
    wv: u64, // bf16 [H, v_dim, kv_lora]: kv_b's value block (expand)
    o_proj: exl3.Linear,
    indexer: ?Indexer,
    router_w: u64 = 0, // bf16 [E, D]
    router_bias: u64 = 0, // fp32 [E]
    experts: ?Experts = null,
    mlp: Mlp, // the dense MLP (layers < first_k_dense_replace) or the shared expert
};

/// The MTP layer (weights.MtpHead): a full decoder layer, its input projection and norms; its head is lm_head.
pub const Mtp = struct {
    layer: Layer,
    eh_proj: u64, // bf16 [D, 2 D]: [enorm(embedding) ; hnorm(hidden)] -> hidden
    enorm: u64,
    hnorm: u64,
    head_norm: u64, // shared_head.norm
};

/// headq's q4 copy of this rank's reduced draft vocabulary (qmm.Q4 packed by shared qmm.pack) and its global ids.
pub const DraftHead = struct {
    w: u64, // int32 [npad / 64, K / 64, 8, 32, 2]
    scales: u64, // bf16 [K / 64, npad]
    biases: u64,
    ids: u64, // int64 [n]: global token ids
    n: usize,
    npad: usize,
    k: usize,
    sk: usize, // glm5_next qmm.split_k(n, k)
};

/// TF_GLM53_HEAD_Q (headq.zig): the head as an EXL3 linear over the Hadamard blocks that cover this rank's
/// vocabulary share; the share starts `off` outputs into the launch's result.
pub const HeadQ = struct { lin: exl3.Linear, off: usize };

/// TF_GLM53_HEAD_FP8 (off by default): this rank's lm_head share packed to the fp8-e4m3 128-block
/// form at load — the canary's qmmf .fp8g weight class (cuda_bind.packHeadFp8), read by the head's
/// quantized-compute launch instead of the bf16 router. The bf16 rows stay bound (the A/B is one env
/// away, no reload). Unlike the EXL3 head the vocab share needs no 128-block cutting: the pack pads
/// rows to `npad` internally and the kernel reads the true `n`.
pub const Fp8Head = struct { codes: u64, scales: u64, npad: u32 };

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: Config,
    rank: usize,
    world: usize,
    heads: usize, // this rank's attention heads
    embed: u64 = 0, // bf16 [V, D]
    final_norm: u64 = 0, // bf16 [D] (model.norm)
    lm_head: u64 = 0, // bf16 [V / world, D]: this rank's vocabulary share (rows vocab_off ..); 0 with head_q
    head_q: ?HeadQ = null, // TF_GLM53_HEAD_Q: the quantized head instead (the same share, the same vocab_off)
    head_fp8: ?Fp8Head = null, // TF_GLM53_HEAD_FP8: the fp8-e4m3 128-block pack of the same share (the bf16 rows stay)
    /// with head_q and Options.head_host: this rank's bf16 lm_head rows [vocab_part, D] on the host, for the
    /// drafters' 4-bit head (cuda_draft.buildDraftHead takes and frees them)
    head_host: []u16 = &.{},
    vocab_part: usize = 0,
    vocab_off: usize = 0,
    layers: []Layer = &.{},
    mtp: ?Mtp = null,
    draft: ?DraftHead = null,
    /// Phase 4: headq.prepare's draft_lm_head - the 4-bit copy of this rank's lm_head share the drafters' block
    /// passes read (cuda_draft.buildDraftHead); null without a drafter
    draft_lm: ?kern.Q4 = null,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    chunk: u64 = 0, // the arena chunk being filled (0: none yet)
    chunk_used: usize = 0,
    device_bytes: usize = 0, // bytes handed out (tensors, not chunk slack)
    digest_checked: usize = 0,
    digest_bad: usize = 0,
    read_bytes: u64 = 0, // bytes the fast loader read (0: mapped reads)
    read_fallbacks: u64 = 0,
    direct_files: usize = 0,

    /// Arena chunks: small tensors share them; anything over a quarter chunk gets its own allocation.
    pub const chunk_bytes: usize = 256 << 20;
    /// torch's caching allocator rounds every block to 512 bytes: the same alignment for every pointer the kernels see.
    pub const alignment: usize = 512;

    /// `bytes` of device memory that lives as long as the weights (not zeroed).
    pub fn alloc(w: *Weights, d: *const cuda.Driver, bytes: usize) !u64 {
        const n = std.mem.alignForward(usize, @max(bytes, 1), alignment);
        w.device_bytes += n;
        if (n > chunk_bytes / 4) {
            var b = try cuda.DeviceBuffer.alloc(d, n);
            errdefer b.free();
            try w.buffers.append(w.gpa, b);
            return b.ptr;
        }
        if (w.chunk == 0 or w.chunk_used + n > chunk_bytes) {
            var b = try cuda.DeviceBuffer.alloc(d, chunk_bytes);
            errdefer b.free();
            try w.buffers.append(w.gpa, b);
            w.chunk = b.ptr;
            w.chunk_used = 0;
        }
        const ptr = w.chunk + w.chunk_used;
        w.chunk_used += n;
        return ptr;
    }

    pub fn hasHead(w: *const Weights) bool {
        return w.lm_head != 0 or w.head_q != null;
    }

    pub fn deinit(w: *Weights) void {
        if (w.head_host.len > 0) w.gpa.free(w.head_host);
        w.head_host = &.{};
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
        w.* = undefined;
    }

    /// The widest EXL3 linear output the scratch must hold (rows x N), and the largest K, over every layer.
    pub fn linearExtents(w: *const Weights) struct { k: usize, zn: usize } {
        var k: usize = 0;
        var zn: usize = 0;
        var all: [config_max_layers + 1]*const Layer = undefined;
        var n: usize = 0;
        for (w.layers) |*L| {
            all[n] = L;
            n += 1;
        }
        if (w.mtp) |*m| {
            all[n] = &m.layer;
            n += 1;
        }
        for (all[0..n]) |L| {
            const lins = [_]*const exl3.Linear{ &L.q_a, &L.kv_a, &L.q_b, &L.o_proj, &L.mlp.gate, &L.mlp.up, &L.mlp.down };
            for (lins) |l| {
                k = @max(k, l.k);
                zn = @max(zn, l.sk * l.n);
            }
            if (L.indexer) |*ix| {
                k = @max(k, ix.wq_b.k);
                zn = @max(zn, ix.wq_b.sk * ix.wq_b.n);
            }
        }
        return .{ .k = k, .zn = zn };
    }
};

pub const Options = struct {
    rank: usize = 0,
    world: usize = 1,
    layers: usize, // load layers 0..layers-1
    embed: bool = true,
    /// the final norm and this rank's lm_head share (Phase 2a: the vocabulary-sharded argmax head)
    head: bool = false,
    /// log progress every this many layers (0: a line every layer, as Phase 1 did)
    log_every: usize = 0,
    /// weights_sha256 of the oracle's meta.json: each prepared tensor's bytes are checked against it when given
    digests: ?*const std.json.ObjectMap = null,
    /// O_DIRECT + io_uring reads a layer ahead (cuda_source.zig); false: the mapped reads of Phases 1 and 2a
    fast: bool = false,
    /// the MTP layer (needs config num_nextn_predict_layers >= 1)
    mtp: bool = false,
    /// fused.DRAFT_VOCAB: the 4-bit draft head over this many lowest ids (+ the last 128); 0: none
    draft_vocab: usize = 0,
    /// TF_GLM53_HEAD_Q: lm_head as an EXL3 linear (headq.zig) from a file of its own or the checkpoint; off: bf16 rows
    head_q: headq.Setting = .off,
    /// with head_q: keep this rank's bf16 lm_head rows on the host (Weights.head_host) for the drafters' 4-bit head
    head_host: bool = false,
    /// TF_GLM53_KVB_Q: a checkpoint whose kv_b_proj is an EXL3 linear is decoded to the bf16 blocks at load
    kvb_q: bool = false,
};

/// libc calls std does not wrap on every target: dropping a mapped file's pages from this process and the page cache.
const libc = struct {
    const madvise = @extern(*const fn (?*anyopaque, usize, c_int) callconv(.c) c_int, .{ .name = "madvise" });
    const posix_fadvise = @extern(*const fn (c_int, i64, i64, c_int) callconv(.c) c_int, .{ .name = "posix_fadvise" });
    const malloc_trim = @extern(*const fn (usize) callconv(.c) c_int, .{ .name = "malloc_trim" });
    const MADV_DONTNEED: c_int = 4;
    const POSIX_FADV_DONTNEED: c_int = 4;
};

/// A mapped checkpoint file's pages out of this process (madvise) and out of the page cache (posix_fadvise): clean
/// pages only, so a later read simply reads the disk again (weights.drop_page_cache).
fn dropFile(f: *core.safetensors.File) void {
    const mem = f.map.memory;
    if (mem.len > 0) _ = libc.madvise(@ptrCast(@constCast(mem.ptr)), mem.len, libc.MADV_DONTNEED);
    _ = libc.posix_fadvise(f.file.handle, 0, 0, libc.POSIX_FADV_DONTNEED);
}

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    ck: *core.Checkpoint,
    w: *Weights,
    o: Options,
    io: std.Io,
    host: std.ArrayList(u8) = .empty,
    host2: std.ArrayList(u8) = .empty,
    touched: []bool = &.{}, // per checkpoint file: read since the last release
    dropped: usize = 0, // file drops so far
    batch: ?*const source.Batch = null, // the unit being loaded, prefetched (Options.fast)

    fn deinit(L: *Loader) void {
        L.host.deinit(L.gpa);
        L.host2.deinit(L.gpa);
        L.gpa.free(L.touched);
    }

    fn has(L: *Loader, name: []const u8) bool {
        for (L.ck.files.items) |*f| if (f.names.get(name) != null) return true;
        return false;
    }

    /// Checkpoint.get, noting which file the bytes come from (release drops its pages).
    fn get(L: *Loader, name: []const u8) !core.checkpoint.Tensor {
        for (L.ck.files.items, 0..) |*f, i| if (f.names.get(name) != null) {
            L.touched[i] = true;
            break;
        };
        return L.ck.get(name);
    }

    /// weights.RankReader.release: every file read since the last call leaves this process and the page cache, and
    /// the freed host heap goes back to the OS (on GB10 host memory is the memory the GPU allocations need).
    fn release(L: *Loader) void {
        for (L.ck.files.items, L.touched) |*f, *t| {
            if (!t.*) continue;
            t.* = false;
            dropFile(f);
            L.dropped += 1;
        }
        _ = libc.malloc_trim(0);
    }

    /// Every file's pages dropped (end of the load: files read outside a layer too).
    fn releaseAll(L: *Loader) void {
        for (L.touched) |*t| t.* = true;
        L.release();
    }

    /// The oracle's digest of a prepared tensor, compared with these bytes (a mismatch is logged and counted).
    fn check(L: *Loader, key: []const u8, bytes: []const u8) void {
        const want_map = L.o.digests orelse return;
        const want = want_map.get(key) orelse return;
        if (want != .string) return;
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        L.w.digest_checked += 1;
        if (!std.mem.eql(u8, &hex, want.string)) {
            L.w.digest_bad += 1;
            std.log.err("weight digest differs from the oracle's: {s}", .{key});
        }
    }

    fn put(L: *Loader, key: []const u8, bytes: []const u8) !u64 {
        L.check(key, bytes);
        const p = try L.w.alloc(L.d, bytes.len);
        try L.upload(p, bytes);
        return p;
    }

    fn upload(L: *Loader, dst: u64, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        try L.d.check(L.d.api.cuMemcpyHtoD_v2(dst, bytes.ptr, bytes.len), "cuMemcpyHtoD");
    }

    fn zeros(L: *Loader, bytes: usize) !u64 {
        const p = try L.w.alloc(L.d, bytes);
        try L.d.check(L.d.api.cuMemsetD8_v2(p, 0, std.mem.alignForward(usize, @max(bytes, 1), Weights.alignment)), "cuMemsetD8");
        return p;
    }

    fn staging(list: *std.ArrayList(u8), gpa: std.mem.Allocator, n: usize) ![]u8 {
        try list.resize(gpa, n);
        return list.items;
    }

    /// A whole tensor's bytes: the prefetched copy when the unit's plan read it whole, else the mapping.
    fn whole(L: *Loader, name: []const u8, t: Tensor) []const u8 {
        if (L.batch) |b| if (source.Prefetch.lookup(b, name)) |f| if (!f.partial and f.bytes.len == t.bytes.len) return f.bytes;
        return t.bytes;
    }

    /// This rank's part of a checkpoint tensor (split.rule): the prefetched or mapped bytes themselves when whole or a
    /// row range, else (second-axis cuts) a copy in the host2 staging buffer.
    fn part(L: *Loader, name: []const u8, shape_out: *[core.safetensors.max_rank]usize) ![]const u8 {
        const t = try L.get(name);
        const kind = try split.rule(name);
        shape_out.* = t.shape;
        const shape = t.shape[0..t.rank];
        if (kind == .rep or L.o.world == 1) return L.whole(name, t);
        try split.checkCut(name, kind, shape, L.o.world);
        const axis: usize = if (kind == .row) 0 else 1;
        shape_out[axis] /= L.o.world;
        if (kind == .row) {
            const per = t.bytes.len / shape[0];
            const n = shape[0] / L.o.world;
            if (L.batch) |b| if (source.Prefetch.lookup(b, name)) |f| if (f.partial and f.bytes.len == n * per) return f.bytes;
            return L.whole(name, t)[L.o.rank * n * per ..][0 .. n * per];
        }
        const out = try staging(&L.host2, L.gpa, t.bytes.len / L.o.world);
        try split.cutBytes(out, L.whole(name, t), shape, t.dtype.size(), kind, L.o.rank, L.o.world);
        return out;
    }

    /// weights.as_bf16: bf16 as stored; fp16 only when every value is exactly a bf16 value.
    fn bf16(L: *Loader, name: []const u8, key: []const u8) !u64 {
        const t = try L.get(name);
        if (t.dtype == .bf16) {
            var shape: [core.safetensors.max_rank]usize = undefined;
            return L.put(key, try L.part(name, &shape));
        }
        if (t.dtype != .f16) return error.UnexpectedTensor;
        var shape: [core.safetensors.max_rank]usize = undefined;
        const raw = try L.part(name, &shape);
        const out = try staging(&L.host, L.gpa, raw.len);
        for (0..raw.len / 2) |i| {
            const h: f16 = @bitCast(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
            const f: f32 = @as(f32, h);
            const bits: u32 = @bitCast(f);
            if (bits & 0xFFFF != 0 or std.math.isNan(f)) {
                std.log.err("{s}: fp16 values that bf16 cannot hold exactly; refusing a rounding cast", .{name});
                return error.InexactBf16;
            }
            std.mem.writeInt(u16, out[2 * i ..][0..2], @intCast(bits >> 16), .little);
        }
        return L.put(key, out);
    }

    /// weights.as_bf16 with the capture-side round cast: bf16 as stored; fp16 round-to-nearest-even
    /// to bf16 (bf16Bits — torch's .to(bfloat16), the exact values the world-8 AOT capture compiled
    /// against). For the bf-named tensors the 4.04bpw-h8 pack stores as fp16 (router gate, indexer
    /// wk / weights_proj / k_norm). Logs once per class per process.
    fn bf16Round(L: *Loader, name: []const u8, key: []const u8, latch: *bool, class: []const u8) !u64 {
        const t = try L.get(name);
        if (t.dtype == .bf16) {
            var shape: [core.safetensors.max_rank]usize = undefined;
            return L.put(key, try L.part(name, &shape));
        }
        if (t.dtype != .f16) return error.UnexpectedTensor;
        var shape: [core.safetensors.max_rank]usize = undefined;
        const raw = try L.part(name, &shape);
        const out = try staging(&L.host, L.gpa, raw.len);
        for (0..raw.len / 2) |i| {
            const h: f16 = @bitCast(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
            std.mem.writeInt(u16, out[2 * i ..][0..2], bf16Bits(@as(f32, h)), .little);
        }
        if (!latch.*) {
            std.log.info("glm53: {s}: fp16 storage -> bf16 round cast (torch .to(bfloat16); the AOT capture compiled these values)", .{class});
            latch.* = true;
        }
        return L.put(key, out);
    }

    /// bf16Round with the K4 hybrid's fp8 form accepted (findings-k4-loader.md): a bf16-loaded
    /// matrix (the indexer's wk - a .rep tensor) stored F8_E4M3 + `.weight_scale_inv` is
    /// dequantized at load; the bf16 / fp16 storages flow through bf16Round unchanged.
    fn bf16OrDequant(L: *Loader, name: []const u8, key: []const u8, latch: *bool, class: []const u8) !u64 {
        const t = try L.get(name);
        if (t.dtype != .f8_e4m3) return L.bf16Round(name, key, latch, class);
        var sb: [512]u8 = undefined;
        const st = try L.get(try std.fmt.bufPrint(&sb, "{s}_scale_inv", .{name}));
        if (st.dtype != .f32 or st.rank != 2) return error.UnexpectedTensor;
        if (st.shape[0] != (t.shape[0] + 127) / 128 or st.shape[1] != (t.shape[1] + 127) / 128) return error.UnexpectedTensor;
        var shape: [core.safetensors.max_rank]usize = undefined;
        const raw = try L.part(name, &shape);
        if (shape[0] % 128 != 0 and L.o.world > 1) return error.CutStraddlesScaleBlock;
        const out = try staging(&L.host, L.gpa, shape[0] * shape[1] * 2);
        dequantFp8(out, raw, L.whole(try std.fmt.bufPrint(&sb, "{s}_scale_inv", .{name}), st), shape[0], shape[1], 0, 0, st.shape[1]);
        if (!latch.*) {
            std.log.info("glm53: {s}: fp8 block-quant storage -> bf16 dequant at load (the K4 hybrid pack)", .{class});
            latch.* = true;
        }
        return L.put(key, out);
    }

    /// `.float()` of a bf16 / fp32 vector (the router's score-correction bias).
    fn f32vec(L: *Loader, name: []const u8, key: []const u8) !u64 {
        const t = try L.get(name);
        var shape: [core.safetensors.max_rank]usize = undefined;
        const raw = try L.part(name, &shape);
        if (t.dtype == .f32) return L.put(key, raw);
        if (t.dtype != .bf16) return error.UnexpectedTensor;
        const out = try staging(&L.host, L.gpa, raw.len * 2);
        for (0..raw.len / 2) |i| {
            const v: u32 = @as(u32, std.mem.readInt(u16, raw[2 * i ..][0..2], .little)) << 16;
            std.mem.writeInt(u32, out[4 * i ..][0..4], v, .little);
        }
        return L.put(key, out);
    }

    fn codebook(L: *Loader, prefix: []const u8) !exl3.Codebook {
        var buf: [256]u8 = undefined;
        if (L.has(try std.fmt.bufPrint(&buf, "{s}.mul1", .{prefix}))) {
            _ = try L.get(try std.fmt.bufPrint(&buf, "{s}.mul1", .{prefix}));
            return .mul1;
        }
        if (L.has(try std.fmt.bufPrint(&buf, "{s}.mcg", .{prefix}))) {
            _ = try L.get(try std.fmt.bufPrint(&buf, "{s}.mcg", .{prefix}));
            return .mcg;
        }
        return .@"3inst";
    }

    /// fp16 scales of `size` values (packed sign words are not in GLM-5.3's packs: refused).
    fn scales(L: *Loader, name: []const u8, key: []const u8, size: usize) !u64 {
        const t = try L.get(name);
        if (t.dtype != .f16) {
            std.log.err("{s}: {t} scales (packed sign words are not implemented)", .{ name, t.dtype });
            return error.UnsupportedScales;
        }
        var shape: [core.safetensors.max_rank]usize = undefined;
        const raw = try L.part(name, &shape);
        if (raw.len != size * 2) return error.UnexpectedTensor;
        return L.put(key, raw);
    }

    /// The K4 hybrid pack's dense projection (findings-k4-loader.md): `{prefix}.weight` F8_E4M3
    /// [N, K] + `{prefix}.weight_scale_inv` F32 [ceil(N/128), ceil(K/128)] - the 128x128-block fp8
    /// layout - dequantized AT LOAD to this rank's bf16 part (the kvb_q precedent: the checkpoint's
    /// storage form is decoded to the bf16 blocks the GEMM sites consume). The cut follows the
    /// trellis classification (split.known of the `.trellis` name): .rep = the whole matrix;
    /// .dim1 (an output-split linear - gate/up/q_b) = the weight's ROWS; .row (an input-split one -
    /// down/o_proj) = the weight's COLUMNS - the same rank boundaries the exl3 path cuts, so the
    /// compute contract is unchanged. Every cut lands on a 128 boundary (asserted), so the scale
    /// blocks cut cleanly with srow0/scol0 offsets. The Linear is DENSE: words holds the plain bf16
    /// [n, k] and the compute is cuBLAS (exl3.Linear.dense).
    fn dequantLinear(L: *Loader, prefix: []const u8, key: []const u8) !exl3.Linear {
        var wb: [512]u8 = undefined; // the weight's name (whole() re-reads it - its own buffer)
        var sb: [512]u8 = undefined;
        var tb: [512]u8 = undefined;
        var kb: [256]u8 = undefined;
        const wname = try std.fmt.bufPrint(&wb, "{s}.weight", .{prefix});
        const sname = try std.fmt.bufPrint(&sb, "{s}.weight_scale_inv", .{prefix});
        const wt = try L.get(wname);
        if (wt.dtype != .f8_e4m3 or wt.rank != 2) return error.UnexpectedTensor;
        const st = try L.get(sname);
        if (st.dtype != .f32 or st.rank != 2) return error.UnexpectedTensor;
        const GN = wt.shape[0];
        const GK = wt.shape[1];
        if (st.shape[0] != (GN + 127) / 128 or st.shape[1] != (GK + 127) / 128) return error.UnexpectedTensor;
        const trellis_name = try std.fmt.bufPrint(&tb, "{s}.trellis", .{prefix});
        const kind = split.known(trellis_name) orelse {
            std.log.err("{s}: no split classification (the .trellis name form has no rule)", .{prefix});
            return error.NoSplitRule;
        };
        var n = GN;
        var k = GK;
        var srow0: usize = 0;
        var scol0: usize = 0;
        var wsrc = L.whole(wname, wt);
        var own: []u8 = &.{};
        defer if (own.len > 0) L.gpa.free(own);
        const world = L.o.world;
        if (world > 1 and kind != .rep) {
            // the torch layout's cut: output-split (.dim1 of the trellis) = the [N, K] ROWS;
            // input-split (.row of the trellis) = the COLUMNS. cutBytes does both.
            const wkind: split.Kind = if (kind == .dim1) .row else .dim1;
            if (wkind == .row) {
                if (GN % world != 0) return error.UnevenCut;
                n /= world;
                if (n % 128 != 0) return error.CutStraddlesScaleBlock;
                srow0 = L.o.rank * n / 128;
            } else {
                if (GK % world != 0) return error.UnevenCut;
                k /= world;
                if (k % 128 != 0) return error.CutStraddlesScaleBlock;
                scol0 = L.o.rank * k / 128;
            }
            own = try L.gpa.alloc(u8, n * k);
            try split.cutBytes(own, L.whole(wname, wt), &[_]usize{ GN, GK }, 1, wkind, L.o.rank, world);
            wsrc = own;
        }
        const out = try staging(&L.host, L.gpa, n * k * 2);
        dequantFp8(out, wsrc, L.whole(sname, st), n, k, srow0, scol0, st.shape[1]);
        var lin: exl3.Linear = .{
            .words = try L.put(try std.fmt.bufPrint(&kb, "{s}.words", .{key}), out),
            .suh = 0,
            .svh = 0,
            .counters = 0,
            .k = k,
            .n = n,
            .k2 = 0,
            .cb = .@"3inst",
            .sk = 1,
            .wk = 0,
            .dense = true,
        };
        // 126.6(at) the fp8 canary (TF_GLM53_DENSE_FP8, value semantics): the pack's OWN e4m3
        // codes and 128-block scales repacked for qmmf's .fp8g mode - NO re-quantization. Every
        // e4m3 value is exact in bf16, and the kernel's f32 scale product skips the bf16 rounding
        // the dequant path adds: the fp8 read is the MORE exact form of the same stored weights.
        if (fp8Arm()) {
            const npad8 = cuda.fp8.padded(n);
            const pc = try L.gpa.alloc(u8, npad8 * k);
            defer L.gpa.free(pc);
            cuda.fp8.packCodes(pc, wsrc[0 .. n * k], n, k);
            const sr = (n + 127) / 128;
            const scr = (k + 127) / 128;
            const sg = try L.gpa.alloc(f32, sr * scr);
            defer L.gpa.free(sg);
            const sld = st.shape[1];
            for (0..sr) |r| for (0..scr) |c| {
                sg[r * scr + c] = @bitCast(std.mem.readInt(u32, st.bytes[((srow0 + r) * sld + scol0 + c) * 4 ..][0..4], .little));
            };
            const pt = try L.gpa.alloc(f32, npad8 * (k / 64));
            defer L.gpa.free(pt);
            cuda.fp8.packScales(pt, sg, n, k);
            lin.qwords = try L.put(try std.fmt.bufPrint(&kb, "{s}.qfp8", .{key}), pc);
            lin.qscales = try L.put(try std.fmt.bufPrint(&kb, "{s}.qfp8s", .{key}), std.mem.sliceAsBytes(pt));
            lin.qfp8 = true;
            std.log.info("glm53: the fp8 arm: {s} packed qmmf fp8g (npad {d}, k {d})", .{ key, npad8, k });
            return lin;
        }
        // Â§125.37 the dense-lane A/B arm (TF_GLM53_DENSE_LANE): the qmmf bf16 packed copy,
        // packed from the SAME host staging. NOTE: pk uses the gpa directly, NOT staging() -
        // the staging ArrayList's resize would realloc and dangle `out` (npad*k*2 >= n*k*2).
        if (std.c.getenv("TF_GLM53_DENSE_LANE") != null) {
            const npad = (n + 127) / 128 * 128;
            const pk = try L.gpa.alloc(u16, npad * k);
            defer L.gpa.free(pk);
            const w16: []const u16 = @as([*]const u16, @ptrCast(@alignCast(out.ptr)))[0 .. n * k];
            cuda.qmmf.packBf16(pk, w16, n, k);
            lin.qwords = try L.put(try std.fmt.bufPrint(&kb, "{s}.qwords", .{key}), std.mem.sliceAsBytes(pk));
            std.log.info("glm53: the dense-lane arm: {s} packed qmmf bf16 (npad {d}, k {d})", .{key, npad, k});
        }
        return lin;
    }

    /// weights.load_linear + Exl3Linear.from_tensors (strips layout) + x3linear.plan's tiles.
    fn linear(L: *Loader, prefix: []const u8, key: []const u8) !exl3.Linear {
        var nb: [256]u8 = undefined;
        var kb: [256]u8 = undefined;
        const tname = try std.fmt.bufPrint(&nb, "{s}.trellis", .{prefix});
        if (!L.has(tname)) {
            // the K4 hybrid: no trellis - the fp8 block-quant dense form (findings-k4-loader.md)
            if (L.has(try std.fmt.bufPrint(&nb, "{s}.weight", .{prefix}))) return L.dequantLinear(prefix, key);
        }
        const t = try L.get(tname);
        if (t.dtype != .i16 or t.rank != 3) return error.UnexpectedTensor;
        var shape: [core.safetensors.max_rank]usize = undefined;
        const raw = try L.part(tname, &shape);
        const last = shape[2];
        if (last % 8 != 0) return error.UnsupportedExl3Width;
        const k2 = last / 8;
        const kk = 16 * shape[0];
        const n = 16 * shape[1];
        if (kk % 128 != 0 or n % 128 != 0) return error.UnexpectedTensor;
        const cbk = try L.codebook(prefix);
        if (k2 % 2 != 0 and cbk != .mul1) return error.UnsupportedExl3Width;
        const words = try staging(&L.host, L.gpa, raw.len);
        exl3.strips(words, raw, kk, n, 2 * last); // a tile: 16 * bits int16 = 2 * last bytes
        const sk_wk = exl3.plan(kk, n);
        return .{
            .words = try L.put(try std.fmt.bufPrint(&kb, "{s}.words", .{key}), words),
            .suh = try L.scales(try std.fmt.bufPrint(&nb, "{s}.suh", .{prefix}), try std.fmt.bufPrint(&kb, "{s}.suh", .{key}), kk),
            .svh = try L.scales(try std.fmt.bufPrint(&nb, "{s}.svh", .{prefix}), try std.fmt.bufPrint(&kb, "{s}.svh", .{key}), n),
            .counters = try L.zeros(8 * (n / 128) * 4),
            .k = kk,
            .n = n,
            .k2 = k2,
            .cb = cbk,
            .sk = sk_wk[0],
            .wk = sk_wk[1],
        };
    }

    /// fused.Weights: kv_b [H (nope + v), kv_lora] -> wk [H, nope, kv_lora] and wv [H, v, kv_lora], contiguous.
    fn kvB(L: *Loader, name: []const u8, key: []const u8, c: Config, heads: usize) ![2]u64 {
        var sb: [512]u8 = undefined;
        const t = try L.get(name);
        const f8in = t.dtype == .f8_e4m3;
        // the 4.04bpw-h8 pack stores kv_b_proj as fp16: accepted with the capture-side round cast;
        // the K4 hybrid stores it fp8 block-quant (findings-k4-loader.md): dequantized at load
        if (t.dtype != .bf16 and t.dtype != .f16 and !f8in) return error.UnexpectedTensor;
        var shape: [core.safetensors.max_rank]usize = undefined;
        const raw = try L.part(name, &shape);
        const per = c.nope + c.v_dim;
        const row = c.kv_lora * 2;
        if (shape[0] != heads * per or shape[1] != c.kv_lora) return error.UnexpectedTensor;
        // host2 may hold `raw` (a cut): copy it aside before the staging buffers are resized
        var own: []u8 = &.{};
        defer if (own.len > 0) L.gpa.free(own);
        var src = raw;
        if (f8in) {
            // the K4 form: this rank's rows (the .row cut, 128-aligned: 28672 / 8 = 3584) dequantized
            // against the FULL scale table (no split rule on the scale's name - read whole), then the
            // bf16 flow below unchanged
            const st = try L.get(try std.fmt.bufPrint(&sb, "{s}_scale_inv", .{name}));
            if (st.dtype != .f32 or st.rank != 2) return error.UnexpectedTensor;
            if (st.shape[0] != (t.shape[0] + 127) / 128 or st.shape[1] != (t.shape[1] + 127) / 128) return error.UnexpectedTensor;
            const srow0 = if (L.o.world > 1) L.o.rank * shape[0] / 128 else 0;
            if (L.o.world > 1 and shape[0] % 128 != 0) return error.CutStraddlesScaleBlock;
            own = try L.gpa.alloc(u8, shape[0] * shape[1] * 2);
            dequantFp8(own, raw, L.whole(try std.fmt.bufPrint(&sb, "{s}_scale_inv", .{name}), st), shape[0], shape[1], srow0, 0, st.shape[1]);
            src = own;
        } else if (L.o.world > 1) {
            own = try L.gpa.dupe(u8, raw);
            src = own;
        }
        const wk = try staging(&L.host, L.gpa, heads * c.nope * row);
        const wv = try staging(&L.host2, L.gpa, heads * c.v_dim * row);
        const f16in = t.dtype == .f16;
        for (0..heads) |h| {
            if (f16in) {
                // fp16 storage: round-cast per element while splitting the blocks (bf16Bits = torch .to(bfloat16),
                // the values the AOT capture compiled; byte layout is identical, only the element decode differs)
                for (0..c.nope * c.kv_lora) |i| {
                    const v: f16 = @bitCast(std.mem.readInt(u16, src[(h * per * c.kv_lora + i) * 2 ..][0..2], .little));
                    std.mem.writeInt(u16, wk[(h * c.nope * c.kv_lora + i) * 2 ..][0..2], bf16Bits(@as(f32, v)), .little);
                }
                const off = (h * per + c.nope) * c.kv_lora;
                for (0..c.v_dim * c.kv_lora) |i| {
                    const v: f16 = @bitCast(std.mem.readInt(u16, src[(off + i) * 2 ..][0..2], .little));
                    std.mem.writeInt(u16, wv[(h * c.v_dim * c.kv_lora + i) * 2 ..][0..2], bf16Bits(@as(f32, v)), .little);
                }
            } else {
                @memcpy(wk[h * c.nope * row ..][0 .. c.nope * row], src[h * per * row ..][0 .. c.nope * row]);
                @memcpy(wv[h * c.v_dim * row ..][0 .. c.v_dim * row], src[(h * per + c.nope) * row ..][0 .. c.v_dim * row]);
            }
        }
        if (f16in and !RoundCastLogged.kv_b) {
            std.log.info("glm53: self_attn.kv_b_proj.weight: fp16 storage -> bf16 round cast (torch .to(bfloat16); the AOT capture compiled these values)", .{});
            RoundCastLogged.kv_b = true;
        }
        var kb: [128]u8 = undefined;
        const a = try L.put(try std.fmt.bufPrint(&kb, "{s}.wk", .{key}), wk);
        const b = try L.put(try std.fmt.bufPrint(&kb, "{s}.wv", .{key}), wv);
        return .{ a, b };
    }

    /// TF_GLM53_KVB_Q: kv_b as an EXL3 linear (kv_lora inputs, H (nope + v) outputs, cut by outputs: this rank's
    /// heads) decoded on the host to the dense matrix (headq.dense) and rounded to bf16 into the same wk / wv
    /// blocks kvB makes from a bf16 `.weight`. The attention reads bf16 as before: the pack is smaller, a token
    /// reads the same bytes.
    fn kvBq(L: *Loader, prefix: []const u8, key: []const u8, c: Config, heads: usize) ![2]u64 {
        var nb: [256]u8 = undefined;
        const tname = try std.fmt.bufPrint(&nb, "{s}.trellis", .{prefix});
        const t = try L.get(tname);
        if (t.dtype != .i16 or t.rank != 3 or t.shape[2] % 8 != 0) return error.UnexpectedTensor;
        var shape: [core.safetensors.max_rank]usize = undefined;
        const raw = try L.part(tname, &shape);
        const k2 = shape[2] / 8;
        const K = 16 * shape[0];
        const N = 16 * shape[1];
        const per = c.nope + c.v_dim;
        if (K != c.kv_lora or N != heads * per) return error.UnexpectedTensor;
        if (try L.codebook(prefix) != .mul1) return error.UnsupportedExl3Codebook;
        const trellis = try L.gpa.alloc(i16, raw.len / 2);
        defer L.gpa.free(trellis);
        @memcpy(std.mem.sliceAsBytes(trellis), raw);
        const suh = try L.gpa.alloc(f16, K);
        defer L.gpa.free(suh);
        const svh = try L.gpa.alloc(f16, N);
        defer L.gpa.free(svh);
        for ([_][]const u8{ "suh", "svh" }, [_][]f16{ suh, svh }) |name, dst| {
            const full = try std.fmt.bufPrint(&nb, "{s}.{s}", .{ prefix, name });
            const sc = try L.get(full);
            if (sc.dtype != .f16) return error.UnsupportedScales;
            const bytes = try L.part(full, &shape);
            if (bytes.len != dst.len * 2) return error.UnexpectedTensor;
            @memcpy(std.mem.sliceAsBytes(dst), bytes);
        }
        const rows = try L.gpa.alloc(f32, N * K);
        defer L.gpa.free(rows);
        try headq.dense(L.gpa, k2, trellis, K, N, suh, svh, rows);
        const wk = try L.gpa.alloc(u16, heads * c.nope * K);
        defer L.gpa.free(wk);
        const wv = try L.gpa.alloc(u16, heads * c.v_dim * K);
        defer L.gpa.free(wv);
        kvSplit(rows, heads, c.nope, c.v_dim, K, wk, wv);
        var kb: [128]u8 = undefined;
        const a = try L.put(try std.fmt.bufPrint(&kb, "{s}.wk", .{key}), std.mem.sliceAsBytes(wk));
        const b = try L.put(try std.fmt.bufPrint(&kb, "{s}.wv", .{key}), std.mem.sliceAsBytes(wv));
        return .{ a, b };
    }

    /// weights.load_experts + experts.prepare: per-expert trellises (gate, up: output tiles; down: input tiles),
    /// int64 pointer tables, int32 widths, stacked fp16 scales; digests in prepare's order (gate, up, down).
    fn experts(L: *Loader, li: usize, key: []const u8, c: Config) !Experts {
        const E = c.experts;
        const D = c.hidden;
        const I = c.expert_width / L.o.world;
        var nb: [256]u8 = undefined;
        var kb: [128]u8 = undefined;
        const projs = [_][]const u8{ "gate", "up", "down" };
        var ptrs: [3][]u64 = undefined;
        var k2s: [3][]i32 = undefined;
        for (0..3) |j| {
            ptrs[j] = try L.gpa.alloc(u64, E);
            k2s[j] = try L.gpa.alloc(i32, E);
        }
        defer for (0..3) |j| {
            L.gpa.free(ptrs[j]);
            L.gpa.free(k2s[j]);
        };
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        var lo_gu: usize = 99;
        var hi_gu: usize = 0;
        var lo_d: usize = 99;
        var hi_d: usize = 0;
        var cbk: ?exl3.Codebook = null;
        for (projs, 0..) |pj, j| {
            // one device allocation for the projection's E trellises (each at a 512-byte boundary, as torch places
            // its tensors): sizes first, then each expert's part uploaded at its offset
            var total: usize = 0;
            for (0..E) |e| {
                var tb: [256]u8 = undefined;
                const t = try L.get(try std.fmt.bufPrint(&tb, "model.layers.{d}.mlp.experts.{d}.{s}_proj.trellis", .{ li, e, pj }));
                total += std.mem.alignForward(usize, t.bytes.len / L.o.world, Weights.alignment);
            }
            const base = try L.w.alloc(L.d, total);
            var off: usize = 0;
            for (0..E) |e| {
                const pre = try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.experts.{d}.{s}_proj", .{ li, e, pj });
                if (cbk == null) cbk = try L.codebook(pre);
                var tb: [256]u8 = undefined;
                const tname = try std.fmt.bufPrint(&tb, "{s}.trellis", .{pre});
                const t = try L.get(tname);
                if (t.dtype != .i16 or t.rank != 3) return error.UnexpectedTensor;
                var shape: [core.safetensors.max_rank]usize = undefined;
                const raw = try L.part(tname, &shape);
                const want_k = if (j == 2) I else D;
                const want_n = if (j == 2) D else I;
                if (shape[0] * 16 != want_k or shape[1] * 16 != want_n) return error.UnexpectedTensor;
                if (shape[2] % 8 != 0) return error.UnsupportedExl3Width;
                const k2 = shape[2] / 8;
                if (k2 < 2 or k2 > 16) return error.UnsupportedExl3Width;
                if (L.o.digests != null) hash.update(raw); // the oracle digest only (Phase 1)
                if (off + raw.len > total) return error.UnexpectedTensor;
                try L.upload(base + off, raw);
                ptrs[j][e] = base + off;
                off += std.mem.alignForward(usize, raw.len, Weights.alignment);
                k2s[j][e] = @intCast(k2);
                if (j < 2) {
                    lo_gu = @min(lo_gu, k2);
                    hi_gu = @max(hi_gu, k2);
                } else {
                    lo_d = @min(lo_d, k2);
                    hi_d = @max(hi_d, k2);
                }
            }
        }
        // the K4 hybrid's experts are MCG (findings-k4-loader.md's census: every .mcg marker
        // 0xCBAC1FED, k2 8 like the 4.04's mul1) - the fatbin carries the CB 1 instantiations;
        // 3inst experts have none anywhere
        if (cbk.? == .@"3inst") return error.UnsupportedExpertCodebook;
        if (L.o.digests) |dg| {
            if (dg.get(try std.fmt.bufPrint(&kb, "{s}.experts.trellis", .{key}))) |want| {
                var digest: [32]u8 = undefined;
                hash.final(&digest);
                const hex = std.fmt.bytesToHex(digest, .lower);
                L.w.digest_checked += 1;
                if (want == .string and !std.mem.eql(u8, &hex, want.string)) {
                    L.w.digest_bad += 1;
                    std.log.err("weight digest differs from the oracle's: {s}.experts.trellis", .{key});
                }
            }
        }
        var ex: Experts = undefined;
        ex.cb = cbk.?;
        ex.count = E;
        ex.dims = D;
        ex.width = I;
        ex.k2_gu = .{ lo_gu, hi_gu };
        ex.k2_d = .{ lo_d, hi_d };
        ex.gate_ptr = try L.put("", std.mem.sliceAsBytes(ptrs[0]));
        ex.up_ptr = try L.put("", std.mem.sliceAsBytes(ptrs[1]));
        ex.down_ptr = try L.put("", std.mem.sliceAsBytes(ptrs[2]));
        ex.gate_k2 = try L.put(try std.fmt.bufPrint(&kb, "{s}.experts.gate_k2", .{key}), std.mem.sliceAsBytes(k2s[0]));
        ex.up_k2 = try L.put(try std.fmt.bufPrint(&kb, "{s}.experts.up_k2", .{key}), std.mem.sliceAsBytes(k2s[1]));
        ex.down_k2 = try L.put(try std.fmt.bufPrint(&kb, "{s}.experts.down_k2", .{key}), std.mem.sliceAsBytes(k2s[2]));
        // stack(mats, j, n): [E, n] fp16 of each expert's suh (1) or svh (2)
        const Stack = struct { proj: []const u8, part: []const u8, n: usize, name: []const u8 };
        const stacks = [_]Stack{
            .{ .proj = "gate", .part = "suh", .n = D, .name = "suh_g" },
            .{ .proj = "up", .part = "suh", .n = D, .name = "suh_u" },
            .{ .proj = "gate", .part = "svh", .n = I, .name = "svh_g" },
            .{ .proj = "up", .part = "svh", .n = I, .name = "svh_u" },
            .{ .proj = "down", .part = "suh", .n = I, .name = "suh_d" },
            .{ .proj = "down", .part = "svh", .n = D, .name = "svh_d" },
        };
        var out: [stacks.len]u64 = undefined;
        for (stacks, 0..) |st, si| {
            const buf = try L.gpa.alloc(u8, E * st.n * 2);
            defer L.gpa.free(buf);
            for (0..E) |e| {
                const name = try std.fmt.bufPrint(&nb, "model.layers.{d}.mlp.experts.{d}.{s}_proj.{s}", .{ li, e, st.proj, st.part });
                const t = try L.get(name);
                if (t.dtype != .f16) return error.UnsupportedScales; // .half() of fp16: as stored
                var shape: [core.safetensors.max_rank]usize = undefined;
                const raw = try L.part(name, &shape);
                if (raw.len != st.n * 2) return error.UnexpectedTensor;
                @memcpy(buf[e * st.n * 2 ..][0 .. st.n * 2], raw);
            }
            out[si] = try L.put(try std.fmt.bufPrint(&kb, "{s}.experts.{s}", .{ key, st.name }), buf);
        }
        ex.suh_g = out[0];
        ex.suh_u = out[1];
        ex.svh_g = out[2];
        ex.svh_u = out[3];
        ex.suh_d = out[4];
        ex.svh_d = out[5];
        return ex;
    }

    fn mlp(L: *Loader, prefix: []const u8, key: []const u8) !Mlp {
        var nb: [256]u8 = undefined;
        var kb: [128]u8 = undefined;
        return .{
            .gate = try L.linear(try std.fmt.bufPrint(&nb, "{s}.gate_proj", .{prefix}), try std.fmt.bufPrint(&kb, "{s}.gate", .{key})),
            .up = try L.linear(try std.fmt.bufPrint(&nb, "{s}.up_proj", .{prefix}), try std.fmt.bufPrint(&kb, "{s}.up", .{key})),
            .down = try L.linear(try std.fmt.bufPrint(&nb, "{s}.down_proj", .{prefix}), try std.fmt.bufPrint(&kb, "{s}.down", .{key})),
        };
    }

    /// Unit 0: the embedding (o.embed) and, with o.head, the final norm and this rank's lm_head share
    /// (fused.Weights: lm_head [V / world, D] split.rule rows, vocab_off = rank * V / world).
    fn globals(L: *Loader, c: Config) !void {
        const w = L.w;
        if (L.o.embed) {
            const t = try L.ck.expect("model.embed_tokens.weight", .bf16, &.{ c.vocab, c.hidden });
            _ = try L.get("model.embed_tokens.weight");
            w.embed = try L.put("embed", L.whole("model.embed_tokens.weight", t));
        }
        if (L.o.head) {
            w.final_norm = try L.bf16("model.norm.weight", "final_norm");
            if (c.vocab % L.o.world != 0) return error.UnevenVocab;
            if (L.o.head_q.on()) return L.headQ(c);
            const t = try L.get("lm_head.weight");
            if (t.dtype != .bf16 or t.rank != 2 or t.shape[0] != c.vocab or t.shape[1] != c.hidden) return error.UnexpectedTensor;
            var shape: [core.safetensors.max_rank]usize = undefined;
            const raw = try L.part("lm_head.weight", &shape);
            w.vocab_part = shape[0];
            w.vocab_off = L.o.rank * shape[0];
            w.lm_head = try L.put("lm_head", raw);
        }
    }

    /// TF_GLM53_HEAD_Q: the EXL3 head's blocks covering this rank's vocabulary share (headq.Host), in the strips
    /// layout with x3linear.plan's tiles (no tile table holds the head); lm_head stays 0. With o.head_host the
    /// checkpoint's bf16 rows of the share are kept on the host for the drafters' 4-bit head.
    fn headQ(L: *Loader, c: Config) !void {
        const w = L.w;
        var file: ?core.safetensors.File = null;
        defer if (file) |*f| {
            dropFile(f);
            f.close(L.io);
        };
        const src: headq.Source = switch (L.o.head_q) {
            .file => |path| blk: {
                file = try headq.openFile(L.gpa, L.io, path);
                break :blk .{ .file = &file.? };
            },
            .pack => .{ .ck = L.ck },
            .off => return error.Invalid,
        };
        var h = try headq.Host.load(L.gpa, src, c.vocab, c.hidden, L.o.world, L.o.rank);
        defer h.deinit();
        if (L.o.head_q == .pack) {
            // the K4 hybrid carries a PLAIN bf16 lm_head (findings-k4-loader.md): headq needs the
            // trellis form - name the fix instead of a bare MissingTensor at the rendezvous wait
            if (!L.has("lm_head.trellis")) {
                std.log.err("glm53: TF_GLM53_HEAD_Q=pack but this checkpoint has no lm_head.trellis (the K4 hybrid's head is plain bf16 .weight) - boot with TF_GLM53_HEAD_Q unset (the bf16 rows head)", .{});
                return error.BadQuantizedHead;
            }
            for ([_][]const u8{ "lm_head.trellis", "lm_head.suh", "lm_head.svh", "lm_head.mul1" }) |name| {
                _ = try L.get(name);
            }
        }
        const n = h.cut.n();
        const sk_wk = exl3.plan(h.k, n);
        // a split-K head would need the launch's partials scratch; GLM-5.3's (303 blocks a rank) takes one split
        if (sk_wk[0] != 1) return error.BadSplit;
        const words = try staging(&L.host, L.gpa, h.trellis.len);
        h.strips(words);
        w.vocab_part = h.cut.part;
        w.vocab_off = L.o.rank * h.cut.part;
        w.head_q = .{ .off = h.cut.off, .lin = .{
            .words = try L.put("lm_head.words", words),
            .suh = try L.put("lm_head.suh", h.suh),
            .svh = try L.put("lm_head.svh", h.svh),
            .counters = try L.zeros(8 * (n / 128) * 4),
            .k = h.k,
            .n = n,
            .k2 = h.k2,
            .cb = .mul1,
            .sk = sk_wk[0],
            .wk = sk_wk[1],
        } };
        std.log.info("glm53 rank {d}: quantized head ({s}): {d}.{d} bits, blocks {d}..{d} of {d} for tokens {d}..{d}, {d} MB", .{ L.o.rank, if (L.o.head_q == .pack) "the checkpoint's" else L.o.head_q.file, h.k2 / 2, (h.k2 % 2) * 5, h.cut.b0, h.cut.b0 + h.cut.nb, c.vocab / 128, w.vocab_off, w.vocab_off + w.vocab_part, words.len >> 20 });
        if (L.o.head_host) {
            if (!L.has("lm_head.weight")) {
                std.log.err("glm53: the drafters' 4-bit head is made from lm_head.weight (bf16), which this checkpoint does not hold: keep it in the pack beside the quantized head, or serve the head from a file of its own (TF_GLM53_HEAD_Q=<file>)", .{});
                return error.NoDraftHeadRows;
            }
            const t = try L.get("lm_head.weight");
            if (t.dtype != .bf16 or t.rank != 2 or t.shape[0] != c.vocab or t.shape[1] != c.hidden) return error.UnexpectedTensor;
            var shape: [core.safetensors.max_rank]usize = undefined;
            const raw = try L.part("lm_head.weight", &shape);
            if (shape[0] != w.vocab_part) return error.UnexpectedTensor;
            const rows = try L.gpa.alloc(u16, raw.len / 2);
            @memcpy(std.mem.sliceAsBytes(rows), raw);
            w.head_host = rows;
        }
    }

    /// weights.load_mtp: layer `num_hidden_layers` (a full decoder layer) and its eh_proj / enorm / hnorm /
    /// shared_head.norm (replicated).
    fn mtpHead(L: *Loader, c: Config) !Mtp {
        var nb: [128]u8 = undefined;
        var kb: [64]u8 = undefined;
        const i = c.layers;
        var out: Mtp = undefined;
        out.layer = try L.layer(i, c);
        const parts = [_][]const u8{ "eh_proj", "enorm", "hnorm", "shared_head.norm" };
        var ptrs: [4]u64 = undefined;
        for (parts, &ptrs) |name, *dst| {
            const full = try std.fmt.bufPrint(&nb, "model.layers.{d}.{s}.weight", .{ i, name });
            dst.* = try L.bf16(full, try std.fmt.bufPrint(&kb, "mtp.{s}", .{name}));
        }
        const eh = try L.get(try std.fmt.bufPrint(&nb, "model.layers.{d}.eh_proj.weight", .{i}));
        if (eh.rank != 2 or eh.shape[0] != c.hidden or eh.shape[1] != 2 * c.hidden) return error.UnexpectedTensor;
        out.eh_proj = ptrs[0];
        out.enorm = ptrs[1];
        out.hnorm = ptrs[2];
        out.head_norm = ptrs[3];
        return out;
    }

    /// Glm53Engine's reduced draft vocabulary + headq.prepare (q4): lm_head rows [rank q, (rank + 1) q) with
    /// q = draft_vocab / world, and on the last rank also the vocabulary's last fused.SPECIALS (128) rows; quantized
    /// and packed on the host, uploaded with their global ids.
    fn draftHead(L: *Loader, c: Config, pre: ?*source.Prefetch) !DraftHead {
        const world = L.o.world;
        const rank = L.o.rank;
        const specials: usize = 128;
        if (L.o.draft_vocab % world != 0 or L.o.draft_vocab > c.vocab) return error.BadDraftVocab;
        const q = L.o.draft_vocab / world;
        const extra: usize = if (rank == world - 1) specials else 0;
        const n = q + extra;
        const k = c.hidden;
        if (k % 64 != 0) return error.BadDraftVocab;
        const t = try L.get("lm_head.weight");
        if (t.dtype != .bf16 or t.shape[0] != c.vocab or t.shape[1] != k) return error.UnexpectedTensor;
        const rows = try L.gpa.alloc(u16, n * k);
        defer L.gpa.free(rows);
        const row_bytes = k * 2;
        const spans = [_][2]usize{ .{ rank * q, (rank + 1) * q }, .{ c.vocab - extra, c.vocab } };
        var at: usize = 0;
        for (spans) |sp| {
            if (sp[1] <= sp[0]) continue;
            const dst = std.mem.sliceAsBytes(rows[at * k ..][0 .. (sp[1] - sp[0]) * k]);
            if (pre) |p| {
                // the file and offset of the rows: the header's entry (the mapping is not read)
                var fi: usize = 0;
                var off: u64 = 0;
                var found = false;
                for (L.ck.files.items, 0..) |*f, i| if (f.names.get("lm_head.weight")) |e| {
                    fi = i;
                    off = f.data + e.begin;
                    found = true;
                    break;
                };
                if (!found) return error.MissingTensor;
                try p.readNow(fi, off + sp[0] * row_bytes, dst);
            } else {
                @memcpy(dst, t.bytes[sp[0] * row_bytes ..][0..dst.len]);
            }
            at += sp[1] - sp[0];
        }
        const kg = k / 64;
        const npad = (n + 127) / 128 * 128;
        const words = try L.gpa.alloc(u32, n * k / 8);
        defer L.gpa.free(words);
        const sc = try L.gpa.alloc(u16, n * kg);
        defer L.gpa.free(sc);
        const bi = try L.gpa.alloc(u16, n * kg);
        defer L.gpa.free(bi);
        quantize4(rows, n, k, words, sc, bi);
        const packed_w = try L.gpa.alloc(u32, npad / 64 * kg * 8 * 32 * 2);
        defer L.gpa.free(packed_w);
        const so = try L.gpa.alloc(u16, kg * npad);
        defer L.gpa.free(so);
        const bo = try L.gpa.alloc(u16, kg * npad);
        defer L.gpa.free(bo);
        pack4(words, sc, bi, n, k, packed_w, so, bo);
        const ids = try L.gpa.alloc(i64, n);
        defer L.gpa.free(ids);
        var j: usize = 0;
        for (spans) |sp| {
            var v = sp[0];
            while (v < sp[1]) : (v += 1) {
                ids[j] = @intCast(v);
                j += 1;
            }
        }
        return .{
            .w = try L.put("draft.w", std.mem.sliceAsBytes(packed_w)),
            .scales = try L.put("draft.scales", std.mem.sliceAsBytes(so)),
            .biases = try L.put("draft.biases", std.mem.sliceAsBytes(bo)),
            .ids = try L.put("draft.ids", std.mem.sliceAsBytes(ids)),
            .n = n,
            .npad = npad,
            .k = k,
            .sk = q4SplitK(n, k),
        };
    }

    /// weights.load_layer for layer i.
    fn layer(L: *Loader, i: usize, c: Config) !Layer {
        var nb: [256]u8 = undefined;
        var kb: [128]u8 = undefined;
        var pb: [64]u8 = undefined;
        var key_buf: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}", .{i});
        const key = try std.fmt.bufPrint(&key_buf, "layers.{d}", .{i});
        const at = struct {
            fn n(buf: []u8, pre: []const u8, s: []const u8) ![]const u8 {
                return std.fmt.bufPrint(buf, "{s}.{s}", .{ pre, s });
            }
        };
        var out: Layer = undefined;
        out.index = i;
        out.input_norm = try L.bf16(try at.n(&nb, p, "input_layernorm.weight"), try at.n(&kb, key, "input_norm"));
        out.post_attn_norm = try L.bf16(try at.n(&nb, p, "post_attention_layernorm.weight"), try at.n(&kb, key, "post_attn_norm"));
        out.q_a = try L.linear(try at.n(&nb, p, "self_attn.q_a_proj"), try at.n(&kb, key, "q_a"));
        out.q_a_norm = try L.bf16(try at.n(&nb, p, "self_attn.q_a_layernorm.weight"), try at.n(&kb, key, "q_a_norm"));
        out.kv_a = try L.linear(try at.n(&nb, p, "self_attn.kv_a_proj_with_mqa"), try at.n(&kb, key, "kv_a"));
        out.kv_a_norm = try L.bf16(try at.n(&nb, p, "self_attn.kv_a_layernorm.weight"), try at.n(&kb, key, "kv_a_norm"));
        out.q_b = try L.linear(try at.n(&nb, p, "self_attn.q_b_proj"), try at.n(&kb, key, "q_b"));
        const kvb = if (L.o.kvb_q and L.has(try at.n(&nb, p, "self_attn.kv_b_proj.trellis")))
            try L.kvBq(try at.n(&nb, p, "self_attn.kv_b_proj"), key, c, L.w.heads)
        else blk: {
            if (!L.o.kvb_q and L.has(try at.n(&nb, p, "self_attn.kv_b_proj.trellis"))) {
                std.log.err("glm53: this checkpoint's kv_b_proj is an EXL3 linear: set TF_GLM53_KVB_Q=1 on every rank to decode it at load", .{});
                return error.UnexpectedTensor;
            }
            break :blk try L.kvB(try at.n(&nb, p, "self_attn.kv_b_proj.weight"), key, c, L.w.heads);
        };
        out.wk = kvb[0];
        out.wv = kvb[1];
        out.o_proj = try L.linear(try at.n(&nb, p, "self_attn.o_proj"), try at.n(&kb, key, "o_proj"));
        out.indexer = null;
        if (L.has(try at.n(&nb, p, "self_attn.indexer.wq_b.trellis"))) {
            out.indexer = .{
                .wq_b = try L.linear(try at.n(&nb, p, "self_attn.indexer.wq_b"), try at.n(&kb, key, "wq_b")),
                .wk = try L.bf16OrDequant(try at.n(&nb, p, "self_attn.indexer.wk.weight"), try at.n(&kb, key, "idx.wk"), &RoundCastLogged.idx_wk, "self_attn.indexer.wk.weight"),
                .weights_proj = try L.bf16Round(try at.n(&nb, p, "self_attn.indexer.weights_proj.weight"), try at.n(&kb, key, "idx.weights_proj"), &RoundCastLogged.idx_weights_proj, "self_attn.indexer.weights_proj.weight"),
                .k_norm_w = try L.bf16Round(try at.n(&nb, p, "self_attn.indexer.k_norm.weight"), try at.n(&kb, key, "idx.k_norm_w"), &RoundCastLogged.idx_k_norm_w, "self_attn.indexer.k_norm.weight"),
                .k_norm_b = try L.bf16Round(try at.n(&nb, p, "self_attn.indexer.k_norm.bias"), try at.n(&kb, key, "idx.k_norm_b"), &RoundCastLogged.idx_k_norm_b, "self_attn.indexer.k_norm.bias"),
            };
        }
        out.router_w = 0;
        out.router_bias = 0;
        out.experts = null;
        if (c.isMoe(i)) {
            out.router_w = try L.bf16Round(try at.n(&nb, p, "mlp.gate.weight"), try at.n(&kb, key, "router.weight"), &RoundCastLogged.gate, "mlp.gate.weight");
            out.router_bias = try L.f32vec(try at.n(&nb, p, "mlp.gate.e_score_correction_bias"), try at.n(&kb, key, "router.bias"));
            out.experts = try L.experts(i, key, c);
            out.mlp = try L.mlp(try at.n(&nb, p, "mlp.shared_experts"), try at.n(&kb, key, "mlp"));
        } else {
            out.mlp = try L.mlp(try at.n(&nb, p, "mlp"), try at.n(&kb, key, "mlp"));
        }
        return out;
    }
};

/// Layers 0..o.layers-1 of rank o.rank of o.world (and the embedding; with o.head the final norm and this rank's
/// lm_head share; with o.mtp the MTP layer; with o.draft_vocab the 4-bit draft head), from the checkpoint folder
/// `dir`. Mapped reads: each layer's files leave the page cache once it is on the GPU. o.fast: the units' bytes are
/// read with O_DIRECT a unit ahead (cuda_source.zig) and the mapping is only used for the headers.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: Config, o: Options) !Weights {
    if (c.heads % o.world != 0) return error.UnevenHeads;
    if (o.layers > c.layers) return error.TooManyLayers;
    if (o.mtp and c.mtp_layers == 0) return error.NoMtpLayer;
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var w: Weights = .{ .gpa = gpa, .config = c, .rank = o.rank, .world = o.world, .heads = c.heads / o.world };
    errdefer w.deinit();
    var L: Loader = .{ .gpa = gpa, .d = d, .ck = &ck, .w = &w, .o = o, .io = io };
    defer L.deinit();
    L.touched = try gpa.alloc(bool, ck.files.items.len);
    @memset(L.touched, false);
    var pre: ?*source.Prefetch = null;
    if (o.fast) pre = source.Prefetch.init(gpa, &ck, o.rank, o.world) catch |e| blk: {
        std.log.warn("glm53: O_DIRECT prefetch unavailable ({t}); mapped reads", .{e});
        break :blk null;
    };
    defer if (pre) |p| p.deinit();
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();

    // units in load order: 0 (embedding, final norm, lm_head), 1 + i (layer i), 1 + c.layers (the MTP layer)
    var units: [config_max_layers + 2]usize = undefined;
    var nu: usize = 0;
    if (o.embed or o.head or o.draft_vocab > 0) {
        units[nu] = 0;
        nu += 1;
    }
    for (0..o.layers) |i| {
        units[nu] = i + 1;
        nu += 1;
    }
    if (o.mtp) {
        units[nu] = c.layers + 1;
        nu += 1;
    }
    const layers = try gpa.alloc(Layer, o.layers);
    var made: usize = 0;
    errdefer gpa.free(layers);
    if (pre) |p| for (0..@min(nu, 2)) |k| try p.request(k, units[k]);
    for (units[0..nu], 0..) |u, ui| {
        if (pre) |p| L.batch = try p.wait(ui, u);
        if (u == 0) {
            try L.globals(c);
        } else if (u <= c.layers and u - 1 < o.layers) {
            layers[u - 1] = try L.layer(u - 1, c);
            made += 1;
        } else {
            w.mtp = try L.mtpHead(c);
        }
        L.batch = null;
        if (pre) |p| {
            p.release(ui);
            if (ui + 2 < nu) try p.request(ui + 2, units[ui + 2]);
        }
        L.release();
        if (u == 0 or u > c.layers) continue;
        const i = u - 1;
        const last = i + 1 == o.layers;
        if (o.log_every == 0) {
            std.log.info("glm53: layer {d} loaded", .{i});
        } else if ((i + 1) % o.log_every == 0 or last) {
            const secs = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e9;
            const gib = @as(f64, @floatFromInt(w.device_bytes)) / (1 << 30);
            const rd = if (pre) |p| @as(f64, @floatFromInt(p.read_bytes.load(.monotonic))) / (1 << 30) else 0;
            std.log.info("glm53 rank {d}: {d} of {d} layers loaded ({d:.0} s, {d:.1} GiB of weights on the device, {d:.1} GiB read direct)", .{ o.rank, i + 1, o.layers, secs, gib, rd });
        }
    }
    if (made != o.layers) return error.Invalid;
    if (o.draft_vocab > 0) w.draft = try L.draftHead(c, pre);
    L.releaseAll();
    if (pre) |p| {
        w.read_bytes = p.read_bytes.load(.monotonic);
        w.read_fallbacks = p.fallbacks.load(.monotonic);
        w.direct_files = p.direct_files;
    }
    w.layers = layers;
    return w;
}

/// kv_b's dense rows [H (nope + v), K] fp32 (a head's nope key rows, then its v value rows) rounded to bf16 into
/// wk [H, nope, K] and wv [H, v, K]: Loader.kvB's split of a bf16 `.weight`.
pub fn kvSplit(rows: []const f32, heads: usize, nope: usize, v_dim: usize, K: usize, wk: []u16, wv: []u16) void {
    const per = nope + v_dim;
    std.debug.assert(rows.len == heads * per * K and wk.len == heads * nope * K and wv.len == heads * v_dim * K);
    for (0..heads) |h| {
        for (0..nope * K) |i| wk[h * nope * K + i] = bf16Bits(rows[h * per * K + i]);
        for (0..v_dim * K) |i| wv[h * v_dim * K + i] = bf16Bits(rows[(h * per + nope) * K + i]);
    }
}

// --------------------------------------------------------------------------------------- the draft head (host) ---

/// torch's float -> bfloat16 on the device (__float2bfloat16: round to nearest even; NaN -> 0x7FFF).
/// e4m3 -> f32, the host decode of the F8_E4M3 storage (the K4 pack's dense matrices,
/// findings-k4-loader.md): bias 7, denormals m * 2^-9, 0x7F/0xFF NaN - the values torch's
/// float8_e4m3fn to(float32) produces.
pub fn fp8ToF32(code: u8) f32 {
    const sign: u32 = if (code & 0x80 != 0) 0x80000000 else 0;
    const e: u32 = (code >> 3) & 0xF;
    const m: u32 = code & 7;
    if (e == 0) {
        if (m == 0) return @bitCast(sign);
        // denormal: (m / 8) * 2^-6
        const v: f32 = @as(f32, @floatFromInt(m)) * 0.001953125;
        return @bitCast(sign | @as(u32, @bitCast(v)));
    }
    if (e == 0xF and m == 7) return std.math.nan(f32);
    // normal: (1 + m / 8) * 2^(e - 7)
    return @bitCast(sign | ((e + 120) << 23) | (m << 20));
}

/// The K4 dequant (findings-k4-loader.md): out bf16 [n, k] row-major = the fp8 codes times the
/// 128x128-block scales - out[i, j] = bf16(fp8(w[i, j]) * scales[srow0 + i/128, scol0 + j/128]).
/// `srow0`/`scol0`: this rank's first scale block (the weight cuts are 128-aligned, asserted by
/// the callers); `sld`: the scale table's row stride, the GLOBAL ceil(K / 128). Pure: the unit
/// test's target.
/// 126.6(at) the fp8 canary's gate (value semantics: unset/empty/"0" = off).
var fp8_arm: ?bool = null;
fn fp8Arm() bool {
    if (fp8_arm == null) fp8_arm = blk: {
        const v = std.c.getenv("TF_GLM53_DENSE_FP8") orelse break :blk false;
        const s = std.mem.span(v);
        break :blk s.len > 0 and !std.mem.eql(u8, s, "0");
    };
    return fp8_arm.?;
}

pub fn dequantFp8(out: []u8, w: []const u8, scales: []const u8, n: usize, k: usize, srow0: usize, scol0: usize, sld: usize) void {
    std.debug.assert(out.len >= n * k * 2 and w.len >= n * k); // callers may pass larger buffers
    for (0..n) |i| {
        const srow = (srow0 + i / 128) * sld * 4;
        const wrow = w[i * k ..][0..k];
        const orow = out[i * k * 2 ..][0 .. k * 2];
        for (0..k) |j| {
            const soff = srow + (scol0 + j / 128) * 4;
            const sc: f32 = @bitCast(std.mem.readInt(u32, scales[soff..][0..4], .little));
            const v = fp8ToF32(wrow[j]) * sc;
            std.mem.writeInt(u16, orow[2 * j ..][0..2], bf16Bits(v), .little);
        }
    }
}

pub fn bf16Bits(f: f32) u16 {
    if (std.math.isNan(f)) return 0x7FFF;
    const u: u32 = @bitCast(f);
    return @truncate((u +% (0x7FFF + ((u >> 16) & 1))) >> 16);
}

fn bf16Value(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// torch.round (nearbyint: halves to even) for the small non-negative-ish values quantize4 rounds.
pub fn roundEven(x: f32) f32 {
    const fl = @floor(x);
    const diff = x - fl;
    if (diff > 0.5) return fl + 1;
    if (diff < 0.5) return fl;
    return if (@mod(fl, 2.0) == 0) fl else fl + 1;
}

/// glm5_next qmm.split_k: K slices of an (n, k) 4-bit weight, from its shape alone.
pub fn q4SplitK(n: usize, k: usize) usize {
    const forced = [_]struct { n: usize, k: usize, sk: usize }{
        .{ .n = 12576, .k = 4096, .sk = 4 }, .{ .n = 4096, .k = 4096, .sk = 2 }, .{ .n = 2048, .k = 4096, .sk = 4 },
        .{ .n = 8192, .k = 1536, .sk = 4 },  .{ .n = 8192, .k = 512, .sk = 4 },  .{ .n = 4096, .k = 8192, .sk = 8 },
    };
    for (forced) |f| if (f.n == n and f.k == k) return f.sk;
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

/// qmm.quantize4 of bf16 rows `w` [n, k]: words [n, k / 8] (nibble j of word i = input 8 i + j), bf16 scales and
/// biases [n, k / 64], with torch's float32 steps (amin / amax, (hi - lo) / 15 clamped at 1e-8 then rounded to bf16,
/// round((x - bias) / scale) halves to even, clamped to 0..15).
pub fn quantize4(w: []const u16, n: usize, k: usize, words: []u32, scales: []u16, biases: []u16) void {
    const kg = k / 64;
    const tiny: f32 = @floatCast(@as(f64, 1e-8));
    for (0..n) |r| {
        for (0..kg) |g| {
            const vals = w[r * k + g * 64 ..][0..64];
            var lo: f32 = bf16Value(vals[0]);
            var hi: f32 = lo;
            for (vals[1..]) |b| {
                const v = bf16Value(b);
                lo = @min(lo, v);
                hi = @max(hi, v);
            }
            var sc: f32 = (hi - lo) / 15.0;
            if (sc < tiny) sc = tiny;
            const sb = bf16Bits(sc);
            const bb = bf16Bits(lo);
            scales[r * kg + g] = sb;
            biases[r * kg + g] = bb;
            const sf = bf16Value(sb);
            const bf = bf16Value(bb);
            for (0..8) |wi| {
                var word: u32 = 0;
                for (0..8) |j| {
                    const x = bf16Value(vals[wi * 8 + j]);
                    var q = roundEven((x - bf) / sf);
                    q = @min(@max(q, 0.0), 15.0);
                    word |= @as(u32, @intFromFloat(q)) << @intCast(4 * j);
                }
                words[r * (k / 8) + g * 8 + wi] = word;
            }
        }
    }
}

/// shared qmm.OFFSETS: nibble slot p of a lane's word holds input 32 v + 2 (lane % 4) + OFFSETS[p] of its group.
const q4_offsets = [8]usize{ 0, 8, 16, 24, 1, 9, 17, 25 };

/// shared qmm.pack (gs 64): words [n, k / 8] -> out [npad / 64, k / 64, 8, 32, 2] int32 (npad = n rounded up to 128,
/// zero columns past n); scales / biases [n, k / 64] -> [k / 64, npad] (zeros past n).
pub fn pack4(words: []const u32, scales: []const u16, biases: []const u16, n: usize, k: usize, out: []u32, s_out: []u16, b_out: []u16) void {
    @memset(out, 0);
    @memset(s_out, 0);
    @memset(b_out, 0);
    pack4Range(words, scales, biases, n, k, 0, n, out, s_out, b_out);
}

/// pack4's columns [c0, c1) only (outputs zeroed by the caller): each column writes its own words, scales and biases,
/// so ranges can be packed side by side (Phase 4's threaded drafter quantization).
pub fn pack4Range(words: []const u32, scales: []const u16, biases: []const u16, n: usize, k: usize, c0: usize, c1: usize, out: []u32, s_out: []u16, b_out: []u16) void {
    const kg = k / 64;
    const npad = (n + 127) / 128 * 128;
    for (c0..c1) |c| {
        const T = c / 64;
        const j = (c % 64) / 8;
        const r = c % 8;
        for (0..kg) |g| {
            s_out[g * npad + c] = scales[c * kg + g];
            b_out[g * npad + c] = biases[c * kg + g];
            for (0..2) |v| for (0..4) |c4| {
                var word: u32 = 0;
                for (q4_offsets, 0..) |o, p| {
                    const idx = g * 64 + 32 * v + 2 * c4 + o; // the input within the row
                    const nib = (words[c * (k / 8) + idx / 8] >> @intCast(4 * (idx % 8))) & 0xF;
                    word |= nib << @intCast(4 * p);
                }
                out[(((T * kg + g) * 8 + j) * 32 + (r * 4 + c4)) * 2 + v] = word;
            };
        }
    }
}

test "quantize4 + pack4: shapes, a constant group, round halves to even" {
    try std.testing.expectEqual(@as(f32, 2), roundEven(2.5));
    try std.testing.expectEqual(@as(f32, 4), roundEven(3.5));
    try std.testing.expectEqual(@as(f32, 3), roundEven(2.6));
    try std.testing.expectEqual(@as(u16, 0x3F80), bf16Bits(1.0));
    try std.testing.expectEqual(@as(usize, 2), q4SplitK(8192, 6144));
    try std.testing.expectEqual(@as(usize, 2), q4SplitK(8320, 6144));
    const n = 3;
    const k = 128;
    var w: [n * k]u16 = undefined;
    for (&w, 0..) |*x, i| x.* = bf16Bits(@floatFromInt(i % 16));
    var words: [n * k / 8]u32 = undefined;
    var sc: [n * 2]u16 = undefined;
    var bi: [n * 2]u16 = undefined;
    quantize4(&w, n, k, &words, &sc, &bi);
    // inputs 0..15 in each group: lo 0, scale 1, codes = the values
    try std.testing.expectEqual(bf16Bits(1.0), sc[0]);
    try std.testing.expectEqual(@as(u32, 0x76543210), words[0]);
    var out: [128 / 64 * 2 * 8 * 32 * 2]u32 = undefined;
    var so: [2 * 128]u16 = undefined;
    var bo: [2 * 128]u16 = undefined;
    pack4(&words, &sc, &bi, n, k, &out, &so, &bo);
    // column 0, group 0, lane 0 (j 0, r 0, c4 0), v 0: inputs 0, 8, 16, 24, 1, 9, 17, 25 -> 0, 8, 0, 8, 1, 9, 1, 9
    try std.testing.expectEqual(@as(u32, 0x91918080), out[0]);
    try std.testing.expectEqual(sc[1], so[128 + 0]);
    try std.testing.expectEqual(@as(u16, 0), so[5]);
}

test "an EXL3 kv_b's dense rows land in the blocks a bf16 kv_b gives" {
    // 2 heads, 3 key rows + 2 value rows a head, 4 inputs: row r holds r + i / 8 (exact in bf16)
    const heads = 2;
    const nope = 3;
    const v_dim = 2;
    const K = 4;
    var rows: [heads * (nope + v_dim) * K]f32 = undefined;
    for (&rows, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i / K)) + @as(f32, @floatFromInt(i % K)) / 8;
    var wk: [heads * nope * K]u16 = undefined;
    var wv: [heads * v_dim * K]u16 = undefined;
    kvSplit(&rows, heads, nope, v_dim, K, &wk, &wv);
    // Loader.kvB's copies of the same rows as bf16 bytes
    for (0..heads) |h| {
        for (0..nope * K) |i| try std.testing.expectEqual(bf16Bits(rows[h * (nope + v_dim) * K + i]), wk[h * nope * K + i]);
        for (0..v_dim * K) |i| try std.testing.expectEqual(bf16Bits(rows[(h * (nope + v_dim) + nope) * K + i]), wv[h * v_dim * K + i]);
    }
    try std.testing.expectEqual(bf16Bits(5.0), wk[nope * K]); // head 1's first key row is row 5
    try std.testing.expectEqual(bf16Bits(3.125), wv[1]); // head 0's first value row is row 3
}

test "k4 fp8ToF32: the e4m3 decode's hand vectors" {
    // the F8_E4M3 conventions (findings-k4-loader.md): bias 7, denormal m * 2^-9, 0x7F NaN
    try std.testing.expectEqual(@as(f32, 0.0), fp8ToF32(0x00));
    try std.testing.expectEqual(@as(f32, 1.0), fp8ToF32(0x38));
    try std.testing.expectEqual(@as(f32, 1.125), fp8ToF32(0x39));
    try std.testing.expectEqual(@as(f32, 2.0), fp8ToF32(0x40));
    try std.testing.expectEqual(@as(f32, 8.0), fp8ToF32(0x50));
    try std.testing.expectEqual(@as(f32, 448.0), fp8ToF32(0x7E));
    try std.testing.expect(std.math.isNan(fp8ToF32(0x7F)));
    try std.testing.expectEqual(@as(f32, -0.0), fp8ToF32(0x80));
    try std.testing.expectEqual(@as(f32, -2.0), fp8ToF32(0xC0));
    try std.testing.expectEqual(@as(f32, 0.001953125), fp8ToF32(0x01)); // the denormal: 1 * 2^-9
    try std.testing.expectEqual(@as(f32, -0.005859375), fp8ToF32(0x83)); // -3 * 2^-9
}

test "k4 dequantFp8: a hand-computed 2x-block plane" {
    // 256x256 (2x2 scale blocks), the codes cycling a few e4m3 values, the scales 1 / 4 / 0.5 / 2 -
    // every (block row, block col) combination exercised
    const N = 256;
    const K = 256;
    var w: [N * K]u8 = undefined;
    for (&w, 0..) |*x, i| x.* = switch (i % 4) {
        0 => 0x38, // 1.0
        1 => 0x40, // 2.0
        2 => 0xC0, // -2.0
        3 => 0x50, // 8.0
        else => unreachable,
    };
    const scales: [16]u8 = @bitCast([_]f32{ 1.0, 4.0, 0.5, 2.0 }); // row-major [2, 2], LE host
    var out: [N * K * 2]u8 = undefined;
    dequantFp8(&out, &w, &scales, N, K, 0, 0, 2);
    for (0..N) |i| {
        for (0..K) |j| {
            const base: f32 = switch ((i * K + j) % 4) {
                0 => 1.0,
                1 => 2.0,
                2 => -2.0,
                3 => 8.0,
                else => unreachable,
            };
            const sc: f32 = switch ((i / 128) * 2 + j / 128) {
                0 => 1.0,
                1 => 4.0,
                2 => 0.5,
                3 => 2.0,
                else => unreachable,
            };
            const bits = std.mem.readInt(u16, out[(i * K + j) * 2 ..][0..2], .little);
            try std.testing.expectEqual(bf16Bits(base * sc), bits);
        }
    }
    // the rank offsets: srow0 1, scol0 1 reads the (1, 1) block only
    dequantFp8(&out, &w, &scales, 128, 128, 1, 1, 2);
    for (0..128) |i| {
        const bits = std.mem.readInt(u16, out[i * 128 * 2 ..][0..2], .little);
        try std.testing.expectEqual(bf16Bits(2.0), bits); // code 0x38 * scale 2.0
    }
}

test "k4 dequantFp8: the numpy cross-check dump" {
    // the synthetic plane the python checker (k4-dequant-check.py, findings-k4-loader.md) rebuilds
    // with torch and compares bit-exact against this dump. KEEP THE INPUT FORMULAS IN SYNC.
    const N = 512;
    const K = 384; // 3 K-blocks: the scol0 arithmetic at a non-square shape
    var w: [N * K]u8 = undefined;
    for (&w, 0..) |*x, i| x.* = @intCast((i * 131 + 17) % 256);
    var scales: [16 * 4]u8 = undefined; // the [4, 4] LE f32 table (row stride 4 = sld)
    const sc = [_]f32{ 1.0, 0.25, 3.0, 0.5, 2.0, 0.125, 8.0, 1.5, 0.75, 6.0, 0.375, 4.5, 1.25, 5.0, 0.625, 7.0 };
    for (sc, 0..) |v, i| std.mem.writeInt(u32, scales[4 * i ..][0..4], @bitCast(v), .little);
    var out: [N * K * 2]u8 = undefined;
    dequantFp8(&out, &w, &scales, N, K, 0, 0, 4);
    // also a cut rank's view: rows 256.., cols 128.. (srow0 2, scol0 1) - gathered per row (a row
    // range + a column range of the row-major plane is not contiguous)
    var wc: [256 * 256]u8 = undefined;
    for (0..256) |i| @memcpy(wc[i * 256 ..][0..256], w[(256 + i) * K + 128 ..][0..256]);
    var cut: [256 * 256 * 2]u8 = undefined;
    dequantFp8(&cut, &wc, &scales, 256, 256, 2, 1, 4);
    // Zig 0.17 has no std.fs.cwd - the snapshot_file fd pattern
    const fd = std.c.open("/tmp/k4-dequant-check.bin", .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.TestFile;
    defer _ = std.c.close(fd);
    for ([_][]const u8{ &w, &scales, &out, &cut }) |bytes| {
        var done: usize = 0;
        while (done < bytes.len) {
            const n = std.c.write(fd, bytes.ptr + done, bytes.len - done);
            if (n <= 0) return error.TestFile;
            done += @intCast(n);
        }
    }
}
