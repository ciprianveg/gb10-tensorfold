//! GLM-5.3's Triton kernels from the captured cubins: each launch has the Python call site's grid, constexprs and
//! launch options (fused.py, glm5_next glue.py / latent.py, topk.py). Only constexprs the call site passes are
//! matched (defaults it leaves out are not: aot.find ignores constexprs a launch does not name).
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

const p = aot.ptr;

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn cb(name: []const u8, v: bool) aot.Const {
    return aot.ci(name, @intFromBool(v));
}

fn u(x: usize) u32 {
    return @intCast(x);
}

pub fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

pub fn pow2(n: usize) usize {
    return std.math.ceilPowerOfTwo(usize, n) catch unreachable;
}

/// A Python float constexpr as Triton stores it: the fp32 nearest the fp64 value.
pub fn f32of(x: f64) f32 {
    return @floatCast(x);
}

/// fused.ATTN_DECODE / ATTN_PROMPT: (keys a chunk program, keys a tile, warps, stages) by window class.
pub const AttnTiling = struct { chk: usize, kt: usize, nw: u32, ns: u32 };
pub const attn_decode: AttnTiling = .{ .chk = 256, .kt = 32, .nw = 4, .ns = 2 };
pub const attn_prompt: AttnTiling = .{ .chk = 2048, .kt = 64, .nw = 8, .ns = 2 };
/// TF_GLM53_KV8_V2's prompt tiling (findings-kv8-v2.md §9): the bf16 prompt tiling's KTT 64 stages the fp8 code
/// tiles beside the bf16 dot tiles and the variant lands at 144,896 B of shared - past the ~100 KB the device
/// opts into (the native prefill kernel budgets SMEM_MAX 96 KB), and cuLaunchKernel refuses it with
/// INVALID_VALUE. KTT 32 (the decode tiling's tile, a proven compile) halves every tile: the bf16 KTT-32 variant
/// measures 46,592 B, the fp8 extras are at most ~36.9 KB double-buffered - ~83 KB worst case, under the line.
/// ns 1 (§9.1/§9.2): the KV8 loop's key rows come from the TOK load - a load whose ADDRESS depends on another
/// load's data. An indirect-load staging hazard was SUSPECTED at ns 2 (§9.1's SASS reading), but the crash that
/// motivated it turned out to be a kv8repro harness bug (§9.2) - ns 2 is unvalidated, not convicted. ns 1 is
/// kept as the safe default (it validated clean; shared drops further); after the p >= K ladder passes, ns 2 is
/// the first perf revisit: flip this, re-sweep, re-run the ladder.
/// Pinned here, not a setting: a tiling the pack has no variant for is a dead boot.
pub const attn_prompt_kv8: AttnTiling = .{ .chk = 2048, .kt = 32, .nw = 4, .ns = 1 };

/// fused.RB: rows a program in the absorb / expand kernels for wide windows.
pub const rb: usize = 16;
/// fused.BT: keys a scoring program.
pub const bt: usize = 128;
/// glue.ROUTER_KS: the router's K slices.
pub const router_ks: usize = 8;

/// The MLA shapes every attention launch compiles in (this rank's heads).
pub const Mla = struct { heads: usize, nope: usize, rope: usize, lw: usize, vd: usize, topk: usize };

pub const Tri = struct {
    set: ?*const aot.Set,
    s: cuda.Stream,
    /// Coverage (cuda_coverage.zig, no GPU): record each launch's variant key here instead of launching.
    probe: ?*aot.Probe = null,

    fn run(t: Tri, name: []const u8, grid: [3]usize, args: []const aot.Arg, consts: []const aot.Const, opts: aot.Launch) !void {
        if (t.probe) |pr| return pr.add(name, args, consts, opts);
        const set = t.set orelse return error.NoTritonSet;
        try set.runWith(t.s, name, .{ u(grid[0]), u(grid[1]), u(grid[2]) }, args, consts, opts);
    }

    /// glue.rmsnorm (no group sums): out = bf16(w * bf16(x * rinv)).
    pub fn rmsnorm(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, o_stride: usize, rows: usize, d: usize, eps: f32) !void {
        const block = pow2(d);
        try t.run("_rmsnorm", .{ rows, 1, 1 }, &.{ p("X", "*bf16", x), int("x_stride", x_stride), p("W", "*bf16", w), p("OUT", "*bf16", out), int("o_stride", o_stride), p("XS", "*bf16", out), aot.float("eps", eps) }, &.{ ci("D", d), ci("BLOCK", block), cb("SUMS", false) }, .{ .num_warps = if (block <= 2048) 4 else 8 });
    }

    /// glue.router: OUT[m, e] = fp32 x[m] . w[e] over ROUTER_KS K slices, the slices added in order. `part`: fp32
    /// scratch of ROUTER_KS * m * ne.
    pub fn router(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, part: u64, m: usize, d: usize, ne: usize) !void {
        if (d % (router_ks * 64) != 0) return error.RouterUnsliced; // TODO: glue's single-pass _router branch
        const bm: usize = if (m <= 16) 16 else if (m <= 32) 32 else if (m <= 64) 64 else 128;
        try t.run("_router_part", .{ cdiv(m, bm), cdiv(ne, 32), router_ks }, &.{ p("X", "*bf16", x), p("W", "*bf16", w), p("PART", "*fp32", part), int("M", m), int("x_stride", x_stride) }, &.{ ci("D", d), ci("NE", ne), ci("BM", bm), ci("BLOCK_E", 32), ci("BK", 64), ci("KS", router_ks) }, .{ .num_warps = 4, .num_stages = 3 });
        const total = m * ne;
        try t.run("_router_sum", .{ cdiv(total, 1024), 1, 1 }, &.{ p("PART", "*fp32", part), p("OUT", "*fp32", out), int("total", total) }, &.{ ci("KS", router_ks), ci("BLOCK", 1024) }, .{ .num_warps = 4 });
    }

    /// fused.route's glue._topk: sigmoid scores + bias pick TOPK experts (lower id on ties), weights normalized and
    /// scaled; SLOTS = TOPK (no shared slot).
    pub fn topk(t: Tri, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, ne: usize, k: usize, scale: f32, norm: bool) !void {
        try t.run("_topk", .{ rows, 1, 1 }, &.{ p("L", "*fp32", logits), p("BIAS", "*fp32", bias), p("PICK", "*i32", pick), p("WTS", "*fp32", wts), aot.float("scale", scale) }, &.{ ci("NE", ne), ci("TOPK", k), ci("SLOTS", k), ci("BLOCK", pow2(ne + 1)), ci("SLOTP", pow2(k + 1)), cb("NORM", norm) }, .{ .num_warps = 4 });
    }

    /// glue.residual_add: X = bf16(X + bf16(the world partials summed rank 0 first)); g [world, rows, d] fp32.
    pub fn residualAdd(t: Tri, x: u64, g: u64, rows: usize, d: usize, world: usize) !void {
        return t.residualAddOf(x, g, "*fp32", rows, d, world);
    }

    /// glue.residual_add with the partials' dtype named: "*fp32" (gathered partials, the RoCE / ring sums in b.red) or
    /// "*bf16" (Phase 3a: a prompt half's bf16 ring all-reduce / reduce-scatter, b.hred).
    pub fn residualAddOf(t: Tri, x: u64, g: u64, g_ty: []const u8, rows: usize, d: usize, world: usize) !void {
        const block = @min(1024, d);
        try t.run("_residual_add", .{ rows, d / block, 1 }, &.{ p("X", "*bf16", x), p("XOUT", "*bf16", x), p("G", g_ty, g), int("RS", rows * d) }, &.{ ci("D", d), ci("WORLD", world), ci("BLOCK", block) }, .{ .num_warps = 4 });
    }

    /// fused._qrope: q_rot of every row at POS + r (the wide window's companion of the batched absorb).
    pub fn qrope(t: Tri, q: u64, inv: u64, qr: u64, pos: u64, rows: usize, m: Mla) !void {
        try t.run("_qrope", .{ rows, m.heads, 1 }, &.{ p("Q", "*bf16", q), p("INV", "*fp32", inv), p("QR", "*bf16", qr), p("POS", "*i32", pos) }, &.{ ci("H", m.heads), ci("QD", m.nope + m.rope), ci("NOPE", m.nope), ci("RD", m.rope), cb("ROWS", false) }, .{ .num_warps = 1 });
    }

    /// x3prefill._gemm: OUT[m, block] = ((xh[m] @ W_q[:, block]) @ H) * SCALE * svh (no bias), a 128-column block a
    /// program; tiles from x3prefill.tiles (the shape's alone). `out_ty` "*bf16" or "*fp32" (the out tensor's dtype).
    pub fn prefillGemm(t: Tri, xh: u64, wq: u64, h: u64, svh: u64, out: u64, out_ty: []const u8, M: usize, o_stride: usize, K: usize, N: usize, bm: usize, bk: usize, group: usize, warps: u32, stages: u32, scale: f32) !void {
        try t.run("_gemm", .{ cdiv(M, bm) * (N / 128), 1, 1 }, &.{ p("X", "*fp16", xh), p("W", "*fp16", wq), p("H", "*bf16", h), p("SVH", "*fp16", svh), p("BIAS", "*fp16", svh), p("OUT", out_ty, out), int("M", M), int("o_stride", o_stride) }, &.{ ci("K", K), ci("N", N), ci("BM", bm), ci("BK", bk), ci("GROUP", group), cb("HAS_BIAS", false), aot.cf("SCALE", scale) }, .{ .num_warps = warps, .num_stages = stages });
    }

    /// fused._kv_write (bf16 cache, DCP 1, one stream): row r's latent RMSNorm and rotated key into cache row POS + r.
    pub fn kvWrite(t: Tri, kva: u64, kva_stride: usize, nw: u64, cache: u64, pos: u64, inv: u64, eps: f32, rows: usize, m: Mla) !void {
        return t.kvWriteDcp(kva, kva_stride, nw, cache, pos, inv, eps, rows, m, 1, 0);
    }

    /// fused.kv_write's _kv_write with DCP / RANK (Phase 3b): only the owner of position p (p % DCP == RANK) stores
    /// its row, at local row p // DCP (DCP 1, RANK 0: every row at p, as kvWrite).
    pub fn kvWriteDcp(t: Tri, kva: u64, kva_stride: usize, nw: u64, cache: u64, pos: u64, inv: u64, eps: f32, rows: usize, m: Mla, dcp: usize, rank: usize) !void {
        try t.run("_kv_write", .{ rows, 1, 1 }, &.{ p("KVA", "*bf16", kva), int("kva_stride", kva_stride), p("NW", "*bf16", nw), p("LC", "*bf16", cache), p("POS", "*i32", pos), p("INV", "*fp32", inv), aot.float("eps", eps) }, &.{ ci("LW", m.lw), ci("RD", m.rope), ci("DCP", dcp), ci("RANK", rank), cb("ROWS", false) }, .{ .num_warps = 4 });
    }

    /// fused._ik_write: the index key's LayerNorm (fp32, bias) and RoPE on its first RD dims into IC[POS + r].
    pub fn ikWrite(t: Tri, ik: u64, nw: u64, nb: u64, inv: u64, ic: u64, pos: u64, rows: usize, d: usize, rd: usize) !void {
        return t.ikWriteDcp(ik, nw, nb, inv, ic, pos, rows, d, rd, 1, 0);
    }

    /// _ik_write with DCP / RANK (Phase 3b): the owner's local row p // DCP only.
    pub fn ikWriteDcp(t: Tri, ik: u64, nw: u64, nb: u64, inv: u64, ic: u64, pos: u64, rows: usize, d: usize, rd: usize, dcp: usize, rank: usize) !void {
        try t.run("_ik_write", .{ rows, 1, 1 }, &.{ p("IK", "*fp32", ik), p("NW", "*bf16", nw), p("NB", "*bf16", nb), p("INV", "*fp32", inv), p("IC", "*bf16", ic), p("POS", "*i32", pos), aot.float("eps", 1e-6) }, &.{ ci("D", d), ci("RD", rd), ci("DCP", dcp), ci("RANK", rank), cb("ROWS", false) }, .{ .num_warps = 4 });
    }

    /// fused._iq_rope: the index query's first RD dims rotated in place.
    pub fn iqRope(t: Tri, q: u64, inv: u64, pos: u64, rows: usize, nh: usize, d: usize, rd: usize) !void {
        try t.run("_iq_rope", .{ rows, nh, 1 }, &.{ p("Q", "*bf16", q), p("INV", "*fp32", inv), p("POS", "*i32", pos) }, &.{ ci("NH", nh), ci("D", d), ci("RD", rd), cb("ROWS", false) }, .{ .num_warps = 1 });
    }

    /// fused._index_scores with PACK False (the radix select's 4-byte order words; later keys 0): OUT [n, T] int32.
    pub fn indexScores(t: Tri, q: u64, w: u64, ic: u64, pos: u64, out: u64, T: usize, r0: usize, n: usize, nh: usize, d: usize) !void {
        const ws = f32of(std.math.pow(f64, @floatFromInt(nh), -0.5));
        const qs = f32of(std.math.pow(f64, @floatFromInt(d), -0.5));
        try t.run("_index_scores", .{ n, cdiv(T, bt), 1 }, &.{ p("Q", "*bf16", q), p("W", "*fp32", w), p("IC", "*bf16", ic), p("POS", "*i32", pos), p("OUT", "*i32", out), int("T", T), int("R0", r0) }, &.{ ci("NH", nh), ci("D", d), ci("BTT", bt), aot.cf("WSCALE", ws), aot.cf("QSCALE", qs), ci("DCP", 1), ci("RANK", 0), cb("PACK", false), cb("ROWS", false) }, .{ .num_warps = 2, .num_stages = 2 });
    }

    /// fused._ik_write8 (TF_GLM53_FP8_INDEX, findings-prefill-fp8-index.md): the index key into the e4m3 plane -
    /// a 132-byte token row (128 codes + one fp32 scale; IC and ICS are the SAME plane pointer, codes *fp8e4nv,
    /// scale *fp32). The plane is carved inside the bf16 ic allocation (132 <= 256 bytes a token; the layouts
    /// never mix within a boot - the digest carries the env). DCP 1 only.
    /// `stride`: the plane's token row - 132 (the legacy layout, _ik_write8) or 144 (inc 3's padded
    /// 16-byte-multiple layout, _ik_write8p; a SEPARATE NAME so the captured pack's legacy variants stay
    /// valid - the recapture adds the padded ones alongside).
    pub fn ikWrite8(t: Tri, ik: u64, nw: u64, nb: u64, inv: u64, ic: u64, ics: u64, pos: u64, rows: usize, d: usize, rd: usize, stride: usize) !void {
        const name = if (stride == 144) "_ik_write8p" else "_ik_write8";
        try t.run(name, .{ rows, 1, 1 }, &.{ p("IK", "*fp32", ik), p("NW", "*bf16", nw), p("NB", "*bf16", nb), p("INV", "*fp32", inv), p("IC", "*fp8e4nv", ic), p("ICS", "*fp32", ics), p("POS", "*i32", pos), aot.float("eps", 1e-6) }, &.{ ci("D", d), ci("RD", rd), cb("ROWS", false) }, .{ .num_warps = 4 });
    }

    /// _ik_write8 ROWS: the stream's own cache rows BASE[r] + POS[r].
    pub fn ikWriteRows8(t: Tri, ik: u64, nw: u64, nb: u64, inv: u64, ic: u64, ics: u64, pos: u64, base: u64, rows: usize, d: usize, rd: usize, stride: usize) !void {
        const name = if (stride == 144) "_ik_write8p" else "_ik_write8";
        try t.run(name, .{ rows, 1, 1 }, &.{ p("IK", "*fp32", ik), p("NW", "*bf16", nw), p("NB", "*bf16", nb), p("INV", "*fp32", inv), p("IC", "*fp8e4nv", ic), p("ICS", "*fp32", ics), p("POS", "*i32", pos), aot.float("eps", 1e-6), p("BASE", "*i32", base) }, &.{ ci("D", d), ci("RD", rd), cb("ROWS", true) }, .{ .num_warps = 4 });
    }

    /// fused._index_scores8 (TF_GLM53_FP8_INDEX): the scores (PACK=False order words, DCP 1) from the e4m3 plane -
    /// IC / ICS the same plane pointer as _ik_write8's; the keys dequantize in-register (the kv8 v2 convention),
    /// the dot stays bf16 (increment 1: the 2x K-read-byte win; the fp8 dot is findings-prefill-fp8-index.md §5).
    pub fn indexScores8(t: Tri, q: u64, w: u64, ic: u64, ics: u64, pos: u64, out: u64, T: usize, r0: usize, n: usize, nh: usize, d: usize, stride: usize) !void {
        const ws = f32of(std.math.pow(f64, @floatFromInt(nh), -0.5));
        const qs = f32of(std.math.pow(f64, @floatFromInt(d), -0.5));
        const name = if (stride == 144) "_index_scores8p" else "_index_scores8";
        try t.run(name, .{ n, cdiv(T, bt), 1 }, &.{ p("Q", "*bf16", q), p("W", "*fp32", w), p("IC", "*fp8e4nv", ic), p("ICS", "*fp32", ics), p("POS", "*i32", pos), p("OUT", "*i32", out), int("T", T), int("R0", r0) }, &.{ ci("NH", nh), ci("D", d), ci("BTT", bt), aot.cf("WSCALE", ws), aot.cf("QSCALE", qs), cb("ROWS", false) }, .{ .num_warps = 2, .num_stages = 2 });
    }

    /// _index_scores8 ROWS: the stream's own rows, its cache at BASE.
    pub fn indexScoresRows8(t: Tri, q: u64, w: u64, ic: u64, ics: u64, pos: u64, base: u64, out: u64, T: usize, r0: usize, n: usize, nh: usize, d: usize, stride: usize) !void {
        const ws = f32of(std.math.pow(f64, @floatFromInt(nh), -0.5));
        const qs = f32of(std.math.pow(f64, @floatFromInt(d), -0.5));
        const name = if (stride == 144) "_index_scores8p" else "_index_scores8";
        try t.run(name, .{ n, cdiv(T, bt), 1 }, &.{ p("Q", "*bf16", q), p("W", "*fp32", w), p("IC", "*fp8e4nv", ic), p("ICS", "*fp32", ics), p("POS", "*i32", pos), p("OUT", "*i32", out), int("T", T), int("R0", r0), p("BASE", "*i32", base) }, &.{ ci("NH", nh), ci("D", d), ci("BTT", bt), aot.cf("WSCALE", ws), aot.cf("QSCALE", qs), cb("ROWS", true) }, .{ .num_warps = 2, .num_stages = 2 });
    }

    /// fused._index_scores, DCP > 1 (Phase 3b; PACK True: fused.select packs whenever DCP > 1): row r's score of this
    /// rank's slots t < Tl (global position t * DCP + RANK) packed with the position into one int64 that orders by
    /// score, then lower position; later positions int64 min + 1. OUT [n, Tl] int64.
    pub fn indexScoresDcp(t: Tri, q: u64, w: u64, ic: u64, pos: u64, out: u64, Tl: usize, r0: usize, n: usize, nh: usize, d: usize, dcp: usize, rank: usize) !void {
        const ws = f32of(std.math.pow(f64, @floatFromInt(nh), -0.5));
        const qs = f32of(std.math.pow(f64, @floatFromInt(d), -0.5));
        try t.run("_index_scores", .{ n, cdiv(Tl, bt), 1 }, &.{ p("Q", "*bf16", q), p("W", "*fp32", w), p("IC", "*bf16", ic), p("POS", "*i32", pos), p("OUT", "*i64", out), int("T", Tl), int("R0", r0) }, &.{ ci("NH", nh), ci("D", d), ci("BTT", bt), aot.cf("WSCALE", ws), aot.cf("QSCALE", qs), ci("DCP", dcp), ci("RANK", rank), cb("PACK", true), cb("ROWS", false) }, .{ .num_warps = 2, .num_stages = 2 });
    }

    /// topk.top_keys: row r's k best packed keys of S [rows, nc] int64 (high word order, ties to the lower column),
    /// int64 in column order -> OUT [rows, k] (the radix branch of fused.select under DCP).
    pub fn topKeys(t: Tri, sc: u64, out: u64, nc: usize, rows: usize, k: usize) !void {
        try t.run("_select_keys", .{ rows, 1, 1 }, &.{ p("S", "*i64", sc), p("OUT", "*i64", out), int("NC", nc) }, &.{ ci("K", k), ci("BLOCK", 1024), cb("KEYS", true) }, .{ .num_warps = 4 });
    }

    /// fused._attn_dcp (bf16 cache; Phase 3b): program (row, chunk, head group g) - rank g's gathered queries (QALL
    /// [G, R, H, LW + RD]) over this rank's keys of row r (its slots of positions 0..p while p < K, else its share
    /// of the selection, TOK / CNT); partials at head g * H + h of [nch, R, G * H]. `direct` (nch == 1, the prompt
    /// chunks at the 2048-key tiling): the single-chunk merge fused into the scorer - `po`/`pm` are OSEND/LSEND
    /// (the normalized bf16 partial + its lse, in the exchange layout) and the caller skips _merge_lse entirely
    /// (findings-dcp8.md section 17: the fp32 partial plane + its extra pass disappear).
    pub fn attnDcp(t: Tri, qall: u64, cache: u64, tok: u64, cnt: u64, pos: u64, po: u64, pm: u64, pl: u64, rows: usize, m: Mla, g: usize, rank: usize, tl: AttnTiling, direct: bool, qfp8: bool, merged: bool) !void {
        const nch = @max(1, m.topk / tl.chk);
        const scale = f32of(std.math.pow(f64, @floatFromInt(m.nope + m.rope), -0.5));
        // QFP8 (TF_GLM53_E1_FP8, section 18): `qall` is the gathered e4m3 plane (584-byte rows: codes + the
        // s0/s1 scales), QSC its fp32 word view (the same storage) - the scorer dequantizes in-register.
        // QFP8: KTT 32 + ns 1 - the u8 staging + the bf16 key tile pushed the Q variant past sm121's 99 KB
        // smem optin at KTT 64 (113 KB at ns 2, 105 KB at ns 1); KTT 32 halves the key tile (the KV8
        // in-kernel-dequant precedent pins ns 1 too).
        try t.run("_attn_dcp", .{ rows, nch, g }, &.{ p("QALL", if (qfp8) "*fp8e4nv" else "*bf16", qall), p("LC", "*bf16", cache), p("TOK", "*i32", tok), p("CNT", "*i32", cnt), p("POS", "*i32", pos), p("PO", if (direct) "*bf16" else "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), p("QSC", "*fp32", qall), int("R", rows) }, &.{ ci("H", m.heads), ci("G", g), ci("LW", m.lw), ci("RD", m.rope), ci("K", m.topk), ci("CHK", tl.chk), ci("KTT", if (qfp8) 32 else tl.kt), aot.cf("SCALE", scale), ci("DCP", g), ci("RANK", rank), cb("DIRECT", direct), cb("QFP8", qfp8), cb("MERGED", merged) }, .{ .num_warps = tl.nw, .num_stages = if (qfp8 and tl.ns > 1) 1 else tl.ns });
    }

    /// fused._merge_lse: program (row, head of all HT = G * H): this rank's chunk partials in chunk order -> the
    /// normalized output (bf16) and its log-sum-exp at [destination rank = head // H, row, head % H].
    pub fn mergeLse(t: Tri, po: u64, pm: u64, pl: u64, osend: u64, lsend: u64, rows: usize, g: usize, m: Mla, nch: usize) !void {
        try t.run("_merge_lse", .{ rows, g * m.heads, 1 }, &.{ p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), p("OSEND", "*bf16", osend), p("LSEND", "*fp32", lsend), int("R", rows) }, &.{ ci("HT", g * m.heads), ci("H", m.heads), ci("LW", m.lw), ci("NCH", nch) }, .{ .num_warps = 4 });
    }

    /// fused._dcp_combine: program (row, own head) - every rank's normalized partial merged in rank order by their
    /// log-sum-exps (ORECV[src] at stride ss elements, LRECV[src] at stride ssl) -> OUT [R, H, LW] bf16.
    pub fn dcpCombine(t: Tri, orecv: u64, lrecv: u64, out: u64, rows: usize, ss: usize, ssl: usize, m: Mla, world: usize) !void {
        try t.run("_dcp_combine", .{ rows, m.heads, 1 }, &.{ p("ORECV", "*bf16", orecv), p("LRECV", "*fp32", lrecv), p("OUT", "*bf16", out), int("R", rows), int("SS", ss), int("SSL", ssl) }, &.{ ci("H", m.heads), ci("LW", m.lw), ci("WORLD", world) }, .{ .num_warps = 4 });
    }

    /// topk.top_columns over int32 order words: row r's K best columns, ascending, ties to the lower column.
    pub fn selectKeys(t: Tri, sc: u64, out: u64, nc: usize, rows: usize, k: usize) !void {
        try t.run("_select_keys", .{ rows, 1, 1 }, &.{ p("S", "*i32", sc), p("OUT", "*i32", out), int("NC", nc) }, &.{ ci("K", k), ci("BLOCK", 1024), cb("KEYS", false), cb("U32", true) }, .{ .num_warps = 4 });
    }

    /// topk._topk_row (TF_GLM53_TOPK_STREAM, the prefill blocks' selection): row r's K best columns of S [rows, NC]
    /// int32 order words, ascending, ties to the lower column - bit-identical to selectKeys, in 2-3 streaming passes
    /// over the row instead of the radix's 5 (findings-prefill-topk.md). SCRATCH: [rows, 2 CAP] int32 (the score
    /// words at [r 2 CAP], the reversed columns at [r 2 CAP + CAP]), transient per call - the caller carves it from
    /// the score buffer's spare half (DCP 1 sized the buffer for packed keys). CAP 1024 + the split i32 pairs: the
    /// §10 register rework (the first capture's [2048] int64 buffer spilled and lost 12% to the radix).
    pub fn topkRow(t: Tri, sc: u64, out: u64, scratch: u64, nc: usize, rows: usize, k: usize) !void {
        try t.run("_topk_row", .{ rows, 1, 1 }, &.{ p("S", "*i32", sc), p("OUT", "*i32", out), p("SCRATCH", "*i32", scratch), int("NC", nc) }, &.{ ci("K", k), ci("CAP", 1024), ci("BLOCK", 1024) }, .{ .num_warps = 4 });
    }

    /// fused._absorb: QA[r, h] = q_nope[r, h] . WK[h] and the rotated q_rot (row-block grid).
    pub fn absorb(t: Tri, q: u64, wk: u64, inv: u64, qa: u64, qr: u64, pos: u64, rows: usize, m: Mla) !void {
        const rbk = if (rows > rb) rb else rows;
        try t.run("_absorb", .{ m.heads, m.lw / 32, cdiv(rows, rbk) }, &.{ p("Q", "*bf16", q), p("WK", "*bf16", wk), p("INV", "*fp32", inv), p("QA", "*bf16", qa), p("QR", "*bf16", qr), p("POS", "*i32", pos), int("R", rows) }, &.{ ci("H", m.heads), ci("QD", m.nope + m.rope), ci("NOPE", m.nope), ci("NOPE_P", pow2(m.nope)), ci("RD", m.rope), ci("LW", m.lw), ci("BN", 32), ci("RBK", rbk), cb("ROWS", false) }, .{ .num_warps = 4 });
    }

    /// fused._attention_local (bf16 cache, one stream): one pass straight into `ol` when one chunk covers index_topk
    /// keys, else chunk partials and latent._merge.
    pub fn attention(t: Tri, qa: u64, qr: u64, cache: u64, tok: u64, pos: u64, po: u64, pm: u64, pl: u64, ol: u64, dummy: u64, rows: usize, m: Mla, tl: AttnTiling) !void {
        const nch = @max(1, m.topk / tl.chk);
        const scale = f32of(std.math.pow(f64, @floatFromInt(m.nope + m.rope), -0.5));
        const opts: aot.Launch = .{ .num_warps = tl.nw, .num_stages = tl.ns };
        const common = [_]aot.Const{ ci("H", m.heads), ci("LW", m.lw), ci("RD", m.rope), ci("K", m.topk), ci("CHK", tl.chk), ci("KTT", tl.kt), aot.cf("SCALE", scale), cb("ROWS", false) };
        if (nch == 1) {
            var direct: [common.len + 1]aot.Const = undefined;
            @memcpy(direct[0..common.len], &common);
            direct[common.len] = cb("DIRECT", true);
            try t.run("_attn_chunks", .{ rows, 1, 1 }, &.{ p("QA", "*bf16", qa), p("QR", "*bf16", qr), p("LC", "*bf16", cache), p("TOK", "*i32", tok), p("POS", "*i32", pos), p("PO", "*bf16", ol), p("PM", "*fp32", pm), p("PL", "*fp32", pl), int("R", rows) }, &direct, opts);
            return;
        }
        try t.run("_attn_chunks", .{ rows, nch, 1 }, &.{ p("QA", "*bf16", qa), p("QR", "*bf16", qr), p("LC", "*bf16", cache), p("TOK", "*i32", tok), p("POS", "*i32", pos), p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), int("R", rows) }, &common, opts);
        try t.run("_merge", .{ rows, m.heads, 1 }, &.{ p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), p("OUT", "*bf16", ol), p("CNT", "*i32", dummy), int("R", rows) }, &.{ ci("H", m.heads), ci("LW", m.lw), ci("NCH", nch), cb("SPARSE", false) }, .{ .num_warps = 4 });
    }

    /// fused._attention_local with KV8 (TF_GLM53_KV8_V2, findings-kv8-v2.md): the wide window reads the e4m3 latent
    /// plane itself - LC the codes (a byte a value, a row LW + RD + 8), SC the same storage viewed fp32 (the row's
    /// two scales) - and dequantizes in-kernel, exactly what the staged mirror held. No dequantPrompt pass, no
    /// bf16 mirror: the kv8 prompt path's O(pos) per-chunk term is gone at every depth. The prompt path only
    /// (non-ROWS, one stream; the decode path stays staged); `_merge` is dtype-blind (fp32 partials) and shared
    /// with the bf16 chunked path. The tiling is FIXED (attn_prompt_kv8, not the caller's): the KTT-64 prompt
    /// tiling's variant needs 144.9 KB of shared - past what the device launches (§9) - and a tiling the pack has
    /// no variant for is a dead boot. New AOT variants (KV8 = true, LC *fp8e4nv + SC *fp32): the recapture plan is
    /// findings-kv8-v2.md §5.
    pub fn attentionKv8(t: Tri, qa: u64, qr: u64, plane: u64, tok: u64, pos: u64, po: u64, pm: u64, pl: u64, ol: u64, dummy: u64, rows: usize, m: Mla) !void {
        const tl = attn_prompt_kv8;
        const nch = @max(1, m.topk / tl.chk);
        const scale = f32of(std.math.pow(f64, @floatFromInt(m.nope + m.rope), -0.5));
        const opts: aot.Launch = .{ .num_warps = tl.nw, .num_stages = tl.ns };
        const common = [_]aot.Const{ ci("H", m.heads), ci("LW", m.lw), ci("RD", m.rope), ci("K", m.topk), ci("CHK", tl.chk), ci("KTT", tl.kt), aot.cf("SCALE", scale), cb("ROWS", false), cb("KV8", true) };
        if (nch == 1) {
            var direct: [common.len + 1]aot.Const = undefined;
            @memcpy(direct[0..common.len], &common);
            direct[common.len] = cb("DIRECT", true);
            try t.run("_attn_chunks", .{ rows, 1, 1 }, &.{ p("QA", "*bf16", qa), p("QR", "*bf16", qr), p("LC", "*fp8e4nv", plane), p("SC", "*fp32", plane), p("TOK", "*i32", tok), p("POS", "*i32", pos), p("PO", "*bf16", ol), p("PM", "*fp32", pm), p("PL", "*fp32", pl), int("R", rows) }, &direct, opts);
            return;
        }
        try t.run("_attn_chunks", .{ rows, nch, 1 }, &.{ p("QA", "*bf16", qa), p("QR", "*bf16", qr), p("LC", "*fp8e4nv", plane), p("SC", "*fp32", plane), p("TOK", "*i32", tok), p("POS", "*i32", pos), p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), int("R", rows) }, &common, opts);
        try t.run("_merge", .{ rows, m.heads, 1 }, &.{ p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), p("OUT", "*bf16", ol), p("CNT", "*i32", dummy), int("R", rows) }, &.{ ci("H", m.heads), ci("LW", m.lw), ci("NCH", nch), cb("SPARSE", false) }, .{ .num_warps = 4 });
    }

    /// fused._expand: OUT[r, h, n] = sum_k OL[r, h, k] WV[h, n, k].
    pub fn expand(t: Tri, ol: u64, wv: u64, out: u64, rows: usize, m: Mla) !void {
        const rbk = if (rows > rb) rb else rows;
        try t.run("_expand", .{ m.heads, m.vd / 16, cdiv(rows, rbk) }, &.{ p("OL", "*bf16", ol), p("WV", "*bf16", wv), p("OUT", "*bf16", out), int("R", rows) }, &.{ ci("H", m.heads), ci("DV", m.vd), ci("LW", m.lw), ci("BN", 16), ci("RBK", rbk) }, .{ .num_warps = 4 });
    }

    // ---- Phase 4b: several streams in one decode window (fused.Rows: ROWS=True, BASE an int32 table) ----------
    // Row r sits at position POS[r] (an int32 table, not the window's first position) of the stream whose cache rows
    // start at row BASE[r]; every row computes what it computes alone (multi.GlmMultiDecoder's windows; *RowsDcp: the
    // same under decode context parallelism, docs/glm53-dcp-streams.md).

    /// _kv_write ROWS: row r's latent and rotated key into cache row BASE[r] + POS[r].
    pub fn kvWriteRows(t: Tri, kva: u64, kva_stride: usize, nw: u64, cache: u64, pos: u64, base: u64, inv: u64, eps: f32, rows: usize, m: Mla) !void {
        return t.kvWriteRowsDcp(kva, kva_stride, nw, cache, pos, base, inv, eps, rows, m, 1, 0);
    }

    /// _kv_write ROWS with DCP / RANK (several streams, the caches sharded): row r is stored only by the owner of its
    /// position (POS[r] % DCP == RANK), at cache row BASE[r] + POS[r] // DCP (BASE: the stream's first local row).
    pub fn kvWriteRowsDcp(t: Tri, kva: u64, kva_stride: usize, nw: u64, cache: u64, pos: u64, base: u64, inv: u64, eps: f32, rows: usize, m: Mla, dcp: usize, rank: usize) !void {
        try t.run("_kv_write", .{ rows, 1, 1 }, &.{ p("KVA", "*bf16", kva), int("kva_stride", kva_stride), p("NW", "*bf16", nw), p("LC", "*bf16", cache), p("POS", "*i32", pos), p("INV", "*fp32", inv), aot.float("eps", eps), p("BASE", "*i32", base) }, &.{ ci("LW", m.lw), ci("RD", m.rope), ci("DCP", dcp), ci("RANK", rank), cb("ROWS", true) }, .{ .num_warps = 4 });
    }

    /// _ik_write ROWS: row r's index key into IC row BASE[r] + POS[r].
    pub fn ikWriteRows(t: Tri, ik: u64, nw: u64, nb: u64, inv: u64, ic: u64, pos: u64, base: u64, rows: usize, d: usize, rd: usize) !void {
        return t.ikWriteRowsDcp(ik, nw, nb, inv, ic, pos, base, rows, d, rd, 1, 0);
    }

    /// _ik_write ROWS with DCP / RANK: the owner's row BASE[r] + POS[r] // DCP only.
    pub fn ikWriteRowsDcp(t: Tri, ik: u64, nw: u64, nb: u64, inv: u64, ic: u64, pos: u64, base: u64, rows: usize, d: usize, rd: usize, dcp: usize, rank: usize) !void {
        try t.run("_ik_write", .{ rows, 1, 1 }, &.{ p("IK", "*fp32", ik), p("NW", "*bf16", nw), p("NB", "*bf16", nb), p("INV", "*fp32", inv), p("IC", "*bf16", ic), p("POS", "*i32", pos), aot.float("eps", 1e-6), p("BASE", "*i32", base) }, &.{ ci("D", d), ci("RD", rd), ci("DCP", dcp), ci("RANK", rank), cb("ROWS", true) }, .{ .num_warps = 4 });
    }

    /// _iq_rope ROWS: the index query of row r rotated at POS[r].
    pub fn iqRopeRows(t: Tri, q: u64, inv: u64, pos: u64, rows: usize, nh: usize, d: usize, rd: usize) !void {
        try t.run("_iq_rope", .{ rows, nh, 1 }, &.{ p("Q", "*bf16", q), p("INV", "*fp32", inv), p("POS", "*i32", pos) }, &.{ ci("NH", nh), ci("D", d), ci("RD", rd), cb("ROWS", true) }, .{ .num_warps = 1 });
    }

    /// _index_scores ROWS (PACK False): window row R0 + r scores the keys t <= POS[R0 + r] of its stream's index-key
    /// rows from BASE[R0 + r]; OUT [n, T] int32 order words.
    pub fn indexScoresRows(t: Tri, q: u64, w: u64, ic: u64, pos: u64, base: u64, out: u64, T: usize, r0: usize, n: usize, nh: usize, d: usize) !void {
        const ws = f32of(std.math.pow(f64, @floatFromInt(nh), -0.5));
        const qs = f32of(std.math.pow(f64, @floatFromInt(d), -0.5));
        try t.run("_index_scores", .{ n, cdiv(T, bt), 1 }, &.{ p("Q", "*bf16", q), p("W", "*fp32", w), p("IC", "*bf16", ic), p("POS", "*i32", pos), p("OUT", "*i32", out), int("T", T), int("R0", r0), p("BASE", "*i32", base) }, &.{ ci("NH", nh), ci("D", d), ci("BTT", bt), aot.cf("WSCALE", ws), aot.cf("QSCALE", qs), ci("DCP", 1), ci("RANK", 0), cb("PACK", false), cb("ROWS", true) }, .{ .num_warps = 2, .num_stages = 2 });
    }

    /// _absorb ROWS: q_rot of row r at its own position POS[r].
    pub fn absorbRows(t: Tri, q: u64, wk: u64, inv: u64, qa: u64, qr: u64, pos: u64, rows: usize, m: Mla) !void {
        const rbk = if (rows > rb) rb else rows;
        try t.run("_absorb", .{ m.heads, m.lw / 32, cdiv(rows, rbk) }, &.{ p("Q", "*bf16", q), p("WK", "*bf16", wk), p("INV", "*fp32", inv), p("QA", "*bf16", qa), p("QR", "*bf16", qr), p("POS", "*i32", pos), int("R", rows) }, &.{ ci("H", m.heads), ci("QD", m.nope + m.rope), ci("NOPE", m.nope), ci("NOPE_P", pow2(m.nope)), ci("RD", m.rope), ci("LW", m.lw), ci("BN", 32), ci("RBK", rbk), cb("ROWS", true) }, .{ .num_warps = 4 });
    }

    /// _index_scores ROWS with DCP > 1 (PACK True): window row R0 + r scores this rank's slots t < Tl of its stream's
    /// index-key rows from BASE[R0 + r] (global position t * DCP + RANK <= POS[R0 + r]); OUT [n, Tl] int64 packed keys.
    pub fn indexScoresRowsDcp(t: Tri, q: u64, w: u64, ic: u64, pos: u64, base: u64, out: u64, Tl: usize, r0: usize, n: usize, nh: usize, d: usize, dcp: usize, rank: usize) !void {
        const ws = f32of(std.math.pow(f64, @floatFromInt(nh), -0.5));
        const qs = f32of(std.math.pow(f64, @floatFromInt(d), -0.5));
        try t.run("_index_scores", .{ n, cdiv(Tl, bt), 1 }, &.{ p("Q", "*bf16", q), p("W", "*fp32", w), p("IC", "*bf16", ic), p("POS", "*i32", pos), p("OUT", "*i64", out), int("T", Tl), int("R0", r0), p("BASE", "*i32", base) }, &.{ ci("NH", nh), ci("D", d), ci("BTT", bt), aot.cf("WSCALE", ws), aot.cf("QSCALE", qs), ci("DCP", dcp), ci("RANK", rank), cb("PACK", true), cb("ROWS", true) }, .{ .num_warps = 2, .num_stages = 2 });
    }

    /// fused._attn_dcp_rows (tools/glm53/patch_fused_dcp_rows.py: _attn_dcp with a position and a cache base a row):
    /// program (row, chunk, head group g) - rank g's gathered queries over this rank's keys of row r's stream (its
    /// slots of positions 0..POS[r] while POS[r] < K, else its share of the row's selection, TOK / CNT), read from
    /// the stream's cache rows at BASE[r]. A function of its own so no pack made before it can answer its launches.
    pub fn attnDcpRows(t: Tri, qall: u64, cache: u64, tok: u64, cnt: u64, pos: u64, base: u64, po: u64, pm: u64, pl: u64, rows: usize, m: Mla, g: usize, rank: usize, tl: AttnTiling) !void {
        const nch = @max(1, m.topk / tl.chk);
        const scale = f32of(std.math.pow(f64, @floatFromInt(m.nope + m.rope), -0.5));
        try t.run("_attn_dcp_rows", .{ rows, nch, g }, &.{ p("QALL", "*bf16", qall), p("LC", "*bf16", cache), p("TOK", "*i32", tok), p("CNT", "*i32", cnt), p("POS", "*i32", pos), p("BASE", "*i32", base), p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), int("R", rows) }, &.{ ci("H", m.heads), ci("G", g), ci("LW", m.lw), ci("RD", m.rope), ci("K", m.topk), ci("CHK", tl.chk), ci("KTT", tl.kt), aot.cf("SCALE", scale), ci("DCP", g), ci("RANK", rank) }, .{ .num_warps = tl.nw, .num_stages = tl.ns });
    }

    /// fused._attention_local ROWS: row r at POS[r] over its stream's cache rows from BASE[r] (its key list: keys
    /// 0..p while p < K, else TOK[r]); one pass when a chunk covers index_topk keys, else partials + latent._merge.
    pub fn attentionRows(t: Tri, qa: u64, qr: u64, cache: u64, tok: u64, pos: u64, base: u64, po: u64, pm: u64, pl: u64, ol: u64, dummy: u64, rows: usize, m: Mla, tl: AttnTiling) !void {
        const nch = @max(1, m.topk / tl.chk);
        const scale = f32of(std.math.pow(f64, @floatFromInt(m.nope + m.rope), -0.5));
        const opts: aot.Launch = .{ .num_warps = tl.nw, .num_stages = tl.ns };
        const common = [_]aot.Const{ ci("H", m.heads), ci("LW", m.lw), ci("RD", m.rope), ci("K", m.topk), ci("CHK", tl.chk), ci("KTT", tl.kt), aot.cf("SCALE", scale), cb("ROWS", true) };
        if (nch == 1) {
            var direct: [common.len + 1]aot.Const = undefined;
            @memcpy(direct[0..common.len], &common);
            direct[common.len] = cb("DIRECT", true);
            try t.run("_attn_chunks", .{ rows, 1, 1 }, &.{ p("QA", "*bf16", qa), p("QR", "*bf16", qr), p("LC", "*bf16", cache), p("TOK", "*i32", tok), p("POS", "*i32", pos), p("PO", "*bf16", ol), p("PM", "*fp32", pm), p("PL", "*fp32", pl), int("R", rows), p("BASE", "*i32", base) }, &direct, opts);
            return;
        }
        try t.run("_attn_chunks", .{ rows, nch, 1 }, &.{ p("QA", "*bf16", qa), p("QR", "*bf16", qr), p("LC", "*bf16", cache), p("TOK", "*i32", tok), p("POS", "*i32", pos), p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), int("R", rows), p("BASE", "*i32", base) }, &common, opts);
        try t.run("_merge", .{ rows, m.heads, 1 }, &.{ p("PO", "*fp32", po), p("PM", "*fp32", pm), p("PL", "*fp32", pl), p("OUT", "*bf16", ol), p("CNT", "*i32", dummy), int("R", rows) }, &.{ ci("H", m.heads), ci("LW", m.lw), ci("NCH", nch), cb("SPARSE", false) }, .{ .num_warps = 4 });
    }

    /// fused._cat2: OUT[r] = [A[r] ; B[r]] (bf16 rows of D each), the MTP layer's [enorm(embedding) ; hnorm(hidden)].
    pub fn cat2(t: Tri, a: u64, b: u64, out: u64, rows: usize, d: usize) !void {
        try t.run("_cat2", .{ rows, d / 1024, 1 }, &.{ p("A", "*bf16", a), p("B", "*bf16", b), p("OUT", "*bf16", out) }, &.{ ci("D", d), ci("BLOCK", 1024) }, .{ .num_warps = 4 });
    }

    /// glm5_next qmm.group_sums: XS[m, g] = fp32 sum of x[m, 64 g .. 64 g + 63] (the draft head's input sums).
    pub fn groupSums(t: Tri, x: u64, x_stride: usize, xs: u64, rows: usize, k: usize) !void {
        try t.run("_group_sums", .{ rows, cdiv(k / 64, 16), 1 }, &.{ p("X", "*bf16", x), p("XS", "*fp32", xs), int("x_stride", x_stride) }, &.{ ci("K", k), ci("GB", 16) }, .{ .num_warps = 2 });
    }

    // ---- Phase 4: the drafters (glm5_next glue / dflash2, cuda/dspark.py, cuda/dflash.py) ----------------------

    /// glue.rmsnorm with group sums (the drafters' _norm): out = bf16(w * bf16(x * rinv)) and XS[r, g] = fp32 sums of
    /// out's 64-groups (the next 4-bit matmul's input sums).
    pub fn rmsnormSums(t: Tri, x: u64, x_stride: usize, w: u64, out: u64, o_stride: usize, xs: u64, rows: usize, d: usize, eps: f32) !void {
        const block = pow2(d);
        try t.run("_rmsnorm", .{ rows, 1, 1 }, &.{ p("X", "*bf16", x), int("x_stride", x_stride), p("W", "*bf16", w), p("OUT", "*bf16", out), int("o_stride", o_stride), p("XS", "*fp32", xs), aot.float("eps", eps) }, &.{ ci("D", d), ci("BLOCK", block), cb("SUMS", true) }, .{ .num_warps = if (block <= 2048) 4 else 8 });
    }

    /// glue.embed over a BF16 table (_embed_b16): rows of the verifier's embedding for int32 ids, one copy.
    pub fn embedB16(t: Tri, ids: u64, table: u64, out: u64, rows: usize, d: usize) !void {
        try t.run("_embed_b16", .{ rows, d / 64, 1 }, &.{ p("IDS", "*i32", ids), p("W", "*bf16", table), p("OUT", "*bf16", out) }, &.{ ci("D", d), ci("COPIES", 1) }, .{ .num_warps = 1 });
    }

    /// glue.swiglu: [gate | up] (bf16) -> bf16(bf16(silu(min(g, L))) * clip(u, -L, L)) and its 64-group sums.
    pub fn swigluSums(t: Tri, gu: u64, out: u64, xs: u64, limit: f32, rows: usize, width: usize) !void {
        const blk = std.math.gcd(@as(usize, 512), width);
        if (blk < 64) return error.Invalid;
        try t.run("_swiglu", .{ rows, width / blk, 1 }, &.{ p("GU", "*bf16", gu), p("OUT", "*bf16", out), p("XS", "*fp32", xs), aot.float("LIMIT", limit) }, &.{ ci("W", width), ci("BLOCK", blk) }, .{ .num_warps = 4 });
    }

    /// dflash2._prep_kernel: q / k normed (q_norm / k_norm) and rotated, v copied, out of [q heads | k heads | v heads]
    /// rows (row stride `stride`) into [heads, rows, head_dim]; heads 0 (context rows): QO is the input itself.
    pub fn prep(t: Tri, qkv: u64, qn: u64, kn: u64, cos: u64, sin: u64, qo: u64, ko: u64, vo: u64, rows: usize, stride: usize, eps: f32, heads: usize, kvh: usize, half: usize) !void {
        try t.run("_prep_kernel", .{ rows, heads + 2 * kvh, 1 }, &.{ p("QKV", "*bf16", qkv), p("QN", "*bf16", qn), p("KN", "*bf16", kn), p("COS", "*fp32", cos), p("SIN", "*fp32", sin), p("QO", "*bf16", qo), p("KO", "*bf16", ko), p("VO", "*bf16", vo), int("L", rows), int("stride", stride), aot.float("eps", eps) }, &.{ ci("H", heads), ci("HKV", kvh), ci("HALF", half) }, .{ .num_warps = 1 });
    }

    /// dflash2._dconv_kernel: the two-tap grouped dynamic convolution over a block (row r mixes rows r and r - 1),
    /// plus the residual when `has_res`.
    pub fn dconv(t: Tri, x: u64, dyn: u64, base: u64, res: u64, out: u64, rows: usize, d: usize, gs: usize, branch: usize, has_res: bool) !void {
        try t.run("_dconv_kernel", .{ rows, cdiv(d, 1024), 1 }, &.{ p("X", "*bf16", x), p("DYN", "*bf16", dyn), p("BASE", "*bf16", base), p("RES", "*bf16", res), p("OUT", "*bf16", out) }, &.{ ci("D", d), ci("G", d / gs), ci("GS", gs), ci("BRANCH", branch), cb("HAS_RES", has_res), ci("BLOCK", 1024) }, .{ .num_warps = 4 });
    }

    /// dspark._block_attn_kernel: one program a head - the block's N queries over the context rows [s - W, s) and the
    /// block's keys (causal: up to their own slot), keys at ring slot position % RING_.
    pub fn blockAttn(t: Tri, q: u64, k: u64, v: u64, out: u64, pos: u64, w: usize, scale: f32, n: usize, heads: usize, hd: usize, ring_rows: usize, causal: bool) !void {
        try t.run("_block_attn_kernel", .{ heads, 1, 1 }, &.{ p("Q", "*bf16", q), p("K", "*bf16", k), p("V", "*bf16", v), p("OUT", "*bf16", out), p("POS", "*i64", pos), int("W", w), aot.float("scale", scale) }, &.{ ci("N", n), ci("MP", @max(16, pow2(n))), ci("NH", heads), ci("HD", hd), ci("RING_", ring_rows), ci("BK", 64), cb("CAUSAL", causal) }, .{ .num_warps = 4 });
    }

    /// dflash._dattn_ring: one program a KV head - its G query heads' N rows over the sliding window of committed
    /// context and the block, keys at ring slot position % RING_.
    pub fn dattnRing(t: Tri, q: u64, k: u64, v: u64, out: u64, pos: u64, window: usize, scale: f32, n: usize, g: usize, heads: usize, hd: usize, ring_rows: usize, kvh: usize, causal: bool) !void {
        try t.run("_dattn_ring", .{ kvh, 1, 1 }, &.{ p("Q", "*bf16", q), p("K", "*bf16", k), p("V", "*bf16", v), p("OUT", "*bf16", out), p("POS", "*i64", pos), int("window", window), aot.float("scale", scale) }, &.{ ci("N", n), ci("G", g), ci("NH", heads), ci("HD", hd), ci("RING_", ring_rows), ci("BK", 64), cb("CAUSAL", causal) }, .{ .num_warps = 4 });
    }

    /// Phase 4c: dflash._dconv_seg - _dconv_kernel over blocks of SEG rows side by side (a block's first row mixes no
    /// row before it: one block computes the solo kernel's bits), plus the residual when `has_res`.
    pub fn dconvSeg(t: Tri, x: u64, dyn: u64, base: u64, res: u64, out: u64, rows: usize, d: usize, gs: usize, branch: usize, has_res: bool, seg: usize) !void {
        try t.run("_dconv_seg", .{ rows, cdiv(d, 1024), 1 }, &.{ p("X", "*bf16", x), p("DYN", "*bf16", dyn), p("BASE", "*bf16", base), p("RES", "*bf16", res), p("OUT", "*bf16", out) }, &.{ ci("D", d), ci("G", d / gs), ci("GS", gs), ci("BRANCH", branch), cb("HAS_RES", has_res), ci("BLOCK", 1024), ci("SEG", seg) }, .{ .num_warps = 4 });
    }

    /// Phase 4c: dflash._dattn_seg - _dattn_ring for `segs` blocks of N rows side by side (program 1: the block): its
    /// queries at rows seg * N of q [heads, L, hd], its ring at slot META[St + seg] (rows slot * RING_ of the pooled
    /// k / v [kv heads, POOL, hd]), its committed length META[2 St + seg] (int64).
    pub fn dattnSeg(t: Tri, q: u64, k: u64, v: u64, out: u64, meta: u64, st: usize, window: usize, scale: f32, L: usize, pool: usize, n: usize, g: usize, heads: usize, hd: usize, ring_rows: usize, kvh: usize, segs: usize, causal: bool) !void {
        try t.run("_dattn_seg", .{ kvh, segs, 1 }, &.{ p("Q", "*bf16", q), p("K", "*bf16", k), p("V", "*bf16", v), p("OUT", "*bf16", out), p("META", "*i64", meta), int("St", st), int("window", window), aot.float("scale", scale), int("L", L), int("POOL", pool) }, &.{ ci("N", n), ci("G", g), ci("NH", heads), ci("HD", hd), ci("RING_", ring_rows), ci("BK", 64), cb("CAUSAL", causal) }, .{ .num_warps = 4 });
    }

    /// glue.residual_add with the gathered slots `rs` words apart (fused.small_gather's padded row stride).
    pub fn residualAddStride(t: Tri, x: u64, g: u64, rows: usize, d: usize, world: usize, rs: usize) !void {
        const block = @min(1024, d);
        try t.run("_residual_add", .{ rows, d / block, 1 }, &.{ p("X", "*bf16", x), p("XOUT", "*bf16", x), p("G", "*fp32", g), int("RS", rs) }, &.{ ci("D", d), ci("WORLD", world), ci("BLOCK", block) }, .{ .num_warps = 4 });
    }

    /// fused._swiglu2: bf16(bf16(silu(g)) * u) for separate gate and up rows of `width`.
    pub fn swiglu2(t: Tri, g: u64, up: u64, out: u64, rows: usize, width: usize) !void {
        const blk = std.math.gcd(@as(usize, 512), width);
        try t.run("_swiglu2", .{ rows, width / blk, 1 }, &.{ p("G", "*bf16", g), p("U", "*bf16", up), p("OUT", "*bf16", out) }, &.{ ci("W", width), ci("BLOCK", blk) }, .{ .num_warps = 4 });
    }
};

/// fused.bucket: the indexer's key range for a window whose last row sits at t - 1 (INDEX_SPLIT 1); null while
/// t <= topk (no selection).
pub fn bucket(t: usize, topk: usize) ?usize {
    if (t <= topk) return null;
    const pw = std.math.ceilPowerOfTwo(usize, t) catch unreachable; // 1 << (t - 1).bit_length()
    return @max(2 * topk, pw);
}

test "index buckets as fused.bucket (INDEX_SPLIT 1)" {
    try std.testing.expectEqual(@as(?usize, null), bucket(2048, 2048));
    try std.testing.expectEqual(@as(?usize, 4096), bucket(2049, 2048));
    try std.testing.expectEqual(@as(?usize, 32768), bucket(20001, 2048));
    try std.testing.expectEqual(@as(?usize, 32768), bucket(32768, 2048));
    try std.testing.expectEqual(@as(?usize, 65536), bucket(32769, 2048));
}

test "constexpr floats: the fp32 nearest Python's float" {
    try std.testing.expectEqual(@as(u32, 0x3d800000), @as(u32, @bitCast(f32of(std.math.pow(f64, 256.0, -0.5)))));
    try std.testing.expectEqual(@as(u32, 0x3e3504f3), @as(u32, @bitCast(f32of(std.math.pow(f64, 32.0, -0.5)))));
}

// ---------------------------------------------------------------- TF_GLM53_TOPK_STREAM (host test) ---
//
// topk._topk_row must pick exactly what the radix _select_keys picks - the K best by (word desc, column asc),
// columns out ascending - or the A/B sha gate breaks. The GPU kernels can't run on the host, so this mirrors the
// kernel's DECISION structure (the pass logic, the digit arithmetic, the composite threshold) as plain Zig and
// checks it against the radix's contract on adversarial score rows (findings-prefill-topk.md §3).

const topk_sim = struct {
    const K: usize = 2048; // the model's index_topk
    const CAP: usize = 1024; // topkRow's collect bound (§10: 1024 + the split i32 pairs)

    fn composite(w: u32, i: usize) i64 {
        return (@as(i64, @intCast(w)) << 32) | (4294967295 - @as(i64, @intCast(i)));
    }

    /// the test's LCG (a deterministic word stream; any u32 pattern is a valid order word)
    fn rnd(s: *u64) u32 {
        s.* = s.* *% 6364136223846793005 +% 1442695040888963407;
        return @truncate(s.* >> 33);
    }

    /// The radix's contract (topk.py's _select_keys, KEYS=False U32=True): the K best by word, ties to the lower
    /// column, output columns ascending - what torch.topk + torch.sort over the composites keeps.
    fn radixRef(gpa: std.mem.Allocator, words: []const u32, out: []u32) !void {
        const Idx = struct { w: u32, i: u32 };
        const idx = try gpa.alloc(Idx, words.len);
        defer gpa.free(idx);
        for (words, 0..) |w, i| idx[i] = .{ .w = w, .i = @intCast(i) };
        std.mem.sort(Idx, idx, {}, struct {
            fn lt(_: void, a: Idx, b: Idx) bool {
                return a.w > b.w or (a.w == b.w and a.i < b.i);
            }
        }.lt);
        for (out, 0..) |*o, j| o.* = idx[j].i; // the first K in (word, column) order ...
        std.mem.sort(u32, out, {}, std.sort.asc(u32)); // ... out in column order
    }

    /// topk._topk_row's structure (§10's rework): one top-byte histogram pass, then the whole-bucket write
    /// (g == need), the collect + in-register mini-radix over the SPLIT i32 pairs (g <= CAP) or the radix's own
    /// remaining passes (g > CAP). Mirrors the kernel statement for statement.
    fn streamSim(gpa: std.mem.Allocator, words: []const u32, out: []u32) !void {
        const n = words.len;
        var hist: [256]i32 = @splat(0);
        for (words) |w| hist[@as(usize, (w >> 24) & 0xFF)] += 1;
        var prefix: u32 = 0;
        var fixed: u32 = 0;
        var need: i64 = @intCast(out.len);
        var digit: u32 = 0;
        {
            var below: i64 = 0;
            const total: i64 = @intCast(n);
            for (0..256) |b| {
                if (total - below >= need) digit = @intCast(b);
                below += hist[b];
            }
        }
        var above: i64 = 0;
        for ((digit + 1)..256) |b| above += hist[b];
        need -= above;
        prefix |= digit << 24;
        fixed |= 0xFF << 24;
        const g: i64 = hist[digit];
        // `allin` takes the WHOLE bucket when g == need (the pair compare alone cannot express it - u > uint max
        // is never true - so the flag short-circuits it; the first sim run caught this, §10)
        const allin = g == need;
        if (g <= CAP) {
            // the threshold pair
            var thr_hw: u32 = 0xFFFFFFFF;
            var thr_lw: u32 = 0xFFFFFFFF;
            if (g > need) {
                // collect the bucket's (word, ~column) pairs - two i32 planes
                const hw = try gpa.alloc(u32, @intCast(g));
                defer gpa.free(hw);
                const lw = try gpa.alloc(u32, @intCast(g));
                defer gpa.free(lw);
                var cnt: usize = 0;
                for (words, 0..) |w, i| {
                    if ((w >> 24) == digit) {
                        hw[cnt] = w;
                        lw[cnt] = ~@as(u32, @intCast(i));
                        cnt += 1;
                    }
                }
                // the mini-radix: 4 byte passes over the high words, then over the low words of the equal-score
                // boundary (pairs are unique, so the low-word cut is exact)
                var hp: u32 = 0;
                var hf: u32 = 0;
                var sft: u5 = 24;
                while (true) {
                    var h: [256]i32 = @splat(0);
                    for (hw) |x| {
                        if ((x & hf) == hp) h[@as(usize, (x >> sft) & 0xFF)] += 1;
                    }
                    var d: u32 = 0;
                    {
                        var below: i64 = 0;
                        var t: i64 = 0;
                        for (h) |x| t += x;
                        for (0..256) |b| {
                            if (t - below >= need) d = @intCast(b);
                            below += h[b];
                        }
                    }
                    var ab: i64 = 0;
                    for ((d + 1)..256) |b| ab += h[b];
                    need -= ab;
                    hp |= d << sft;
                    hf |= @as(u32, 0xFF) << sft;
                    if (sft == 0) break;
                    sft -= 8;
                }
                thr_hw = hp;
                thr_lw = 0xFFFFFFFF;
                if (need > 0) {
                    // the equal-score boundary: the lower columns first = the higher low words
                    var lp: u32 = 0;
                    var lf: u32 = 0;
                    var s2: u5 = 24;
                    while (true) {
                        var h: [256]i32 = @splat(0);
                        for (hw, 0..) |x, j| {
                            if (x == hp and (lw[j] & lf) == lp) h[@as(usize, (lw[j] >> s2) & 0xFF)] += 1;
                        }
                        var d: u32 = 0;
                        {
                            var below: i64 = 0;
                            var t: i64 = 0;
                            for (h) |x| t += x;
                            for (0..256) |b| {
                                if (t - below >= need) d = @intCast(b);
                                below += h[b];
                            }
                        }
                        var ab: i64 = 0;
                        for ((d + 1)..256) |b| ab += h[b];
                        need -= ab;
                        lp |= d << s2;
                        lf |= @as(u32, 0xFF) << s2;
                        if (s2 == 0) break;
                        s2 -= 8;
                    }
                    thr_lw = lp;
                }
            }
            // the write pass: the residual `need` (0 or 1: pairs are unique) says whether the threshold pair
            // itself is the last winner - the radix's equal-cumsum is inclusive, so this must be too
            var written: usize = 0;
            for (words, 0..) |w, i| {
                const lwi = ~@as(u32, @intCast(i));
                if ((w >> 24) > digit or ((w >> 24) == digit and
                    (allin or w > thr_hw or (w == thr_hw and
                        (lwi > thr_lw or (need > 0 and lwi == thr_lw))))))
                {
                    out[written] = @intCast(i);
                    written += 1;
                }
            }
            try std.testing.expectEqual(out.len, written);
        } else {
            var sft: u5 = 16;
            while (true) {
                var h: [256]i32 = @splat(0);
                for (words) |w| {
                    if ((w & fixed) == prefix) h[@as(usize, (w >> sft) & 0xFF)] += 1;
                }
                var d: u32 = 0;
                {
                    var below: i64 = 0;
                    var t: i64 = 0;
                    for (h) |x| t += x;
                    for (0..256) |b| {
                        if (t - below >= need) d = @intCast(b);
                        below += h[b];
                    }
                }
                var ab: i64 = 0;
                for ((d + 1)..256) |b| ab += h[b];
                need -= ab;
                prefix |= d << sft;
                fixed |= @as(u32, 0xFF) << sft;
                if (sft == 0) break;
                sft -= 8;
            }
            var written: usize = 0;
            var equal_seen: i64 = 0;
            for (words, 0..) |w, i| {
                const eq = w == prefix;
                const take = w > prefix or (eq and equal_seen < need);
                if (eq) equal_seen += 1;
                if (take) {
                    out[written] = @intCast(i);
                    written += 1;
                }
            }
            try std.testing.expectEqual(out.len, written);
        }
    }
};

test "TF_GLM53_TOPK_STREAM: the streaming select picks the radix's set, ties included (findings-prefill-topk.md)" {
    const gpa = std.testing.allocator;
    var seed: u64 = 0x1234_5678_9abc_def0;

    const Case = struct { name: []const u8, n: usize, fill: fn (words: []u32, s: *u64) void };
    const cases = [_]Case{
        .{ .name = "random full-range words", .n = 8192, .fill = struct {
            fn f(words: []u32, s: *u64) void {
                for (words) |*w| w.* = topk_sim.rnd(s);
            }
        }.f },
        .{ .name = "heavy ties (97 distinct)", .n = 16384, .fill = struct {
            fn f(words: []u32, s: *u64) void {
                for (words, 0..) |*w, i| w.* = (@as(u32, @intCast(@mod(i, 97))) << 20) | (topk_sim.rnd(s) & 0xFFF);
            }
        }.f },
        .{ .name = "all equal (the radix fallback)", .n = 5000, .fill = struct {
            fn f(words: []u32, s: *u64) void {
                _ = s;
                for (words) |*w| w.* = 0x5A5A_0000;
            }
        }.f },
        .{ .name = "T == K exactly (the whole row taken; g > CAP -> the fallback)", .n = topk_sim.K, .fill = struct {
            fn f(words: []u32, s: *u64) void {
                for (words) |*w| w.* = 0x0100_0000 | (topk_sim.rnd(s) & 0xFF);
            }
        }.f },
        .{ .name = "g == need exactly (the 2-stream fast path)", .n = 3000, .fill = struct {
            fn f(words: []u32, s: *u64) void {
                // 1552 above the boundary byte, exactly need = 2048 - 1552 = 496 in it
                for (words, 0..) |*w, i| {
                    w.* = if (i < 1552) 0xFF00_0000 | (topk_sim.rnd(s) & 0xFFFFFF) else if (i < 1552 + 2048 - 1552) 0xFE00_0000 | (topk_sim.rnd(s) & 0xFFFF) else topk_sim.rnd(s) & 0xFDFF_FFFF;
                }
            }
        }.f },
        .{ .name = "later-key sentinel mix (60% zero words)", .n = 4096, .fill = struct {
            fn f(words: []u32, s: *u64) void {
                for (words, 0..) |*w, i| w.* = if (i % 5 < 3) 0 else 0x4000_0000 | (topk_sim.rnd(s) & 0xFFFFFF);
            }
        }.f },
        .{ .name = "monotone increasing (recency-biased)", .n = 32768, .fill = struct {
            fn f(words: []u32, s: *u64) void {
                for (words, 0..) |*w, i| w.* = @intCast(i *% 0x9E37 +% (topk_sim.rnd(s) & 0xFF));
            }
        }.f },
        .{ .name = "boundary bucket exactly CAP (collect full)", .n = 4096, .fill = struct {
            fn f(words: []u32, s: *u64) void {
                // 1600 above the boundary byte, exactly 1024 in it, the rest below: g == CAP, need == 448
                for (words, 0..) |*w, i| {
                    w.* = if (i < 1600) 0xFF00_0000 | (topk_sim.rnd(s) & 0xFFFFFF) else if (i < 1600 + topk_sim.CAP) 0xFE00_0000 | (topk_sim.rnd(s) & 0xFFFF) else topk_sim.rnd(s) & 0xFDFF_FFFF;
                }
            }
        }.f },
        .{ .name = "boundary bucket CAP + 1 (the fallback edge)", .n = 4096, .fill = struct {
            fn f(words: []u32, s: *u64) void {
                for (words, 0..) |*w, i| {
                    w.* = if (i < 1600) 0xFF00_0000 | (topk_sim.rnd(s) & 0xFFFFFF) else if (i < 1600 + topk_sim.CAP + 1) 0xFE00_0000 | (topk_sim.rnd(s) & 0xFFFF) else topk_sim.rnd(s) & 0xFDFF_FFFF;
                }
            }
        }.f },
        .{ .name = "equal scores at the cut (the low-word refine)", .n = 4096, .fill = struct {
            fn f(words: []u32, s: *u64) void {
                // 1600 above, 500 IDENTICAL words at the boundary byte (the low word decides)
                for (words, 0..) |*w, i| {
                    w.* = if (i < 1600) 0xFF00_0000 | (topk_sim.rnd(s) & 0xFFFFFF) else if (i < 2100) 0xFE00_1234 else topk_sim.rnd(s) & 0xFDFF_FFFF;
                }
            }
        }.f },
        .{ .name = "random at 131072 (the 128K depth)", .n = 131072, .fill = struct {
            fn f(words: []u32, s: *u64) void {
                for (words) |*w| w.* = topk_sim.rnd(s);
            }
        }.f },
    };
    inline for (cases) |cs| {
        const words = try gpa.alloc(u32, cs.n);
        defer gpa.free(words);
        cs.fill(words, &seed);
        const want = try gpa.alloc(u32, topk_sim.K);
        defer gpa.free(want);
        const got = try gpa.alloc(u32, topk_sim.K);
        defer gpa.free(got);
        try topk_sim.radixRef(gpa, words, want);
        topk_sim.streamSim(gpa, words, got) catch |e| {
            std.debug.print("streamSim case \"{s}\": {t}\n", .{ cs.name, e });
            return e;
        };
        std.testing.expectEqualSlices(u32, want, got) catch |e| {
            std.debug.print("case {s}: {t}\n", .{ cs.name, e });
            return e;
        };
    }
}

// ------------------------------------------------------- TF_GLM53_FP8_INDEX (host test) ---
//
// The fp8 index-key path is NOT bit-identical (e4m3 keys change the scores), so its host gate is a numerics test
// against the bf16 cache's scores on synthetic data (the kv8repro roundtrip precedent): the e4m3 roundtrip's
// per-element bound, the score bound it implies, and a statistical check. The GPU gates are the quality bench and
// the perf A/B (findings-prefill-fp8-index.md).

const fp8_index_sim = struct {
    const D: usize = 128; // the model's index_dim
    const NH: usize = 32; // the model's index_heads
    const E4MAX: f32 = 448.0;

    var grid: [127]f32 = undefined; // e4m3's positive finite values, ascending (the NaN code excluded)
    var grid_ready = false;

    fn gridInit() void {
        if (grid_ready) return;
        var i: usize = 0;
        var code: u16 = 0;
        while (code <= 0x7E) : (code += 1) { // 0x7F is the NaN code
            const e: u16 = (code >> 3) & 15;
            const m: u16 = code & 7;
            const v: f32 = if (e == 0)
                @as(f32, @floatFromInt(m)) * std.math.pow(f32, 2, -9) // the subnormals: m/8 * 2^-6
            else
                (1.0 + @as(f32, @floatFromInt(m)) / 8.0) * std.math.pow(f32, 2, @as(f32, @floatFromInt(@as(i32, e) - 7)));
            grid[i] = v;
            i += 1;
        }
        std.debug.assert(i == 127);
        grid_ready = true;
    }

    /// The nearest e4m3 representable value (the grid enumerator: ties resolved to the lower neighbor - either
    /// neighbor satisfies the bound the test asserts; the GPU's round-to-nearest is the quality bench's to check).
    fn quant(v: f32) f32 {
        gridInit();
        const a = @abs(v);
        if (a > E4MAX) return if (v < 0) -E4MAX else E4MAX;
        var best = grid[0];
        var bd = a - grid[0];
        if (bd < 0) bd = -bd;
        for (grid[1..]) |g| {
            const d = a - g;
            const ad = if (d < 0) -d else d;
            if (ad < bd) {
                bd = ad;
                best = g;
            }
        }
        return if (v < 0) -best else best;
    }

    /// bf16 round-to-nearest-even (Triton's .to(bf16); Zig has no bf16 primitive, so the 16-bit truncation with
    /// the carried lsb rounds it here)
    fn bf(v: f32) f32 {
        const bits: u32 = @bitCast(v);
        const lsb: u32 = (bits >> 16) & 1;
        return @bitCast((bits +% 0x7FFF +% lsb) & 0xFFFF0000);
    }

    var seed: u64 = 0x9e37_79b9_7f4a_7c15;
    fn rnd() f32 {
        seed = seed *% 6364136223846793005 +% 1442695040888963407;
        const ur: u32 = @truncate(seed >> 33);
        return @as(f32, @floatFromInt(ur % 20001)) / 10000.0 - 1.0; // [-1, 1)
    }
};

test "TF_GLM53_FP8_INDEX: the e4m3 key roundtrip and its score sit inside the proven bounds (findings-prefill-fp8-index.md)" {
    fp8_index_sim.gridInit();
    const ws: f32 = @floatCast(std.math.pow(f64, @floatFromInt(fp8_index_sim.NH), -0.5));
    const qs: f32 = @floatCast(std.math.pow(f64, @floatFromInt(fp8_index_sim.D), -0.5));

    // one query token's heads and weights, 64 synthetic key rows
    var q: [fp8_index_sim.NH][fp8_index_sim.D]f32 = undefined;
    var w: [fp8_index_sim.NH]f32 = undefined;
    for (&q) |*row| {
        for (row) |*v| v.* = fp8_index_sim.bf(fp8_index_sim.rnd());
    }
    for (&w) |*v| v.* = fp8_index_sim.rnd();

    var worst_rel: f32 = 0;
    var sum_rel: f32 = 0;
    var n_rel: usize = 0;
    for (0..64) |_| {
        // the key row the bf16 cache would hold, its amax, the plane's quantization of it
        var x: [fp8_index_sim.D]f32 = undefined;
        var amax: f32 = 0;
        for (&x) |*v| {
            v.* = fp8_index_sim.bf(fp8_index_sim.rnd() * 3.0); // a wider spread than the queries
            amax = @max(amax, @abs(v.*));
        }
        const scale = @max(amax / fp8_index_sim.E4MAX, 1e-12);
        var s16: f32 = 0;
        var s8: f32 = 0;
        var dq_max: f32 = 0; // the roundtrip's worst |k8 - k16|
        for (&x, 0..) |v, d| {
            const k16 = v; // the cache's bf16 value
            const k8 = fp8_index_sim.bf(fp8_index_sim.quant(v / scale) * scale); // the plane's dequant -> bf16
            dq_max = @max(dq_max, @abs(k8 - k16));
            for (0..fp8_index_sim.NH) |h| {
                s16 += (w[h] * ws) * @max(q[h][d] * k16 * qs, 0); // the kernel's sum, evaluated per dim
                s8 += (w[h] * ws) * @max(q[h][d] * k8 * qs, 0);
            }
        }
        // (a) the roundtrip bound: the e4m3 quantization (half an ulp at the top binade = 16 scales = amax / 28)
        // PLUS the dequant's bf16 rounding (half an ulp of bf16, 2^-8 relative)
        const rt: f32 = 16.0 / 448.0 + 1.0 / 256.0;
        try std.testing.expect(dq_max <= amax * rt + 1e-9);
        // (b) the score bound it implies (relu is 1-Lipschitz; the weights scale the heads)
        var wq: f32 = 0;
        for (0..fp8_index_sim.NH) |h| {
            var l1: f32 = 0;
            for (&q[h]) |v| l1 += @abs(v);
            wq += @abs(w[h] * ws) * l1;
        }
        try std.testing.expect(@abs(s8 - s16) <= wq * (amax * rt) * qs + 1e-6);
        // (c) the statistical check (informative; the deterministic seed keeps it stable)
        const rel = @abs(s8 - s16) / @max(@abs(s16), 1e-3);
        worst_rel = @max(worst_rel, rel);
        sum_rel += rel;
        n_rel += 1;
    }
    try std.testing.expect(sum_rel / @as(f32, @floatFromInt(n_rel)) < 0.02); // the mean sits under 2%
    // the worst element may exceed 2% (a small score with cancellation) but not the e4m3 noise floor's reach
    try std.testing.expect(worst_rel < 0.10);
}
