//! Full GLM-5.3 served from Zig (Phase 3b): what Glm53Engine sets up and runs a request with, shared by the native
//! server (native/glm53_cuda.zig: rank 0's HTTP side, ranks 1..3 following its requests) and tf-glm53-generate
//! --mode 3b (every rank reading the same prompts file).
//!
//! `Boot`: one rank of the TP world in its served configuration (Phase 3a's run3a setup as one object: NCCL through
//! the TCP rendezvous, the weights, the reference boot's tiles, kernels, the RoCE one-shot, cuBLAS as torch sets it
//! up, the prompt GEMM's workspaces, the MTP layer's dense prompt experts, the decode side stream and L2 prefetch,
//! the sequence-parallel chunk, the Runner at the context's capacity (limit + k + 1), its graphs prewarmed). Decode
//! context parallelism (DCP = the world) turns on past DCP_AUTO tokens of context unless TF_GLM53_DCP says.
//! `Session`: a request on every rank alike (Glm53Engine._run): prompt reuse (cuda_reuse.zig) around Runner.run,
//! and the learned states' files (--learn: each rank writes and reads its own shard of a kept state).
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const W = @import("cuda_weights.zig");
const fw = @import("cuda_forward.zig");
const kern = @import("cuda_kernels.zig");
const rn = @import("cuda_runner.zig");
const smp = @import("sampling.zig");
const prompt = @import("cuda_prompt.zig");
const dec = @import("cuda_decode.zig");
const roce = @import("cuda_roce.zig");
const tiles = @import("tiles.zig");
const reuse = @import("cuda_reuse.zig");
const draft = @import("cuda_draft.zig");
const dhost = @import("draft_host.zig");
const dlogmod = @import("draft_log.zig");
const copies = @import("copies.zig");
const multi = @import("cuda_multi.zig");
const mdraft = @import("cuda_mdraft.zig");
const dcpm = @import("cuda_dcp.zig");
const headq = @import("headq.zig");
const binder = @import("cuda_bind.zig"); // §98: the 1.1->r2 binder, in the module
const glm53 = @import("glm53").family; // §98: the 1.1 loader (the NVFP4-CSF weights source)
const core = @import("core");

pub const world_default: usize = 4;
/// engine.DCP_AUTO: contexts past this interleave the KV cache over the ranks (bf16 latent)
pub const dcp_auto: usize = 200_000;

fn env(name: [:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    const t = std.mem.trim(u8, std.mem.span(v), " ");
    return if (t.len == 0) null else t;
}

fn envFlag(name: [:0]const u8, default: bool) !bool {
    const v = env(name) orelse return default;
    if (std.mem.eql(u8, v, "1") or std.ascii.eqlIgnoreCase(v, "true") or std.ascii.eqlIgnoreCase(v, "on")) return true;
    if (std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "off")) return false;
    std.log.err("{s}={s}: 0 or 1", .{ name, v });
    return error.BadSetting;
}

fn envInt(comptime T: type, name: [:0]const u8, default: T) !T {
    const v = env(name) orelse return default;
    return std.fmt.parseInt(T, v, 10) catch {
        std.log.err("{s}={s}: a whole number", .{ name, v });
        return error.BadSetting;
    };
}

fn envFloat(name: [:0]const u8, default: f64) !f64 {
    const v = env(name) orelse return default;
    return std.fmt.parseFloat(f64, v) catch {
        std.log.err("{s}={s}: a number", .{ name, v });
        return error.BadSetting;
    };
}

/// TF_GLM53_L2PF: -1 off, 0 bulk, 1 lines, 2 touch (l2pf.MODES).
pub fn l2pfMode(v: []const u8) !c_int {
    if (std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off")) return -1;
    if (std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "on") or std.mem.eql(u8, v, "bulk")) return 0;
    if (std.mem.eql(u8, v, "lines")) return 1;
    if (std.mem.eql(u8, v, "touch")) return 2;
    return error.BadL2pfMode;
}

/// roce.hca_names: "mlx5_0,mlx5_1" (an NCCL_IB_HCA spelling: =^ prefixes, :port suffixes) -> at most two names.
pub fn parseHcas(text: []const u8, out: *[2][]const u8) [][]const u8 {
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |raw| {
        if (n == out.len) break;
        var name = std.mem.trim(u8, raw, " =^");
        if (std.mem.indexOfScalar(u8, name, ':')) |c| name = name[0..c];
        if (name.len == 0) continue;
        out[n] = name;
        n += 1;
    }
    return out[0..n];
}

/// One rank's served configuration: tf-glm53-generate --mode 3a's flags, or the TF_GLM53_* variables the server
/// reads (`fromEnv`); every rank must be started with the same ones (checked at the rendezvous).
pub const Options = struct {
    rank: usize = 0,
    world: usize = world_default,
    master: []const u8 = "127.0.0.1:29581",
    model: []const u8 = "",
    aot: []const u8 = "",
    nccl_lib: []const u8 = "libnccl.so.2",
    cublas_lib: []const u8 = "libcublas.so.13",
    cublas_ws: ?usize = null,
    /// Â§125.37 the dense-lane A/B arm: the qmmf bf16 lane for the dense smalls' decode windows.
    dense_lane: bool = false,
    cublas_math: c_int = 0,
    /// the longest prompt plus reply a request may use (Glm53Engine.limit); the runner holds limit + k + 1
    context: usize = 32768,
    k: usize = 2,
    draft_vocab: usize = 32768,
    mtp_reuse: u32 = 2,
    graphs: bool = true,
    prewarm: bool = true,
    fast_load: bool = true,
    roce: bool = true,
    hcas: []const u8 = "",
    gid: u32 = 3,
    roce_health: bool = true,
    spin_limit: u32 = 20_000_000,
    tiles: ?[]const u8 = null,
    inv_ref: ?[]const u8 = null,
    bmm_probe: ?[]const u8 = null,
    prompt_rows: usize = 8192,
    prompt_rows_short: usize = 4096,
    /// §128.27az: the shared sequence plane's max aggregate (null: the uniform slots).
    pool_agg: ?usize = null,
    dcp_prompt_rows: usize = 4096,
    /// TF_GLM53_DCP_PROMPT_GATHER: prompt chunks under DCP gather each layer's cache prefix (cuda_forward.PromptGather)
    dcp_prompt_gather: bool = false,
    prompt_sp: bool = true,
    experts: []const u8 = "shared",
    pe_det: []const u8 = "slots16",
    unpack_mb: usize = 384,
    side: []const u8 = "af",
    l2pf: []const u8 = "1",
    l2pf_mb: f64 = 8.0,
    multi_select: bool = true,
    device_cands: bool = true,
    mtp_dense: bool = true,
    /// TF_GLM53_L2PF_AFTER: the L2 prefetch sites queued after their exchange instead of before it (timing only)
    l2pf_after: bool = false,
    /// TF_GLM53_HEAD_Q (headq.zig): lm_head as an EXL3 linear from a safetensors file of its own (the value: its
    /// path) or from the checkpoint ("pack"); off by default (the bf16 rows)
    head_q: headq.Setting = .off,
    /// TF_GLM53_KVB_Q: a checkpoint whose kv_b_proj is an EXL3 linear is decoded to the bf16 blocks at load
    kvb_q: bool = false,
    /// TF_GLM53_DCP: 0 auto (the world past dcp_auto tokens of context, else 1), 1, or the world
    dcp: usize = 0,
    /// TF_GLM53_DCP_STREAMS=1: --parallel N > 1 with decode context parallelism on (this port's own; unset: refused
    /// as the Python engine refuses it). docs/glm53-dcp-streams.md
    dcp_streams: bool = false,
    timeout_s: f64 = 5400,
    reuse: reuse.Settings = .{},
    /// Phase 4: the drafters (checkpoint directories, "" none: TF_GLM53_DSPARK / TF_GLM53_DFLASH), DSpark's settings
    /// (TF_GLM53_DSPARK_*), DFlash2's (TF_GLM53_DFLASH_DEPTH / _CONFIDENCE), copy drafts (TF_GLM53_COPY_*), the
    /// default draft mode (TF_GLM53_MTP: dspark | dflash; unset: a drafter when --mtp-drafts 0 (k 0) loads one, else
    /// the MTP head), the drafters' verify rows (TF_GLM53_VERIFY_ROWS)
    dspark: []const u8 = "",
    dflash: []const u8 = "",
    dspark_env: dhost.DSparkEnv = .{},
    dspark_cal: ?[]const u8 = null, // 125.43e: TF_GLM53_DSPARK_CAL - the measured landing table (a file path)
    dspark_log: ?[]const u8 = null, // 125.43e: TF_GLM53_DSPARK_LOG - the walked-tree JSONL (rank 0)
    dflash_set: dhost.DFlashSettings = .{},
    copy: copies.Settings = .{ .on = false, .match = 0, .most = 0 },
    mode: ?[]const u8 = null,
    verify_rows: usize = 8,
    /// Phase 4b: concurrent streams (--parallel N / TF_GLM53_PARALLEL; 1: one request at a time) and the concurrent
    /// decoder's settings (TF_GLM53_DRAFT_CUT / _STREAMS, TF_GLM53_BATCH_FILL, TF_GLM53_QUICK_ROWS,
    /// TF_GLM53_PRECAPTURE)
    parallel: usize = 1,
    multi_set: multi.Settings = .{},
    /// TF_GLM53_KV8: the latent KV cache in e4m3 + two fp32 scales a row (584 B vs 1152 bf16; DCP 1 only for now).
    /// Lossy - bits differ from the bf16 cache (the VERIFY_HEAD=fp8 gating style); per-mode determinism holds within
    /// one setting. Halves the dominant cache term: ~94.4 -> ~50.6 KiB a token (findings-kv8.md).
    kv8: bool = false,
    /// TF_GLM53_KV8_V2: the prompt (wide-window) attention reads the e4m3 plane through the KV8 variants of
    /// _attn_chunks - in-kernel dequant, no dequantPrompt mirror, no O(pos) per-chunk term (findings-kv8-v2.md).
    /// Needs kv8 and the v2 AOT variants (the recapture plan, findings-kv8-v2.md §5); default off: the staged v1
    /// prompt path until the pack is recaptured and validated.
    kv8_v2: bool = false,
    /// 119.2: TF_GLM53_KV8_V3 - the 656-byte row: the rope half the staging's own bf16 (exact, v2
    /// quantized it e4m3 with one whole-half scale), the latent half a scale a 128-element block.
    /// +12.3% plane bytes; the anchors re-roll (a legitimate format shift, the old kept as history).
    kv8_v3: bool = false,

    /// TF_GLM53_PROMPT_PROF: the serve wires glm.prompt.Prof into its Forward - per-chunk and per-prompt phase
    /// reports (PROMPT-PROF lines) on every rank. Deliberately NOT in the settings digest: it changes no numerics
    /// and no serving settings (CUDA events only, and a chunk-boundary sync while it is on; off = today's bits).
    prompt_prof: bool = false,

    /// TF_GLM53_TOPK_STREAM: the prefill blocks' indexer selection runs topk._topk_row - the radix's picks,
    /// bit-identical (the same tie rule; host-tested), in 2-3 streaming passes over the score row instead of the
    /// radix's 5 (findings-prefill-topk.md - the prefill-decline fix A). Needs the _topk_row variant in the AOT
    /// pack; decode windows keep the radix. IN the digest (like kv8_v2 / multi_select): the selection kernel is a
    /// numerics-adjacent path, and ranks must agree on it even though the picks are identical by construction.
    topk_stream: bool = false,

    /// TF_GLM53_FP8_INDEX: the index keys in an e4m3 plane (128 codes + an fp32 scale a token - stock vLLM's ABI)
    /// carved inside the bf16 ic allocation; the scoring dequantizes in-kernel (the dot stays bf16 - increment 1,
    /// the 2x K-read-byte win; findings-prefill-fp8-index.md). Changes the scores (e4m3 keys): NOT bit-identical -
    /// the gates are the host numerics test + the quality bench. Needs DCP 1 (the engine refuses more) and the
    /// _ik_write8 / _index_scores8 AOT variants; coexists with kv8 and topk_stream (independent caches and sites).
    /// IN the digest: the cache layout is a numerics change every rank must agree on.
    fp8_index: bool = false,
    e1_fp8: bool = false,
    e3_merge: bool = true,
    fp8_stride: usize = 132, // inc 3: the plane's token row - 132 legacy, 144 the padded 16-byte-multiple layout

    /// The server's settings: TF_GLM53_RANK / _WORLD / _MASTER / _AOT / ... (the script's names), `context` from
    /// --context. Unset variables keep tf-glm53-generate --mode 3a's defaults (the image's served ones).
    pub fn fromEnv(model: []const u8, context: usize) !Options {
        var o: Options = .{ .model = model, .context = context };
        o.rank = try envInt(usize, "TF_GLM53_RANK", 0);
        o.world = try envInt(usize, "TF_GLM53_WORLD", world_default);
        o.master = env("TF_GLM53_MASTER") orelse o.master;
        o.aot = env("TF_GLM53_AOT") orelse env("TENSORFOLD_CUDA_KERNELS") orelse "";
        o.nccl_lib = env("TF_GLM53_NCCL_LIB") orelse o.nccl_lib;
        o.cublas_lib = env("TF_GLM53_CUBLAS_LIB") orelse o.cublas_lib;
        if (env("TF_GLM53_CUBLAS_WS")) |_| o.cublas_ws = try envInt(usize, "TF_GLM53_CUBLAS_WS", 0);
        o.dense_lane = env("TF_GLM53_DENSE_LANE") != null;
        o.k = try envInt(usize, "TF_GLM53_K", o.k);
        o.draft_vocab = try envInt(usize, "TF_GLM53_DRAFT_VOCAB", o.draft_vocab);
        o.mtp_reuse = try envInt(u32, "TF_GLM53_MTP_REUSE", o.mtp_reuse);
        o.graphs = try envFlag("TF_GLM53_GRAPHS", true);
        o.prewarm = try envFlag("TF_GLM53_PREWARM", true);
        o.fast_load = try envFlag("TF_GLM53_FAST_LOAD", true);
        o.roce = try envFlag("TF_GLM53_ROCE", true);
        o.hcas = env("TF_GLM53_HCAS") orelse env("TF_GLM53_ROCE_HCA") orelse "";
        o.gid = try envInt(u32, "TF_GLM53_GID", o.gid);
        o.roce_health = try envFlag("TF_GLM53_ROCE_HEALTH", true);
        o.tiles = env("TF_GLM53_TILES");
        if (o.tiles) |t| if (std.mem.startsWith(u8, t, "load:")) {
            o.tiles = t[5..];
        };
        o.inv_ref = env("TF_GLM53_INV_REF");
        o.bmm_probe = env("TF_GLM53_BMM_PROBE");
        o.prompt_rows = try envInt(usize, "TF_GLM53_PROMPT_ROWS", o.prompt_rows);
        o.prompt_rows_short = try envInt(usize, "TF_GLM53_PROMPT_ROWS_SHORT", o.prompt_rows_short);
        if (env("TF_GLM53_POOL_AGG")) |v| o.pool_agg = std.fmt.parseInt(usize, v, 10) catch null;
        o.dcp_prompt_rows = try envInt(usize, "TF_GLM53_DCP_PROMPT_ROWS", o.dcp_prompt_rows);
        o.dcp_prompt_gather = try envFlag("TF_GLM53_DCP_PROMPT_GATHER", false);
        o.prompt_sp = try envFlag("TF_GLM53_PROMPT_SP", true);
        o.experts = env("TF_GLM53_EXPERTS") orelse o.experts;
        o.pe_det = env("TF_GLM53_PE_DET") orelse env("TF_EXL3_PROMPT_DET") orelse o.pe_det;
        o.unpack_mb = try envInt(usize, "TF_GLM53_UNPACK_MB", o.unpack_mb);
        o.side = env("TF_GLM53_SIDE") orelse o.side;
        o.l2pf = env("TF_GLM53_L2PF") orelse o.l2pf;
        o.l2pf_mb = try envFloat("TF_GLM53_L2PF_MB", o.l2pf_mb);
        o.l2pf_after = try envFlag("TF_GLM53_L2PF_AFTER", false);
        o.multi_select = try envFlag("TF_GLM53_MULTI_SELECT", true);
        o.device_cands = try envFlag("TF_GLM53_DEVICE_CANDS", true);
        o.mtp_dense = try envFlag("TF_GLM53_MTP_DENSE", true);
        o.head_q = headq.Setting.parse(env("TF_GLM53_HEAD_Q"));
        o.kvb_q = try envFlag("TF_GLM53_KVB_Q", false);
        o.dcp = try envInt(usize, "TF_GLM53_DCP", 0);
        o.dcp_streams = try envFlag("TF_GLM53_DCP_STREAMS", false);
        o.timeout_s = try envFloat("TF_GLM53_TIMEOUT", o.timeout_s);
        o.reuse = try reuse.Settings.fromEnv();
        o.dspark = env("TF_GLM53_DSPARK") orelse "";
        o.dflash = env("TF_GLM53_DFLASH") orelse "";
        o.dspark_cal = env("TF_GLM53_DSPARK_CAL");
        if (o.dspark_cal != null and o.dspark_cal.?.len == 0) o.dspark_cal = null; // the launchers pass ${VAR:-}: empty is unset
        o.dspark_log = env("TF_GLM53_DSPARK_LOG");
        if (o.dspark_log != null and o.dspark_log.?.len == 0) o.dspark_log = null;
        o.dspark_env = .{ .depth = env("TF_GLM53_DSPARK_DEPTH"), .policy = env("TF_GLM53_DSPARK_POLICY"), .confidence = env("TF_GLM53_DSPARK_CONFIDENCE"), .top_k = env("TF_GLM53_DSPARK_TOPK"), .noise = env("TF_GLM53_DSPARK_NOISE"), .filter = env("TF_GLM53_DSPARK_FILTER"), .row_cost = env("TF_GLM53_DSPARK_ROW_COST"), .quant = env("TF_GLM53_DSPARK_QUANT"), .costs = env("TF_GLM53_DSPARK_COSTS") };
        o.dflash_set = .{ .depth = try envInt(usize, "TF_GLM53_DFLASH_DEPTH", 7), .confidence = try envFloat("TF_GLM53_DFLASH_CONFIDENCE", 0.3) };
        // §100: honor the Boot's own default (copy off) when TF_GLM53_COPY_DRAFTS is unset — parse's
        // engine default is ON (copies.zig: the Python family drafts by prompt lookup), and the unconditional
        // parse silently overrode the off default: copy widths 8/16 entered the verify mask, the prewarm's
        // verify(16) launched the head at 16 rows, and the pack lacks that variant (the §98.6 crash).
        if (env("TF_GLM53_COPY_DRAFTS") != null) {
            o.copy = copies.Settings.parse(env("TF_GLM53_COPY_DRAFTS"), env("TF_GLM53_COPY_MIN"), env("TF_GLM53_COPY_MAX")) catch {
                std.log.err("TF_GLM53_COPY_DRAFTS (0 / 1), _MIN (2..64), _MAX (1..31)", .{});
                return error.BadSetting;
            };
        }
        o.mode = env("TF_GLM53_MTP");
        o.verify_rows = try envInt(usize, "TF_GLM53_VERIFY_ROWS", 8);
        o.parallel = @max(1, try envInt(usize, "TF_GLM53_PARALLEL", 1));
        // Phase 4c: TF_GLM53_CONC_MODE (engine.CONC_MODE) - the drafts of concurrent requests (dflash, dspark, ...)
        if (o.parallel > 1) if (env("TF_GLM53_CONC_MODE")) |m| {
            o.mode = m;
        };
        o.multi_set = multi.Settings.fromEnv() catch {
            std.log.err("TF_GLM53_DRAFT_CUT (a number), _STREAMS / TF_GLM53_QUICK_ROWS (whole numbers)", .{});
            return error.BadSetting;
        };
        o.kv8 = try envFlag("TF_GLM53_KV8", false);
        o.kv8_v2 = try envFlag("TF_GLM53_KV8_V2", false);
        o.kv8_v3 = try envFlag("TF_GLM53_KV8_V3", false);
        if (o.kv8_v3) o.kv8 = true; // the v3 row rides kv8
        o.prompt_prof = try envFlag("TF_GLM53_PROMPT_PROF", false);
        o.topk_stream = try envFlag("TF_GLM53_TOPK_STREAM", false);
        o.fp8_index = try envFlag("TF_GLM53_FP8_INDEX", false);
        // section 18: the DCP E1 query gather as e4m3 (kv8Quant-packed rows, the in-kernel q dequant) - half
        // the E1 wire; the scores change q-side (~0.4%), the needle/coherence gates decide
        o.e1_fp8 = try envFlag("TF_GLM53_E1_FP8", false);
        o.e3_merge = try envFlag("TF_GLM53_E3_MERGE", true);
        // inc 3 (findings-prefill-fp8-cuda.md section 20): the padded plane - a 16-byte-multiple token row,
        // what the native scorer's 16-byte cp.async staging (and a TMA descriptor) needs. 132 = the legacy
        // layout, byte-compatible with the shipped pack and the shipped kernel's plateau configuration.
        o.fp8_stride = try envInt(usize, "TF_GLM53_FP8_STRIDE", 132);
        if (o.fp8_stride != 132 and o.fp8_stride != 144) {
            std.log.err("TF_GLM53_FP8_STRIDE must be 132 (the legacy plane) or 144 (the padded plane), got {d}", .{o.fp8_stride});
            return error.BadFp8Stride;
        }
        if (o.fp8_stride == 144 and !o.fp8_index) {
            std.log.warn("glm53 rank {d}: TF_GLM53_FP8_STRIDE=144 without TF_GLM53_FP8_INDEX - ignored (the padded layout only exists under the e4m3 plane)", .{o.rank});
            o.fp8_stride = 132;
        }
        if (o.aot.len == 0) {
            std.log.err("glm53: no Triton kernel pack: set TF_GLM53_AOT to the directory with aot.json", .{});
            return error.NoKernelPack;
        }
        return o;
    }

    /// The draft mode of a request without its own (Glm53Engine: TF_GLM53_MTP; a drafter alone with --mtp-drafts 0).
    pub fn defaultMode(o: Options) !rn.Mode {
        if (o.mode) |m| {
            if (std.mem.eql(u8, m, "dspark")) return if (o.dspark.len > 0) .dspark else error.NoDrafter;
            if (std.mem.eql(u8, m, "dflash")) return if (o.dflash.len > 0) .dflash else error.NoDrafter;
            return .mtp; // an MTP input mode (normed/normed, ...): the MTP head
        }
        if (o.k == 0 and o.dspark.len > 0) return .dspark;
        if (o.k == 0 and o.dflash.len > 0) return .dflash;
        return .mtp;
    }

    /// The decode context parallelism a context gets (Glm53Engine: TF_GLM53_DCP, else the world past DCP_AUTO).
    pub fn dcpFor(o: Options) !usize {
        const d = if (o.dcp != 0) o.dcp else if (o.context > dcp_auto) o.world else 1;
        // the parity-mirror (findings-dcp2-divisor.md §4.1): the divisors of world too
        if (d != 1 and d != o.world and (d > o.world or o.world % d != 0)) return error.BadDcp;
        return d;
    }
};

/// Runner._check_cache_fits: the target layers' bf16 latent rows of `slots` streams of `capacity` tokens (`need`
/// bytes, a drafter's `extra` included) against what is free less the reserve (TF_GLM53_CACHE_RESERVE_GB, default 6,
/// at most half of what is free): on GB10's unified memory an overcommitted cache swaps the node instead of failing.
/// Null when it fits, else the most context a stream that would fit (Python's "--context <= fit").
pub const CacheFit = struct { need: u64, free: u64, spare: u64, fit: u64 };

pub fn cacheFit(row: u64, local: u64, slots: u64, dcp: u64, extra: u64, free: u64, reserve_gb: f64) ?CacheFit {
    const need = row * local * slots + extra;
    const spare: u64 = @intFromFloat(@min(reserve_gb * (1 << 30), @as(f64, @floatFromInt(free)) / 2));
    if (need + spare <= free) return null;
    const room = free -| spare -| extra;
    return .{ .need = need, .free = free, .spare = spare, .fit = room / (row * slots) * dcp };
}

/// The settings every rank must share, hashed (the rendezvous compares them before anything loads).
fn settingsDigest(o: Options, dcp: usize, drafts: u64) u64 {
    var h = std.hash.Wyhash.init(0x6a3b);
    h.update(std.mem.asBytes(&drafts));
    const cw = [_]usize{ @intFromBool(o.copy.on), o.copy.match, o.copy.most, o.verify_rows, o.parallel };
    h.update(std.mem.asBytes(&cw));
    if (o.parallel > 1) o.multi_set.digest(&h);
    const shape = [_]usize{ o.world, o.context, o.pool_agg orelse 0, o.k, o.draft_vocab, o.mtp_reuse, @intFromBool(o.graphs), @intFromBool(o.prewarm), @intFromBool(o.roce), o.prompt_rows, o.prompt_rows_short, o.dcp_prompt_rows, @intFromBool(o.dcp_prompt_gather), @intFromBool(o.prompt_sp), o.unpack_mb, @intFromBool(o.multi_select), @intFromBool(o.device_cands), @intFromBool(o.mtp_dense), dcp, @intFromBool(o.roce_health), @intFromBool(o.reuse.on), o.reuse.gap, o.reuse.stride, o.reuse.entries, @intFromBool(o.reuse.loose), @intFromFloat(o.l2pf_mb * 1024), @intFromBool(o.kv8), @intFromBool(o.kv8_v2), @intFromBool(o.kv8_v3), @intFromBool(o.topk_stream), @intFromBool(o.fp8_index), o.fp8_stride, @intFromBool(o.e1_fp8), @intFromBool(o.e3_merge), @intFromBool(o.dspark_cal != null) };
    h.update(std.mem.asBytes(&shape));
    // TF_GLM53_KV8 normalizes the side letters to off, so the raw TF_GLM53_SIDE no longer changes behavior: hash
    // it empty then (kv8 itself is in `shape`), or ranks that differ only in the ignored setting would refuse.
    for ([_][]const u8{ o.experts, o.pe_det, if (o.kv8) "" else o.side, o.l2pf }) |t| {
        h.update(t);
        h.update("|");
    }
    return h.final();
}

fn now(io: std.Io) i128 {
    return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
}

fn since(io: std.Io, t0: i128) f64 {
    return @as(f64, @floatFromInt(now(io) - t0)) / 1e9;
}

/// One rank in its served configuration (run3a's setup), heap-held: the Forward and the Runner point into it.
pub const Boot = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    o: Options,
    dcp: usize,
    driver: cuda.Driver,
    ctx: cuda.Context,
    lib: cuda.nccl.Library,
    rdv: cuda.rendezvous.Rendezvous,
    comm: cuda.nccl.Communicator,
    cfg: Config,
    w: W.Weights,
    k: kern.Kernels,
    stream: cuda.Stream,
    side_stream: cuda.Stream,
    inv: cuda.DeviceBuffer,
    fast: ?*roce.Roce = null,
    blas_lib: cuda.cublas.Library,
    ws: cuda.DeviceBuffer,
    qlane: ?cuda.qmmf.Lane = null,
    qpart: cuda.DeviceBuffer = undefined,
    blas: cuda.cublas.Blas,
    cache: ?prompt.UnpackCache = null,
    main_wide: prompt.Wide,
    side_wide: prompt.Wide,
    mtp_dense: ?prompt.MtpDense = null,
    side: ?dec.Side = null,
    l2pf: ?dec.L2pf = null,
    sp: ?prompt.Sp = null,
    pg: ?fw.PromptGather = null,
    kv8: ?fw.Kv8 = null, // TF_GLM53_KV8's staging / scratch / ramp tables (one serves every layer)
    prof: ?prompt.Prof = null, // TF_GLM53_PROMPT_PROF's timers (the Forward and the Runner's copy point here)
    // §98 TF_GLM53_BIND: the 1.1 pack's loaded state. The bound `w` REFERENCES pw's device buffers, so
    // everything here stays alive with the Boot and is undone at stage 6, before b.w's own buffers.
    bind_on: bool = false,
    bind_asm: *glm53.blocks.Assembled = undefined, // heap-held: rebind's set pointers aim into it (§98)
    bind_m_pack: cuda.Module = undefined,
    bind_m_quant: cuda.Module = undefined,
    bind_ck: core.Checkpoint = undefined,
    bind_pw: glm53.weights.Weights = undefined,
    bind_side: []?glm53.moe.Layer = &.{},
    bind_arena: cuda.DeviceBuffer = undefined,
    bind_hook: ?binder.Hook = null,
    bind_moe_prof: ?*binder.Hook.MoeProf = null, // §99: TF_GLM_PROF's 1.1 moe Prof (heap-held; the hook points at it)
    fwd: fw.Forward, // the sequence-parallel chunk points at it (Sp.main); the Runner holds a copy
    r: rn.Runner,
    // Phase 4: the drafters' configs (read and checked before anything loads), their tap plan, the drafters, the
    // default draft mode
    dspark_cfg: ?dhost.DSparkConfig = null,
    dflash_cfg: ?dhost.DFlashConfig = null,
    dspark_set: dhost.DSparkSettings = .{ .depth = 0 },
    plan: dhost.TapPlan = .{},
    dspark_dr: ?*draft.Drafter = null,
    dcal: ?dhost.CalTables = null, // 125.43e: the loaded calibration (the dspark chain's gate)
    window_lane: bool = true, // 125.43(d): the dispatch cell (the verify captures' mode stamp)
    dlog: ?dlogmod.DraftLog = null, // 125.43e: the walked-tree log (the runner + the mdraft point at it)
    dflash_dr: ?*draft.Drafter = null,
    /// Phase 4b: the concurrent decoder (--parallel N > 1; cuda_multi.zig), on every rank
    multi: ?*multi.Multi = null,
    mode: rn.Mode = .mtp,
    load_s: f64 = 0,
    prewarm_graphs: usize = 0,
    prewarm_s: f64 = 0,
    nccl_version: c_int = 0,
    stage: u8 = 0, // how far init got (deinit undoes that much)

    /// 128.27w THE PHASE MARKS (TF_GLM53_ALLOC_DUMP): the device state at each init milestone —
    /// PRE-LOAD (the fixed bucket: contexts, NCCL, driver), PRE-PREWARM (+ the weights, caches,
    /// drafter), POST-PREWARM (the existing non-funnel line at the prewarm end) — the deltas
    /// itemize the terms for the >=1M resize design (the user's capacity mandate). Logging-only:
    /// never fails the boot, never changes a byte of compute.
    fn dumpPhase(b: *Boot, tag: []const u8) void {
        if (!cuda.envOn("TF_GLM53_ALLOC_DUMP")) return;
        const mi = b.ctx.memInfo() catch return;
        const used: f64 = @as(f64, @floatFromInt(mi.total -| mi.free)) / (1 << 30);
        const fun: f64 = @as(f64, @floatFromInt(cuda.deviceBytes())) / (1 << 30);
        const free_gib: f64 = @as(f64, @floatFromInt(mi.free)) / (1 << 30);
        std.log.info("alloc-dump phase {s}: device used {d:.2} GiB (funnel live {d:.2}, non-funnel {d:.2}), free {d:.2} GiB", .{ tag, used, fun, used - fun, free_gib });
    }

    /// Every rank of the world loaded and warmed, in step with the others (rendezvous barriers between stages).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, o: Options) !*Boot {
        const b = try gpa.create(Boot);
        errdefer gpa.destroy(b);
        b.* = undefined;
        b.gpa = gpa;
        b.io = io;
        b.o = o;
        b.stage = 0;
        // 118.4: the undo reads these before any stage gate (the bind block's `if (b.bind_on)`), and
        // `b.* = undefined` leaves them 0xAA - a truthy bind_on ran the block on garbage and the moe
        // prof's Event pool segfaulted on EVERY init-error path (the receipt: the three trace-boot
        // crashes, addr2line Event.deinit <- Prof.deinit <- undo:936). Listed here, the block skips.
        b.bind_on = false;
        b.bind_moe_prof = null;
        b.fast = null;
        b.cache = null;
        b.mtp_dense = null;
        b.side = null;
        b.l2pf = null;
        b.sp = null;
        b.kv8 = null;
        b.dcal = null; // 125.43e: the 0xAA class (the block's own receipt above)
        b.dlog = null;
        b.dspark_cfg = null;
        b.dflash_cfg = null;
        b.dspark_set = .{ .depth = 0 };
        b.plan = .{};
        b.dspark_dr = null;
        b.dflash_dr = null;
        b.multi = null;
        b.mode = try o.defaultMode();
        b.dcp = try o.dcpFor();
        errdefer b.undo();
        try b.bringUp();
        return b;
    }

    fn bringUp(b: *Boot) !void {
        const gpa = b.gpa;
        const io = b.io;
        const o = b.o;
        if (o.rank >= o.world) return error.BadRank;
        try checkParallel(o, b.dcp);
        const det16 = std.mem.eql(u8, o.pe_det, "slots16");
        if (!det16 and !std.mem.eql(u8, o.pe_det, "0")) return error.UnsupportedPromptDet;
        const shared = std.mem.eql(u8, o.experts, "shared");
        if (!shared and !std.mem.eql(u8, o.experts, "tf")) return error.UnsupportedExperts;
        b.driver = try cuda.Driver.open();
        b.stage = 1;
        b.ctx = try cuda.Context.init(&b.driver, 0);
        b.stage = 2;
        const master = try cuda.rendezvous.parseMaster(o.master);
        b.lib = try cuda.nccl.Library.openPath(o.nccl_lib);
        b.stage = 3;
        b.nccl_version = try b.lib.version();
        b.rdv = try cuda.rendezvous.Rendezvous.open(io, master.ip, master.port, o.rank, o.world, o.timeout_s);
        b.stage = 4;
        var uid: cuda.nccl.UniqueId = undefined;
        if (o.rank == 0) uid = try cuda.nccl.Communicator.uniqueId(&b.lib);
        try b.rdv.broadcast(std.mem.asBytes(&uid));
        b.comm = try cuda.nccl.Communicator.init(&b.lib, uid, o.rank, o.world);
        b.stage = 5;
        std.log.info("glm53 rank {d} of {d}: NCCL {d} up; context {d}, MTP drafts {d}, DCP {d}, prompt reuse {s}", .{ o.rank, o.world, b.nccl_version, o.context, o.k, b.dcp, if (o.reuse.on) "on" else "off" });
        b.cfg = try Config.load(gpa, io, o.model);
        // Phase 4: the drafters' configs, checked against the target before anything big loads; the tap plan
        // (dspark_host.tap_plan: DFlash2's layers first, DSpark's after)
        var dh = std.hash.Wyhash.init(0x44);
        var dspark_layers: [dhost.max_taps]usize = undefined;
        var want: [2]?[]const usize = .{ null, null };
        if (o.dflash.len > 0) {
            const text = try readText(gpa, io, o.dflash, "config.json");
            defer gpa.free(text);
            const c = try dhost.DFlashConfig.parse(gpa, text);
            try c.check(o.world, b.cfg.hidden, b.cfg.layers);
            b.dflash_cfg = c;
            want[0] = b.dflash_cfg.?.taps[0..c.n_taps];
            dh.update(std.mem.asBytes(&c.crc));
            o.dflash_set.digest(&dh);
        }
        if (o.dspark.len > 0) {
            const text = try readText(gpa, io, o.dspark, "config.json");
            defer gpa.free(text);
            const c = try dhost.DSparkConfig.parse(gpa, text);
            try c.check(o.world, b.cfg.hidden, b.cfg.vocab, b.cfg.layers);
            b.dspark_cfg = c;
            b.dspark_set = try dhost.DSparkSettings.parse(c.block, o.dspark_env);
            want[1] = c.tapLayers(&dspark_layers);
            dh.update(std.mem.asBytes(&c.crc));
            b.dspark_set.digest(&dh);
        }
        b.plan = try dhost.TapPlan.make(want);
        try checkParallelDrafters(o.parallel, if (b.dflash_cfg) |c| c.block else null, if (b.dspark_cfg) |c| c.block else null);
        const mode_word: u64 = @backingInt(b.mode);
        dh.update(std.mem.asBytes(&mode_word));
        // a quantized head's bytes and the kv_b setting join the record only when on: off, the digest is unchanged
        const head_id = try headq.identityOf(gpa, io, o.head_q, o.model, b.cfg.vocab, b.cfg.hidden);
        if (head_id != 0) dh.update(std.mem.asBytes(&head_id));
        if (o.kvb_q) dh.update("kvb_q");
        try b.rdv.sync("settings", settingsDigest(o, b.dcp, dh.final()), o.timeout_s);

        // The env-combination preflight: refuse BEFORE the weights load, not after 300 s of it
        // (the user, 2026-10-18: "i would expect the dry runs to catch these, not waste load time").
        if (o.kv8_v2 and !o.kv8) {
            std.log.err("TF_GLM53_KV8_V2 (the in-kernel dequant prompt path) needs TF_GLM53_KV8=1", .{});
            return error.Kv8V2NeedsKv8;
        }
        // The SP chain composes at dcp < world since findings-dcp2-build.md §D: its E1/E3 take the
        // subgroup p2p (gatherFrom / pairExchange on the side stream, the same fixed member order).
        if (o.fp8_index and b.dcp > 1) {
            std.log.err("TF_GLM53_FP8_INDEX needs decode context parallelism off (DCP 1): unset TF_GLM53_DCP / stay under {d} tokens of context", .{dcpm.dcp_auto_tokens});
            return error.Fp8IndexNeedsDcp1;
        }
        const t_load = now(io);
        b.dumpPhase("PRE-LOAD"); // 128.27w: the fixed bucket before the weights (contexts, NCCL, driver)
        // §98 THE BINDER ADAPTER (§95.4): TF_GLM53_BIND=<pack> loads the NVFP4-CSF pack through the 1.1
        // loader + the module binder (the run lane's proven path: same loader, same alias table, the E2
        // hook's §92 serve-rows scratch) instead of r2's native W.load. `w` is the same type; the loaded
        // 1.1 state stays on the Boot (its buffers are what w points at).
        if (env("TF_GLM53_BIND")) |pack| {
            b.w = try b.bindLoad(pack);
        } else {
            b.w = try W.load(gpa, io, &b.driver, o.model, b.cfg, .{ .layers = b.cfg.layers, .rank = o.rank, .world = o.world, .head = true, .log_every = 10, .fast = o.fast_load, .mtp = o.k > 0, .draft_vocab = if (o.k > 0) o.draft_vocab else 0, .head_q = o.head_q, .head_host = o.head_q.on() and b.plan.n > 0, .kvb_q = o.kvb_q });
        }
        b.stage = 6;
        if (o.tiles) |path| {
            if (b.bind_on) return error.BindNoTiles; // the binder's dense linears are not tile-patched
            const t = try tiles.load(gpa, io, &b.w, path);
            std.log.info("glm53 rank {d}: tiles from {s}: {d} linears, {d} differ from x3linear.plan", .{ o.rank, path, t.linears, t.changed });
        }
        if (b.plan.n > 0) { // headq.prepare(drafts=True): the 4-bit draft copy of this rank's head share
            b.w.draft_lm = try draft.buildDraftHead(gpa, &b.driver, &b.w);
        }
        b.load_s = since(io, t_load);
        try b.rdv.sync("loaded", 0, o.timeout_s);

        b.k = try kern.Kernels.load(gpa, io, &b.ctx, o.aot);
        b.stage = 7;
        if (b.w.head_q) |hq| if (!b.k.haveWidth(hq.lin.k2)) {
            std.log.err("glm53 rank {d}: this build's glm53_exl3_linear fatbin has no {d}-bit linear kernel for the quantized head (rebuild the fatbins: buildspeed.sh fatbins)", .{ o.rank, hq.lin.k2 / 2 });
            return error.UnsupportedExl3Width;
        };
        b.stream = try cuda.Stream.init(&b.driver, true);
        b.stage = 8;
        const half = b.cfg.rope / 2;
        b.inv = try cuda.DeviceBuffer.alloc(&b.driver, half * 4);
        b.stage = 9;
        if (b.bind_on and b.bind_pw.inv != 0) {
            try b.driver.check(b.driver.api.cuMemcpyDtoDAsync_v2(b.inv.ptr, b.bind_pw.inv, half * 4, b.stream.handle), "cuMemcpyDtoDAsync");
        } else {
            try b.k.invFreq(b.stream, b.inv.ptr, b.cfg.rope, b.cfg.rope_theta);
        }
        try b.stream.synchronize();
        if (o.inv_ref) |path| {
            const ref = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 16));
            defer gpa.free(ref);
            const mine = try gpa.alloc(u8, half * 4);
            defer gpa.free(mine);
            try b.inv.download(0, mine);
            if (ref.len != mine.len) return error.BadInvReference;
            if (!std.mem.eql(u8, ref, mine)) {
                std.log.warn("glm53: inv_freq differs from torch's (the reference's bytes are used)", .{});
                try b.inv.upload(0, ref);
            }
        }
        if (o.roce and o.world > 1) {
            var hb: [2][]const u8 = undefined;
            const hcas = parseHcas(o.hcas, &hb);
            b.fast = try roce.Roce.start(gpa, &b.driver, &b.rdv, o.rank, o.world, .{ .hcas = hcas, .gid_index = o.gid, .spin_limit = o.spin_limit });
            if (!try b.fast.?.verify(gpa, &b.comm, b.stream, b.cfg.hidden, fw.decode_rows)) return error.RoceCheckFailed;
        }
        b.stage = 10;
        b.blas_lib = try cuda.cublas.Library.openPath(o.cublas_lib);
        b.stage = 11;
        const major: c_int = @intCast((try b.ctx.capability()) / 10);
        const ws_bytes = o.cublas_ws orelse cuda.cublas.torchWorkspaceBytes(major, null);
        b.ws = try cuda.DeviceBuffer.alloc(&b.driver, @max(ws_bytes, 16));
        // Â§125.37: the dense-lane A/B arm - the lane (sm121: clusters on) + the fp32 partials
        // scratch at the max decode gate (sk<=8, R<=64, n<=6144 -> 12.6 MB; 16 MB carved).
        if (o.dense_lane) {
            b.qlane = try cuda.qmmf.Lane.load(&b.driver, 12);
            b.qpart = try cuda.DeviceBuffer.alloc(&b.driver, 16 << 20);
        }
        b.stage = 12;
        b.blas = try cuda.cublas.Blas.init(&b.blas_lib, b.ws.ptr, ws_bytes, o.cublas_math);
        b.stage = 13;
        if (o.bmm_probe) |path| if (!try bmmProbe(gpa, io, &b.driver, b.stream, &b.blas, path)) return error.BmmBitsDiffer;
        if (o.unpack_mb > 0) b.cache = try prompt.UnpackCache.init(gpa, &b.driver, o.unpack_mb << 20);
        b.main_wide = try prompt.Wide.init(&b.driver, &b.k, b.stream, &b.w, o.prompt_rows, if (b.cache) |*c| c else null, &b.blas);
        b.stage = 14;
        b.side_stream = try cuda.Stream.init(&b.driver, true);
        b.stage = 15;
        b.side_wide = try prompt.Wide.init(&b.driver, &b.k, b.side_stream, &b.w, o.prompt_rows / 2 + 1, null, null);
        b.stage = 16;
        try b.stream.synchronize();
        try b.side_stream.synchronize();
        if (o.mtp_dense and o.k > 0 and shared and b.w.mtp != null) {
            if (prompt.MtpDense.init(gpa, &b.driver, &b.k, b.stream, &b.w, &b.blas, o.prompt_rows)) |x| {
                b.mtp_dense = x;
            } else |e| std.log.warn("glm53 rank {d}: MTP prompt rows dense unavailable ({t})", .{ o.rank, e });
        }
        // TF_GLM53_SIDE's letters default ON ("af"); TF_GLM53_KV8 turns the side stream off instead of refusing
        // the boot (findings-kv8.md §9): the letters are decode-window overlap only - the same kernels with the
        // same bits on a second stream - so off costs speed, never quality, and an operator should not need to
        // know the default to boot the opt-in cache mode. The staged kv8 path is simply not VERIFIED against the
        // side fork yet (§9.3); flip this off (and drop the digest normalization below) to run that gate.
        var letters = try dec.parseSide(o.side);
        var kv8_side_off = false;
        if (o.kv8 and letters.any()) {
            kv8_side_off = true;
            letters = .{ .a = false, .w = false, .f = false };
        }
        if (letters.any()) b.side = try dec.Side.init(&b.driver, letters);
        const pf_mode = try l2pfMode(o.l2pf);
        if (pf_mode >= 0) {
            if (!(o.l2pf_mb > 0 and o.l2pf_mb <= 64)) return error.BadL2pfBudget;
            b.l2pf = try dec.L2pf.init(gpa, &b.driver, &b.w, .{ .mode = pf_mode, .mb = o.l2pf_mb, .after = o.l2pf_after });
        }
        // TF_GLM53_KV8: the staged quantize / dequantize side buffers. v1 limits: DCP 1 (the _attn_dcp read path is
        // not staged yet; vLLM's fp8 MLA carries the same constraint on its DCP path). The sequence-parallel prompt
        // chunks are RESTORED under kv8 (findings-prefill-sp-restore.md): the halves stage through the side
        // stream's OWN Kv8 (Sp.init), so frontAttn's side-stream cache writes no longer race the main's decode.
        // The decode side letters stay auto-off above (the side stream is the SP comm stream during prompts).
        const capacity = o.context + o.k + 1; // Glm53Engine: Runner(limit + k + 1)
        if (o.kv8) {
            // The kv8+DCP port is in (findings-dcp8.md §1: the stage tables' DCP compensation, the
            // quant's shard dest, the dequant mode-3 mirror; gate-10 a/b/c PASS) - the Kv8NeedsDcp1
            // refusal is LIFTED; the boot itself gates the serve path (the 500Kx4 incremental first).
            if (kv8_side_off) std.log.info("glm53 rank {d}: side stream off (kv8): TF_GLM53_SIDE '{s}' ignored - the side stream is the sequence-parallel comm stream during prompts, and the decode side fork stays unverified against the staged path (findings-kv8.md §9.3)", .{ o.rank, o.side });
            // v2: the wide windows read the plane itself, so the staged path's capacity-sized prompt mirror is dead
            // weight - not allocated. The decode scratch and the staging stay (the decode path is still staged).
            // DCP + several streams (findings-dcp8.md section 10, bug B): the multi windows' mode-4 mirror
            // covers the whole slots plane - its row count, beside the caches (0: single stream, mode 3 only).
            // The parity-mirror (findings-dcp2-build.md §C) can need MORE scratch rows than the one-stream
            // full-context plane: at DCP 2 x 2 streams each rank holds slots x ceil(capacity / dcp) = the whole
            // capacity (plus the per-shard +1s), 600002 > 600001 at 600K - the scratch must cover the max.
            const kv_plane: usize = if (o.parallel > 1 and b.dcp > 1) dcpm.localCapacity(capacity, b.dcp) * o.parallel else 0;
            var kv = try fw.Kv8.init(gpa, &b.driver, b.cfg, @max(o.prompt_rows, fw.max_rows), fw.max_rows, if (o.kv8_v2) 0 else @max(dcpm.localCapacity(capacity, 1), kv_plane));
            if (kv_plane > 0) {
                kv.plane_rows = kv_plane;
                kv.local = dcpm.localCapacity(capacity, b.dcp);
            }
            const rows_cap = dcpm.localCapacity(capacity, 1);
            const per_tok = fw.State.bytesForDcpV3(&b.w, capacity, 1, true, o.kv8_v3) / rows_cap;
            const per_tok16 = fw.State.bytesForDcpKv8(&b.w, capacity, 1, false) / rows_cap;
            std.log.info("glm53 rank {d}: 8-bit KV on (TF_GLM53_KV8{s}): e4m3 latent + {s}, index keys bf16 - {d:.1} KB a token of caches ({d:.0}% of bf16), staging + scratch {d:.0} MiB", .{ o.rank, if (o.kv8_v3) " v3: the rope bf16 exact, a scale a 128-block (656 B)" else "", if (o.kv8_v3) "4 fp32 block scales" else "2 fp32 scales", @as(f64, @floatFromInt(per_tok)) / 1024.0, per_tok * 100 / per_tok16, @as(f64, @floatFromInt(kv.bytes())) / (1 << 20) });
            // 119.1: the paged base (the kv8-v3 memo's step 2/3) - the latent plane's rows through a
            // 512-row page table. shuffle: a fixed-seed permutation (the same every rank), so the serve
            // anchors under it prove no reader bypasses the table. One stream only (the multi paths -
            // rows_mode quant, dequant modes 2/4 - keep the direct addressing; the global table is the
            // v2 seam's follow-up).
            kv.v3 = o.kv8_v3;
            if (o.parallel <= 1) {
                if (std.c.getenv("TF_GLM53_KV_PAGES")) |m| {
                    if (std.mem.eql(u8, std.mem.span(m), "shuffle")) {
                        try kv.pages(dcpm.localCapacity(capacity, @max(b.dcp, 1)));
                        std.log.info("glm53 rank {d}: the latent plane on shuffled 512-row pages (TF_GLM53_KV_PAGES=shuffle) - {d} pages a layer", .{ o.rank, (dcpm.localCapacity(capacity, @max(b.dcp, 1)) + 511) / 512 });
                    }
                }
            }
            if (o.kv8_v2) std.log.info("glm53 rank {d}: kv8 v2 on (TF_GLM53_KV8_V2): the prompt kernels dequantize the plane in-kernel - no mirror, no per-chunk dequant pass; needs the v2 AOT variants of _attn_chunks (the recapture plan, findings-kv8-v2.md §5); the decode path stays staged", .{o.rank});
            b.kv8 = kv;
        }
        // TF_GLM53_PROMPT_PROF: the Prof exists before the Forward (which points at it); `on` stays false through
        // the prewarm below (its captures must hold no events) - the served fill paths turn it on per chunk
        if (o.prompt_prof) {
            b.prof = prompt.Prof.init(gpa, &b.driver, b.stream);
            std.log.info("glm53 rank {d}: prompt prof on (TF_GLM53_PROMPT_PROF): per-chunk and per-prompt phase reports on every rank (a chunk-boundary sync while on; not in the settings digest - no numeric change)", .{o.rank});
        }
        if (o.topk_stream) std.log.info("glm53 rank {d}: prefill top-k streaming on (TF_GLM53_TOPK_STREAM): the prompt blocks' selection in 2-3 passes over the score row (the radix's picks, bit-identical - findings-prefill-topk.md); decode windows keep the radix; needs the _topk_row AOT variant", .{o.rank});
        if (o.fp8_index) std.log.info("glm53 rank {d}: fp8 index keys on (TF_GLM53_FP8_INDEX): the e4m3 plane inside the ic allocation (128 codes + an fp32 scale a token, 132 bytes), the scoring dequantizes in-kernel (the dot bf16 - the 2x K-read-byte win); every selection switches, the scores change (NOT bit-identical - the numerics test + the quality bench gate it); needs the _ik_write8 / _index_scores8 AOT variants (findings-prefill-fp8-index.md)", .{o.rank});
        if (o.e1_fp8) std.log.info("glm53 rank {d}: e4m3 E1 query wire on (TF_GLM53_E1_FP8): the DCP query gather ships kv8Quant-packed rows (584 B: codes + s0/s1), half the bytes; the scorer dequantizes in-register - the scores change q-side, the needle/coherence gates decide (findings-dcp8.md section 18)", .{o.rank});
        if (o.fp8_index and o.fp8_stride == 144) std.log.info("glm53 rank {d}: padded fp8 index plane on (TF_GLM53_FP8_STRIDE=144): 144-byte token rows - a 16-byte-multiple stride, the native scorer's 16-byte staging path (glm53_fp8_scores_p, 2 CTAs/SM) and TMA-expressible; needs the _ik_write8p / _index_scores8p AOT variants (the recapture); the scores are BIT-IDENTICAL to the 132 layout (only the addresses move)", .{o.rank});
        if (o.prompt_sp and o.world > 1 and !o.fp8_index) std.log.info("glm53 rank {d}: sequence-parallel prompt chunks on (TF_GLM53_PROMPT_SP{s}{s}): the halves' comm chains on the side stream overlapped with the main-stream compute - under kv8 the halves stage through the side stream's OWN Kv8 (the per-stream staging, findings-prefill-sp-restore.md); under DCP the chains split attentionDcp (E1/E3 on the side, the scoring on the main - findings-sp-under-dcp.md section 2)", .{ o.rank, if (o.kv8) " + KV8" else "", if (b.dcp > 1) " + DCP" else "" });
        b.fwd = .{ .w = &b.w, .k = &b.k, .s = b.stream, .inv = b.inv.ptr, .comm = &b.comm, .fast = b.fast, .ring = true, .wide = &b.main_wide, .blas = &b.blas, .qlane = if (b.qlane) |*x| x else null, .qpart = if (o.dense_lane) b.qpart.ptr else 0, .shared_experts = shared, .pe_det16 = det16, .side = if (b.side) |*x| x else null, .l2pf = if (b.l2pf) |*x| x else null, .multi_select = o.multi_select, .topk_stream = o.topk_stream, .fp8_index = o.fp8_index, .fp8_stride = o.fp8_stride, .e1_fp8 = o.e1_fp8, .e3_merge = o.e3_merge, .lane_pick = &b.window_lane, .prof = if (b.prof) |*x| x else null, .mtp_dense = if (b.mtp_dense) |*x| x else null, .dcp = b.dcp, .taps = try fw.Taps.of(&b.plan), .kv8 = if (b.kv8) |*x| x else null, .kv8_v2 = o.kv8_v2 };
        if (b.bind_hook) |*hk| b.fwd.moe_hook = .{ .ctx = hk, .has = binder.Hook.has, .call = binder.Hook.call }; // §98: the port's MoE block over the bound weights
        // SP under DCP: on when the world splits and fp8_index is off (the e4m3 index plane has no DCP port -
        // findings-dcp8.md section 5; the refusal is Sp's chunk, this only avoids building buffers that cannot run)
        if (o.prompt_sp and o.world > 1 and (b.dcp == 1 or !o.fp8_index)) b.sp = try prompt.Sp.init(gpa, &b.driver, &b.fwd, b.side_stream, &b.side_wide);
        b.stage = 17;
        if (o.dcp_prompt_gather and b.dcp > 1) {
            b.pg = try fw.PromptGather.init(&b.driver, &b.w, capacity, b.dcp);
            b.fwd.pg = &b.pg.?;
            std.log.info("glm53 rank {d}: DCP prompt chunks gather the cache prefix ({d:.2} GiB of scratch)", .{ o.rank, @as(f64, @floatFromInt(b.pg.?.bytes())) / (1024.0 * 1024.0 * 1024.0) });
        }
        try b.checkCacheFits(capacity);
        // --parallel N: N slots of the caches; copy drafts are one stream's (Runner: slots == 1)
        const copy_set: copies.Settings = if (o.parallel > 1) .{ .on = false, .match = 0, .most = 0 } else o.copy;
        b.r = try rn.Runner.init(gpa, io, &b.driver, b.fwd, capacity, .{ .k = o.k, .graphs = o.graphs, .window = 128, .reuse = o.mtp_reuse, .served = true, .prompt_rows = o.prompt_rows, .prompt_rows_short = o.prompt_rows_short, .dcp_prompt_rows = o.dcp_prompt_rows, .prompt_sp = o.prompt_sp, .roce_health = o.roce_health, .sp = if (b.sp) |*x| x else null, .device_cands = o.device_cands, .dcp = b.dcp, .verify_rows = o.verify_rows, .draft_rows = if (b.dspark_cfg != null) b.dspark_set.depth + 1 else 0, .copy = copy_set, .slots = o.parallel, .kv8 = o.kv8, .kv8_v3 = o.kv8_v3, .pool_agg = o.pool_agg }); // 126.6r: the RUNTIME depth keys the verify-window enumeration (not the checkpoint's block - the 126.6(i) trap); at depth 6 the graphs capture 7-row windows; the 21 bound (d6 x numseq-3) is the multi-stream seam's guard
        b.stage = 18;
        // Phase 4: the drafters on every rank (each its heads / MLP share), attached to the runner before its prewarm
        if (b.dflash_cfg != null) {
            b.dflash_dr = try draft.Drafter.load(gpa, io, &b.driver, &b.k, b.stream, &b.w, o.dflash, .dflash, b.dspark_set, o.dflash_set);
            try b.dflash_dr.?.setTaps(b.plan.cols[0][0..b.plan.n_cols[0]], b.plan.n);
            b.r.dflash = b.dflash_dr;
            std.log.info("glm53 rank {d}: DFlash2 drafter {s} (block {d}, taps after layers {any}, {d:.0} MiB on the device)", .{ o.rank, std.fs.path.basename(o.dflash), b.dflash_dr.?.sp.block, b.dflash_cfg.?.taps[0..b.dflash_cfg.?.n_taps], @as(f64, @floatFromInt(b.dflash_dr.?.deviceBytes())) / (1 << 20) });
        }
        if (b.dspark_cfg != null) {
            b.dspark_dr = try draft.Drafter.load(gpa, io, &b.driver, &b.k, b.stream, &b.w, o.dspark, .dspark, b.dspark_set, o.dflash_set);
            try b.dspark_dr.?.setTaps(b.plan.cols[1][0..b.plan.n_cols[1]], b.plan.n);
            b.r.dspark = b.dspark_dr;
            std.log.info("glm53 rank {d}: DSpark drafter {s} (block {d}, policy {t}, confidence {d}, depth {d}, {d:.0} MiB on the device)", .{ o.rank, std.fs.path.basename(o.dspark), b.dspark_cfg.?.block, b.dspark_set.policy, b.dspark_set.confidence, b.dspark_set.depth, @as(f64, @floatFromInt(b.dspark_dr.?.deviceBytes())) / (1 << 20) });
        }
        if (o.parallel > 1) {
            b.multi = try multi.Multi.init(gpa, io, &b.r, o.parallel, o.multi_set, o.rank == 0);
            const per = b.multi.?.bytesPerStream();
            std.log.info("glm53 rank {d}: {d} concurrent streams (MTP drafts {d}, windows up to {d} rows), {d:.2} GiB of caches each ({d} tokens); draft cut {d} from {d} active streams", .{ o.rank, o.parallel, o.k, b.multi.?.rows, @as(f64, @floatFromInt(per)) / (1 << 30), o.context, o.multi_set.draft_cut, o.multi_set.cut_streams });
            if (b.multi.?.frows > 0) std.log.info("glm53 rank {d}: concurrent drafters: DFlash2 {s}, DSpark {s}; up to {d} drafts a stream at {d} streams (adaptive: by the round's active streams), confidence floor {d} from {d} active streams; requests draft with {t}", .{ o.rank, if (b.dflash_dr != null) "on" else "off", if (b.dspark_dr != null) "on" else "off", b.multi.?.frows - 1, o.parallel, o.multi_set.cutDFlash(), o.multi_set.cut_streams, b.mode });
        }
        // 125.43e: the DSpark calibration + the walked-tree log - upstream #280's answer to the
        // target-numerics acceptance shift (the dense lane's sampled fall, 125.43): the MEASURED
        // landing table by (depth, ln confidence) gates the walk's continuation (the scalar floor
        // stays the threshold; the head's estimate stands when no table covers the regime); the
        // log emits the py contract's records for tools/fit_draft_calibration.py. The calibration
        // is in the settings digest: a rank without it refuses. OUTSIDE the parallel branch: the
        // DCP2 lane runs single-stream (the runner's own draft path) - the wiring must arm there.
        if (o.dspark_cal) |path| {
            b.dcal = try dhost.CalTables.load(gpa, io, path);
            if (b.dspark_dr) |dr| dr.chain.cal = &b.dcal.?;
            std.log.info("glm53 rank {d}: DSpark calibration on (TF_GLM53_DSPARK_CAL {s}): the measured landing table gates the walk ({s} greedy / {s} sampled)", .{ o.rank, path, if (b.dcal.?.greedy != null) "yes" else "no", if (b.dcal.?.sampled != null) "yes" else "no" });
        }
        if (o.dspark_log) |path| {
            if (o.rank == 0) {
                // 125.43e round 2: the log covers BOTH planes - the multi (the non-DCP streams) and
                // the Runner's single-stream path (the DCP2 lane's own form; the requests get their
                // own tree-stream ids so the tool's grouping never merges prompts).
                b.dlog = dlogmod.DraftLog.init();
                if (b.multi) |mu| {
                    mu.dlog = &b.dlog.?;
                    for (mu.md) |m| {
                        if (m) |x| x.dlog = &b.dlog.?;
                    }
                } else {
                    b.r.dlog = &b.dlog.?;
                }
                std.log.info("glm53 rank {d}: DSpark draft log on (TF_GLM53_DSPARK_LOG {s}): the walked trees + the kept picks ({s} plane), tools/fit_draft_calibration.py's contract", .{ o.rank, path, if (b.multi != null) "multi" else "single-stream" });
            }
        }
        b.stage = 19;
        b.dumpPhase("PRE-PREWARM"); // 128.27w: + the weights, caches, drafter — the prewarm (the graph term) follows
        if (o.prewarm) {
            const tp = now(io);
            if (b.multi) |mu| {
                // Runner.prewarm(capture=False) then the concurrent windows (multi.prewarm)
                _ = try b.r.prewarmWith(false);
                b.prewarm_graphs = try mu.prewarm();
            } else {
                b.prewarm_graphs = try b.r.prewarm();
            }
            b.prewarm_s = since(io, tp);
            // the END-OF-PREWARM device bound (the user's envelope, round 2): >= 11 GiB free with the
            // graphs live - the receipt's own end state was 11.59. The steady-state runtime (NCCL window
            // buffers, request scratch) draws from what is left; a boot that lands below this refuses to
            // serve rather than degrade silently. TF_GLM53_PREWARM_END_FREE_GB overrides the floor.
            const end_free_gb: f64 = if (std.c.getenv("TF_GLM53_PREWARM_END_FREE_GB")) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch 11) else 11;
            const ef = try b.ctx.memInfo();
            // §104: the same GB10 page-cache confound as checkCacheFits (§98.2) — the prewarm-end
            // reading runs right after the load's cache fill (measured 2.59-2.73 GiB "free" vs 8.14
            // across boots of one config, §102.3). The effective free is the host's MemAvailable
            // when it reads higher; the raw reading rides the log for diagnosis.
            var ef_free = ef.free;
            if (memAvailable()) |avail| {
                if (avail > ef_free) ef_free = avail;
            }
            const ef_gib = @as(f64, @floatFromInt(ef_free)) / (1 << 30);
            const raw_gib = @as(f64, @floatFromInt(ef.free)) / (1 << 30);
            if (raw_gib < ef_gib)
                std.log.info("glm53 rank {d}: prewarm end free read {d:.2} GiB raw, {d:.2} GiB effective (the load's page cache held; §98.2 class)", .{ o.rank, raw_gib, ef_gib });
            std.log.info("glm53 rank {d}: prewarm end state: {d} decode graphs, {d:.2} GiB device free (floor {d:.0})", .{ o.rank, b.prewarm_graphs, ef_gib, end_free_gb });
            // 126t6 the non-funnel receipt: the device's used (total - free) vs the funnel's live
            // bytes - the difference is the CUDA-fixed block outside DeviceBuffer.alloc (the contexts,
            // NCCL buffers, the graph capture pools, the driver reserve). The recovery mandate's target.
            if (cuda.envOn("TF_GLM53_ALLOC_DUMP")) {
                const fb_gib = @as(f64, @floatFromInt(cuda.deviceBytes())) / (1 << 30);
                const used_gib = @as(f64, @floatFromInt(ef.total -| ef.free)) / (1 << 30);
                std.log.info("alloc-dump: prewarm end device used {d:.2} GiB, funnel live {d:.2} GiB, NON-FUNNEL {d:.2} GiB (contexts, NCCL, graph pools, driver)", .{ used_gib, fb_gib, used_gib - fb_gib });
            }
            if (ef_gib < end_free_gb) {
                std.log.err("glm53 rank {d}: REFUSED - prewarm ended at {d:.2} GiB device free, below the {d:.0} GiB end-state floor: the graph/pool footprint must shrink (findings-dcp2-build.md §F.5a)", .{ o.rank, ef_gib, end_free_gb });
                return error.PrewarmEndFreeLow;
            }
        }
        try b.rdv.sync("warm", b.prewarm_graphs, o.timeout_s);
        std.log.info("glm53 rank {d}: loaded in {d:.0} s, {d} decode graphs captured in {d:.1} s{s}", .{ o.rank, b.load_s, b.prewarm_graphs, b.prewarm_s, if (b.dcp > 1) "; decode context parallel" else "" });
    }

    /// §98: the host's MemAvailable (reclaimable-adjusted) from /proc/meminfo, or null when unreadable.
    fn memAvailable() ?u64 {
        var buf: [4096]u8 = undefined;
        const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY });
        if (fd < 0) return null;
        defer _ = std.c.close(fd);
        const n = std.c.read(fd, &buf, buf.len);
        if (n <= 0) return null;
        const text = buf[0..@intCast(n)];
        const ix = std.mem.indexOf(u8, text, "MemAvailable:") orelse return null;
        const rest = text[ix + "MemAvailable:".len ..];
        const line_end = std.mem.indexOfScalar(u8, rest, '\n') orelse return null;
        const val = std.mem.trim(u8, rest[0..line_end], " kB\r\t");
        const kb = std.fmt.parseInt(u64, val, 10) catch return null;
        return kb * 1024;
    }

    /// Runner._check_cache_fits before the caches allocate (every rank; Python's message on the refusing rank).
    fn checkCacheFits(b: *Boot, capacity: usize) !void {
        const mem = try b.ctx.memInfo();
        // §98: GB10 unified memory — cuMemGetInfo.free counts the reclaimable page cache as held: the boot's
        // own NFS load fills it (this check measured 1.6 GiB "free" with 60+ GiB of clean cache the kernel
        // yields to real allocations — the same false-refusal class as findings-kv8.md §10). The effective
        // free is the host's MemAvailable when it reads higher; the `held` diagnostic below keeps the raw
        // reading.
        var free = mem.free;
        if (memAvailable()) |avail| {
            if (avail > free) free = avail;
        }
        const reserve_gb: f64 = if (std.c.getenv("TF_GLM53_CACHE_RESERVE_GB")) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch 6) else 6;
        const row: u64 = b.cfg.layers * fw.State.latentRowV3(b.o.kv8, b.o.kv8_v3, b.cfg);
        const local: u64 = dcpm.localCapacity(capacity, b.dcp);
        // TF_GLM53_KV8's side buffers (staging, the decode scratch, the prompt mirror) sit beside the caches
        // the side stream's own kv8 staging (SP under kv8) sits beside the caches too
        const extra: u64 = (if (b.kv8) |*v| v.bytes() else 0) + (if (b.sp) |*sp| (if (sp.kv8) |kv| kv.bytes() else 0) else 0);
        const pool_agg: ?usize = blk: {
            const e = std.c.getenv("TF_GLM53_POOL_AGG") orelse break :blk null;
            break :blk std.fmt.parseInt(usize, std.mem.sliceTo(e, 0), 10) catch null; // the empty-string trap: the launcher's unset -e values parse-fail to null
        };
        if (b.o.parallel > 1 and (b.dcp > 1 or pool_agg != null)) {
            // several streams under DCP: the whole of what the slots allocate (index keys and the MTP layer too),
            // against a reserve that is not halved when little is free - a node that runs out here does not fail,
            // it swaps or panics
            // the default grows with the streams (the windows, the graphs and the rings do): 20 GiB at two, 40 at four.
            // Measured 2026-10-12: two streams took 14 to 21 GiB after this check; four at 125K (the caches of 250K x 2)
            // ran a node down to 4 GB with 24 GiB left here, so four streams are refused until that is measured smaller.
            const sgb_default: f64 = dcpm.streams_reserve_gb * @as(f64, @floatFromInt(b.o.parallel)) / 2;
            const sgb: f64 = if (std.c.getenv("TF_GLM53_DCP_STREAMS_RESERVE_GB")) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch sgb_default) else sgb_default;
            // 126t5: the end-state floor JOINS the fit-check — the reserve is S-only (the post-check
            // spend: graphs, the cache pool, windows, the host growth, measured 17.3 at 1-2 streams),
            // and the floor (11) keeps the prewarm/graph-capture off the memory wall. The 3x316K wedge
            // receipt: reserve 25 passed the fit, the capture wedged at ~8.4 free.
            const s_floor: f64 = if (std.c.getenv("TF_GLM53_PREWARM_END_FREE_GB")) |v| (std.fmt.parseFloat(f64, std.mem.span(v)) catch 11) else 11;
            const per: u64 = fw.State.bytesForDcpV3(&b.w, if (pool_agg) |agg| agg else capacity, b.dcp, b.o.kv8, b.o.kv8_v3);
            const sf = dcpm.streamsFit(per, if (pool_agg != null) 1 else b.o.parallel, free, sgb + s_floor);
            const gib_f: f64 = 1 << 30;
            std.log.info("glm53 rank {d}: {d} streams x {d} tokens, cache split over {d} ranks: {d:.1} GiB of caches here, {d:.1} GiB free, {d:.1} GiB left after them (reserve {d:.1} = S {d:.1} + floor {d:.1})", .{ b.o.rank, b.o.parallel, capacity, b.dcp, @as(f64, @floatFromInt(sf.need)) / gib_f, @as(f64, @floatFromInt(sf.free)) / gib_f, (@as(f64, @floatFromInt(sf.free)) - @as(f64, @floatFromInt(sf.need))) / gib_f, sgb + s_floor, sgb, s_floor });
            if (!sf.ok) {
                std.log.err("context {d} x {d} streams needs {d:.1} GiB of caches on this rank and {d:.1} GiB is free; {d:.1} GiB must stay free (TF_GLM53_DCP_STREAMS_RESERVE_GB): use --context <= {d} with --parallel {d}, TF_GLM53_MTP_DENSE=0, or fewer streams", .{ capacity, b.o.parallel, @as(f64, @floatFromInt(sf.need)) / gib_f, @as(f64, @floatFromInt(sf.free)) / gib_f, sgb, sf.fit_tokens(capacity), b.o.parallel });
                return error.CacheDoesNotFit;
            }
            return;
        }
        const x = cacheFit(row, local, b.o.parallel, b.dcp, extra, free, reserve_gb) orelse return;
        const gib = 1 << 30;
        std.log.err("context {d} x {d} streams needs {d:.1} GiB of caches a rank, {d:.1} of {d:.1} GiB is free (keeping {d:.0} GiB spare): use --context <= {d} with --parallel {d}, or fewer streams", .{ capacity, b.o.parallel, @as(f64, @floatFromInt(x.need)) / gib, @as(f64, @floatFromInt(x.free)) / gib, @as(f64, @floatFromInt(mem.total)) / gib, @as(f64, @floatFromInt(x.spare)) / gib, x.fit, b.o.parallel });
        // the 2026-10-16 anomalies (findings-kv8.md §10): a boot racing a live or not-yet-released previous
        // serve measures ~40 GiB less free and refuses shapes the node fits clean. A clean node holds ~67-70 GiB
        // at this check (weights 47.5 + the MTP layer's dense experts 18 + the wide / kv8 side buffers); much
        // more than that is somebody else's allocation.
        const held: u64 = mem.total -| mem.free;
        if (held > 90 * gib) std.log.warn("the device reports {d:.1} GiB held - far past this boot's ~67 GiB of weights and side buffers: another serve's caches (or its page cache) are likely still allocated - check every node (docker ps, nvidia-smi) before rebooting smaller", .{@as(f64, @floatFromInt(held)) / gib});
        return error.CacheDoesNotFit;
    }

    /// §98: the 1.1 pack through the loader + the binder, plus the E2 hook's scratch at the serve's prompt
    /// rows (the §92 sizing: moe-only + the shared mm_part, rows_max = o.prompt_rows). Everything the hook and
    /// the bound weights reference is stored on the Boot before the return.
    fn bindLoad(b: *Boot, pack: []const u8) !W.Weights {
        const gpa = b.gpa;
        const io = b.io;
        const o = b.o;
        // §107: the tap drafters LIFTED through the binder — the head rows (pw.head -> w.lm_head, w.vocab_part)
        // feed buildDraftHead at bringUp's plan.n>0 branch (the D2H download + quantizePack path); the tap plan
        // (TapPlan.make) and the Drafter.load/setTaps wiring are weights-agnostic. buildDraftHead's own NoHead
        // guard catches a headless pack cleanly.
        if (o.head_q.on()) return error.BindNoHeadQ; // the binder owns the head (the pack's bf16 rows)
        if (o.kvb_q) return error.BindNoKvbQ;
        if (!std.mem.eql(u8, o.experts, "shared")) return error.BindNeedsShared; // the side table carries the shared expert
        const t0 = now(io);

        // the 1.1 config from the pack's own config.json (the same schema r2's Config parses)
        const json = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(gpa, &.{ pack, "config.json" }), gpa, .limited(1 << 20));
        defer gpa.free(json);
        var cfg = try glm53.config.parse(gpa, json);
        cfg.run = cfg.layers;
        cfg.mtp = if (o.k > 0) 1 else 0;
        const share = try glm53.shard.Share.of(&cfg, o.rank, o.world);

        const asm_ = try gpa.create(glm53.blocks.Assembled);
        errdefer {
            asm_.deinit();
            gpa.destroy(asm_);
        }
        asm_.* = try glm53.blocks.assemble(&b.driver, &b.ctx, false);
        glm53.blocks.rebind(asm_);
        var m_pack = try cuda.Module.load(&b.driver, cuda.kernels.affine4_pack);
        errdefer m_pack.unload();
        var m_quant = try cuda.Module.load(&b.driver, cuda.kernels.glm_quant);
        errdefer m_quant.unload();
        const qk = try glm53.quant.Kernels.load(m_quant);
        const tools: glm53.weights.Tools = .{ .pack_dense = try m_pack.function("tf_pack_dense"), .quant = &qk };
        const shards = try std.fs.path.join(gpa, &.{ pack, "tensors" });
        defer gpa.free(shards);
        var ck = try core.Checkpoint.openModel(gpa, io, shards);
        errdefer ck.close();
        var stream = try cuda.Stream.init(&b.driver, true);
        defer @constCast(&stream).deinit();

        var pw = try glm53.weights.load(gpa, &b.driver, stream, tools, &ck, &cfg, share);
        errdefer pw.deinit();
        try stream.synchronize();
        const load_ms = @divTrunc(now(io) - t0, 1_000_000);
        std.log.info("glm53 rank {d}: BIND pack {s}: 1.1 load {d} ms, layers {d}, mtp {d}", .{ o.rank, std.fs.path.basename(pack), load_ms, cfg.layers, cfg.mtp });

        // §91 (the heads fix): the loaded per-rank MLA head count is the truth; the wide path reads w.heads.
        const loaded_heads = pw.layers[0].attn.heads;
        const side = try gpa.alloc(?glm53.moe.Layer, pw.layers.len);
        errdefer gpa.free(side);
        @memset(side, null);
        var bstats: binder.Stats = .{};
        const w = try binder.bind(gpa, &b.driver, stream, &pw, o.rank, o.world, loaded_heads, loaded_heads, &bstats, side);
        errdefer gpa.free(w.layers);

        // the E2 hook's scratch (§92): only the moe block + the shared mm_part are placed; rows_max = the
        // serve's prompt rows (the wide-path fix - glm53.moe.block writes `rows` rows unblocked).
        const rows_max = @max(128, o.prompt_rows);
        const scores_t = rows_max + 8;
        const blocks_n = (scores_t + glm53.indexer.topk_chunk - 1) / glm53.indexer.topk_chunk;
        var layout = glm53.forward.scratchLayoutNch(&cfg, share, rows_max, scores_t, blocks_n, glm53.latent.chunksFor(scores_t));
        inline for (@typeInfo(@TypeOf(layout)).@"struct".field_names) |fname| {
            if (comptime (std.mem.startsWith(u8, fname, "moe_") or std.mem.eql(u8, fname, "mm_part") or std.mem.eql(u8, fname, "dummy") or std.mem.eql(u8, fname, "at") or std.mem.eql(u8, fname, "total"))) continue;
            @field(layout, fname) = 0;
        }
        layout.total = glm53.forward.arenaTotal(&layout);
        var arena_buf = try cuda.DeviceBuffer.alloc(&b.driver, layout.total);
        errdefer arena_buf.free();
        var arena: glm53.forward.Arena = .{ .base = arena_buf.ptr, .size = layout.total };
        const placed = try glm53.forward.placeBlocks(&arena, layout);
        try arena_buf.fill8(0, null);

        b.bind_asm = asm_;
        b.bind_m_pack = m_pack;
        b.bind_m_quant = m_quant;
        b.bind_ck = ck;
        b.bind_pw = pw;
        b.bind_side = side;
        b.bind_arena = arena_buf;
        var hook = binder.Hook{ .kernels = asm_.blocks.moe, .scratch = placed.moe, .side = side };
        // §99: TF_GLM_PROF (the §87 decode profile) — the moe sections (router/plan/gate_up/down/shared/
        // combine) per report; without it the hook's GPU time lands unmarked inside the round's "idle".
        if (binder.Hook.MoeProf.wanted()) {
            const mprof = try binder.Hook.MoeProf.init(&b.driver, io); // self-allocated; deinit frees it
            mprof.every = 156; // a report every ~2 decode rounds (78 routed layers a pass)
            hook.kernels.prof = mprof;
            hook.prof = mprof;
            b.bind_moe_prof = mprof;
        }
        b.bind_hook = hook;
        b.bind_on = true;
        std.log.info("glm53 rank {d}: BIND layers={d} moe={d} experts={d} wk={d} heads={d}; hook scratch {d:.0} MiB at rows_max {d}", .{ o.rank, bstats.layers, bstats.moe, bstats.experts, bstats.wk, loaded_heads, @as(f64, @floatFromInt(layout.total)) / (1 << 20), rows_max });
        return w;
    }

    fn undo(b: *Boot) void {
        const s = b.stage;
        if (b.multi) |x| x.deinit();
        if (b.dspark_dr) |x| x.deinit();
        if (b.dflash_dr) |x| x.deinit();
        if (s >= 18) b.r.deinit();
        if (s >= 17) if (b.sp) |*x| x.deinit();
        if (s >= 17) if (b.pg) |*x| x.deinit();
        if (s >= 16) {
            if (b.prof) |*x| x.deinit(); // the Runner (undone at 18) is dead; nothing reads the pointer anymore
            if (b.kv8) |*x| x.deinit();
            if (b.l2pf) |*x| x.deinit();
            if (b.side) |*x| x.deinit();
            if (b.mtp_dense) |*x| x.deinit();
            b.side_wide.deinit();
        }
        if (s >= 15) b.side_stream.deinit();
        if (s >= 14) b.main_wide.deinit();
        if (s >= 13) {
            if (b.cache) |*c| c.deinit();
            b.blas.deinit();
        }
        if (s >= 12) b.ws.free();
        if (b.qlane != null) b.qpart.free();
        if (s >= 11) b.blas_lib.close();
        if (s >= 10) if (b.fast) |rc| rc.deinit(b.gpa);
        if (s >= 9) b.inv.free();
        if (s >= 8) b.stream.deinit();
        if (s >= 7) b.k.deinit();
        if (s >= 6) {
            if (b.bind_on) {
                if (b.bind_moe_prof) |mp| mp.deinit(); // self-destroying (page_allocator)
                b.bind_arena.free();
                b.gpa.free(b.bind_side);
                b.bind_pw.deinit();
                b.bind_ck.close();
                b.bind_asm.deinit();
                b.gpa.destroy(b.bind_asm);
                b.bind_m_quant.unload();
                b.bind_m_pack.unload();
                b.bind_on = false;
            }
            b.w.deinit();
        }
        if (s >= 5) b.comm.deinit();
        if (s >= 4) b.rdv.close();
        if (s >= 3) b.lib.close();
        if (s >= 2) b.ctx.deinit();
        if (s >= 1) b.driver.close();
    }

    pub fn deinit(b: *Boot) void {
        b.undo();
        b.gpa.destroy(b);
    }
};

/// Glm53Engine's refusals of --parallel N > 1 (before anything loads): decode context parallelism must be off, the
/// verify window of N x (k + 1) rows must be a decode window. Phase 4c: DSpark and DFlash2 draft concurrent streams
/// too (cuda_mdraft.zig; the drafters' own check, N blocks in one pass, follows once their configs are read).
pub fn checkParallel(o: Options, dcp: usize) !void {
    if (o.parallel <= 1) return;
    const quiet = @import("builtin").is_test;
    if (o.pool_agg != null) return;
    if (dcp > 1) {
        if (!o.dcp_streams) {
            if (!quiet) std.log.err("--parallel {d} needs decode context parallelism off: a context of at most {d} tokens a stream (--context), TF_GLM53_DCP unset or 1 - or TF_GLM53_DCP_STREAMS=1 on every rank (several streams with the cache split over the ranks)", .{ o.parallel, dcp_auto });
            return error.ParallelNeedsDcpOff;
        }
        // nothing larger than the one-stream shape has ever been held: N streams share its tokens
        if (!dcpm.streamsTokensOk(o.context, o.parallel)) {
            if (!quiet) std.log.err("{d} streams x {d} tokens = {d} tokens of cache: at most {d} in all with the cache split over the ranks (--context <= {d} at --parallel {d})", .{ o.parallel, o.context, o.parallel * o.context, dcpm.streams_tokens, dcpm.streams_tokens / o.parallel, o.parallel });
            return error.DcpStreamsTooManyTokens;
        }
        if (o.mtp_reuse != 0 and o.k > 0 and !quiet) std.log.info("several streams: MTP index reuse is off (each MTP step selects its own keys)", .{});
    }
    _ = multi.windowRows(o.parallel, o.k) catch {
        if (!quiet) std.log.err("{d} streams x {d} rows = {d} > {d}: a window that wide is not a decode window (lower --parallel or --mtp-drafts)", .{ o.parallel, o.k + 1, o.parallel * (o.k + 1), fw.decode_rows });
        return error.WindowTooWide;
    };
}

/// Phase 4c: a loaded drafter drafts N concurrent streams when N of its blocks fit one pass (cuda_mdraft.fits:
/// N x block <= 64 rows; block 8: up to 8 streams).
pub fn checkParallelDrafters(parallel: usize, dflash_block: ?usize, dspark_block: ?usize) !void {
    if (parallel <= 1) return;
    const quiet = @import("builtin").is_test;
    for ([_]?usize{ dflash_block, dspark_block }, [_][]const u8{ "TF_GLM53_DFLASH", "TF_GLM53_DSPARK" }) |maybe, name| {
        const block = maybe orelse continue;
        if (parallel * block > mdraft.max_pass_rows or parallel > mdraft.max_streams) {
            if (!quiet) std.log.err("{s} with --parallel {d}: {d} streams x a block of {d} rows > {d} rows a draft pass; use --parallel <= {d}", .{ name, parallel, parallel, block, mdraft.max_pass_rows, mdraft.max_pass_rows / block });
            return error.DrafterTooWide;
        }
    }
}

/// --bmm-probe FILE (tf-glm53-generate's): torch.bmm's bits on fused.attention_core's wide absorb / expand shapes
/// against cuBLAS here with torch's arguments; true when every case matches.
fn bmmProbe(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, blas: *const cuda.cublas.Blas, path: []const u8) !bool {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    var all_ok = true;
    for ((parsed.value.object.get("cases") orelse return error.BadProbe).array.items) |cv| {
        const o = cv.object;
        const kind = o.get("kind").?.string;
        const H: usize = @intCast(o.get("H").?.integer);
        const R: usize = @intCast(o.get("R").?.integer);
        const nope: usize = @intCast(o.get("nope").?.integer);
        const qd: usize = @intCast(o.get("qd").?.integer);
        const lw: usize = @intCast(o.get("lw").?.integer);
        const vd: usize = @intCast(o.get("vd").?.integer);
        const want = o.get("sha256").?.string;
        const absorb = std.mem.eql(u8, kind, "absorb");
        const na = if (absorb) R * H * qd else R * H * lw;
        const nb = if (absorb) H * nope * lw else H * vd * lw;
        const nc = if (absorb) H * R * lw else H * R * vd;
        const ha = try gpa.alloc(u16, na);
        defer gpa.free(ha);
        const hb = try gpa.alloc(u16, nb);
        defer gpa.free(hb);
        probeFill(ha, if (absorb) 1 else 3);
        probeFill(hb, if (absorb) 2 else 4);
        var da = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(ha));
        defer da.free();
        var db = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(hb));
        defer db.free();
        var dc = try cuda.DeviceBuffer.alloc(d, nc * 2);
        defer dc.free();
        if (absorb) {
            try blas.bgemmBf16(s.handle, .n, .n, lw, R, nope, db.ptr, lw, nope * lw, da.ptr, H * qd, qd, dc.ptr, lw, R * lw, H);
        } else {
            try blas.bgemmBf16(s.handle, .t, .n, vd, R, lw, db.ptr, lw, vd * lw, da.ptr, H * lw, lw, dc.ptr, vd, R * vd, H);
        }
        try s.synchronize();
        const hc = try gpa.alloc(u8, nc * 2);
        defer gpa.free(hc);
        try dc.download(0, hc);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(hc, &digest, .{});
        const got = std.fmt.bytesToHex(digest, .lower);
        const ok = std.mem.eql(u8, &got, want);
        all_ok = all_ok and ok;
        std.log.info("glm53 bmm probe {s} H {d} R {d}: {s}", .{ kind, H, R, if (ok) "equal" else "DIFFERS" });
    }
    return all_ok;
}

fn probeFill(out: []u16, salt: u64) void {
    for (out, 0..) |*v, i| {
        const h: u64 = ((@as(u64, i) +% salt *% 1000003) *% 2654435761) & 0xFFFFFFFF;
        const x: f32 = @as(f32, @floatFromInt(@as(i64, @intCast((h >> 8) % 4001)) - 2000)) / @as(f32, 2000.0);
        v.* = W.bf16Bits(x);
    }
}

/// The model's end ids as the native server takes them (hf_text.eosIds): config.json's eos_token_id (an id or a
/// list; text_config's when the top level names none), plus tokenizer_config.json's eos_token when it is an added
/// token of tokenizer.json.
pub fn eosIds(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    if (readJson(a, io, dir, "config.json")) |doc| if (doc == .object) {
        const holders = [_]?std.json.Value{ doc.object.get("eos_token_id"), if (doc.object.get("text_config")) |tc| (if (tc == .object) tc.object.get("eos_token_id") else null) else null };
        for (holders) |h| {
            const v = h orelse continue;
            switch (v) {
                .integer => |i| try out.append(gpa, @intCast(i)),
                .array => |list| for (list.items) |x| if (x == .integer) try out.append(gpa, @intCast(x.integer)),
                else => {},
            }
            if (out.items.len > 0) break;
        }
    };
    if (readJson(a, io, dir, "tokenizer_config.json")) |doc| if (doc == .object) if (doc.object.get("eos_token")) |e| {
        const piece = switch (e) {
            .string => |s| s,
            .object => |o| if (o.get("content")) |c| (if (c == .string) c.string else "") else "",
            else => "",
        };
        if (piece.len > 0) if (readJson(a, io, dir, "tokenizer.json")) |tok| if (tok == .object) if (tok.object.get("added_tokens")) |added| if (added == .array) {
            for (added.array.items) |t| {
                if (t != .object) continue;
                const c = t.object.get("content") orelse continue;
                const id = t.object.get("id") orelse continue;
                if (c == .string and id == .integer and std.mem.eql(u8, c.string, piece)) {
                    const v: u32 = @intCast(id.integer);
                    if (std.mem.indexOfScalar(u32, out.items, v) == null) try out.append(gpa, v);
                }
            }
        };
    };
    return out.toOwnedSlice(gpa);
}

fn readJson(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ?std.json.Value {
    const path = std.fs.path.join(a, &.{ dir, name }) catch return null;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(256 << 20)) catch return null;
    return std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch null;
}

/// One request as every rank runs it (Glm53Engine.generate's header, after rank 0's choices).
pub const Exec = struct {
    prompt: []const u32,
    max_tokens: usize,
    sampling: ?smp.Sampling = null,
    /// false: "draft": false - serial (k = 0), the cold reference: cut alike, never resumed, keeps nothing
    draft: bool = true,
    /// tokens that end the reply (empty: ignore_eos)
    eos: []const u32 = &.{},
    /// prompt reuse (rank 0's PromptReuse.choose): resume point, cuts, kept points (n * 2 + shared), flags
    begin: usize = 0,
    stops: []const usize = &.{},
    keeps: []const u32 = &.{},
    flags: u32 = 0,
    hooks: ?rn.Hooks = null,
    /// Phase 4: the draft mode (null: the boot's default) and copy drafts (drafted requests only)
    mode: ?rn.Mode = null,
    copies: bool = true,
    /// rn.Gen.prompt_votes (the server at --parallel 1, every rank alike)
    prompt_votes: bool = false,
};

pub const Done = struct {
    stats: rn.Stats,
    kept: []usize = &.{}, // keep points this request kept (gpa-owned)
    replay: bool = false,
};

/// A request on one rank: prompt reuse (when on) around Runner.run, as Glm53Engine._run.
pub const Session = struct {
    gpa: std.mem.Allocator,
    boot: *Boot,
    eos: []u32, // the model's end ids (eosIds)
    reuse: ?*reuse.Reuse = null,
    plan: reuse.Plan,

    /// The session over a booted rank: its prompt reuse with the ranks' least budget (every rank saves and drops
    /// alike: the settings were compared at the rendezvous, the budget is agreed here).
    pub fn init(gpa: std.mem.Allocator, boot: *Boot) !Session {
        const o = boot.o;
        var s: Session = .{ .gpa = gpa, .boot = boot, .eos = try eosIds(gpa, boot.io, o.model), .plan = .{ .ids = reuse.specialIds(gpa, boot.io, o.model), .gap = o.reuse.gap, .stride = o.reuse.stride } };
        errdefer gpa.free(s.eos);
        const wanted: u64 = @intFromFloat(o.reuse.gib * (1 << 30));
        var mine: u64 = 0;
        if (o.reuse.on) {
            const mem = try boot.ctx.memInfo();
            mine = reuse.Reuse.keptBudget(mem.free, wanted) >> 20;
        }
        var all: [cuda.rendezvous.max_world * 8]u8 = undefined;
        try boot.rdv.allGather(std.mem.asBytes(&mine), all[0 .. boot.o.world * 8]);
        var least: u64 = std.math.maxInt(u64);
        for (0..boot.o.world) |i| least = @min(least, std.mem.readInt(u64, all[i * 8 ..][0..8], .little));
        if (o.reuse.on and boot.multi != null) {
            // --parallel N: the kept states live in the streams' slots (prefixes.SlotReuse); the one-stream store
            // never runs
            try boot.multi.?.enableReuse(s.plan, least << 20, o.reuse.entries, o.reuse.loose);
            if (o.rank == 0) std.log.info("glm53: prompt reuse on in the {d} streams' slots (never saved out of them): up to {d} entries, keep points every >= {d} tokens", .{ o.parallel, o.reuse.entries, o.reuse.gap });
        } else if (o.reuse.on) {
            s.reuse = try reuse.Reuse.init(gpa, &boot.r, s.plan, least << 20, o.reuse.entries, o.reuse.loose);
            if (o.rank == 0) std.log.info("glm53: prompt reuse on: kept prompt states up to {d:.2} GiB a rank and {d} entries, keep points every >= {d} tokens at assistant openers; <|user|> {?d}, <|assistant|> {?d}, <think> {?d}", .{ @as(f64, @floatFromInt(least << 20)) / (1 << 30), o.reuse.entries, o.reuse.gap, s.plan.ids.user, s.plan.ids.assistant, s.plan.ids.think });
        }
        return s;
    }

    pub fn deinit(s: *Session) void {
        if (s.reuse) |r| r.deinit();
        s.gpa.free(s.eos);
    }

    /// Glm53Engine.generate's clamps: the prompt under the context, max_tokens within what is left.
    pub fn clampTokens(s: *const Session, prompt_len: usize, max_tokens: usize) !usize {
        const limit = s.boot.o.context;
        if (prompt_len == 0 or prompt_len >= limit) return error.PromptTooLong;
        return @max(1, @min(max_tokens, limit - prompt_len));
    }

    /// Rank 0: PromptReuse.choose for a request (no reuse: nothing resumed, nothing cut).
    pub fn choose(s: *Session, prompt_ids: []const u32, drafting: bool) !reuse.Choice {
        const pr = s.reuse orelse return .{ .begin = 0, .stops = &.{}, .keeps = &.{} };
        return reuse.choose(s.gpa, s.plan, &pr.store, prompt_ids, drafting, 0);
    }

    /// Glm53Engine._run on this rank: the request through prompt reuse (when on) and Runner.run. Tokens go to `out`.
    pub fn exec(s: *Session, x: Exec, out: *std.ArrayList(u32)) !Done {
        const k = if (x.draft) s.boot.o.k else 0;
        const mode: rn.Mode = if (x.draft) x.mode orelse s.boot.mode else .mtp;
        const g: rn.Gen = .{ .max_tokens = x.max_tokens, .sampling = x.sampling, .k = k, .eos = x.eos, .hooks = x.hooks, .mode = mode, .copies = x.draft and x.copies, .prompt_votes = x.prompt_votes };
        if (s.reuse) |pr| {
            const res = try pr.run(x.prompt, x.begin, x.stops, x.keeps, x.flags, x.draft, 0, g, out);
            return .{ .stats = res.stats, .kept = res.kept, .replay = res.replay };
        }
        if (x.begin != 0) return error.ReuseOff;
        return .{ .stats = try s.boot.r.run(x.prompt, g, out) };
    }

    // ---------------------------------------------------------------------------------- learned states (--learn) ---

    /// A learned state's file for this rank: <dir>/<key>.r<rank>.bin.
    pub fn statePath(buf: []u8, dir: []const u8, key: u64, rank: usize) ![:0]const u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.r{d}.bin", .{ dir, key, rank }, 0);
    }

    const file_magic: u32 = 0x4c333547; // "G53L"
    const file_version: u32 = 1;
    pub const Head = extern struct { magic: u32, version: u32, rank: u32, world: u32, dcp: u32, n: u32, key: u64, identity: u64, views: u32, carry: u32, bytes: u64 };

    /// This rank's shard of kept state `e` (its rows - saved, or still live - and its carry) to `file` through a
    /// temporary file renamed into place; returns the file's bytes. The rows go through a pinned staging buffer.
    pub fn writeState(s: *Session, e: *reuse.Kept, file: [:0]const u8, key: u64, identity: u64) !u64 {
        const pr = s.reuse orelse return error.ReuseOff;
        const b = s.boot;
        const D = b.cfg.hidden;
        const vs = try pr.views(e.n());
        defer s.gpa.free(vs);
        var src_offs: ?[]usize = null;
        defer if (src_offs) |x| s.gpa.free(x);
        var src_base: u64 = 0;
        if (e.saved) |sv| {
            // the copy's parts sit at its own (longer) state's offsets
            const big = try pr.views(sv.n);
            defer s.gpa.free(big);
            src_offs = try s.gpa.alloc(usize, big.len);
            _ = reuse.savedSize(big, src_offs.?);
            src_base = sv.mem.ptr;
        } else if (!pr.store.inLive(e)) return error.StateGone;
        var total: u64 = 0;
        for (vs) |v| total += v.row * v.rows;
        const head: Head = .{ .magic = file_magic, .version = file_version, .rank = @intCast(b.o.rank), .world = @intCast(b.o.world), .dcp = @intCast(b.dcp), .n = @intCast(e.n()), .key = key, .identity = identity, .views = @intCast(vs.len), .carry = @intCast(D * 2), .bytes = total };
        var tmp_buf: [1200]u8 = undefined;
        const tmp = try std.fmt.bufPrintSentinel(&tmp_buf, "{s}.part", .{file}, 0);
        const fd = std.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.StateWrite;
        var ok = false;
        defer if (!ok) {
            _ = std.c.unlink(tmp);
        };
        {
            defer _ = std.c.close(fd);
            try putAll(fd, std.mem.asBytes(&head));
            for (vs) |v| {
                const sz = [2]u64{ v.row, v.rows };
                try putAll(fd, std.mem.sliceAsBytes(&sz));
            }
            try b.r.f.s.synchronize(); // the copies into the saved rows / the live rows' last chunk have landed
            var stage = try cuda.HostBuffer.alloc(&b.driver, 64 << 20);
            defer stage.free();
            for (vs, 0..) |v, i| {
                const src = if (src_offs) |offs| src_base + offs[i] else v.ptr;
                var done: usize = 0;
                const n = v.row * v.rows;
                while (done < n) {
                    const part = @min(n - done, stage.bytes.len);
                    try b.driver.check(b.driver.api.cuMemcpyDtoH_v2(stage.bytes.ptr, src + done, part), "cuMemcpyDtoH");
                    try putAll(fd, stage.bytes[0..part]);
                    done += part;
                }
            }
            try b.driver.check(b.driver.api.cuMemcpyDtoH_v2(stage.bytes.ptr, e.carry.ptr, D * 2), "cuMemcpyDtoH");
            try putAll(fd, stage.bytes[0 .. D * 2]);
            if (std.c.fsync(fd) != 0) return error.StateWrite;
        }
        if (std.c.rename(tmp, file) != 0) return error.StateWrite;
        ok = true;
        return @sizeOf(Head) + vs.len * 16 + total + D * 2;
    }

    /// This rank's shard of learned state `ids` read back from `file` as a new shared kept state (not yet an entry:
    /// the caller remembers it once every rank has read its own). A file of another rank, world, DCP, size, key or
    /// identity is refused.
    pub fn readState(s: *Session, ids: []const u32, file: [:0]const u8, key: u64, identity: u64) !*reuse.Kept {
        const pr = s.reuse orelse return error.ReuseOff;
        const b = s.boot;
        const D = b.cfg.hidden;
        const fd = std.c.open(file, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.StateRead;
        defer _ = std.c.close(fd);
        var head: Head = undefined;
        try getAll(fd, std.mem.asBytes(&head), 0);
        const vs = try pr.views(ids.len);
        defer s.gpa.free(vs);
        var total: u64 = 0;
        for (vs) |v| total += v.row * v.rows;
        if (head.magic != file_magic or head.version != file_version or head.rank != b.o.rank or head.world != b.o.world or head.dcp != b.dcp or head.n != ids.len or head.key != key or head.identity != identity or head.views != vs.len or head.carry != D * 2 or head.bytes != total) return error.StateRefused;
        const sizes = try s.gpa.alloc([2]u64, vs.len);
        defer s.gpa.free(sizes);
        try getAll(fd, std.mem.sliceAsBytes(sizes), @sizeOf(Head));
        for (vs, sizes) |v, z| if (z[0] != v.row or z[1] != v.rows) return error.StateRefused;
        var stage = try cuda.HostBuffer.alloc(&b.driver, 64 << 20);
        defer stage.free();
        const data_at: u64 = @sizeOf(Head) + vs.len * 16;
        const Fill = struct {
            d: *const cuda.Driver,
            fd: c_int,
            at: u64,
            stage: []u8,
            pub fn fill(f: *const @This(), views: []const reuse.View, offs: []const usize, base: u64) !void {
                var pos = f.at;
                for (views, offs) |v, off| {
                    const n = v.row * v.rows;
                    var done: usize = 0;
                    while (done < n) {
                        const part = @min(n - done, f.stage.len);
                        try getAll(f.fd, f.stage[0..part], pos);
                        try f.d.check(f.d.api.cuMemcpyHtoD_v2(base + off + done, f.stage.ptr, part), "cuMemcpyHtoD");
                        done += part;
                        pos += part;
                    }
                }
            }
        };
        const filler: Fill = .{ .d = &b.driver, .fd = fd, .at = data_at, .stage = stage.bytes };
        const carry = try s.gpa.alloc(u8, D * 2);
        defer s.gpa.free(carry);
        try getAll(fd, carry, data_at + total);
        try b.r.f.s.synchronize(); // pooled buffers may still be read by queued copies
        return pr.adopt(ids, carry, &filler);
    }

    /// The learned states' identity probe (prompt_imprint's: a fixed prompt pass whose state is hashed): a 4,143-token
    /// prompt (a full chunk past the sparse index's reach, a middle and a short one) prefilled cold, this rank's rows
    /// and carry hashed, every rank's hashes combined - two builds share learned states only when every rank's prompt
    /// pass gives equal bits. Leaves the live caches holding the probe (nothing kept refers to them).
    pub fn probe(s: *Session) !u64 {
        const b = s.boot;
        var toks: [4096 + 40 + 7]u32 = undefined;
        for (&toks, 0..) |*t, i| t.* = @intCast(1000 + i * 7919 % 50000);
        const len = @min(toks.len, b.o.context - 2);
        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(s.gpa);
        var cuts = [_]usize{ 4096, 4136 };
        _ = try b.r.run(toks[0..len], .{ .max_tokens = 1, .k = 0, .cuts = &cuts }, &out);
        if (s.reuse) |pr| try pr.store.setLive(&.{});
        const vs = try reuse.rowViews(s.gpa, &b.r.st, b.cfg, len, b.dcp);
        defer s.gpa.free(vs);
        var h = std.hash.Wyhash.init(0x70);
        var stage = try cuda.HostBuffer.alloc(&b.driver, 64 << 20);
        defer stage.free();
        for (vs) |v| {
            const n = v.row * v.rows;
            var done: usize = 0;
            while (done < n) {
                const part = @min(n - done, stage.bytes.len);
                try b.driver.check(b.driver.api.cuMemcpyDtoH_v2(stage.bytes.ptr, v.ptr + done, part), "cuMemcpyDtoH");
                h.update(stage.bytes[0..part]);
                done += part;
            }
        }
        try b.driver.check(b.driver.api.cuMemcpyDtoH_v2(stage.bytes.ptr, b.r.carry.ptr, b.cfg.hidden * 2), "cuMemcpyDtoH");
        h.update(stage.bytes[0 .. b.cfg.hidden * 2]);
        h.update(std.mem.sliceAsBytes(out.items));
        const mine = h.final();
        var all: [cuda.rendezvous.max_world * 8]u8 = undefined;
        try b.rdv.allGather(std.mem.asBytes(&mine), all[0 .. b.o.world * 8]);
        return std.hash.Wyhash.hash(0x71, all[0 .. b.o.world * 8]);
    }
};

/// A small file of `dir` (a drafter's config.json).
fn readText(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
}

fn putAll(fd: c_int, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + done, @min(bytes.len - done, 1 << 30));
        if (n <= 0) return error.StateWrite;
        done += @intCast(n);
    }
}

fn getAll(fd: c_int, bytes: []u8, at: u64) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.pread(fd, bytes.ptr + done, @min(bytes.len - done, 1 << 30), @intCast(at + done));
        if (n <= 0) return error.StateRead;
        done += @intCast(n);
    }
}

test "hca names as roce.hca_names" {
    var buf: [2][]const u8 = undefined;
    const got = parseHcas("=mlx5_0:1,^mlx5_1, ,mlx5_2", &buf);
    try std.testing.expectEqual(@as(usize, 2), got.len);
    try std.testing.expectEqualStrings("mlx5_0", got[0]);
    try std.testing.expectEqualStrings("mlx5_1", got[1]);
}

test "DCP turns on past DCP_AUTO unless named" {
    try std.testing.expectEqual(@as(usize, 1), try (Options{ .context = 131072 }).dcpFor());
    try std.testing.expectEqual(@as(usize, 4), try (Options{ .context = 1048576 }).dcpFor());
    try std.testing.expectEqual(@as(usize, 1), try (Options{ .context = 1048576, .dcp = 1 }).dcpFor());
    try std.testing.expectEqual(@as(usize, 2), try (Options{ .world = 8, .dcp = 2 }).dcpFor()); // the divisor (the parity-mirror)
    try std.testing.expectEqual(@as(usize, 4), try (Options{ .world = 8, .dcp = 4 }).dcpFor());
    try std.testing.expectError(error.BadDcp, (Options{ .world = 8, .dcp = 3 }).dcpFor()); // not a divisor
}

test "the cache check refuses a context x streams that leaves less than the reserve free (Python's figures)" {
    // 78 layers x 576 x 2 bytes a row; 4 streams x 140K context (~47 GiB) on 50 GiB free: refused, fit ~131K
    const row: u64 = 78 * 576 * 2;
    const gib: u64 = 1 << 30;
    const x = cacheFit(row, 140_000 + 4, 4, 1, 0, 50 * gib, 6).?;
    try std.testing.expect(x.need > 45 * gib);
    try std.testing.expectEqual(@as(u64, 6 * gib), x.spare);
    try std.testing.expect(x.fit < 140_000 and x.fit > 100_000);
    // 4 x 32K fits easily
    try std.testing.expectEqual(@as(?CacheFit, null), cacheFit(row, 32768 + 4, 4, 1, 0, 60 * gib, 6));
    // the reserve is at most half of what is free
    try std.testing.expectEqual(@as(?CacheFit, null), cacheFit(row, 1000, 1, 1, 0, 4 * gib, 6));
}

test "TF_GLM53_KV8's cache check: the halved row plus the side buffers' extra (findings-kv8.md §2.4)" {
    const gib: u64 = 1 << 30;
    // 78 layers of 584-byte rows (e4m3 + 2 fp32 scales) against bf16's 1152; the side buffers (staging 9 MiB,
    // the decode scratch 297 MiB, the 32K prompt mirror 36 MiB, tables) ~ 344 MiB at the 32K shape
    const row8: u64 = 78 * 584;
    const row16: u64 = 78 * 1152;
    const extra: u64 = 344 * (1 << 20);
    // 8 x 32K on 24 GiB free (the live serve's figure): refused in bf16, fits with kv8's row and the extra
    try std.testing.expect(cacheFit(row16, 32768 + 4, 8, 1, 0, 24 * gib, 6) != null);
    try std.testing.expectEqual(@as(?CacheFit, null), cacheFit(row8, 32768 + 4, 8, 1, extra, 24 * gib, 6));
    // one stream on 44 GiB free: bf16 refuses 500K, kv8 fits it - and kv8's own refusal says ~827K, ~1.8x bf16's
    try std.testing.expect(cacheFit(row16, 500_004, 1, 1, 0, 44 * gib, 6) != null);
    try std.testing.expectEqual(@as(?CacheFit, null), cacheFit(row8, 500_004, 1, 1, extra, 44 * gib, 6));
    const bf = cacheFit(row16, 500_004, 1, 1, 0, 44 * gib, 6).?;
    const kf = cacheFit(row8, 900_004, 1, 1, extra, 44 * gib, 6).?;
    try std.testing.expect(bf.fit < 500_000 and bf.fit > 400_000); // ~454K
    try std.testing.expect(kf.fit > 800_000); // ~827K: the doubled DCP 1 ceiling
}

test "TF_GLM53_KV8 auto-offs the side stream: the ignored TF_GLM53_SIDE leaves the rendezvous digest alone (§9)" {
    // kv8 on: the side letters are normalized to off, so two ranks that differ only in TF_GLM53_SIDE agree
    const a = settingsDigest(.{ .model = "m", .context = 32768, .kv8 = true, .side = "af" }, 1, 2);
    const b = settingsDigest(.{ .model = "m", .context = 32768, .kv8 = true, .side = "0" }, 1, 2);
    try std.testing.expectEqual(a, b);
    // kv8 off: the setting matters again, and kv8 itself is in the digest
    const c = settingsDigest(.{ .model = "m", .context = 32768, .kv8 = false, .side = "af" }, 1, 2);
    const d = settingsDigest(.{ .model = "m", .context = 32768, .kv8 = false, .side = "0" }, 1, 2);
    try std.testing.expect(c != d);
    try std.testing.expect(a != c);
}

test "TF_GLM53_KV8_V2 is in the rendezvous digest: ranks must agree on the prompt path (findings-kv8-v2.md)" {
    // the in-kernel dequant path changes the wide windows' bits' sources (the plane, not the mirror): a rank
    // running v1 beside one running v2 must refuse at the rendezvous, not mix
    const a = settingsDigest(.{ .model = "m", .context = 32768, .kv8 = true }, 1, 2);
    const b = settingsDigest(.{ .model = "m", .context = 32768, .kv8 = true, .kv8_v2 = true }, 1, 2);
    const c = settingsDigest(.{ .model = "m", .context = 32768, .kv8 = false, .kv8_v2 = false }, 1, 2);
    try std.testing.expect(a != b);
    try std.testing.expect(a != c);
}

test "TF_GLM53_PROMPT_PROF is not in the rendezvous digest: profiling changes no numerics, no settings" {
    // a rank profiling beside one not profiling must agree (the events and the chunk-boundary sync are timing
    // only); unset = exactly today's Boot (prof null)
    const a = settingsDigest(.{ .model = "m", .context = 32768, .kv8 = true }, 1, 2);
    const b = settingsDigest(.{ .model = "m", .context = 32768, .kv8 = true, .prompt_prof = true }, 1, 2);
    try std.testing.expectEqual(a, b);
}

test "TF_GLM53_TOPK_STREAM is in the rendezvous digest: ranks agree on the prefill selection kernel" {
    // the streaming select picks the radix's set by construction (host-tested in cuda_triton), but the selection
    // kernel is numerics-adjacent: a rank pair that disagrees on it refuses at the rendezvous, never mixes
    const a = settingsDigest(.{ .model = "m", .context = 32768, .kv8 = true }, 1, 2);
    const b = settingsDigest(.{ .model = "m", .context = 32768, .kv8 = true, .topk_stream = true }, 1, 2);
    try std.testing.expect(a != b);
}

test "TF_GLM53_FP8_INDEX is in the rendezvous digest: the cache layout is a numerics change every rank shares" {
    // the e4m3 plane re-lays the ic allocation and changes the scores: a rank pair that disagrees must refuse at
    // the rendezvous, never read each other's caches
    const a = settingsDigest(.{ .model = "m", .context = 32768, .kv8 = true }, 1, 2);
    const b = settingsDigest(.{ .model = "m", .context = 32768, .kv8 = true, .fp8_index = true }, 1, 2);
    try std.testing.expect(a != b);
}

test "--parallel refusals: DCP, a window wider than a decode window; the drafters draft concurrent streams" {
    try checkParallel(.{ .parallel = 1, .dspark = "/x" }, 4);
    try checkParallel(.{ .parallel = 4, .k = 2 }, 1);
    try std.testing.expectError(error.ParallelNeedsDcpOff, checkParallel(.{ .parallel = 4 }, 4));
    // TF_GLM53_DCP_STREAMS: several streams with the cache split, up to the one-stream shape's tokens in all
    try checkParallel(.{ .parallel = 4, .context = 250_000, .dcp_streams = true, .dspark = "/x" }, 4);
    try checkParallel(.{ .parallel = 2, .context = 500_000, .dcp_streams = true, .dspark = "/x" }, 4);
    // the cap is streams_tokens = 1<<22 (findings-dcp8.md): 4 x 500K and 2 x 1M now FIT
    try checkParallel(.{ .parallel = 8, .context = 500_000, .dcp_streams = true, .dspark = "/x" }, 4);
    try std.testing.expectError(error.DcpStreamsTooManyTokens, checkParallel(.{ .parallel = 4, .context = 1_100_000, .dcp_streams = true }, 4));
    try std.testing.expectError(error.DcpStreamsTooManyTokens, checkParallel(.{ .parallel = 2, .context = 2_200_000, .dcp_streams = true }, 4));
    try std.testing.expectError(error.WindowTooWide, checkParallel(.{ .parallel = 16, .k = 2, .context = 60_000, .dcp_streams = true }, 4));
    try checkParallel(.{ .parallel = 4, .dspark = "/x" }, 1);
    try checkParallel(.{ .parallel = 2, .dflash = "/x", .mode = "dflash" }, 1);
    try checkParallel(.{ .parallel = 4, .dspark = "/x", .mode = "dspark", .k = 0 }, 1);
    try std.testing.expectError(error.WindowTooWide, checkParallel(.{ .parallel = 16, .k = 2 }, 1));
}

test "Phase 4c: a drafter drafts N streams when N blocks fit one pass" {
    try checkParallelDrafters(1, 64, null);
    try checkParallelDrafters(4, 8, 8);
    try checkParallelDrafters(8, 8, null);
    try std.testing.expectError(error.DrafterTooWide, checkParallelDrafters(9, 8, null));
    try std.testing.expectError(error.DrafterTooWide, checkParallelDrafters(4, null, 17));
}
