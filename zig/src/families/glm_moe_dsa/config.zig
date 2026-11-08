//! GLM-5.3 (model_type glm_moe_dsa) on CUDA: the 744B's shape from config.json, and one rank's
//! share of it under tensor parallelism. Self-contained so the CUDA family is a module of its own;
//! the Metal family's config.zig parses the same checkpoint (unifying the two parsers is an
//! upstream-PR candidate: one glm_moe_dsa config for both backends).
const std = @import("std");

pub const max_layers = 96;

pub const Kind = enum { mla };

pub const Config = struct {
    hidden: u32 = 6144,
    layers: u32 = 78,
    run: u32 = 78, // backbone layers loaded and run: all, or a shorter prefix for a check that must fit
    vocab: u32 = 154880,
    eps: f32 = 1e-5,
    dense_layers: u32 = 3,
    dense_inter: u32 = 12288,
    experts: u32 = 256,
    topk: u32 = 8,
    moe_inter: u32 = 2048,
    shared_inter: u32 = 2048, // n_shared_experts x moe_intermediate_size
    routed_scale: f32 = 2.5,
    norm_topk: bool = true,
    swiglu_limit: f32 = 10.0,
    heads: u32 = 64, // MLA heads this rank runs: all, or its TP share
    nope: u32 = 192,
    rope: u32 = 64,
    v_dim: u32 = 256,
    q_lora: u32 = 2048,
    kv_lora: u32 = 512,
    rope_theta: f64 = 8e6,
    i_heads: u32 = 32,
    i_dim: u32 = 128,
    i_topk: u32 = 2048,
    mtp: u32 = 1,
    eos: [3]u32 = .{ 154820, 154827, 154829 },
    eos_n: u32 = 3,
    /// Layer i runs its own indexer when set; otherwise it reuses the last full layer's top-k.
    full: std.StaticBitSet(max_layers) = .empty,
    tp: u32 = 1,
    tp_rank: u32 = 0,

    pub fn kind(_: *const Config, _: u32) Kind {
        return .mla;
    }

    pub fn isMoe(c: *const Config, i: u32) bool {
        return i >= c.dense_layers;
    }

    /// A head's query width: its non-positional part plus the RoPE part.
    pub fn qHead(c: *const Config) u32 {
        return c.nope + c.rope;
    }

    /// Keys a sparse row attends: the indexer's top-k, with no pool-compress tail (Flash's kpool).
    pub fn keyWidth(c: *const Config) u32 {
        return c.i_topk;
    }

    /// The context width below which every visible key is attended (the indexer picks all of them anyway).
    pub fn denseLimit(c: *const Config) u32 {
        return c.i_topk;
    }

    /// The full layer whose top-k layer i uses: itself when it has an indexer.
    pub fn indexFrom(c: *const Config, i: u32) u32 {
        var l = i;
        while (l > 0 and !c.full.isSet(@intCast(l))) l -= 1;
        return l;
    }
};

fn int(o: std.json.ObjectMap, key: []const u8) !i64 {
    const v = o.get(key) orelse return error.BadConfig;
    return if (v == .integer) v.integer else error.BadConfig;
}

fn float(o: std.json.ObjectMap, key: []const u8) !f64 {
    const v = o.get(key) orelse return error.BadConfig;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BadConfig,
    };
}

fn str(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

/// config.json of a glm_moe_dsa checkpoint; a shape the kernels do not take is refused.
pub fn parse(gpa: std.mem.Allocator, json: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    const t = parsed.value.object;
    if (!std.mem.eql(u8, str(t, "model_type") orelse "", "glm_moe_dsa")) return error.NotGlmMoeDsa;
    var c: Config = .{};
    c.hidden = @intCast(try int(t, "hidden_size"));
    c.layers = @intCast(try int(t, "num_hidden_layers"));
    c.run = c.layers;
    c.vocab = @intCast(try int(t, "vocab_size"));
    c.eps = @floatCast(try float(t, "rms_norm_eps"));
    c.dense_layers = @intCast(try int(t, "first_k_dense_replace"));
    c.dense_inter = @intCast(try int(t, "intermediate_size"));
    c.experts = @intCast(try int(t, "n_routed_experts"));
    c.topk = @intCast(try int(t, "num_experts_per_tok"));
    c.moe_inter = @intCast(try int(t, "moe_intermediate_size"));
    c.shared_inter = c.moe_inter * @as(u32, @intCast(if (t.get("n_shared_experts")) |v| v.integer else 1));
    c.routed_scale = @floatCast(try float(t, "routed_scaling_factor"));
    c.heads = @intCast(try int(t, "num_attention_heads"));
    c.nope = @intCast(try int(t, "qk_nope_head_dim"));
    c.rope = @intCast(try int(t, "qk_rope_head_dim"));
    c.v_dim = @intCast(try int(t, "v_head_dim"));
    c.q_lora = @intCast(try int(t, "q_lora_rank"));
    c.kv_lora = @intCast(try int(t, "kv_lora_rank"));
    c.i_heads = @intCast(try int(t, "index_n_heads"));
    c.i_dim = @intCast(try int(t, "index_head_dim"));
    c.i_topk = @intCast(try int(t, "index_topk"));
    c.mtp = @intCast(try int(t, "num_nextn_predict_layers"));
    if (t.get("ew_scale")) |_| return error.UnsupportedGlmShape; // not this checkpoint family
    const rp = t.get("rope_parameters") orelse return error.BadConfig;
    if (rp != .object or !std.mem.eql(u8, str(rp.object, "rope_type") orelse "", "default")) return error.UnsupportedRope;
    c.rope_theta = try float(rp.object, "rope_theta");
    const types = t.get("indexer_types") orelse return error.BadConfig;
    if (types != .array or types.array.items.len != c.layers or c.layers + c.mtp > max_layers) return error.BadConfig;
    for (types.array.items, 0..) |v, i| {
        if (v != .string) return error.BadConfig;
        if (std.mem.eql(u8, v.string, "full")) c.full.set(i) else if (!std.mem.eql(u8, v.string, "shared")) return error.BadConfig;
    }
    if (!c.full.isSet(0)) return error.BadConfig;
    // the MTP layer runs its own indexer (its weights carry one)
    if (c.mtp > 0) c.full.set(c.layers);
    if (t.get("eos_token_id")) |v| if (v == .array and v.array.items.len <= 3) {
        c.eos_n = @intCast(v.array.items.len);
        for (v.array.items, 0..) |e, i| c.eos[i] = @intCast(e.integer);
    };
    const plain = (if (t.get("n_group")) |v| v.integer else 1) == 1 and (if (t.get("topk_group")) |v| v.integer else 1) == 1;
    const sigmoid = std.mem.eql(u8, str(t, "scoring_func") orelse "sigmoid", "sigmoid");
    const shaped = c.hidden == 6144 and c.kv_lora == 512 and c.rope == 64 and c.nope == 192 and c.v_dim == 256 and
        c.q_lora == 2048 and c.i_heads == 32 and c.i_dim == 128 and c.i_topk == 2048 and c.moe_inter == 2048;
    if (!plain or !sigmoid or !shaped) return error.UnsupportedGlmShape;
    return c;
}

/// This rank of `ranks`: heads, dense rows, expert rows, shared rows and vocabulary rows split evenly.
pub fn split(c: *Config, rank: u32, ranks: u32) !void {
    // heads, expert rows and dense rows must divide; the vocabulary's 64-blocks (2420 at TP8) may not,
    // and the shard plan gives the remainder to the first ranks.
    if (ranks == 0 or rank >= ranks or c.heads % ranks != 0 or (c.moe_inter / 64) % ranks != 0 or
        (c.dense_inter / 64) % ranks != 0) return error.BadSplit;
    c.tp = ranks;
    c.tp_rank = rank;
    c.heads /= ranks;
}

test "GLM-5.3 is all-MLA, and TP8 splits heads 8 ways" {
    var c = Config{};
    for (0..c.layers + c.mtp) |i| try std.testing.expectEqual(Kind.mla, c.kind(@intCast(i)));
    try std.testing.expectEqual(@as(u32, 2048), c.keyWidth());
    try split(&c, 3, 8);
    try std.testing.expectEqual(@as(u32, 8), c.heads);
    try std.testing.expectEqual(@as(u32, 3), c.tp_rank);
    try std.testing.expectError(error.BadSplit, split(&c, 0, 7));
}

test "the real GLM-5.3 NVFP4-CSF pack config (TF_GLM53_NVFP4_DIR) parses to the literal" {
    const dirz = std.c.getenv("TF_GLM53_NVFP4_DIR") orelse return error.SkipZigTest;
    const dir: []const u8 = std.mem.span(dirz);
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    const json = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 20));
    const c = try parse(a, json);
    try std.testing.expectEqual(@as(u32, 78), c.layers);
    try std.testing.expectEqual(@as(u32, 3), c.dense_layers);
    try std.testing.expectEqual(@as(u32, 256), c.experts);
    try std.testing.expectEqual(@as(u32, 2048), c.moe_inter);
    try std.testing.expectEqual(@as(u32, 2048), c.shared_inter);
    try std.testing.expectEqual(@as(u32, 154880), c.vocab);
    var full: u32 = 0;
    for (0..c.layers) |i| full += @intFromBool(c.full.isSet(@intCast(i)));
    try std.testing.expectEqual(@as(u32, 21), full);
    try std.testing.expect(c.full.isSet(c.layers));
    try std.testing.expect(!c.isMoe(2) and c.isMoe(3));
    var t8 = c;
    try split(&t8, 5, 8);
    try std.testing.expectEqual(@as(u32, 8), t8.heads);
}