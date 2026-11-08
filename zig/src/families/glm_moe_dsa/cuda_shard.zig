//! A TP rank's share of GLM-5.3 on CUDA: attention heads one by one, widths and the vocabulary in 64s
//! (the same rules the Flash family's cuda_shard.zig uses, minus KDA which GLM-5.3 does not have).
const std = @import("std");
const canon = @import("canon");
const Config = @import("config.zig").Config;

/// Elements of a width or the vocabulary a rank takes whole (split.UNIT).
pub const unit = 64;

pub const Span = struct {
    lo: usize,
    hi: usize,

    pub fn len(s: Span) usize {
        return s.hi - s.lo;
    }
};

pub const Share = struct {
    rank: usize,
    world: usize,
    mla: Span, // DSA heads
    dense: Span, // the dense MLP's intermediate rows
    moe: Span, // each routed expert's intermediate rows (64s: NVFP4 groups of 16 divide 64)
    shared: Span, // the shared expert's intermediate rows
    vocab: Span, // the head's rows

    /// Rank `rank` of `world`; refused where a rank would be left empty.
    pub fn of(c: *const Config, rank: usize, world: usize) !Share {
        if (world == 0 or rank >= world) return error.BadRank;
        return .{
            .rank = rank,
            .world = world,
            .mla = try span(c.heads, 1, rank, world),
            .dense = try span(c.dense_inter, unit, rank, world),
            .moe = try span(c.moe_inter, unit, rank, world),
            .shared = try span(c.moe_inter, unit, rank, world), // n_shared_experts x moe_intermediate_size
            .vocab = try span(c.vocab, unit, rank, world),
        };
    }
};

/// Rank `rank`'s [lo, hi) of `total` elements taken in whole `per`s.
fn span(total: u32, per: u32, rank: usize, world: usize) !Span {
    if (total % per != 0 or world > 256) return error.BadSplit;
    const ones: [256]u64 = @splat(1);
    var out: [256]canon.Range = undefined;
    canon.partition(@intCast(total / per), ones[0..world], out[0..world]);
    const r = out[rank];
    if (r.len() == 0) return error.EmptyShare;
    return .{ .lo = r.begin * per, .hi = r.end * per };
}

test "TP8 splits the 744B evenly except the vocabulary's 64-blocks: 8 heads, 256 expert rows, 302/303 vocab blocks" {
    const c = Config{};
    var shares: [8]Share = undefined;
    for (0..8) |r| shares[r] = try Share.of(&c, r, 8);
    for (shares, 0..) |s, r| {
        try std.testing.expectEqual(Span{ .lo = r * 8, .hi = r * 8 + 8 }, s.mla);
        try std.testing.expectEqual(Span{ .lo = r * 256, .hi = r * 256 + 256 }, s.moe);
        try std.testing.expectEqual(@as(usize, 1536), s.dense.len());
        try std.testing.expectEqual(@as(usize, 256), s.shared.len());
    }
    // 154880 / 64 = 2420 blocks: 303 for the first four ranks, 302 for the rest
    try std.testing.expectEqual(@as(usize, 303 * 64), shares[0].vocab.len());
    try std.testing.expectEqual(@as(usize, 302 * 64), shares[7].vocab.len());
    try std.testing.expectEqual(@as(usize, c.vocab), shares[7].vocab.hi);
    // every share is 64-aligned where the kernels need it
    for (shares) |s| {
        try std.testing.expectEqual(@as(usize, 0), s.vocab.lo % 64);
        try std.testing.expectEqual(@as(usize, 0), s.moe.lo % 64);
        try std.testing.expectEqual(@as(usize, 0), s.dense.lo % 64);
    }
    try std.testing.expectError(error.BadRank, Share.of(&c, 8, 8));
}