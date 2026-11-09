//! §The structured pick's host seat: the masked argmax, the exchange record, and the rank
//! reduce — pure host logic, no CUDA, unit-tested here. pickMasked (cuda_runner.zig) is the
//! thin shell around it: the row download, this file's argmax and record, the exchange, this
//! file's reduce.
//!
//! THE TRANSPORT RULING (the design's intent over the poison receipts): NO 16-byte mid-round
//! exchange on the RoCE fast path. Every PROVEN RoCE collective in this engine lives inside
//! the captured graphs (the rounds' reductions replay as graph work); an EAGER collective
//! between replays is the poison class, receipted twice — the raw fast allGather timed out
//! mid-round, the raw NCCL form wedged all eight ranks. The masked pick's gather can never be
//! captured (the masks are host-side and per-request), so its records ride the RENDEZVOUS
//! channel instead: host TCP, rank-0-mediated, every rank ends with the identical bytes, and
//! the reduce below runs on each rank over the same input — the winner bit-identical by
//! construction. The channel's own error set (PeerClosed, Timeout, Disagree) is the loud
//! gate; no GPU-side state machine is touched mid-round.
//!
//! The copy-engine race and its fence retire with the device record: the record is BUILT on
//! the host from the downloaded row (the download's own synchronize is the only sync), so
//! there is no H2D upload for a device gather to read stale. The dead-grammar verdict is
//! COLLECTIVE: a rank whose row allows nothing still joins the exchange (its record carries
//! the dead flag), so no rank errors out of the collective alone — GrammarDead lands on all
//! ranks or none.

const std = @import("std");

/// A local masked pick: the winning global id and its raw value.
pub const Pick = struct { id: u32, val: f32 };

/// The local masked argmax over this rank's vocab share of `row`: the best ALLOWED global id
/// (the mask's bit form = lanes.grammar.allows': word gid/32, bit gid%32; ids at or past the
/// mask's words are not allowed). null: the row allows nothing — the dead-grammar class; the
/// caller still exchanges (the dead record) so the collective completes on every rank.
pub fn maskedArgmax(row: []const f32, mask: []const u32, vocab_off: u32) ?Pick {
    var bv: f32 = -std.math.inf(f32);
    var bi: u32 = 0;
    var any = false;
    for (row, 0..) |v, j| {
        const gid = vocab_off + @as(u32, @intCast(j));
        const word = gid / 32;
        if (word >= mask.len or (mask[word] >> @intCast(gid % 32)) & 1 != 1) continue;
        if (!any or v > bv) {
            bv = v;
            bi = gid;
            any = true;
        }
    }
    return if (any) .{ .id = bi, .val = bv } else null;
}

/// The exchange record, 4 words: the value's raw bits, the id as f32 (exact — ids < 2^24),
/// the live flag (0: this rank's row allowed nothing), a zero pad. The layout descends from
/// argmaxRec's; the spare words carry the live flag so a dead row rides the same exchange.
pub fn record(pick: ?Pick) [4]u32 {
    const p = pick orelse return .{ @bitCast(-std.math.inf(f32)), 0, 0, 0 };
    return .{ @bitCast(p.val), @bitCast(@as(f32, @floatFromInt(p.id))), 1, 0 };
}

/// The rank-reduce — pickRank's rule on the host: among LIVE records the maximum value wins,
/// an exact tie goes to the LOWEST rank (the device kernel's own tie rule). Every rank
/// reduces the identical gathered bytes, so the winner is bit-identical on all of them.
/// null: no live record — the collective GrammarDead (every rank sees it, none hangs).
pub fn reduce(all: []const u32, world: usize) ?Pick {
    var best: ?Pick = null;
    for (0..world) |rk| {
        if (rk * 4 + 4 > all.len) break;
        const rec = all[rk * 4 ..][0..4];
        if (rec[2] == 0) continue; // the dead flag: this rank's row allowed nothing
        const val: f32 = @bitCast(rec[0]);
        const id: u32 = @intFromFloat(@as(f32, @bitCast(rec[1])));
        if (best == null or val > best.?.val) best = .{ .id = id, .val = val };
    }
    return best;
}

// ---------------------------------------------------------------------------------------------------------------------
// The unit classes (CPU-only; the shell — the download, the rendezvous call — is exercised by
// the family's build and the window's gates, not here).
// ---------------------------------------------------------------------------------------------------------------------

test "the masked argmax takes the best allowed id and masks the forbidden high" {
    const vocab_off: u32 = 10;
    const row = [_]f32{ 5.0, 9.0, 1.0, 8.0, 2.0, 7.0 }; // gids 10..15
    // allow gid 10 and gid 13; the 9.0 at gid 11 is masked out, so the 8.0 at gid 13 wins
    var mask = [_]u32{ 0, 0, 0 };
    mask[0] = (1 << 10) | (1 << 13);
    const p = maskedArgmax(&row, &mask, vocab_off).?;
    try std.testing.expectEqual(@as(u32, 13), p.id);
    try std.testing.expectEqual(@as(f32, 8.0), p.val);

    // a mask shorter than the share's span: the allowed word still selects
    var short_mask = [_]u32{0};
    short_mask[0] = 1 << 11;
    const p2 = maskedArgmax(&row, &short_mask, vocab_off).?;
    try std.testing.expectEqual(@as(u32, 11), p2.id);
    try std.testing.expectEqual(@as(f32, 9.0), p2.val);
}

test "the masked argmax of a row that allows nothing is null (the dead-grammar class)" {
    const row = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expect(maskedArgmax(&row, &[_]u32{0}, 0) == null); // a zero word allows nothing
    try std.testing.expect(maskedArgmax(&row, &.{}, 0) == null); // an empty mask allows nothing
    // a mask bit past the share's rows selects nothing there
    const row2 = [_]f32{ 1.0, 2.0 };
    var mask2 = [_]u32{ 1 << 9 }; // gid 9 — not in the share (gids 0..1)
    try std.testing.expect(maskedArgmax(&row2, &mask2, 0) == null);
}

test "the record round-trips the id exactly at the vocab's top and at 2^24-1, live and dead" {
    // the id as f32 is exact below 2^24: the served vocab's top (154880) and the form's bound
    for ([_]u32{ 1, 154_880, (1 << 24) - 1 }) |test_id| {
        const rec = record(.{ .id = test_id, .val = -0.5 });
        try std.testing.expectEqual(@as(u32, 1), rec[2]); // live
        try std.testing.expectEqual(test_id, @as(u32, @intFromFloat(@as(f32, @bitCast(rec[1])))));
        try std.testing.expectEqual(@as(f32, -0.5), @as(f32, @bitCast(rec[0])));
    }
    // the dead record: flag 0, value -inf, id 0
    const dead = record(null);
    try std.testing.expectEqual(@as(u32, 0), dead[2]);
    try std.testing.expectEqual(-std.math.inf(f32), @as(f32, @bitCast(dead[0])));
}

test "the reduce takes the max over live records, ties to the lowest rank, skips the dead" {
    // 4 ranks: r0 dead, r1 val 1.0 id 100, r2 val 3.0 id 200, r3 val 2.0 id 300 -> r2 wins
    var all: [16]u32 = undefined;
    all[0..4].* = record(null);
    all[4..8].* = record(.{ .id = 100, .val = 1.0 });
    all[8..12].* = record(.{ .id = 200, .val = 3.0 });
    all[12..16].* = record(.{ .id = 300, .val = 2.0 });
    const p = reduce(&all, 4).?;
    try std.testing.expectEqual(@as(u32, 200), p.id);
    try std.testing.expectEqual(@as(f32, 3.0), p.val);

    // an exact tie: the LOWEST rank wins (the device pickRank's own rule)
    all[4..8].* = record(.{ .id = 100, .val = 3.0 });
    const tied = reduce(&all, 4).?;
    try std.testing.expectEqual(@as(u32, 100), tied.id); // rank 1's id, not rank 2's

    // all dead: the collective GrammarDead — null on every rank, none hangs
    all[4..8].* = record(null);
    all[8..12].* = record(null);
    all[12..16].* = record(null);
    try std.testing.expect(reduce(&all, 4) == null);

    // one live among the dead: it wins
    all[8..12].* = record(.{ .id = 200, .val = -12.5 });
    const lone = reduce(&all, 4).?;
    try std.testing.expectEqual(@as(u32, 200), lone.id);
}

test "the reduce is bit-identical across ranks given the gathered bytes (the collective's contract)" {
    // the same bytes on every rank -> the same winner: the determinism the followers ride
    var all: [32]u32 = undefined;
    for (0..8) |rk| {
        all[rk * 4 ..][0..4].* = record(if (rk == 5) .{ .id = 154_880, .val = 0.25 } else if (rk == 3) .{ .id = 7, .val = 0.125 } else null);
    }
    const p = reduce(&all, 8).?;
    try std.testing.expectEqual(@as(u32, 154_880), p.id); // the top-of-vocab id, exact through the f32 form
    try std.testing.expectEqual(@as(f32, 0.25), p.val);
}
