//! 125.43e: the DSpark walked-tree log (TF_GLM53_DSPARK_LOG, rank 0) - upstream's TF_DRAFT_LOG
//! contract, consumed by tools/fit_draft_calibration.py UNCHANGED: a JSONL record a drafting round -
//! {"stream": sid, "position": the first drafted token's absolute position, "greedy": the walk's
//! regime, "kept": the PREVIOUS round's committed picks along the drafts, "tokens"/"parents"/
//! "scores": the walked chain (the parents linear; the scores ln(confidence) a slot)}. The NEXT
//! record's kept labels THIS tree (the tool's union map across a stream's rounds). The records go
//! out as log lines ("draft-tree: {...}") - the 0.17 File API is Io-threaded; the docker logs carry
//! them and the extraction greps the prefix into the JSONL the tool reads.

const std = @import("std");

const max_slots = 16;
const max_chain = 16;

pub const DraftLog = struct {
    pos: [max_slots]u64 = @splat(0),
    n: [max_slots]usize = @splat(0),
    greedy: [max_slots]bool = @splat(true),
    ids: [max_slots][max_chain]u32 = @splat(@splat(0)),
    score: [max_slots][max_chain]f64 = @splat(@splat(@as(f64, 0.0))),
    kept: [max_slots][max_chain]u32 = @splat(@splat(0)),
    kept_n: [max_slots]usize = @splat(0),
    last_sid: [max_slots]u32 = @splat(0),
    seen: [max_slots]bool = @splat(false),
    rounds: u64 = 0,
    runs: u64 = 0,

    /// The single-stream path's request id (the tool groups by stream: sequential requests on one
    /// server must not merge - their position frames restart at each prompt). High base: no collision
    /// with the multi's stream ids.
    pub fn nextRun(l: *DraftLog) u64 {
        l.runs += 1;
        return 1000000 + l.runs;
    }

    pub fn init() DraftLog {
        return .{};
    }

    /// The mdraft's walk stash (the propose loop): slot s's returned chain and per-slot confidences.
    pub fn tree(l: *DraftLog, slot: usize, position: u64, greedy: bool, got: []const u32, conf: []const f64) void {
        if (slot >= max_slots) return;
        const m = @min(@min(got.len, conf.len), max_chain);
        l.pos[slot] = position;
        l.greedy[slot] = greedy;
        l.n[slot] = m;
        for (0..m) |j| {
            l.ids[slot][j] = got[j];
            l.score[slot][j] = @log(conf[j]);
        }
    }

    /// The round's close (the multi's applyRound): emit the stashed record (kept: the previous
    /// round's picks along the drafts), then hold this round's picks for the next record. A slot's
    /// stream change clears the held picks (the new stream's first tree stays unlabeled - harmless:
    /// the tool skips hit=None nodes).
    pub fn round(l: *DraftLog, stream: u32, slot: usize, picks_along: []const u32) void {
        if (slot >= max_slots) return;
        if (l.seen[slot] and l.last_sid[slot] != stream) l.kept_n[slot] = 0;
        l.last_sid[slot] = stream;
        l.seen[slot] = true;
        if (l.n[slot] > 0) {
            l.writeRecord(stream, slot);
            l.rounds += 1;
        }
        const m = @min(picks_along.len, max_chain);
        for (0..m) |j| l.kept[slot][j] = picks_along[j];
        l.kept_n[slot] = m;
        l.n[slot] = 0;
    }

    fn writeRecord(l: *DraftLog, stream: u32, slot: usize) void {
        var buf: [4096]u8 = undefined;
        var s: []u8 = buf[0..];
        const p = std.fmt.bufPrint;
        const g = p(s, "{{\"stream\":{d},\"position\":{d},\"greedy\":{s},\"kept\":[", .{ stream, l.pos[slot], if (l.greedy[slot]) "true" else "false" }) catch return;
        s = s[g.len..];
        const k = l.kept_n[slot];
        for (l.kept[slot][0..k], 0..) |t, j| {
            if (j + 1 < k) {
                const g2 = p(s, "{d},", .{t}) catch return;
                s = s[g2.len..];
            } else {
                const g2 = p(s, "{d}", .{t}) catch return;
                s = s[g2.len..];
            }
        }
        const g3 = p(s, "],\"tokens\":[", .{}) catch return;
        s = s[g3.len..];
        const n = l.n[slot];
        for (l.ids[slot][0..n], 0..) |t, j| {
            if (j + 1 < n) {
                const g4 = p(s, "{d},", .{t}) catch return;
                s = s[g4.len..];
            } else {
                const g4 = p(s, "{d}", .{t}) catch return;
                s = s[g4.len..];
            }
        }
        const g5 = p(s, "],\"parents\":[", .{}) catch return;
        s = s[g5.len..];
        for (0..n) |j| {
            const q: i64 = if (j == 0) -1 else @intCast(j - 1);
            if (j + 1 < n) {
                const g6 = p(s, "{d},", .{q}) catch return;
                s = s[g6.len..];
            } else {
                const g6 = p(s, "{d}", .{q}) catch return;
                s = s[g6.len..];
            }
        }
        const g7 = p(s, "],\"scores\":[", .{}) catch return;
        s = s[g7.len..];
        for (l.score[slot][0..n], 0..) |v, j| {
            if (j + 1 < n) {
                const g8 = p(s, "{d:.4},", .{v}) catch return;
                s = s[g8.len..];
            } else {
                const g8 = p(s, "{d:.4}", .{v}) catch return;
                s = s[g8.len..];
            }
        }
        const g9 = p(s, "]}}", .{}) catch return;
        s = s[g9.len..];
        std.log.info("draft-tree: {s}", .{buf[0 .. buf.len - s.len]});
    }
};
