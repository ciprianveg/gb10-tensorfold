//! Decode context parallelism for the CSF family: position sharding, the parity-mirror's assignment, the
//! exchanges and the cache sizing. Ported from r2's cuda_dcp.zig (snapshot 70e68f66, md5
//! 02dd8c834908a1a785833ac65d094d54) with two adaptations: our NCCL surface (tp_group.Group's library,
//! not r2's Communicator) and cache formats as parameters — the NVFP4 pack changes weights only, so the
//! cache bytes are the EXL3 line's at the same dcp (fp8 latent plane under kv8, fp8 index plane).
//!
//! Sharding (fused.State): position p lives on rank p % dcp at local row p / dcp; a rank's caches hold
//! ceil(capacity / dcp) + 1 rows. Attention: every rank's absorbed queries gathered, each rank attends all
//! heads over its own keys, partials and log-sum-exps go to the heads' owners (decode: one all-gather;
//! prompt: all-to-all), which merge them in rank order. Selection: each rank scores its own slots, keeps
//! its top index_topk keys, the candidates are gathered and every rank takes the same global top.
const std = @import("std");

/// engine.DCP_AUTO: past this many tokens the caches interleave over the ranks.
pub const dcp_auto_tokens: usize = 200_000;
/// The most tokens of cache N streams may hold in all (the one-stream shape that has been served: 4M).
pub const streams_tokens: usize = 1 << 22;
/// GiB that must stay free on a rank after the slots' caches (TF_GLM53_DCP_STREAMS_RESERVE_GB).
pub const streams_reserve_gb: f64 = 20.0;

/// The cache's per-token bytes: what the family stores, so NVFP4 and EXL3 siblings at one dcp size alike.
pub const Formats = struct {
    latent_row: usize, // bf16 bf16 latent: lw * 2; kv8: lw + 8 (e4m3 codes + two fp32 scales)
    index_row: usize, // fp8 plane: 132 or 144; bf16 keys: index_dim * 2
    index_dim: usize,
};

/// TF_GLM53_DCP when set (1, a divisor of world, or world), else world past dcpAuto, else 1.
pub fn choose(context: usize, world: usize, env: ?usize) error{BadDcp}!usize {
    if (env) |v| {
        if (v != 0) {
            if (v != 1 and v != world and (v > world or world % v != 0)) return error.BadDcp;
            return v;
        }
    }
    return if (context > dcp_auto_tokens) world else 1;
}

/// The parity a rank writes and scores for: rank % dcp (at dcp == world the rank itself).
pub fn parity(dcp: usize, rank: usize) usize {
    return rank % dcp;
}

/// The parity-mirror's scoring assignment: rank r of block q = r / dcp scores head-groups
/// dcp*q .. dcp*q + dcp - 1 over its parity's shard. At dcp == world q = 0: every rank scores all groups.
pub fn scoredBase(dcp: usize, rank: usize) usize {
    return dcp * (rank / dcp);
}

pub fn groupOwner(dcp: usize, rank: usize, b: usize) usize {
    return scoredBase(dcp, rank) + b;
}

pub fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

/// Rows a rank's caches hold for `capacity` tokens.
pub fn localCapacity(capacity: usize, dcp: usize) usize {
    return cdiv(capacity, dcp) + 1;
}

/// The rank that holds position p, and its row there.
pub fn owner(p: usize, dcp: usize) struct { rank: usize, row: usize } {
    return .{ .rank = p % dcp, .row = p / dcp };
}

/// The keys of a row at position p < T this rank holds: its slots of positions 0..p.
pub fn keysBelow(p: usize, dcp: usize, rank: usize) usize {
    return if (p >= rank) (p - rank) / dcp + 1 else 0;
}

/// The slots a rank scores for a key range of T positions.
pub fn localCols(T: usize, dcp: usize) usize {
    return cdiv(T, dcp);
}

/// The attention exchange's layout: where this rank's heads sit in the received partials and the per-source
/// strides. r2's current caller (round 2, findings-dcp2-build.md §F) takes the subgroup p2p gather for the small
/// decode windows too, so the small path reads the [G] layout at offset 0; the base-shifted layout below survives
/// for dcp == world, where the group IS the world and the shift is 0. The function is r2's verbatim, so a re-pin
/// carries any further change mechanically.
pub const Exchange = struct { o0: usize, l0: usize, ss: usize, ssl: usize };

pub fn exchange(small: bool, dcp: usize, rank: usize, R: usize, H: usize, lw: usize) Exchange {
    const base = scoredBase(dcp, rank);
    const blk = rank - base;
    if (small) return .{ .o0 = (base * dcp + blk) * R * H * lw, .l0 = (base * dcp + blk) * R * H, .ss = dcp * R * H * lw, .ssl = dcp * R * H };
    return .{ .o0 = 0, .l0 = 0, .ss = R * H * lw, .ssl = R * H };
}

pub fn streamsTokensOk(context: usize, streams: usize) bool {
    return streams >= 1 and context <= streams_tokens / streams;
}

/// Row r of a window: stream `slot` at position p -> (the rank that stores its keys, the cache row there).
pub fn slotOwner(p: usize, dcp: usize, slot: usize, local: usize) struct { rank: usize, row: usize } {
    return .{ .rank = p % dcp, .row = slot * local + p / dcp };
}

/// Bytes of cache one token of context takes over all ranks: a latent row a layer, an index row a
/// full-indexer layer, and the MTP layer's.
pub fn tokenBytes(layers: usize, full_indexers: usize, f: Formats, mtp: bool, mtp_indexer: bool) usize {
    var per = layers * f.latent_row + full_indexers * f.index_row;
    if (mtp) per += f.latent_row + (if (mtp_indexer) f.index_row else 0);
    return per;
}

/// What `streams` slots of `context` tokens take on one rank (capacity = context + k + 1 as the engine sizes).
pub fn rankCacheBytes(per_token: usize, context: usize, k: usize, streams: usize, dcp: usize) usize {
    return per_token * localCapacity(context + k + 1, dcp) * streams;
}

pub const StreamsFit = struct {
    need: u64,
    free: u64,
    reserve: u64,
    ok: bool,

    pub fn fit_tokens(f: StreamsFit, capacity: usize) u64 {
        if (f.free <= f.reserve or f.need == 0) return 0;
        const room = f.free - f.reserve;
        if (room >= f.need) return capacity;
        return @intFromFloat(@as(f64, @floatFromInt(capacity)) * @as(f64, @floatFromInt(room)) / @as(f64, @floatFromInt(f.need)) * 0.98);
    }
};

/// The boot check of several streams under DCP: `per` = a rank's bytes for one stream, `free` the device's
/// report before the caches allocate. The reserve is whole (never halved).
pub fn streamsFit(per: u64, streams: u64, free: u64, reserve_gb: f64) StreamsFit {
    const need = per * streams;
    const reserve: u64 = @intFromFloat(@max(reserve_gb, 0) * (1 << 30));
    return .{ .need = need, .free = free, .reserve = reserve, .ok = need + reserve <= free };
}

// ------------------------------------------------------------------------------- the NCCL exchanges ---
// Our family's communicator: the library and handle tp_group.Group already opens; send/recv resolved from the
// library once (the same code r2 runs through its Communicator).
pub const Comm = struct { lib: *const @import("cuda").nccl.Library, comm: @import("cuda").nccl.Comm, rank: usize, world: usize };

const cuda = @import("cuda");
const SendFn = *const fn (u64, usize, cuda.nccl.DataType, c_int, cuda.nccl.Comm, cuda.abi.Stream) callconv(.c) cuda.nccl.Result;
const RecvFn = *const fn (u64, usize, cuda.nccl.DataType, c_int, cuda.nccl.Comm, cuda.abi.Stream) callconv(.c) cuda.nccl.Result;

var p2p_lib: ?*const cuda.nccl.Library = null;
var p2p_send: ?SendFn = null;
var p2p_recv: ?RecvFn = null;

fn p2p(lib: *const cuda.nccl.Library) !struct { send: SendFn, recv: RecvFn } {
    if (p2p_lib != lib) {
        const dl = @constCast(&lib.lib);
        p2p_send = dl.lookup(SendFn, "ncclSend") orelse return error.MissingSymbol;
        p2p_recv = dl.lookup(RecvFn, "ncclRecv") orelse return error.MissingSymbol;
        p2p_lib = lib;
    }
    return .{ .send = p2p_send.?, .recv = p2p_recv.? };
}

/// all_to_all: block k of `send` (count x elem bytes) to rank k, rank k's block into block k of `recv`.
pub fn allToAll(cm: *const Comm, send: u64, recv: u64, count: usize, dt: cuda.nccl.DataType, elem: usize, stream: cuda.abi.Stream) !void {
    const f = try p2p(cm.lib);
    try cm.lib.check(cm.lib.api.ncclGroupStart(), "ncclGroupStart");
    var failed: ?anyerror = null;
    for (0..cm.world) |k| {
        const off = k * count * elem;
        cm.lib.check(f.send(send + off, count, dt, @intCast(k), cm.comm, stream), "ncclSend") catch |e| {
            failed = e;
            break;
        };
        cm.lib.check(f.recv(recv + off, count, dt, @intCast(k), cm.comm, stream), "ncclRecv") catch |e| {
            failed = e;
            break;
        };
    }
    try cm.lib.check(cm.lib.api.ncclGroupEnd(), "ncclGroupEnd");
    if (failed) |e| return e;
}

/// The block gather: every member's `count` elements into slot `member - base` of `out` (the self slot is the
/// caller's copy on the same stream). The E1 queries and the small windows' E3 take this at dcp < world.
pub fn gatherFrom(cm: *const Comm, send: u64, out: u64, count: usize, dt: cuda.nccl.DataType, elem: usize, dcp: usize, rank: usize, stream: cuda.abi.Stream) !void {
    const f = try p2p(cm.lib);
    const base = scoredBase(dcp, rank);
    const mine = rank - base;
    const bytes = count * elem;
    try cm.lib.check(cm.lib.api.ncclGroupStart(), "ncclGroupStart");
    var failed: ?anyerror = null;
    for (0..dcp) |b| {
        if (b == mine) continue;
        const peer = base + b;
        cm.lib.check(f.send(send, count, dt, @intCast(peer), cm.comm, stream), "ncclSend") catch |e| {
            failed = e;
            break;
        };
        cm.lib.check(f.recv(out + b * bytes, count, dt, @intCast(peer), cm.comm, stream), "ncclRecv") catch |e| {
            failed = e;
            break;
        };
    }
    try cm.lib.check(cm.lib.api.ncclGroupEnd(), "ncclGroupEnd");
    if (failed) |e| return e;
}

/// The pair exchange (the mirror's E3): block b of `send` to member base+b, landing at that rank's
/// `recv + parity(sender) * dst_stride`; the self block is the caller's copy into slot `mine`.
pub fn pairExchange(cm: *const Comm, send: u64, recv: u64, count: usize, dt: cuda.nccl.DataType, elem: usize, dcp: usize, rank: usize, dst_stride: usize, stream: cuda.abi.Stream) !void {
    const f = try p2p(cm.lib);
    const base = scoredBase(dcp, rank);
    const mine = rank - base;
    try cm.lib.check(cm.lib.api.ncclGroupStart(), "ncclGroupStart");
    var failed: ?anyerror = null;
    for (0..dcp) |b| {
        if (b == mine) continue;
        const peer = base + b;
        cm.lib.check(f.send(send + b * count * elem, count, dt, @intCast(peer), cm.comm, stream), "ncclSend") catch |e| {
            failed = e;
            break;
        };
    }
    if (failed == null) {
        for (0..dcp) |b| {
            if (b == mine) continue;
            const peer = base + b;
            cm.lib.check(f.recv(recv + (peer % dcp) * dst_stride * elem, count, dt, @intCast(peer), cm.comm, stream), "ncclRecv") catch |e| {
                failed = e;
                break;
            };
        }
    }
    try cm.lib.check(cm.lib.api.ncclGroupEnd(), "ncclGroupEnd");
    if (failed) |e| return e;
}

// -------------------------------------------------------------------------------------------- tests ---

test "positions shard round robin: owner, local rows, slots below a position" {
    const o0 = owner(0, 8);
    try std.testing.expectEqual(@as(usize, 0), o0.rank);
    try std.testing.expectEqual(@as(usize, 0), o0.row);
    const o1 = owner(11, 8);
    try std.testing.expectEqual(@as(usize, 3), o1.rank);
    try std.testing.expectEqual(@as(usize, 1), o1.row);
    try std.testing.expectEqual(@as(usize, 17), localCapacity(128, 8)); // ceil(128/8) + 1
    try std.testing.expectEqual(@as(usize, 2), keysBelow(11, 8, 3)); // positions 3 and 11
    try std.testing.expectEqual(@as(usize, 0), keysBelow(2, 8, 3));
}

test "keys below a position count the rank's own slots" {
    for (0..40) |p| {
        for (0..8) |rank| {
            const got = keysBelow(p, 8, rank);
            var want: usize = 0;
            var x: usize = 0;
            while (x <= p) : (x += 1) if (x % 8 == rank) {
                want += 1;
            };
            try std.testing.expectEqual(want, got);
        }
    }
}

test "T shards into ceil(T/dcp) slots a rank" {
    try std.testing.expectEqual(@as(usize, 16), localCols(128, 8));
    try std.testing.expectEqual(@as(usize, 15), localCols(113, 8));
}

test "the parity-mirror: every (head-group, parity) pair is scored by world/dcp ranks" {
    for ([_]usize{ 2, 4, 8 }) |dcp| {
        var counts: [64]usize = @splat(0);
        for (0..8) |rank| for (0..dcp) |b| {
            counts[groupOwner(dcp, rank, b) * 8 + parity(dcp, rank)] += 1;
        };
        for (0..8) |g| for (0..dcp) |sh| try std.testing.expectEqual(@as(usize, 1), counts[g * 8 + sh]);
    }
}

test "cache bytes a rank: N streams with the cache split take what one stream of N x the tokens takes" {
    const f: Formats = .{ .latent_row = 512 * 2, .index_row = 132, .index_dim = 128 };
    const per = tokenBytes(79, 22, f, true, true);
    const one = rankCacheBytes(per, 1 << 20, 0, 1, 8);
    const many = rankCacheBytes(per, (1 << 20) / 4, 0, 4, 8);
    try std.testing.expectApproxEqRel(@as(f64, @floatFromInt(one)), @as(f64, @floatFromInt(many)), 0.01);
    // the fp8 plane is what makes 1M feasible: 84,956 B/token -> 89.0 GB at 1M, 11.1 GB a rank at DCP8
    const total = per * (1 << 20);
    try std.testing.expectEqual(@as(usize, 84_956), per);
    try std.testing.expect(total > 88_000_000_000 and total < 90_000_000_000);
}

test "the boot check of several streams: a whole reserve, the token cap, a fitting context" {
    const per: u64 = 40 << 30;
    try std.testing.expect(streamsFit(per, 1, 120 << 30, streams_reserve_gb).ok);
    try std.testing.expect(!streamsFit(per, 3, 120 << 30, streams_reserve_gb).ok);
    try std.testing.expect(streamsTokensOk(500_000, 8));
    try std.testing.expect(!streamsTokensOk(600_000, 8));
}

test "the DCP choice: env wins, else the world past the auto threshold" {
    try std.testing.expectEqual(@as(usize, 1), try choose(1000, 8, null));
    try std.testing.expectEqual(@as(usize, 8), try choose(300_000, 8, null));
    try std.testing.expectEqual(@as(usize, 2), try choose(300_000, 8, 2));
    try std.testing.expectEqual(@as(usize, 8), try choose(1000, 8, 8));
    try std.testing.expectError(error.BadDcp, choose(1000, 8, 3));
}