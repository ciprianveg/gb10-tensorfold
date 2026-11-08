//! TP8 for GLM-5.3 on 1.1's collectives: a rank writes its fp32 partials, the group all-gathers them
//! and the device sums rank 0 first. This is 1.1's own model (`tp_group.Group` + `glue.rankSum` /
//! `glue.residualAdd(..., world)`), not a second collective stack:
//!  - decode: 1 row x 6144 x 4 B = 24 KiB a partial, twice a layer — NCCL all-gather, no RoCE;
//!  - prefill: a 128-row chunk is 3.1 MiB a partial, twice a layer — the same call, bandwidth-bound;
//!  - 1.1's RoCE gather is two-rank-only (`Group.openRoce` refuses other worlds), so TP8 stays NCCL.
//! The GB10 notes apply: NCCL_CUMEM_HOST_ENABLE=0 / NCCL_CUMEM_ENABLE=0 and NCCL_RUNTIME_CONNECT=0 (set
//! by the Flash family's native path for the same driver-pool reason, #995).
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("glue/cuda_glue.zig");

pub const Tp = struct {
    group: *const cuda.tp_group.Group,
    peers: u64, // fp32 [world][rows][dims]: the gathered partials, one slot a rank
    peer_stride: usize, // rows * dims * 4
    me: usize,
    world: usize,

    pub fn bytes(world: usize, rows: usize, dims: usize) usize {
        return world * rows * dims * 4;
    }

    /// Gather every rank's fp32 [rows][dims] partial at `mine` into `peers`, then sum with rank 0's
    /// first into `out` (bf16), as glue.rank_sum does.
    pub fn sumPartial(t: *const Tp, set: glue.Set, stream: cuda.Stream, mine: u64, out: u64, rows: usize, dims: usize) !void {
        try t.group.allGather(stream, mine, t.peers, rows * dims * 4);
        try glue.rankSum(set, stream, mine, t.peers, t.peer_stride, out, t.me, rows, dims, t.world);
    }

    /// x = bf16(x + the gathered partial's ranks summed rank 0 first), the residual site.
    pub fn residual(t: *const Tp, set: glue.Set, stream: cuda.Stream, x: u64, mine: u64, rows: usize, dims: usize) !void {
        try t.group.allGather(stream, mine, t.peers, rows * dims * 4);
        try glue.residualAdd(set, stream, x, x, t.peers, rows, dims, t.world);
    }
};

test "TP8 partial sizing: 24 KiB a decode row, 3.1 MiB a 128-row chunk, per rank slot" {
    const rows = 1;
    try std.testing.expectEqual(@as(usize, 8 * 1 * 6144 * 4), Tp.bytes(8, rows, 6144));
    try std.testing.expectEqual(@as(usize, 8 * 128 * 6144 * 4), Tp.bytes(8, 128, 6144));
}