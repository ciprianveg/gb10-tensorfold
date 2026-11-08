//! Step 3's gate: the TP group's bring-up, byte-verified. `tf-glm53 tp-smoke [WORLD] [RANK]` (or
//! TF_WORLD / TF_RANK) opens 1.1's NCCL `tp_group.Group` (NCCL_COMM_ID names rank 0's address), gathers
//! one value a rank with `Group.gather` and a 4 KiB pattern with `allGather`, and checks every byte:
//! rank r's slot must hold rank r's value/pattern, in rank order. World 1 runs on one node (NCCL still
//! needs a COMM_ID); world 8 is the fleet run the serve will use.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

fn envInt(name: [:0]const u8, fallback: usize) usize {
    const v = std.c.getenv(name) orelse return fallback;
    return std.fmt.parseInt(usize, std.mem.sliceTo(v, 0), 10) catch fallback;
}

pub fn run(gpu: check.Gpu, args: []const []const u8) !void {
    const world: usize = if (args.len > 0) try std.fmt.parseInt(usize, args[0], 10) else envInt("TF_WORLD", 1);
    const rank: usize = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else envInt("TF_RANK", 0);
    if (world == 0 or rank >= world) return error.BadRank;

    // NCCL 2.2x dropped NCCL_COMM_ID: when TF_NCCL_ID_FILE is set, the id travels through that file
    const g = if (std.c.getenv("TF_NCCL_ID_FILE")) |f|
        try cuda.tp_group.Group.openShared(gpu.io, rank, world, std.mem.sliceTo(f, 0))
    else
        try cuda.tp_group.Group.open(rank, world);
    defer @constCast(&g).close();
    const stream = try cuda.Stream.init(gpu.d, true);
    defer @constCast(&stream).deinit();

    // ---- one u64 a rank, gathered in rank order
    const got = try gpu.gpa.alloc(u64, world);
    defer gpu.gpa.free(got);
    const mine: u64 = 0xABCD0000_00000000 | (@as(u64, rank) * 0x9E37);
    try g.gather(u64, gpu.d, stream, mine, got);
    var value_ok = true;
    for (0..world) |r| {
        const want: u64 = 0xABCD0000_00000000 | (@as(u64, r) * 0x9E37);
        if (got[r] != want) {
            value_ok = false;
            std.debug.print("  value slot {d}: {x} for {x}\n", .{ r, got[r], want });
        }
    }

    // ---- a 4 KiB pattern a rank, all-gathered, every byte checked
    const bytes = 4096;
    var one = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    defer one.free();
    var all = try cuda.DeviceBuffer.alloc(gpu.d, bytes * world);
    defer all.free();
    var pat: [bytes]u8 = undefined;
    for (&pat, 0..) |*b, i| b.* = @truncate(rank * 31 + i);
    try one.uploadAsync(0, &pat, stream.handle);
    try g.allGather(stream, one.ptr, all.ptr, bytes);
    try stream.synchronize();
    try g.check();
    const shot = try gpu.gpa.alloc(u8, bytes * world);
    defer gpu.gpa.free(shot);
    try all.download(0, shot);
    var bytes_ok = true;
    var first_bad: ?usize = null;
    for (0..world) |r| {
        for (0..bytes) |i| {
            const want: u8 = @truncate(r * 31 + i);
            if (shot[r * bytes + i] != want) {
                bytes_ok = false;
                if (first_bad == null) first_bad = r * bytes + i;
            }
        }
    }

    std.debug.print(
        "RESULT tp-smoke rank={d}/{d} gather={s} allgather4096={s} first_bad={any}\n",
        .{ rank, world, if (value_ok) "ok" else "BAD", if (bytes_ok) "ok" else "BAD", first_bad },
    );
    if (!value_ok or !bytes_ok) return error.TpParity;
    check.pass("tp-smoke: rank {d}/{d}, gather and allGather byte-verified", .{ rank, world });
}