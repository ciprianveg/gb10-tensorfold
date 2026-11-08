//! NCCL opened at run time for tensor-parallel ranks: communicators and the collectives the lane engines use.

const std = @import("std");
const platform = @import("sys");
const abi = @import("abi.zig");
const tp_watch = @import("tp_watch.zig");

pub const minimum_version: c_int = 23000;

pub fn requireVersion(loaded: c_int) error{NcclVersion}!void {
    if (loaded < minimum_version) return error.NcclVersion;
}

pub const Result = c_int;
pub const Comm = ?*opaque {};
pub const UniqueId = extern struct { internal: [128]u8 };

pub const DataType = enum(c_int) { i8 = 0, u8 = 1, i32 = 2, u32 = 3, i64 = 4, u64 = 5, f16 = 6, f32 = 7, f64 = 8, bf16 = 9 };
pub const RedOp = enum(c_int) { sum = 0, prod = 1, max = 2, min = 3, avg = 4 };

/// NCCL 2.30 public ncclConfig_v23000 ABI. New nonblocking rank owners explicitly require 2.30 or later.
pub const Config = extern struct {
    size: usize = @sizeOf(Config),
    magic: u32 = 0xcafebeef,
    version: u32 = 23000,
    blocking: c_int = 0,
    cga_cluster_size: c_int = std.math.minInt(c_int),
    min_ctas: c_int = std.math.minInt(c_int),
    max_ctas: c_int = std.math.minInt(c_int),
    net_name: ?[*:0]const u8 = null,
    split_share: c_int = std.math.minInt(c_int),
    traffic_class: c_int = std.math.minInt(c_int),
    comm_name: ?[*:0]const u8 = null,
    collnet_enable: c_int = std.math.minInt(c_int),
    cta_policy: c_int = std.math.minInt(c_int),
    shrink_share: c_int = std.math.minInt(c_int),
    nvls_ctas: c_int = std.math.minInt(c_int),
    n_channels_per_net_peer: c_int = std.math.minInt(c_int),
    nvlink_centric_sched: c_int = std.math.minInt(c_int),
    graph_usage_mode: c_int = std.math.minInt(c_int),
    num_rma_ctx: c_int = std.math.minInt(c_int),
    max_p2p_peers: c_int = std.math.minInt(c_int),
    graph_stream_ordering: c_int = std.math.minInt(c_int),
};

pub const Error = error{ LibraryUnavailable, MissingSymbol, NcclFailed, NcclVersion };

pub const in_progress: Result = 7; // ncclInProgress: a non-blocking communicator's call is still running

const R = Result;
const D = abi.DevicePtr;

pub const Api = struct {
    ncclGetVersion: *const fn (*c_int) callconv(.c) R,
    ncclGetUniqueId: *const fn (*UniqueId) callconv(.c) R,
    ncclCommInitRank: *const fn (*Comm, c_int, UniqueId, c_int) callconv(.c) R,
    ncclCommInitAll: *const fn ([*]Comm, c_int, ?[*]const c_int) callconv(.c) R,
    ncclCommDestroy: *const fn (Comm) callconv(.c) R,
    ncclCommAbort: *const fn (Comm) callconv(.c) R,
    ncclCommGetAsyncError: *const fn (Comm, *R) callconv(.c) R,
    ncclGetErrorString: *const fn (R) callconv(.c) ?[*:0]const u8,
    ncclAllReduce: *const fn (D, D, usize, DataType, RedOp, Comm, abi.Stream) callconv(.c) R,
    ncclAllGather: *const fn (D, D, usize, DataType, Comm, abi.Stream) callconv(.c) R,
    ncclBroadcast: *const fn (D, D, usize, DataType, c_int, Comm, abi.Stream) callconv(.c) R,
    ncclSend: *const fn (D, usize, DataType, c_int, Comm, abi.Stream) callconv(.c) R,
    ncclRecv: *const fn (D, usize, DataType, c_int, Comm, abi.Stream) callconv(.c) R,
    ncclReduceScatter: *const fn (D, D, usize, DataType, RedOp, Comm, abi.Stream) callconv(.c) R,
    ncclGroupStart: *const fn () callconv(.c) R,
    ncclGroupEnd: *const fn () callconv(.c) R,
};

pub const Library = struct {
    lib: platform.DynLib,
    api: Api,

    pub fn open() Error!Library {
        return openPath("libnccl.so.2");
    }

    pub fn openPath(path: []const u8) Error!Library {
        var lib = platform.DynLib.open(path) catch return error.LibraryUnavailable;
        errdefer lib.close();
        var api: Api = undefined;
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(api, name) = lib.lookup(T, name) orelse {
                std.log.err("{s} has no {s}", .{ path, name });
                return error.MissingSymbol;
            };
        }
        return .{ .lib = lib, .api = api };
    }

    pub fn close(self: *Library) void {
        self.lib.close();
    }

    pub fn check(self: *const Library, r: Result, what: []const u8) Error!void {
        if (r == 0) return;
        if (@import("builtin").is_test) return error.NcclFailed; // a test's injected failure: the runner fails error logs
        const s = self.api.ncclGetErrorString(r);
        std.log.err("{s}: {s} ({d})", .{ what, if (s) |p| std.mem.span(p) else "?", r });
        tp_watch.fatal(what); // with the ranks' lines open, every rank stops with it; with none, the caller hears it
        return error.NcclFailed;
    }

    /// `post(ctx)`'s sends and receives as one group; a failed post still ends it, so later calls stay out of it.
    pub fn group(self: *const Library, ctx: anytype, comptime post: fn (@TypeOf(ctx)) anyerror!void) anyerror!void {
        try self.check(self.api.ncclGroupStart(), "ncclGroupStart");
        post(ctx) catch |err| {
            _ = self.api.ncclGroupEnd();
            return err;
        };
        try self.check(self.api.ncclGroupEnd(), "ncclGroupEnd");
    }

    /// NCCL's version as 10000 * major + 100 * minor + patch.
    pub fn version(self: *const Library) Error!c_int {
        var v: c_int = 0;
        try self.check(self.api.ncclGetVersion(&v), "ncclGetVersion");
        return v;
    }

    /// Refuses a library older than tensor parallelism supports (upstream's gate).
    pub fn requireSupportedVersion(self: *const Library) Error!void {
        const loaded = try self.version();
        requireVersion(loaded) catch |err| {
            std.log.err("NCCL {d}.{d}.{d} loaded; tensor parallelism requires at least {d}.{d}.{d}", .{
                @divTrunc(loaded, 10000),          @mod(@divTrunc(loaded, 100), 100),          @mod(loaded, 100),
                @divTrunc(minimum_version, 10000), @mod(@divTrunc(minimum_version, 100), 100), @mod(minimum_version, 100),
            });
            return err;
        };
    }
};

test "nonblocking communicator config preserves the public ABI defaults" {
    const c: Config = .{};
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(Config));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(Config, "net_name"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(Config, "comm_name"));
    try std.testing.expect(c.size == 96 and c.blocking == 0 and c.max_ctas == std.math.minInt(c_int));
}

/// One rank's NCCL communicator (ncclCommInitRank) and the collectives the GLM-5.3 engine runs on its compute stream:
/// every call is enqueued on `stream` (stream order is the only ordering; nothing here synchronizes).
pub const Communicator = struct {
    lib: *const Library,
    comm: Comm,
    rank: usize,
    world: usize,

    /// Rank 0 makes the id (ncclGetUniqueId) and hands its 128 bytes to the other ranks (rendezvous.zig).
    pub fn uniqueId(lib: *const Library) Error!UniqueId {
        var id: UniqueId = undefined;
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        return id;
    }

    /// Joins the world: blocks until every rank has called it with the same id (the CUDA context must be current).
    pub fn init(lib: *const Library, id: UniqueId, rank: usize, world: usize) Error!Communicator {
        var c: Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&c, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = c, .rank = rank, .world = world };
    }

    pub fn deinit(self: *Communicator) void {
        _ = self.lib.api.ncclCommDestroy(self.comm);
        self.* = undefined;
    }

    /// recv [world * count] <- every rank's send [count] in rank order (comm.NCCL.all_gather).
    pub fn allGather(self: *const Communicator, send: D, recv: D, count: usize, dt: DataType, stream: abi.Stream) Error!void {
        try self.lib.check(self.lib.api.ncclAllGather(send, recv, count, dt, self.comm, stream), "ncclAllGather");
    }

    /// recv <- the sum of every rank's send (NCCL's ring: ranks alike, the order depends on the size).
    pub fn allReduce(self: *const Communicator, send: D, recv: D, count: usize, dt: DataType, op: RedOp, stream: abi.Stream) Error!void {
        try self.lib.check(self.lib.api.ncclAllReduce(send, recv, count, dt, op, self.comm, stream), "ncclAllReduce");
    }

    /// recv [count] <- block `rank` of the sum of every rank's send [world * count] (comm.NCCL.reduce_scatter: NCCL's
    /// ring, so the summation order depends on the size; every rank of one configuration gets the same bits).
    pub fn reduceScatter(self: *const Communicator, send: D, recv: D, count: usize, dt: DataType, op: RedOp, stream: abi.Stream) Error!void {
        try self.lib.check(self.lib.api.ncclReduceScatter(send, recv, count, dt, op, self.comm, stream), "ncclReduceScatter");
    }

    /// One in-place all-gather a buffer, as one NCCL group (comm.NCCL.all_gather_group, fused._ag_rows): buffer i
    /// holds world blocks of `counts[i]` elements and this rank's block (rank * counts[i]) is the send.
    pub fn allGatherInPlace(self: *const Communicator, bufs: []const D, counts: []const usize, dts: []const DataType, elem_bytes: []const usize, stream: abi.Stream) Error!void {
        try self.lib.check(self.lib.api.ncclGroupStart(), "ncclGroupStart");
        var failed: ?Error = null;
        for (bufs, counts, dts, elem_bytes) |b, n, dt, eb| {
            const send = b + @as(D, self.rank * n * eb);
            self.lib.check(self.lib.api.ncclAllGather(send, b, n, dt, self.comm, stream), "ncclAllGather") catch |e| {
                failed = e;
                break;
            };
        }
        try self.lib.check(self.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        if (failed) |e| return e;
    }
};
