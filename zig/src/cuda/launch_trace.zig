//! TF_GLM53_LAUNCHES=1: the audit trail. One line a unique GEMM site, naming the exact kernel entry the call
//! resolved to, so "which GEMM is this layer actually running" is answered from the process, not from a reading
//! of the code (§39's top audit question: are the experts on the native NVFP4 MMA or on a bf16 fallback?).
const std = @import("std");

const Entry = struct { site: [40]u8, kernel: [56]u8, rows: usize, n: usize, k: usize };
var entries: [64]Entry = undefined;
var count: usize = 0;
var on: bool = false;
var probed: bool = false;

pub fn enabled() bool {
    if (!probed) {
        probed = true;
        on = if (std.c.getenv("TF_GLM53_LAUNCHES")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else false;
    }
    return on;
}

/// Record (site, kernel, shape) once. `rows` is bucketed so a 64-row chunk does not print 64 lines.
pub fn hit(site: []const u8, kernel: []const u8, rows: usize, n: usize, k: usize) void {
    if (!enabled()) return;
    const rb: usize = if (rows >= 256) 256 else if (rows >= 64) 64 else if (rows >= 16) 16 else 1;
    for (entries[0..count]) |e| {
        if (e.rows == rb and e.n == n and e.k == k and std.mem.eql(u8, e.site[0..std.mem.indexOfScalar(u8, &e.site, 0) orelse e.site.len], site)) return;
    }
    if (count == entries.len) return;
    var e: Entry = .{ .site = @splat(0), .kernel = @splat(0), .rows = rb, .n = n, .k = k };
    @memcpy(e.site[0..@min(site.len, e.site.len - 1)], site[0..@min(site.len, e.site.len - 1)]);
    @memcpy(e.kernel[0..@min(kernel.len, e.kernel.len - 1)], kernel[0..@min(kernel.len, e.kernel.len - 1)]);
    entries[count] = e;
    count += 1;
    std.debug.print("LAUNCH site={s} kernel={s} rows={d} n={d} k={d}\n", .{ site, kernel, rb, n, k });
}