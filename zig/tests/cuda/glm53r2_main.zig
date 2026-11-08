//! `tf-glm53r2` — the A1 transplant's entry point and A2's home.
//!
//! The family under this binary is `zig/src/families/glm53r2/`: r2's engine (`glm_moe_dsa` @ 70e68f66) copied
//! verbatim as its own module, with the 1.1 loader staying the weights source. A1's gate is that this binary
//! compiles, links the whole family and reports the blob census; A2 adds the binder command here.
//!
//! `TF_GLM53_KV8` / `TF_GLM53_DCP` are r2's own switches: the census matrix boots this binary with them set, so
//! the arms are wired from the first boot rather than retrofitted (§43's note).
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm53r2 = @import("glm53r2");
const glm53r2_bind = @import("glm53r2_bind.zig");
const glm53r2_run = @import("glm53r2_run.zig");
const glm53r2_serve = @import("glm53r2_serve.zig");

const usage =
    \\tf-glm53r2: GLM-5.3 on r2's engine (the A1 transplant; the 1.1 loader stays the weights source)
    \\  selftest   the cache arithmetic (§43) and the blob census - no GPU
    \\  bind <pack> <rank> <world> <rows>   A2/E2: the 1.1 loader's weights through r2's forward, a digest a layer
    \\  run <pack> <text> <ngen> <rank> <world>   A4a: generation on r2's engine (chunked prefill + kv8 decode)
 \\  serve <pack> <port> [context]   §98: the HTTP surface — the served Boot+Session, rank 0's HTTP side,
 \\      the ranks following (TF_GLM53_RANK/_WORLD/_MASTER + TF_GLM53_BIND for the 1.1 loader's weights)
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    // §70: the ARENA here was the 2x-residency bug. `bind` hands this allocator to the 1.1 loader, whose
    // per-layer staging (packExperts' packed up/down tables ~0.8 GiB, the nvfp4_slice codes) is `free`d after
    // upload — an arena makes every free a no-op, so 1.12 GiB accumulated per layer (measured: anon 2.68 ->
    // 19.43 GiB over 15 layers) and the 78-layer load crossed the 3 GiB guard line. The port binary has always
    // used init.gpa (glm53_main.zig:36); this makes the two mains agree.
    const gpa = init.gpa;

    // the census arms, read exactly as r2's Runner reads them (its settings digest covers both)
    const kv8 = if (std.c.getenv("TF_GLM53_KV8")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else false;
    const dcp: usize = if (std.c.getenv("TF_GLM53_DCP")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 1 else 1;
    const kv8_v2 = if (std.c.getenv("TF_GLM53_KV8_V2")) |v| !std.mem.eql(u8, std.mem.span(v), "0") else false;

    std.debug.print("tf-glm53r2: family module up (r2 compute path, {d} files)\n", .{glm53r2.files});
    std.debug.print("tf-glm53r2: census arms kv8={} dcp={d} kv8_v2={} (TF_GLM53_KV8 / TF_GLM53_DCP / TF_GLM53_KV8_V2)\n", .{ kv8, dcp, kv8_v2 });

    if (args.len >= 2 and std.mem.eql(u8, args[1], "bind")) {
        var driver = try cuda.Driver.open();
        defer driver.close();
        var ctx = try cuda.Context.init(&driver, 0);
        defer ctx.deinit();
        const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = gpa, .io = init.io };
        glm53r2_bind.run(gpu, args[2..]) catch |e| {
            std.debug.print("FAIL bind: {t}\n", .{e});
            return 1;
        };
        return 0;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "serve")) {
        var driver = try cuda.Driver.open();
        defer driver.close();
        var ctx = try cuda.Context.init(&driver, 0);
        defer ctx.deinit();
        const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = gpa, .io = init.io };
        glm53r2_serve.serve(gpu, args[2..]) catch |e| {
            std.debug.print("FAIL serve: {t}\n", .{e});
            return 1;
        };
        return 0;
    }
    if (args.len >= 2 and std.mem.eql(u8, args[1], "run")) {
        var driver = try cuda.Driver.open();
        defer driver.close();
        var ctx = try cuda.Context.init(&driver, 0);
        defer ctx.deinit();
        const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = gpa, .io = init.io };
        glm53r2_run.run(gpu, args[2..]) catch |e| {
            std.debug.print("FAIL run: {t}\n", .{e});
            return 1;
        };
        return 0;
    }
    if (args.len < 2 or !std.mem.eql(u8, args[1], "selftest")) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    glm53r2.selftest(gpa, init.io) catch |e| {
        std.debug.print("FAIL selftest: {t}\n", .{e});
        return 1;
    };
    std.debug.print("PASS selftest: the family's cache arithmetic and blob census\n", .{});
    return 0;
}