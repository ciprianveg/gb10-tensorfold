//! `tf-glm53 <command>` — the GLM-5.3 NVFP4-CSF executable (its own binary: the family imports the GLM
//! glue's files relatively, so it cannot share tf-cuda-test's module graph).
//!
//! Commands today: `load` (Q1/Q3's device load smoke). The serve and the step commands land at Q4.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm53_load = @import("glm53_load.zig");
const glm53_tp = @import("glm53_tp.zig");
const glm53_head = @import("glm53_head.zig");
const glm53_step = @import("glm53_step.zig");
const glm53_run = @import("glm53_run.zig");
const glm53_abench = @import("glm53_abench.zig");

const usage =
    \\usage: tf-glm53 <command>
    \\  load [DIR] [RANK] [WORLD] [LAYERS]  one rank's real device load of the NVFP4-CSF pack
    \\                                      (DIR defaults to TF_GLM53_NVFP4_DIR; LAYERS trims to a smoke)
    \\  tp-smoke [WORLD] [RANK]            the NCCL TP group's bring-up, gather/allGather byte-verified
    \\                                      (TF_NCCL_ID_FILE carries the id; TF_WORLD/TF_RANK set rank)
    \\  blocks                             STEP 1: assemble the forward's block set and print its census
    \\  head [DIR] [RANK] [WORLD]          STEP 4: the sharded head logits, the reduced argmax and the sampler
    \\  step [DIR] [LAYERS] [IDS]          STEP 5: prefill + one decode through the layer loop, with digests
    \\  run [DIR] [TEXT] [NGEN] [RANK] [WORLD]  the pack's tokenizer, a prompt in chunks, greedy decode to text
    \\  abench [--rows R] [--heads H] [--keys K] [--bench N]  the wide attention entry alone (§39 harness)
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    const cmd = args[1];
    const rest = args[2..];
    run(gpu, cmd, rest) catch |e| {
        std.debug.print("FAIL {s}: {t}\n", .{ cmd, e });
        return 1;
    };
    return 0;
}

fn run(gpu: check.Gpu, cmd: []const u8, rest: []const [:0]const u8) !void {
    if (std.mem.eql(u8, cmd, "load")) return glm53_load.run(gpu, rest);
    if (std.mem.eql(u8, cmd, "tp-smoke")) return glm53_tp.run(gpu, rest);
    if (std.mem.eql(u8, cmd, "blocks")) return glm53_load.blocks(gpu, rest);
    if (std.mem.eql(u8, cmd, "head")) return glm53_head.run(gpu, rest);
    if (std.mem.eql(u8, cmd, "step")) return glm53_step.run(gpu, rest);
    if (std.mem.eql(u8, cmd, "run")) return glm53_run.run(gpu, rest);
    if (std.mem.eql(u8, cmd, "abench")) return glm53_abench.run(gpu, rest);
    std.debug.print("{s}", .{usage});
    return error.UnknownCommand;
}