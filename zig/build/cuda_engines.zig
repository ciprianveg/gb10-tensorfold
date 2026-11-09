//! The CUDA engines a native server opens (native/cuda.zig) and the family modules they import.
const std = @import("std");

/// The GLM-5.3-Flash CUDA family over the runtime, the core, the lane core, canon's shares, core/heat.zig, engine_api.
pub fn glmCuda(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, heat: *std.Build.Module, api: *std.Build.Module) *std.Build.Module {
    const canon = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon.zig"), .target = target, .optimize = optimize });
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("canon", canon);
    m.addImport("lanes", lanes);
    m.addImport("heat", heat);
    m.addImport("tp", @import("tp.zig").module(b, target, optimize, cuda)); // over the engine's own cuda module
    m.addImport("engine_api", api); // TF_GLM_KV_SHARE's prompt_radix store: family()'s, the server's
    return m;
}

/// LFM2 (d1's decision model) on CUDA: config, weights, kernels and the forward that scores labels.
pub fn lfm2Module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/lfm2/lfm2.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
}

/// Flash Next's CUDA family, sharing the maintained runtime and lane core.
pub fn flashnextModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, qwen_vision: *std.Build.Module, api: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_launch.zig"), .target = target, .optimize = optimize, .link_libc = true });
    m.addImport("cuda", cuda);
    m.addImport("core", core);
    m.addImport("lanes", lanes);
    m.addImport("qwen_vision", qwen_vision);
    m.addImport("living", core.import_table.get("living").?);
    m.addImport("engine_api", api); // its disk image (cuda_disk.zig) names the rows store's types: family()'s, the server's
    return m;
}

/// The CUDA engines a native server opens (native/cuda.zig), over the given runtime and families.
pub fn engines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, heat: *std.Build.Module, qwen35: *std.Build.Module, qwen_vision: *std.Build.Module) struct { api: *std.Build.Module, engines: *std.Build.Module } {
    const api = qwen35.import_table.get("engine_api").?; // family()'s, so the server and qwen3_5 share one
    const mod = b.createModule(.{
        .root_source_file = b.path("zig/src/native/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "engine_api", .module = api },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "nemotron", .module = nemotron },
            .{ .name = "drama3", .module = @import("drama3_host.zig").module(b, target, optimize, cuda, core) },
            .{ .name = "lfm2", .module = lfm2Module(b, target, optimize, cuda, core) },
            .{ .name = "core", .module = core },
            .{ .name = "flashnext", .module = flashnextModule(b, target, optimize, cuda, core, lanes, qwen_vision, api) },
            .{ .name = "minimax_m3_cuda", .module = @import("minimax_cuda.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "qwen35", .module = qwen35 },
            .{ .name = "kolibri1", .module = @import("kolibri.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "gemma4_cuda", .module = @import("gemma4_cuda.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "spark25", .module = @import("spark25.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "dory", .module = @import("dory.zig").module(b, target, optimize, cuda, core, lanes) },
        },
    });
    mod.addImport("qwen4", @import("qwen4.zig").module(b, target, optimize, core));
    // One lane or the other — the GLM glue files cannot sit in two modules of one binary.
    // The lane build (TF_BUILD_GLM53_LANE) serves glm53r2 in place of the GLM-5.3-Flash family.
    const lane_env = b.graph.environ_map.get("TF_BUILD_GLM53_LANE");
    std.debug.print("LANE-DEBUG: env-seen={any}\n", .{lane_env != null});
    const lane_options = b.addOptions();
    lane_options.addOption(bool, "glm53_lane", lane_env != null);
    mod.addOptions("lane_options", lane_options);
    if (lane_env != null) {
        mod.addImport("glm53r2", glm53r2Cuda(b, target, optimize, cuda, core, glm53Cuda(b, target, optimize, cuda, core), lanes));
    } else {
        mod.addImport("glm_cuda", glmCuda(b, target, optimize, cuda, core, lanes, heat, api));
    }
    return .{ .api = api, .engines = mod };
}

/// The cluster canon module (glm53's base module imports it).
fn canonModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon.zig"), .target = target, .optimize = optimize });
}

/// GLM-5.3 (glm_moe_dsa) on CUDA from the NVFP4-CSF pack: its own family module (self-contained).
pub fn glm53Cuda(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module) *std.Build.Module {
    const canon = canonModule(b, target, optimize);
    return b.createModule(.{ .root_source_file = b.path("zig/src/families/glm53_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "cuda", .module = cuda },
        .{ .name = "core", .module = core },
        .{ .name = "canon", .module = canon },
    } });
}

/// GLM-5.3's r2 engine (families/glm53r2): r2's forward/prompt/decode/engine verbatim,
/// with the 1.1 loader still the weights source. Its own module because r2's family compiles as a unit.
pub fn glm53r2Cuda(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, glm53: *std.Build.Module, lanes_mod: *std.Build.Module) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm53r2/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{
        .{ .name = "cuda", .module = cuda },
        .{ .name = "core", .module = core },
        .{ .name = "glm53", .module = glm53 },
        .{ .name = "lanes", .module = lanes_mod },
    } });
    // r2's RoCE one-shot (Phase 3a): plain C over libibverbs, as the EXL tree compiles it. Only the serve/run
    // paths reference it (cuda_runner/cuda_engine), but the module links it once for all its users.
    m.addCSourceFile(.{ .file = b.path("zig/src/families/glm53r2/roce_proxy.c"), .flags = &.{ "-O2", "-std=gnu11" } });
    m.linkSystemLibrary("ibverbs", .{});
    // The L2 shim: mlx5dv_create_qp/ibv_wr_* live in libmlx5 (the host's rdma-core).
    m.linkSystemLibrary("mlx5", .{});
    m.linkSystemLibrary("pthread", .{});
    return m;
}
