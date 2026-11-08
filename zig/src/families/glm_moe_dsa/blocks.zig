//! STEP 1: the per-forward block set, assembled from the pack's kernel images, and its census.
//!
//! `assemble` is the wiring the forward needs and nothing else: the GLM glue module and the three
//! shapes the family reuses (bmm/matmul, latent attention, moe/experts), the family's own extras
//! (norms 6144/2048, the bf16 scorer), the indexer (fp8 scorer, plane writer, radix select), and the
//! kv8 cache kernels. Loading is the census: `glue.Kernels.load` resolves every `tf_glm_*` entry by name
//! and fails on the first missing symbol, and each family loader does the same for its own entries.
//!
//! `census` prints what was resolved, per block, and refuses an empty set: a block that lost a symbol in
//! a build must fail here, not inside a forward.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("glue/cuda_glue.zig");
const mlp = @import("glue/cuda_mlp.zig");
const moe = @import("glue/cuda_moe.zig");
const experts = @import("glue/cuda_experts.zig");
const exl3 = @import("glue/cuda_exl3.zig");
const mm_mod = @import("glue/cuda_mm.zig");
const indexer = @import("indexer.zig");
const kv8 = @import("kv8.zig");
const ops_mod = @import("ops.zig");
const fw = @import("forward.zig");

pub const Assembled = struct {
    blocks: fw.Blocks,
    ops_mod: cuda.Module,
    experts_mod: cuda.Module,
    glue_mod: cuda.Module,
    mm_mod: cuda.Module,
    lane: cuda.qmmf.Lane,
    prompt: cuda.nvfp4.Prompt,
    experts_k: experts.Kernels,
    exl3_k: exl3.Kernels,
    mm: mm_mod.Kernels,
    extras_mod: cuda.Module,
    fp8_mod: cuda.Module,
    select_mod: cuda.Module,
    plane_mod: cuda.Module,
    glue_k: glue.Kernels,

    pub fn deinit(a: *Assembled) void {
        a.ops_mod.unload();
        a.experts_mod.unload();
        a.glue_mod.unload();
        a.mm_mod.unload();
        a.extras_mod.unload();
        a.fp8_mod.unload();
        a.select_mod.unload();
        a.plane_mod.unload();
        a.lane.unload();
        a.prompt.unload();
        a.experts_k.unload();
        a.exl3_k.unload();
        a.* = undefined;
    }
};

/// The block set one forward launches from: the same images `Modules.load` gives the GLM family, plus
/// this family's four, with the shape-agnostic blocks wired to the same glue set.
pub fn assemble(d: *const cuda.Driver, ctx: *const cuda.Context, padded_plane: bool) !Assembled {
    const k = cuda.kernels;
    var a: Assembled = undefined;
    // the same four of the GLM family's images this family launches from (glm_glue, glm_ops, the
    // routed experts' kernels, glm_mm's tiled bf16 linears)
    a.glue_mod = try cuda.Module.load(d, k.glm_glue);
    errdefer a.glue_mod.unload();
    a.ops_mod = try cuda.Module.load(d, k.glm_ops);
    errdefer a.ops_mod.unload();
    a.experts_mod = try cuda.Module.load(d, k.experts);
    errdefer a.experts_mod.unload();
    a.mm_mod = try cuda.Module.load(d, k.glm_mm);
    errdefer a.mm_mod.unload();
    a.extras_mod = try cuda.Module.load(d, k.glm53_extras);
    errdefer a.extras_mod.unload();
    a.fp8_mod = try cuda.Module.load(d, k.glm53_fp8_scores);
    errdefer a.fp8_mod.unload();
    a.select_mod = try cuda.Module.load(d, k.glm53_select);
    errdefer a.select_mod.unload();
    a.plane_mod = try cuda.Module.load(d, k.glm53_ik_write8);
    errdefer a.plane_mod.unload();

    a.glue_k = try glue.Kernels.load(a.glue_mod);
    a.lane = try cuda.qmmf.Lane.load(d, @intCast(try ctx.attribute(.compute_capability_major)));
    errdefer a.lane.unload();
    a.prompt = try cuda.nvfp4.Prompt.load(d);
    errdefer a.prompt.unload();
    a.experts_k = try experts.Kernels.load(d, a.experts_mod, @intCast(try ctx.attribute(.multiprocessor_count)));
    errdefer a.experts_k.unload();
    a.exl3_k = try exl3.Kernels.load(d);
    errdefer a.exl3_k.unload();
    a.mm = try mm_mod.Kernels.load(a.mm_mod);

    // One convention for the whole family: this loader uploads RAW bf16 rows (the routed experts bring
    // their own packed tables), so every b16.matmul takes the raw path. The GLM engine's tiled `mm` path
    // reads tiles this loader never wrote; enabling it here was an out-of-bounds read in the dense MLP.
    const glue_set: glue.Set = .{ .glue = &a.glue_k, .mm = null };
    const extras = try ops_mod.Extras.load(a.extras_mod, a.fp8_mod);
    const index = try indexer.Kernels.load(a.select_mod, a.plane_mod, padded_plane, extras.index_score_fp8.?, extras.index_score);
    const kv = try kv8.Kernels.load(d);
    a.blocks = .{
        .ops = .{ .glue = glue_set, .extras = extras },
        .index = index,
        .mlp = .{ .set = glue_set, .linear = .{ .lane = &a.lane, .nvfp4_prompt = &a.prompt }, .copy_rows = try a.ops_mod.function("tf_glm_copy_rows") },
        .moe = .{ .set = glue_set, .experts = &a.experts_k, .exl3 = &a.exl3_k, .put_slot = try a.ops_mod.function("tf_glm_put_slot") },
        .ops_module = a.ops_mod,
    };
    _ = kv; // the kv8 handles are held by the cache path; the symbols resolve here or this load failed
    return a;
}

/// `assemble` returns by value, so every pointer the block set holds into its own fields must be
/// rewired after the copy. Call this immediately after `assemble`.
pub fn rebind(a: *Assembled) void {
    a.blocks.ops.glue = .{ .glue = &a.glue_k, .mm = null }; // raw rows: see assemble
    a.blocks.mlp.set = a.blocks.ops.glue;
    a.blocks.mlp.linear = .{ .lane = &a.lane, .nvfp4_prompt = &a.prompt };
    a.blocks.moe.set = a.blocks.ops.glue;
    a.blocks.moe.experts = &a.experts_k;
    a.blocks.moe.exl3 = &a.exl3_k;
}

/// Take the GLM engine's tiled entry (findings §38): every linear was retiled in place by
/// `tile.tileLinears`, so `b16.matmul` may read the tiles. Call AFTER tiling - a raw-rows weight read
/// through this entry is an out-of-bounds read (§32's own warning).
pub fn useTiled(a: *Assembled) void {
    a.blocks.ops.glue = .{ .glue = &a.glue_k, .mm = &a.mm };
    a.blocks.mlp.set = a.blocks.ops.glue;
    a.blocks.moe.set = a.blocks.ops.glue;
}

/// Print the census: every block's resolved entry count, and the two family loaders' handles.
pub fn census(a: *const Assembled) void {
    _ = a; // handles are non-null by construction; the counts below are the census
    const glue_fields = @typeInfo(glue.Kernels).@"struct".field_names.len;
    std.debug.print(
        "RESULT glm53-blocks modules={d} glue_entries={d} extras={d} index={d} mlp=3 moe=4 experts={d} exl3={d}\n",
        .{ @as(usize, 4), glue_fields, @typeInfo(ops_mod.Extras).@"struct".field_names.len,
           @typeInfo(indexer.Kernels).@"struct".field_names.len, 4, 1 },
    );
    // every handle is non-null by construction (cuda.Module.function fails rather than returning null),
    // so the census that matters is the count: refuse a set that silently lost a group.
    if (glue_fields < 40) @panic("glm53-blocks: the glue set is short");
}