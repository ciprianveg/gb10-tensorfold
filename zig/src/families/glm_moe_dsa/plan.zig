//! The GLM-5.3 NVFP4-CSF pack's tensor inventory, walked once: every module the CUDA loader reads,
//! with its full (unsharded) geometry and how a TP rank cuts it. The loader and the pack contract
//! test both walk this list, so a name or a shape can only be wrong in one place.
//!
//! Names are composed under `prefix` (the pack's own: "model."); a checkpoint opened with that prefix
//! stripped (`Checkpoint.openModelPrefix`) answers the names this module composes without it.
const std = @import("std");
const Config = @import("config.zig").Config;

/// The pack's tensor prefix, carried in every `Item.name`: the loader opens the checkpoint unfiltered
/// (`Checkpoint.openModel`) and looks names up exactly as the pack stores them (lm_head has no prefix).
pub const prefix = "model.";
/// The names a module's quantized form adds beside `.weight`.
pub const scale_2 = ".weight_scale_2";
pub const csf_fixed = ".weight_scale.nvfp4_csf_fixed";
pub const csf_exceptions = ".weight_scale.nvfp4_csf_exceptions";

/// A module path from its stored tensor name ("...gate_proj.weight" -> "...gate_proj"): the reader
/// takes modules, not tensors.
pub fn module(name: []const u8) []const u8 {
    return if (std.mem.endsWith(u8, name, ".weight")) name[0 .. name.len - ".weight".len] else name;
}

pub const Role = enum {
    embed,
    head,
    final_norm,
    in_norm,
    post_norm,
    q_a,
    q_a_norm,
    kv_a,
    kv_a_norm,
    q_b,
    kv_b,
    o_proj,
    idx_wk,
    idx_wproj,
    idx_wqb,
    idx_kn_w,
    idx_kn_b,
    dense_gate,
    dense_up,
    dense_down,
    router,
    router_bias,
    shared_gate,
    shared_up,
    shared_down,
    expert_gate,
    expert_up,
    expert_down,
    eh_proj,
    enorm,
    hnorm,
    shared_head_norm,
};

/// Stored form: bf16/f32 tensors ship as they are; fp4 modules ship `.weight` (u8 nibbles) plus
/// `.weight_scale_2`, `.weight_scale.nvfp4_csf_fixed` and `.weight_scale.nvfp4_csf_exceptions`.
pub const Flavor = enum { bf16, f32, fp4 };

pub const Item = struct {
    name: []const u8, // borrowed for the call only
    role: Role,
    flavor: Flavor,
    n: usize, // rows of the logical matrix (or elements of a vector); the full width, not the rank's
    k: usize, // columns; 0 for a vector
    layer: u32 = 0,
};

/// A vector's element count, or a matrix's (rows, columns).
pub fn dims(it: Item) [2]usize {
    return .{ it.n, it.k };
}

/// Walk every module the loader reads, in load order. `ctx` provides `fn item(self, Item) !void`.
pub fn walk(c: *const Config, ctx: anytype) !void {
    try visit(ctx, .{ .name = prefix ++ "embed_tokens.weight", .role = .embed, .flavor = .bf16, .n = c.vocab, .k = c.hidden });
    try visit(ctx, .{ .name = "lm_head.weight", .role = .head, .flavor = .bf16, .n = c.vocab, .k = c.hidden });
    try visit(ctx, .{ .name = prefix ++ "norm.weight", .role = .final_norm, .flavor = .bf16, .n = c.hidden, .k = 0 });
    for (0..c.layers + c.mtp) |li| try walkLayer(c, @intCast(li), ctx);
}

/// One layer's modules (the backbone or the MTP layer at index `layers`).
pub fn walkLayer(c: *const Config, i: u32, ctx: anytype) !void {
    var buf: [96]u8 = undefined;
    {
        const mtp = i == c.layers;
        inline for (.{
            .{ "input_layernorm.weight", Role.in_norm, c.hidden, 0 },
            .{ "post_attention_layernorm.weight", Role.post_norm, c.hidden, 0 },
            .{ "self_attn.q_a_proj.weight", Role.q_a, c.q_lora, c.hidden },
            .{ "self_attn.q_a_layernorm.weight", Role.q_a_norm, c.q_lora, 0 },
            .{ "self_attn.kv_a_proj_with_mqa.weight", Role.kv_a, c.kv_lora + c.rope, c.hidden },
            .{ "self_attn.kv_a_layernorm.weight", Role.kv_a_norm, c.kv_lora, 0 },
            .{ "self_attn.q_b_proj.weight", Role.q_b, c.heads * (c.nope + c.rope), c.q_lora },
            .{ "self_attn.kv_b_proj.weight", Role.kv_b, c.heads * (c.nope + c.v_dim), c.kv_lora },
            .{ "self_attn.o_proj.weight", Role.o_proj, c.hidden, c.heads * c.v_dim },
        }) |spec| {
            const name = try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.{s}", .{ i, spec[0] });
            try visit(ctx, .{ .name = name, .role = spec[1], .flavor = .bf16, .n = spec[2], .k = spec[3], .layer = i });
        }
        // the DSA indexer's five tensors on a full layer (GLM-5.3 has no pool-compress gate/ape)
        if (c.full.isSet(i)) {
            inline for (.{
                .{ "self_attn.indexer.wk.weight", Role.idx_wk, c.i_dim, c.hidden },
                .{ "self_attn.indexer.weights_proj.weight", Role.idx_wproj, c.i_heads, c.hidden },
                .{ "self_attn.indexer.wq_b.weight", Role.idx_wqb, c.i_heads * c.i_dim, c.q_lora },
                .{ "self_attn.indexer.k_norm.weight", Role.idx_kn_w, c.i_dim, 0 },
                .{ "self_attn.indexer.k_norm.bias", Role.idx_kn_b, c.i_dim, 0 },
            }) |spec| {
                const name = try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.{s}", .{ i, spec[0] });
                try visit(ctx, .{ .name = name, .role = spec[1], .flavor = .bf16, .n = spec[2], .k = spec[3], .layer = i });
            }
        }
        if (i < c.dense_layers) {
            inline for (.{
                .{ "mlp.gate_proj.weight", Role.dense_gate, c.dense_inter, c.hidden },
                .{ "mlp.up_proj.weight", Role.dense_up, c.dense_inter, c.hidden },
                .{ "mlp.down_proj.weight", Role.dense_down, c.hidden, c.dense_inter },
            }) |spec| {
                const name = try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.{s}", .{ i, spec[0] });
                try visit(ctx, .{ .name = name, .role = spec[1], .flavor = .bf16, .n = spec[2], .k = spec[3], .layer = i });
            }
        } else {
            inline for (.{
                .{ "mlp.gate.weight", Role.router, c.experts, c.hidden },
                .{ "mlp.gate.e_score_correction_bias", Role.router_bias, c.experts, 0 },
                .{ "mlp.shared_experts.gate_proj.weight", Role.shared_gate, c.moe_inter, c.hidden },
                .{ "mlp.shared_experts.up_proj.weight", Role.shared_up, c.moe_inter, c.hidden },
                .{ "mlp.shared_experts.down_proj.weight", Role.shared_down, c.hidden, c.moe_inter },
            }) |spec| {
                const name = try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.{s}", .{ i, spec[0] });
                try visit(ctx, .{ .name = name, .role = spec[1], .flavor = if (spec[1] == .router_bias) .f32 else .bf16, .n = spec[2], .k = spec[3], .layer = i });
            }
            for (0..c.experts) |e| {
                inline for (.{
                    .{ "gate_proj", Role.expert_gate, c.moe_inter, c.hidden },
                    .{ "up_proj", Role.expert_up, c.moe_inter, c.hidden },
                    .{ "down_proj", Role.expert_down, c.hidden, c.moe_inter },
                }) |spec| {
                    const name = try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.mlp.experts.{d}." ++ spec[0] ++ ".weight", .{ i, e });
                    // the MTP layer's experts stay BF16 (the pack's quant ignore-list)
                    try visit(ctx, .{ .name = name, .role = spec[1], .flavor = if (mtp) .bf16 else .fp4, .n = spec[2], .k = spec[3], .layer = i });
                }
            }
        }
        if (mtp) {
            inline for (.{
                .{ "eh_proj.weight", Role.eh_proj, c.hidden, 2 * c.hidden },
                .{ "enorm.weight", Role.enorm, c.hidden, 0 },
                .{ "hnorm.weight", Role.hnorm, c.hidden, 0 },
                .{ "shared_head.norm.weight", Role.shared_head_norm, c.hidden, 0 },
            }) |spec| {
                const name = try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.{s}", .{ i, spec[0] });
                try visit(ctx, .{ .name = name, .role = spec[1], .flavor = .bf16, .n = spec[2], .k = spec[3], .layer = i });
            }
        }
    }
}

/// An expert projection's stored tensor name.
pub fn expertName(buf: []u8, c: *const Config, i: u32, e: usize, comptime proj: []const u8) ![]const u8 {
    _ = c;
    return std.fmt.bufPrint(buf, prefix ++ "layers.{d}.mlp.experts.{d}." ++ proj ++ ".weight", .{ i, e });
}

fn visit(ctx: anytype, it: Item) !void {
    return ctx.item(it);
}

test "the walk yields the pack's inventory: 3 + 75 MoE layers with 256 FP4 experts each, 22 indexers, one BF16 MTP layer" {
    const Coll = struct {
        items: usize = 0,
        fp4: usize = 0,
        bf16: usize = 0,
        f32: usize = 0,
        roles: std.enums.EnumSet(Role) = .{},

        pub fn item(self: *@This(), it: Item) !void {
            self.items += 1;
            switch (it.flavor) {
                .fp4 => self.fp4 += 1,
                .bf16 => self.bf16 += 1,
                .f32 => self.f32 += 1,
            }
            self.roles.insert(it.role);
        }
    };
    var c = Config{};
    // the pack's indexer pattern: layers 0-2 full, then every fourth from 6 (the MTP layer's own is set by the parser)
    for ([_]u32{ 0, 1, 2 }) |i| c.full.set(i);
    var i: u32 = 6;
    while (i < c.layers) : (i += 4) c.full.set(i);
    c.full.set(c.layers); // the MTP layer's own indexer, as the parser sets it
    var coll: Coll = .{};
    try walk(&c, &coll);
    // 76 router biases: layers 3..77 (75 quantized) and the MTP layer 78
    try std.testing.expectEqual(@as(usize, 76), coll.f32);
    try std.testing.expect(coll.roles.contains(.idx_wqb));
    try std.testing.expectEqual(@as(usize, 75 * 256 * 3), coll.fp4); // layers 3..77, 256 experts, 3 linears
    // 3 top-level + 79 layers * (2 norms + 7 MLA) + 22 * 5 indexer + 3 dense * 3 + 76 MoE * 5 routing/shared
    // + 75 * 256 * 3 FP4 experts + the MTP layer's 256 * 3 BF16 experts + 4 MTP extras
    const expect = 3 + 79 * 9 + 22 * 5 + 3 * 3 + 76 * 5 + coll.fp4 + 256 * 3 + 4;
    try std.testing.expectEqual(expect, coll.items);
}