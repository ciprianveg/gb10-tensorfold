//! This family's linears tiled at load for glm_mm (§38): every bf16 weight `b16.matmul` reads is
//! retiled in place by the GLM engine's own `glm_mm_tile16`, so the forward can take the tiled entry
//! (`glue.Set.mm`) instead of the raw-rows fallback §32 chose. Same bytes, same shape - only the
//! layout the MMA reads changes, which is what makes the prompt path tensor-core bound instead of
//! 0.36 ms a layer-row.
//!
//! Reference: the GLM family's `cuda_tile.zig::tileLinears` (its `Lin` list and its tmp-buffer tiling);
//! the field list below is this loader's counterpart of that one, and `residualAdd`/norms/embed/
//! `absorbQ`/`expandV`/the routed experts are excluded because no `b16.matmul` reads them.
const std = @import("std");
const cuda = @import("cuda");
const mm_mod = @import("glue/cuda_mm.zig");
const weights = @import("weights.zig");

const Weights = weights.Weights;
const Layer = weights.Layer;

/// A linear b16.matmul reads: the field holding its pointer, and its shape [n][k].
const Lin = struct { field: *u64, n: usize, k: usize };

fn add(l: *std.ArrayList(Lin), gpa: std.mem.Allocator, field: *u64, n: usize, k: usize) !void {
    if (field.* != 0) try l.append(gpa, .{ .field = field, .n = n, .k = k });
}

fn layerLinears(gpa: std.mem.Allocator, c: *const @import("config.zig").Config, l: *Layer, list: *std.ArrayList(Lin)) !void {
    try add(list, gpa, &l.attn.proj, c.q_lora + c.kv_lora + c.rope, c.hidden);
    try add(list, gpa, &l.attn.q_b, l.attn.heads * l.attn.qk, c.q_lora);
    try add(list, gpa, &l.attn.o, c.hidden, l.attn.heads * c.v_dim);
    if (c.full.isSet(l.index)) {
        try add(list, gpa, &l.attn.index.kw, c.i_dim, c.hidden);
        try add(list, gpa, &l.attn.index.wproj, c.i_heads, c.hidden);
        try add(list, gpa, &l.attn.index.qb, c.i_heads * c.i_dim, c.q_lora);
    }
    switch (l.ffn) {
        .dense => |*m| switch (m.w) {
            .bf16 => |*b| {
                try add(list, gpa, &b.gu, 2 * m.width, c.hidden);
                try add(list, gpa, &b.down, c.hidden, m.width);
            },
            .nvfp4 => {},
        },
        .moe => |*m| {
            try add(list, gpa, &m.shared_gu, 2 * m.shared, c.hidden);
            try add(list, gpa, &m.shared_down, c.hidden, m.shared);
        },
    }
}

/// Every linear b16.matmul reads, this rank's head last (the GLM engine's order).
fn linears(w: *Weights, list: *std.ArrayList(Lin)) !void {
    for (w.layers) |*l| try layerLinears(w.gpa, &w.config, l, list);
    if (w.mtp) |*m| {
        try layerLinears(w.gpa, &w.config, &m.layer, list);
        try add(list, w.gpa, &m.eh, w.config.hidden, 2 * w.config.hidden);
    }
    try add(list, w.gpa, &w.head, w.share.vocab.len(), w.config.hidden);
}

/// Every linear retiled in place (bf16: `glm_mm_tile16` through one tmp buffer), the count.
pub fn tileLinears(w: *Weights, mk: *const mm_mod.Kernels, s: cuda.Stream, d: *const cuda.Driver) !usize {
    var list: std.ArrayList(Lin) = .empty;
    defer list.deinit(w.gpa);
    try linears(w, &list);
    var most: usize = 0;
    for (list.items) |e| {
        if (e.n % 8 != 0 or e.k % 64 != 0) return error.BadShape;
        most = @max(most, e.n * e.k * 2);
    }
    var tmp = try cuda.DeviceBuffer.alloc(d, most);
    defer tmp.free();
    for (list.items) |e| {
        try mm_mod.tile(mk, s, e.field.*, tmp.ptr, e.n, e.k);
        try d.check(d.api.cuMemcpyDtoDAsync_v2(e.field.*, tmp.ptr, e.n * e.k * 2, s.handle), "cuMemcpyDtoDAsync");
        try s.synchronize(); // tmp is the next linear's
    }
    return list.items.len;
}