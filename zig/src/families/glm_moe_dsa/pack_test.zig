//! The GLM-5.3 NVFP4-CSF pack against this family's loader, on a CPU-only box:
//! (a) every module the plan walks exists with the sampled dtype/shape (the full inventory, no bulk reads);
//! (b) the loader's own gather path cuts an expert's codes for all eight TP ranks exactly as the
//!     independent decode of the CSF planes says (the milestone-1 digests, through the production code).
//! Env-gated: TF_GLM53_NVFP4_DIR names the pack.
const std = @import("std");
const core = @import("core");
const ckpt = core.checkpoint;
const config = @import("config.zig");
const plan = @import("plan.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("weights.zig");

const Sample = struct {
    items: usize = 0,
    fp4: usize = 0,
    layers: std.AutoHashMapUnmanaged(u32, void) = .empty,

    fn item(self: *Sample, a: std.mem.Allocator, ck: *ckpt.Checkpoint, it: plan.Item) !void {
        self.items += 1;
        if (it.flavor == .fp4) self.fp4 += 1;
        try self.layers.put(a, it.layer, {});
        // the vector/matrix classes carry their exact stored dtype; fp4 modules are checked here for
        // their `.weight` only (their scale planes are nvfp4_slice's admission, exercised in (b))
        const t = try ck.get(it.name);
        if (it.flavor == .fp4) {
            // the quantized triad must be complete and well-formed too (weight_scale_2 + the CSF pair)
            const mod = plan.module(it.name);
            const s2 = try ck.get(try std.fmt.allocPrint(a, "{s}" ++ plan.scale_2, .{mod}));
            if (s2.dtype != .f32 or s2.numel() != 1) return error.UnexpectedTensor;
            const fx = try ck.get(try std.fmt.allocPrint(a, "{s}" ++ plan.csf_fixed, .{mod}));
            const ex = try ck.get(try std.fmt.allocPrint(a, "{s}" ++ plan.csf_exceptions, .{mod}));
            // fixed: (n/16) * (16 + 8 k/16) bytes; exceptions: a u32 list
            const stride = 16 + 8 * (it.k / 16);
            if (fx.dtype != .u8 or fx.numel() != (it.n / 16) * stride) return error.UnexpectedTensor;
            if (ex.dtype != .u32 or ex.rank != 1) return error.UnexpectedTensor;
            // the W4A4 static input scale lives in the pack's sidecar (model-inputscales.safetensors):
            // part of the checkpoint, unread by W4A16 math, so it is only marked used here
            if (ck.has(try std.fmt.allocPrint(a, "{s}.input_scale", .{mod}))) _ = try ck.get(try std.fmt.allocPrint(a, "{s}.input_scale", .{mod}));
        }
        const want_elems = if (it.k == 0) it.n else it.n * it.k;
        const stored = switch (it.flavor) {
            .bf16 => t.dtype == .bf16,
            .f32 => t.dtype == .f32,
            .fp4 => t.dtype == .u8 and t.numel() == want_elems / 2,
        };
        if (!stored) {
            std.debug.print("bad dtype/size: {s}: {t} numel {d}, want {d}\n", .{ it.name, t.dtype, t.numel(), want_elems });
            return error.UnexpectedTensor;
        }
        if (it.flavor != .fp4 and t.numel() != want_elems) {
            std.debug.print("bad numel: {s}: {d}, want {d}\n", .{ it.name, t.numel(), want_elems });
            return error.UnexpectedTensor;
        }
    }
};

test "the real pack (TF_GLM53_NVFP4_DIR) holds every module the loader reads, at the plan's shapes" {
    const dirz = std.c.getenv("TF_GLM53_NVFP4_DIR") orelse return error.SkipZigTest;
    const dir: []const u8 = std.mem.span(dirz);
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const cfg_path = try std.fs.path.join(a, &.{ dir, "config.json" });
    const json = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, a, .limited(1 << 20));
    const c = try config.parse(a, json);
    const root = try std.fs.path.join(a, &.{ dir, "tensors" });
    var ck = try ckpt.Checkpoint.openModel(a, io, root);
    defer ck.close();

    var sample: Sample = .{};
    defer sample.layers.deinit(a);
    var walker = struct {
        sample: *Sample,
        a: std.mem.Allocator,
        ck: *ckpt.Checkpoint,
        pub fn item(self: *@This(), it: plan.Item) !void {
            return self.sample.item(self.a, self.ck, it);
        }
    }{ .sample = &sample, .a = a, .ck = &ck };
    try plan.walk(&c, &walker);

    try std.testing.expectEqual(@as(usize, 75 * 256 * 3), sample.fp4);
    try std.testing.expectEqual(@as(usize, 79), sample.layers.count()); // 78 backbone + the MTP layer
    // the whole inventory, nothing left unused: the checkpoint's names are exactly the plan's
    try std.testing.expectEqual(@as(usize, 0), ck.unused());
}

test "the loader's gather cuts layer 3 expert 0's codes for all eight TP ranks as the independent decode says" {
    const dirz = std.c.getenv("TF_GLM53_NVFP4_DIR") orelse return error.SkipZigTest;
    const dir: []const u8 = std.mem.span(dirz);
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const cfg_path = try std.fs.path.join(a, &.{ dir, "config.json" });
    const json = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, a, .limited(1 << 20));
    var c = try config.parse(a, json);
    const root = try std.fs.path.join(a, &.{ dir, "tensors" });
    var ck = try ckpt.Checkpoint.openModel(a, io, root);
    defer ck.close();

    // rank-r digests of gate_proj's (2048 x 6144 -> 2048 x 3072 nibble bytes) row cut and
    // down_proj's (6144 x 2048 -> 6144 x 1024) input-column cut; milestone 1's oracle, re-verified
    const gate_codes = [8][]const u8{
        "748fa352af235b8cde6a287d1bbea71ce32654d02df78b0b37bafd2909fc17df",
        "5a8ba7c8a227501dce4682051e6ca3e11ac4ca69e8e4e9fc545b270d05ff6660",
        "8036468119711c7f30f0d22dea59d0ddfa2b5401eaef140f105de27261a7e70f",
        "07deb487311850e1cc5aa7926957109b3592f4fa38c69fddab03506926b7bd1b",
        "163e6e8aace6458c4fc6e972926ae6717ad38d6e3d4c8f18fbff9ca230953a30",
        "562a279f8f3efbe4eede5a46eedf13266bfba7b5f488b61c34bba4a8651d39bc",
        "f4bb18aa95c7b276b59f886b52bf6f90888d25561a377290f92d9cca521dfa0d",
        "3c3ebaefe97f0fb3f349e8b483108d6384beb2e49f8eeb32cad0b5234e01585c",
    };
    const down_codes = [8][]const u8{
        "d75b339da37db1e1f2ef2504c89fc710fdb457429ee0d72902efd5095d15b1ef",
        "4db6c2896d4a148dfad8043f9c9280d2795c10f58844cc9b88247499de8ae859",
        "234f20bfad1cccdb8866f24e830734727dfe97b2feecd2b3f03d74390d3b7b15",
        "2d9b4638c5077189aaf07473b15303b2e320310fcf01e42d87296f7f5f642f90",
        "e2774211aec8d69fdda4eba42507170c5da4752ad86634a37e4141d30a305f72",
        "b4548a6a81829d9e884bde90ffbd3d0a0b4f43da077b800ea2749669514fa947",
        "4272d36bcfd88a386de0fb84087bda8b19ecbd03f26512ae216ce833c7860134",
        "15b911d5a399bab3e039da101491e713028109e976277af58314f9de6ec538bd",
    };
    var w: weights.Weights = .{ .gpa = a, .config = c, .share = undefined };
    defer w.buffers.deinit(a);
    defer { for (w.named.items) |n| a.free(n.name); w.named.deinit(a); }
    for (0..8) |r| {
        const share = try shard.Share.of(&c, r, 8);
        w.share = share;
        var L: weights.Loader = .{ .gpa = a, .d = undefined, .stream = undefined, .tools = .{ .pack_dense = undefined }, .w = &w, .ck = &ck };
        defer L.host.deinit(a);
        const gate = plan.Item{ .name = "model.layers.3.mlp.experts.0.gate_proj.weight", .role = .expert_gate, .flavor = .fp4, .n = c.moe_inter, .k = c.hidden, .layer = 3 };
        try L.gather(gate);
        try digestMatches(gate_codes[r], L.host.items);
        L.host.clearRetainingCapacity();
        const down = plan.Item{ .name = "model.layers.3.mlp.experts.0.down_proj.weight", .role = .expert_down, .flavor = .fp4, .n = c.hidden, .k = c.moe_inter, .layer = 3 };
        try L.gather(down);
        try digestMatches(down_codes[r], L.host.items);
    }
}

fn digestMatches(hex: []const u8, bytes: []const u8) !void {
    var want: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&want, hex);
    var got: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &got, .{});
    try std.testing.expectEqualSlices(u8, &want, &got);
}
test "the sidecar's W4A4 input scales attach per quantized projection: counted, named, parsed" {
    const dirz = std.c.getenv("TF_GLM53_NVFP4_DIR") orelse return error.SkipZigTest;
    const dir: []const u8 = std.mem.span(dirz);
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const cfg_path = try std.fs.path.join(a, &.{ dir, "config.json" });
    const json = try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, a, .limited(1 << 20));
    const c = try config.parse(a, json);
    const root = try std.fs.path.join(a, &.{ dir, "tensors" });
    var ck = try ckpt.Checkpoint.openModel(a, io, root);
    defer ck.close();

    // the pack's sidecar: exactly one f32 scalar per quantized projection (75 layers x 256 experts x 3)
    var n: usize = 0;
    for (ck.files.items) |*f| {
        var it = f.names.iterator();
        while (it.next()) |e| if (std.mem.endsWith(u8, e.key_ptr.*, ".input_scale")) {
            n += 1;
        };
    }
    try std.testing.expectEqual(@as(usize, 75 * 256 * 3), n);

    // layer 3's first expert: gate and up share the expert's input, down has its own (the sidecar's values)
    var s = try weights.expertInputScales(a, &ck, &c, 3);
    defer s.deinit(a);
    try std.testing.expectEqual(@as(usize, 512), s.up.len);
    try std.testing.expectEqual(@as(usize, 256), s.down.len);
    try std.testing.expectApproxEqRel(@as(f32, 0.00024995350395329297), s.up[0], 1e-6);
    try std.testing.expectEqual(s.up[0], s.up[1]); // gate == up
    try std.testing.expectApproxEqRel(@as(f32, 0.0007672991487197578), s.down[0], 1e-6);
    // and the last layer/expert through the same naming path
    var last = try weights.expertInputScales(a, &ck, &c, c.layers - 1);
    defer last.deinit(a);
    try std.testing.expectApproxEqRel(@as(f32, 0.0018252418376505375), last.up[2 * 255], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 0.8392857313156128), last.down[255], 1e-6);
}
