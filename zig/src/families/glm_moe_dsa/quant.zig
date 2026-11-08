//! The load-time weight quantizers the MTP layer needs (BF16 experts -> NVFP4), the same bits as the
//! Flash family's glm_quant entries; the kernel names are the fatbin's.
const cuda = @import("cuda");

/// glm_quant.cu's entries.
pub const Kernels = struct {
    amax: cuda.Function,
    nvfp4: cuda.Function,
    q4: cuda.Function,

    pub fn load(m: cuda.Module) !Kernels {
        return .{
            .amax = try m.function("tf_glm_nvfp4_amax"),
            .nvfp4 = try m.function("tf_glm_nvfp4_quantize"),
            .q4 = try m.function("tf_glm_quantize4"),
        };
    }
};

/// experts.quantize: w bf16 [e][n][k] -> codes u8 [e][n][k/2], e4m3 scales [e][n][k/16], g fp32 [e]; 16-byte aligned.
pub fn nvfp4(ks: *const Kernels, stream: cuda.Stream, w: u64, e: usize, n: usize, k: usize, codes: u64, scales: u64, g: u64) !void {
    if (k % 16 != 0) return error.KNotMultipleOf16;
    if (e == 0 or n == 0) return;
    const per = n * k;
    var a: cuda.Args = .{};
    a.add(w);
    a.add(@as(c_longlong, @intCast(per)));
    a.add(g);
    try cuda.launch.launch(ks.amax, .{ .grid = .{ .x = @intCast(e) }, .block = .{ .x = 1024 } }, stream, &a);
    const total = e * per / 16;
    var b: cuda.Args = .{};
    b.add(w);
    b.add(g);
    b.add(@as(c_longlong, @intCast(per / 16)));
    b.add(@as(c_longlong, @intCast(total)));
    b.add(codes);
    b.add(scales);
    try cuda.launch.launch(ks.nvfp4, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, stream, &b);
}