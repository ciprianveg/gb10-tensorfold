//! The GLM-5.3 (glm_moe_dsa) CUDA family's module root. It lives one level above the family's own
//! directory so the family can import the shape-agnostic blocks the Flash family already carries
//! (`glm/cuda_glue.zig`, `cuda_latent.zig`, `cuda_mlp.zig`, `cuda_moe.zig`); the module boundary is
//! the repository's `families/` scope, not a family directory (upstream-PR candidate #3 makes those
//! blocks named modules instead).
pub const family = @import("glm_moe_dsa/cuda.zig");

test {
    _ = family;
}
