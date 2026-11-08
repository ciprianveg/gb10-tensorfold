//! Full GLM-5.3 (glm_moe_dsa) on CUDA in Zig: the Python glm_moe_dsa engine's fused path (MLA latent attention, the DSA
//! indexer, EXL3 dense linears and routed experts), its kernels and layouts, so each layer matches it bit for bit.
//! Phase 1: the first N layers on one GB10 as rank 0 of world 4 with the other ranks absent (identity comm: own
//! partial, zeros elsewhere), zig/tests/cuda/glm53_layers.zig against tools/glm53/oracle.py.
//! Phase 2a: the full model at TP over NCCL (Forward.comm), the vocabulary-sharded argmax head (Forward.head), greedy,
//! zig/tests/cuda/glm53_generate.zig against tools/glm53/reference.py.
//! Phase 2b: the MTP layer, the 4-bit draft head, MTP-drafted rounds replayed as CUDA graphs and the Python engine's
//! keyed sampling (cuda_runner.zig, sampling.zig), the O_DIRECT / io_uring load (cuda_source.zig);
//! zig/tests/cuda/glm53_generate.zig --runner against tools/glm53/reference2b.py.
//! Phase 3a: the RoCE one-shot all-reduce / gather for decode windows (cuda_roce.zig over roce_proxy.c and
//! glm53_roce.cu), the served prompt path (cuda_prompt.zig: sequence-parallel chunks of thousands of rows, the prompt
//! GEMM, torch.bmm through cuBLAS, prompt experts) and the served tile table (tiles.zig); zig/tests/cuda/
//! glm53_generate.zig --mode 3a against tools/glm53/reference3a.py (the Python engine in its served configuration).
//! Phase 3a speed: the decode windows' side stream and L2 prefetch (cuda_decode.zig: TF_GLM53_SIDE / TF_GLM53_L2PF as
//! the served image runs them), the decode-sized multi-block top-k and device-side sampling candidates
//! (glm53_decode.cu), the MTP layer's prompt rows in bigger blocks, the prompt path's timers (cuda_prompt.Prof).
//! Phase 3b: the served engine (cuda_engine.zig: Boot, Session; native/glm53_cuda.zig serves it over the native
//! server's OpenAI routes, ranks 1..3 following rank 0's requests), EOS and rank 0's stop vote (cuda_runner.run),
//! prompt reuse (cuda_reuse.zig: prefixes.PromptReuse) and its learned states on disk (--learn), decode context
//! parallelism (cuda_dcp.zig, DCP = 4 past 200K tokens of context).
//! Phase 4: the speculative drafters (cuda_draft.zig: DSpark and DFlash2 on the device, draft_host.zig: their host
//! stages), copy drafts (copies.zig), the target's tap rows (Forward.taps), drafter / copy rounds in cuda_runner.run.
//! Phase 4b: concurrent streams (cuda_multi.zig: --parallel N, multi.GlmMultiDecoder - the caches' slots, shared
//! decode rounds with per-row positions and cache bases, the batched MTP chain, the draft cut, prompt reuse in the
//! slots).
//! Phase 4c: the drafters for concurrent streams (cuda_mdraft.zig: DFlash2 as dflash.MultiDrafter, DSpark with the
//! drafter's KV keyed by the stream slot and one ragged batched block pass) in cuda_multi.zig's rounds.
//! Scoring (score.zig, cuda_score.zig; tf-glm53-generate --mode score): KL(reference || served) and top-1 agreement
//! over fixed token rows through the served prompt path, the head on every row. Not reached by the server.
//! Quantized head (headq.zig, TF_GLM53_HEAD_Q, off by default): lm_head as an EXL3 linear over the blocks covering a
//! rank's vocabulary share; an EXL3 kv_b decoded to the bf16 blocks at load (TF_GLM53_KVB_Q). Not yet run on a GPU.

const std = @import("std");

pub const Config = @import("config.zig").Config;
pub const exl3 = @import("exl3.zig");
pub const split = @import("split.zig");
pub const weights = @import("cuda_weights.zig");
pub const kernels = @import("cuda_kernels.zig");
pub const triton = @import("cuda_triton.zig");
pub const forward = @import("cuda_forward.zig");
pub const Forward = forward.Forward;
pub const Buffers = forward.Buffers;
pub const RouteTap = forward.RouteTap;
pub const State = forward.State;
pub const sampling = @import("sampling.zig");
pub const native = @import("cuda_native.zig"); // 118: the standard-surface registration (the lane-host lone driver)
pub const source = @import("cuda_source.zig");
pub const runner = @import("cuda_runner.zig");
pub const Runner = runner.Runner;
pub const roce = @import("cuda_roce.zig");
pub const prompt = @import("cuda_prompt.zig");
pub const tiles = @import("tiles.zig");
pub const decode = @import("cuda_decode.zig");
pub const reuse = @import("cuda_reuse.zig");
pub const engine = @import("cuda_engine.zig");
pub const dcp = @import("cuda_dcp.zig");
pub const coverage = @import("cuda_coverage.zig");
pub const draft = @import("cuda_draft.zig");
pub const draft_host = @import("draft_host.zig");
pub const copies = @import("copies.zig");
pub const multi = @import("cuda_multi.zig");
pub const mdraft = @import("cuda_mdraft.zig");
pub const headq = @import("headq.zig");
pub const score = @import("score.zig");
pub const score_cuda = @import("cuda_score.zig");

test {
    _ = @import("config.zig");
    _ = exl3;
    _ = split;
    _ = triton;
    _ = forward;
    _ = weights;
    _ = kernels;
    _ = sampling;
    _ = source;
    _ = runner;
    _ = roce;
    _ = prompt;
    _ = tiles;
    _ = decode;
    _ = reuse;
    _ = engine;
    _ = dcp;
    _ = coverage;
    _ = draft;
    _ = draft_host;
    _ = copies;
    _ = multi;
    _ = mdraft;
    _ = headq;
    _ = score;
    _ = score_cuda;
}

// --- A1 transplant: the census, in executable form ------------------------------------------------------------
// The 1.1 loader stays the weights source; this module is r2's compute path verbatim (glm_moe_dsa @ 70e68f66,
// `*.bak`/`*.kv8bak` excluded). `files` is the transplant's file count, the A1 census's own denominator.

pub const files = 29;

/// The fatbin names the family and its shared cuda modules resolve at load (the pruned set, §44). Kept as data so
/// the census can be printed and asserted without a GPU.
pub const blob_names = [_][]const u8{
    "glm53_prompt",     "glm53_prompt_experts", "glm53_decode",     "glm53_draft",   "glm53_head",
    "glm53_mtp",        "glm53_rope",           "glm53_roce",       "glm53_kv8",     "glm53_fp8_scores",
    "glm53_exl3_experts", "glm53_exl3_linear",  "qmm_group",        "qmmf",          "nvfp4_experts",
};

fn blobBytes(name: []const u8) usize {
    inline for (blob_names) |n| {
        if (std.mem.eql(u8, name, n)) return @field(@import("cuda").kernels, n).len;
    }
    return 0;
}

/// The A1 gate in executable form: the config's cache arithmetic (§43) and the blob census. The config comes
/// from the real pack (`TF_GLM53_NVFP4_DIR`, the 1.1 loader's own variable) when it is set, else from the
/// canonical GLM-5.3 dimensions with the same shape rules.
pub fn selftest(gpa: std.mem.Allocator, io: std.Io) !void {
    const cfg = blk: {
        if (std.c.getenv("TF_GLM53_NVFP4_DIR")) |d| {
            if (Config.load(gpa, io, std.mem.span(d))) |c| break :blk c else |_| {}
        }
        const json =
            \\{"model_type":"glm_moe_dsa","hidden_size":6144,"num_hidden_layers":2,"first_k_dense_replace":3,
            \\ "intermediate_size":12288,"moe_intermediate_size":2048,"n_routed_experts":256,"n_shared_experts":1,
            \\ "num_experts_per_tok":8,"routed_scaling_factor":2.5,"norm_topk_prob":true,"num_attention_heads":64,
            \\ "q_lora_rank":2048,"kv_lora_rank":512,"qk_nope_head_dim":192,"qk_rope_head_dim":64,"v_head_dim":256,
            \\ "rope_parameters":{"rope_theta":8000000,"rope_type":"default"},"index_n_heads":32,"index_head_dim":128,
            \\ "index_topk":2048,"indexer_types":["full","full"],"rms_norm_eps":1e-05,"vocab_size":154880}
        ;
        break :blk try Config.parse(gpa, json);
    };
    const lat_kv8 = forward.State.latentRow(true, cfg);
    const lat_bf16 = forward.State.latentRow(false, cfg);
    std.debug.print("tf-glm53r2: config layers={d} hidden={d} kv_lora={d} rope={d} index_dim={d}\n", .{ cfg.layers, cfg.hidden, cfg.kv_lora, cfg.rope, cfg.index_dim });
    std.debug.print("tf-glm53r2: latentWidth={d} kv8 row={d} bf16 row={d} (ratio {d:.1}%)\n", .{ cfg.latentWidth(), lat_kv8, lat_bf16, 100.0 * @as(f64, @floatFromInt(lat_kv8)) / @as(f64, @floatFromInt(lat_bf16)) });
    if (lat_kv8 != 584 or lat_bf16 != 1152) return error.CensusMismatch;
    var missing: usize = 0;
    inline for (blob_names) |n| {
        const len = @field(@import("cuda").kernels, n).len;
        std.debug.print("tf-glm53r2: blob {s} {d} bytes\n", .{ n, len });
        if (len == 0) missing += 1;
    }
    if (missing != 0 and @import("cuda").kernels.available) return error.BlobCensusMiss;
    std.debug.print("tf-glm53r2: census {d} blobs, {d} missing\n", .{ blob_names.len, missing });
}
