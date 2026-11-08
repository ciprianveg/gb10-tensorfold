// GLM-5.3's DSA indexer scoring on the sm121 fp8 tensor cores (TF_GLM53_FP8_INDEX increment 2,
// ours): the all-prior-keys MQA-logits GEMM of fused._index_scores8 as a native CUDA kernel.
//
// The math (findings-prefill-fp8-cuda.md): for window row r at position p, key t <= p:
//   score[r, t] = sum_h W[r, h] * WSCALE * relu(dot(q[r, h, :], k[t, :]) * QSCALE)
// with q read bf16 and quantized per row (amax / 448 -> e4m3, the per-row scale in the epilogue)
// and k the e4m3 index-key plane increment 1 built (a 132-byte token row: 128 codes then one
// fp32 scale - the same storage read two ways: CODES as bytes, SCALES as floats at word 32 of
// the row's 33). The result packs as _index_scores8's int32 order word (the monotone int of the
// fp32 score, xor 0x80000000; later keys t > p and the range past T are the 0 sentinel), so the
// selection downstream is unchanged.
//
// The formulation is DeepGEMM's sm120_fp8_mqa_logits pattern (vLLM's reference, measured 2.85x
// our Triton at 100K), read in the live container (vllm/third_party/deep_gemm/include/...):
//   * mma/sm120.cuh: the MMA itself - mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32, the
//     non-block-scaled variant (their fp8_mma), inline PTX with a[4] / b[2] / d[4] fragments.
//   * impls/sm120_fp8_paged_mqa_logits.cuh: the INVERTED shapes - A is the KEY tile (M x K =
//     keys x head_dim), B is the row's QUERY (K x N = head_dim x heads, N = 32 = 4 MMA_N tiles),
//     so one warp's MMAs produce a [16 keys x 8 heads] fp32 tile per N-tile; their epilogue maps
//     the C fragment (lane l: g = l/4, t = l%4; d[0..3] = keys {g, g+8} x heads {2t, 2t+1}),
//     accumulates relu(d) * w[h] over the N-tiles, reduces the four t lanes with shfl_xor 1|2,
//     and multiplies the per-key scale after the reduction. Borrowed verbatim (cited line
//     ranges in the findings); the C-fragment mapping is the PTX ISA's own for m16n8k32.
//   * The STAGING STRUCTURE: their double-buffered SMEM pipeline with the group-commit /
//     wait-group cadence, in our no-TMA context (section 13), plus their SWIZZLED-fragment idea
//     shrunk to a manual pair-swizzle (section 14): each lane's {col, col+16} word pair staged
//     adjacent so the A fragment is one 8-byte SMEM load per key row (was two 4-byte, 4-way
//     bank-conflicted). NOT borrowed: TMA, cluster/mbarrier pipelines, the 128B XOR ldmatrix
//     layouts, the paged scheduler.
//   * The query is re-quantized per CTA (4096 elements against the CTA's ~8M MMA MACs, ~0.05%)
//     and the whole increment stays off the AOT pack (the fatbin ships outside it, the
//     glm53_kv8.cu precedent).
//
// THE TWO STRIDE VARIANTS (inc 3, findings-prefill-sp-restore.md's successor lane / section 20): the plane's
// token row is 132 bytes (128 codes + the fp32 scale at byte 128) in the legacy layout; the PADDED layout is
// 144 bytes (+ 12 pad) - a 16-byte-multiple stride, what TMA-grade staging needs (the fetch as one bulk
// descriptor per tile instead of thousands of cp.async issues - findings-prefill-fp8-cuda.md section 19's
// verdict that the fetch's issue+drain is the critical path). STRIDE_W (words a row: 33 or 36) is a TEMPLATE
// parameter - two entry points, glm53_fp8_scores (33: the shipped plateau configuration verbatim) and
// glm53_fp8_scores_p (36: the padded path). The two paths differ ONLY in the memory schedule:
//   * 33: the pair-swizzled 16,384 B code-only tiles, 4-byte cp.async staging, 8-byte pair lds, 32,804 B
//     SMEM = 3 CTAs/SM (the 1.166 ms plateau).
//   * 36: the identity 144-byte SMEM rows (the +16 pad a row), 16-BYTE cp.async staging (4 issues a thread,
//     was 16), and CONFLICT-FREE 4-byte fragment lds (the lane map 4g + t4 is a bijection mod 32 banks) -
//     36,900 B SMEM = 2 CTAs/SM. The bench decides which wins; the swizzle cannot take 16-byte staging (the
//     pair members sit 16 bytes apart in the plane - a gather cp.async cannot do).
// The arithmetic is identical in both (same fragment bytes, same MMA order, bit-identical scores).
//
// Occupancy (the GB10's measured limits, section 14: 48 SMs, 102,400 B SMEM/SM, 64K regs):
// 3 CTAs/SM needs <= 33,333 B SMEM and <= 85 regs a CTA-slot. The tiles stage ONLY the 128 code
// bytes a row (the per-key scale is read straight from global in the epilogue - never from SMEM),
// so a tile is 16,384 B and a CTA 32,804 B: 98.4 KB of the 100 KB = 3 CTAs/SM. launch_bounds
// (256, 3) holds ptxas to the 85-register budget. A 3-stage pipeline was REJECTED on the
// numbers: 3 x 16.9 KB blew the 48 KB static limit and dropped SMEM occupancy to 1 CTA/SM.
//
// Grid (ceil(T / (128 * TILES_PER_CTA)) CTA columns, n rows), 256 threads = 8 warps: warp w owns
// local keys [16w, +16) of the 128-key SMEM tile, all 32 heads; each lane holds its A/B fragments
// and 4 N-tiles x 4 K-steps of MMAs per tile. Deterministic: the quantization, the MMA and the
// epilogue are all order-fixed; the pipeline and the swizzle change ONLY the memory schedule -
// the a-fragment words for a valid key are the same plane bytes in the same registers, so the
// scores are bit-identical to the unpipelined kernel (gates 7-8 are the contract).
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {
constexpr float FP8_MAX = 448.0f;
constexpr int NH = 32;      // the model's index_heads (MMA_N tiles: 4 x 8)
constexpr int D = 128;      // the model's index_dim (MMA K steps: 4 x 32)
constexpr int KEYS_PER_CTA = 128;
constexpr int TILES_PER_CTA = 32;                      // the SMEM pipeline walk per CTA (section 15:
                                                      // at T=100K this makes grid.x = 25, so a row's
                                                      // q-quant + B-build prologue runs 25x, not 98x -
                                                      // the prologue was ~32% of all issue slots)

// the STRIDE_W template: the plane's words a row. 33 = the legacy 132-byte token row; 36 = inc 3's padded
// 144-byte row (TMA-grade, see the header). Everything downstream derives from it.
constexpr int TILE_BYTES_33 = KEYS_PER_CTA * 128;      // 16384: the swizzled code-only tiles (128B SMEM rows)
constexpr int TILE_WORDS_33 = TILE_BYTES_33 / 4;       // 4096 4-byte cp.asyncs, 256 threads x 16
constexpr int TILE_BYTES_36 = KEYS_PER_CTA * 144;      // 18432: the padded identity tiles (144B SMEM rows)

__device__ __forceinline__ int order_word(float v) {
    const int bits = __float_as_int(v);
    const int key = bits >= 0 ? bits : (bits ^ 0x7FFFFFFF);  // the monotone int of the fp32
    return key ^ (-2147483647 - 1);                          // ... in unsigned order
}

// DeepGEMM's fp8_mma (mma/sm120.cuh): m16n8k32, e4m3 x e4m3 -> f32
__device__ __forceinline__ void fp8_mma(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

// The pair-swizzled SMEM layout (section 14): the row's code word u (ks = u/8, p = u%4, h = the
// +16 half) lands at position ((ks ^ (row & 3)) << 3) | (p << 1) | h - so a lane's {col, col+16}
// pair for one K-step is TWO CONSECUTIVE words (one 8-byte load), the row's XOR de-correlates the
// banks (2-way conflicts, was 4-way), and the map is a bijection inside the 32-word row.
__device__ __forceinline__ int ks_of(int u) { return u >> 3; }
__device__ __forceinline__ int smem_pos(int lr, int u) {
    return ((ks_of(u) ^ (lr & 3)) << 3) | ((u & 3) << 1) | ((u >> 2) & 1);
}

// stage one 128-key tile into SMEM, codes only (the scale word stays global - the epilogue reads
// it there): thread t stages its code word u = t&31 of tile-local rows lr = (t>>5) + 8k, k = 0..15
// (256 threads x 16 = the 4096 words). The section-11 clamp lives HERE (the Triton contract never
// reads an invalid key row; a fully invalid tile stages row kcap repeatedly and every key of it
// fails the store guard). The plane's 132-byte rows are only 4-byte aligned, so the staging is
// 4-byte cp.async (16-byte needs aligned sources; 8-byte cannot gather the swizzled pair).
// THE HOIST (section 15): u, the swizzled position smem_pos(lr, u), and lr&3 are all INVARIANT
// across a thread's 16 iterations (lr walks +8, and the swizzle only reads lr&3 - 8k vanishes
// mod 4) - one address computation up front, then 16 x (min, mad, cp.async).
template <int STRIDE_W>
__device__ __forceinline__ void stage_tile(uint8_t* dst, const __nv_fp8_e4m3* kcodes, int tile, int kcap) {
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(dst);
    if (STRIDE_W == 36) {
        // the padded path: 16-BYTE cp.asyncs into the identity 144-byte SMEM rows - 1024 chunks
        // (128 rows x 8), 4 a thread (the legacy path needs 16 four-byte issues for the same bytes;
        // section 19's verdict: the fetch's issue+drain is the critical path). Both source and
        // destination 16-aligned (the 144-byte strides); the swizzle is GONE in this layout - the
        // fragment loads read it plain and land conflict-free (the header's bank analysis).
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int q = (int)threadIdx.x + k * 256;
            const int lr = q >> 3, c = q & 7;
            const int srow = tile + lr < kcap ? tile + lr : kcap;  // the section-11 clamp
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sbase + (uint32_t)(lr * 144 + c * 16)), "l"(kcodes + (size_t)srow * 144 + (size_t)c * 16));
        }
    } else {
        const int lr0 = threadIdx.x >> 5;
        const int u = threadIdx.x & 31;
        const uint32_t dboff = sbase + (uint32_t)smem_pos(lr0, u) * 4;  // the same word of every row
        const int cbyte = u * 4;
        #pragma unroll
        for (int k = 0; k < TILE_WORDS_33 / 256; ++k) {
            const int lr = lr0 + k * 8;
            const int srow = tile + lr < kcap ? tile + lr : kcap;     // the section-11 clamp
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(dboff + (uint32_t)lr * 128), "l"(kcodes + (size_t)srow * 132 + cbyte));
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
}  // namespace

// Q: bf16 [n, NH, D] (the caller pre-offsets the block's rows), W: fp32 [n, NH] (same, WSCALE not
// folded - the epilogue multiplies), CODES/SCALES: increment 1's e4m3 plane, one storage (a token
// row 132 bytes: CODES + t * 132, its scale SCALES + t * 33 + 32). POS: i32 positions; R0: this
// block's first window row; ROWS_MODE: several streams - row r's position POS[R0 + r], its cache
// rows at BASE[R0 + r] (else p = POS[0] + R0 + r, the cache at the plane's base). OUT: int32
// [n, T], row stride T - the order words, 0 for t > p (later keys) exactly as the Triton kernel.
template <int STRIDE_W>
__device__ __forceinline__ void scores_body(const __nv_bfloat16* __restrict__ Q, const float* __restrict__ W,
                                            const __nv_fp8_e4m3* __restrict__ CODES, const float* __restrict__ SCALES,
                                            const int* __restrict__ POS, const int* __restrict__ BASE, int* __restrict__ OUT,
                                            int T, int R0, int ROWS_MODE, int n, float QSCALE, float WSCALE) {
    const int r = blockIdx.y;               // window row R0 + r
    const int tile0 = blockIdx.x * (KEYS_PER_CTA * TILES_PER_CTA);
    if (tile0 >= T || r >= n) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t4 = lane & 3;  // the C fragment's (key, head-pair) coordinates

    long long p;
    const __nv_fp8_e4m3* kcodes;
    if (ROWS_MODE) {
        p = (long long)POS[R0 + r];
        kcodes = CODES + (long long)BASE[R0 + r] * (STRIDE_W * 4);
    } else {
        p = (long long)POS[0] + R0 + r;
        kcodes = CODES;
    }
    const float* kscales = reinterpret_cast<const float*>(kcodes);
    const int kcap = p < T ? (int)p : T - 1;  // the last plane row any load may touch (section 11)

    // ---- the row's query: amax over [NH, D] (256 threads x 16 values) -> the per-row scale
    constexpr int TILEB = (STRIDE_W == 33) ? TILE_BYTES_33 : TILE_BYTES_36;   // the path's SMEM tile
    __shared__ __align__(16) uint8_t sbuf[2][TILEB];
    __shared__ float smem_sq, smem_w[8];
    {
        float amax = 0.0f;
        const __nv_bfloat16* qrow = Q + (size_t)r * NH * D;
        #pragma unroll
        for (int j = threadIdx.x; j < NH * D / 4; j += 256) {   // 4 bf16 a load (section 16)
            const uint2 u = *reinterpret_cast<const uint2*>(qrow + 4 * j);
            const float2 f0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
            const float2 f1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y))));
        }
        #pragma unroll
        for (int off = 16; off; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        if ((lane & 31) == 0) smem_w[warp] = amax;
        __syncthreads();
        if (threadIdx.x == 0) {
            float m = 0.0f;
            #pragma unroll
            for (int wq = 0; wq < 8; ++wq) m = fmaxf(m, smem_w[wq]);
            smem_sq = fmaxf(m / FP8_MAX, 1e-12f);
        }
        __syncthreads();
    }
    const float sq = smem_sq;
    const float ir = 1.0f / sq;             // quantize by the reciprocal (one division, not 4096)
    const float qsc = sq * QSCALE;          // relu(x * c) = relu(x) * c: the scales ride the epilogue

    // ---- B fragments: the row's query, quantized e4m3 - the PTX ISA's m16n8k32 col-major B
    //      layout, b[0] = column (head) g, K rows (l % 4) * 4 .. +3; b[1] = the same, K + 16.
    //      PER K-STEP (the first gate-7 run fed one ks-invariant B to all four MMAs - a
    //      systematic mispairing, findings section 8). Built ONCE per CTA - the TILES_PER_CTA
    //      walk amortizes it.
    uint32_t bfrag[4][4][2];
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const int h = nt * 8 + g;           // this lane's head of N-tile nt
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const __nv_bfloat16* qh = Q + ((size_t)r * NH + h) * D + ks * 32 + t4 * 4;
            __nv_bfloat16 c0[4], c1[4];
            *reinterpret_cast<uint2*>(c0) = *reinterpret_cast<const uint2*>(qh);      // 4 bf16
            *reinterpret_cast<uint2*>(c1) = *reinterpret_cast<const uint2*>(qh + 16);
            uint32_t b0 = 0, b1 = 0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                // the SCALAR cvt only (section 16's postmortem: the fp8x2 two-at-a-time variant broke
                // gate 7 at the deep shape - every valid word wrong - while the scalar path is the one
                // the golden and gates are built on; this CUDA's scalar cvt "returns the raw storage
                // byte" per its own header note, a different code path than the x2 hardware op)
                const unsigned char q0 = __nv_cvt_float_to_fp8(__bfloat162float(c0[i]) * ir, __NV_SATFINITE, __NV_E4M3);
                const unsigned char q1 = __nv_cvt_float_to_fp8(__bfloat162float(c1[i]) * ir, __NV_SATFINITE, __NV_E4M3);
                b0 |= (uint32_t)q0 << (8 * i);    // the code byte (this CUDA's cvt returns the storage byte)
                b1 |= (uint32_t)q1 << (8 * i);
            }
            bfrag[nt][ks][0] = b0;
            bfrag[nt][ks][1] = b1;
        }
    }

    // ---- the row's W x WSCALE, hoisted out of the tile walk (row-constant; the products are the
    //      same fp32 multiplies the epilogue always did, just issued once)
    float wv[4][2];
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        wv[nt][0] = W[(size_t)r * NH + nt * 8 + t4 * 2] * WSCALE;
        wv[nt][1] = W[(size_t)r * NH + nt * 8 + t4 * 2 + 1] * WSCALE;
    }

    // ---- the double-buffered walk over this CTA's tiles (DeepGEMM's cadence - THE PLATEAU
    //      CONFIGURATION, section 19): the three structural probes around it all LOST - the
    //      one-barrier variant 1.218 (the fetch must overlap wait+sync+compute, not just compute),
    //      the early scale loads 1.221 (the per-warp stalls were already hidden by the interleave;
    //      the early loads just added LSU pressure), the 3-stage/2-CTA pipeline 1.343 (the 8 lost
    //      warps/SM cost more than the second in-flight fetch bought). 1.166 ms is the plateau of
    //      this plane contract; the next class needs TMA-grade staging (a 16-byte-multiple plane
    //      stride - inc 3 scope). wait_group 1 leaves exactly the newest stage in flight; the
    //      trailing syncthreads keeps the next stage's writes out of the buffer laggard threads
    //      are still computing from.
    stage_tile<STRIDE_W>(sbuf[0], kcodes, tile0, kcap);
    #pragma unroll 1
    for (int i = 0; i < TILES_PER_CTA; ++i) {
        const int ti = tile0 + i * KEYS_PER_CTA;
        if (ti >= T) break;
        const bool next = (i + 1 < TILES_PER_CTA) && (ti + KEYS_PER_CTA < T);
        if (next) stage_tile<STRIDE_W>(sbuf[(i + 1) & 1], kcodes, ti + KEYS_PER_CTA, kcap);
        if (next) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
        else      asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        const uint8_t* sb = sbuf[i & 1];

        // ---- A fragments from SMEM + the MMAs: this warp's 16 local keys x 128 dims, the PTX
        //      ISA's row-major A layout - a[0]/a[1] = rows g / g+8 at columns (l%4)*4, a[2]/a[3]
        //      at +16 - fetched as ONE 8-byte pair load a row (the pair-swizzled layout puts the
        //      {col, col+16} words adjacent). The local row is always staged (the clamp lives in
        //      stage_tile), so the reads need no guard; the store below keeps the section-11
        //      validity contract.
        const int kbase = ti + warp * 16;   // the GLOBAL key index of local row warp*16
        const int lr = warp * 16 + g, lr2 = lr + 8;
        float d[4][4] = {};
        #pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            uint32_t afrag[4];
            if (STRIDE_W == 36) {
                // the padded path: plain 4-byte loads off the identity 144-byte rows - CONFLICT-FREE
                // (the lane word 36*lr + 8*ks + t4 folds to 4g + t4 mod 32 banks, a bijection)
                const int col = ks * 32 + t4 * 4;
                afrag[0] = *reinterpret_cast<const uint32_t*>(sb + lr * 144 + col);
                afrag[1] = *reinterpret_cast<const uint32_t*>(sb + lr2 * 144 + col);
                afrag[2] = *reinterpret_cast<const uint32_t*>(sb + lr * 144 + col + 16);
                afrag[3] = *reinterpret_cast<const uint32_t*>(sb + lr2 * 144 + col + 16);
            } else {
                // the legacy path: ONE 8-byte pair load a row (the pair-swizzled layout puts the
                // {col, col+16} words adjacent)
                const int pos0 = (((ks ^ (lr & 3)) << 3) | (t4 << 1)) * 4;   // lr2 & 3 == lr & 3
                const uint2 p0 = *reinterpret_cast<const uint2*>(sb + lr * 128 + pos0);
                const uint2 p1 = *reinterpret_cast<const uint2*>(sb + lr2 * 128 + pos0);
                afrag[0] = p0.x; afrag[1] = p1.x; afrag[2] = p0.y; afrag[3] = p1.y;
            }
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) fp8_mma(d[nt], afrag, bfrag[nt][ks]);
        }

        // ---- the epilogue (DeepGEMM's mapping): keys {g, g+8} of the warp's 16, heads {2t4, 2t4+1}
        float v0 = 0.0f, v1 = 0.0f;        // v0: key g, v1: key g + 8
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            v0 += fmaxf(d[nt][0], 0.0f) * wv[nt][0] + fmaxf(d[nt][1], 0.0f) * wv[nt][1];
            v1 += fmaxf(d[nt][2], 0.0f) * wv[nt][0] + fmaxf(d[nt][3], 0.0f) * wv[nt][1];
        }
        v0 *= qsc;
        v1 *= qsc;
        #pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {  // the four t4 lanes of each key hold partials
            v0 += __shfl_xor_sync(0xffffffffu, v0, off);
            v1 += __shfl_xor_sync(0xffffffffu, v1, off);
        }

        // ---- the per-key scale (global, 4 bytes x 2 keys a warp), causality and the word. (The
        //      section-18 prefetch of these benched SLOWER - 1.221 vs 1.166: the per-warp
        //      long-scoreboard stalls the NCU capture showed were already hidden by the 24
        //      interleaved warps at 49% issue; the early loads just added LSU pressure and a spill.)
        if (t4 == 0) {
            const int kt0 = kbase + g, kt1 = kbase + g + 8;
            if (kt0 < T) OUT[(size_t)r * T + kt0] = (kt0 <= p) ? order_word(v0 * kscales[(size_t)kt0 * STRIDE_W + 32]) : 0;
            if (kt1 < T) OUT[(size_t)r * T + kt1] = (kt1 <= p) ? order_word(v1 * kscales[(size_t)kt1 * STRIDE_W + 32]) : 0;
        }
        __syncthreads();
    }
}

// the two fatbin entry points (the Zig launch picks by the boot's TF_GLM53_FP8_STRIDE): the legacy
// 132-byte plane (the shipped plateau configuration, 3 CTAs/SM) and inc 3's padded 144-byte plane
// (the 16-byte staging + the identity layout, 2 CTAs/SM). Identical arithmetic, identical results.
extern "C" __global__ void __launch_bounds__(256, 3)
glm53_fp8_scores(const __nv_bfloat16* Q, const float* W,
                 const __nv_fp8_e4m3* CODES, const float* SCALES,
                 const int* POS, const int* BASE, int* OUT,
                 int T, int R0, int ROWS_MODE, int n, float QSCALE, float WSCALE) {
    scores_body<33>(Q, W, CODES, SCALES, POS, BASE, OUT, T, R0, ROWS_MODE, n, QSCALE, WSCALE);
}

extern "C" __global__ void __launch_bounds__(256, 2)
glm53_fp8_scores_p(const __nv_bfloat16* Q, const float* W,
                   const __nv_fp8_e4m3* CODES, const float* SCALES,
                   const int* POS, const int* BASE, int* OUT,
                   int T, int R0, int ROWS_MODE, int n, float QSCALE, float WSCALE) {
    scores_body<36>(Q, W, CODES, SCALES, POS, BASE, OUT, T, R0, ROWS_MODE, n, QSCALE, WSCALE);
}
