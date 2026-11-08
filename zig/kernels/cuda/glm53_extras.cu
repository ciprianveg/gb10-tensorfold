// GLM-5.3's family extras for the CSF lane: the widths 1.1's glue does not instantiate (hidden 6144,
// q_lora 2048) and the DSA indexer's scoring entry. The norm arms are the SAME hand-CUDA template as
// 1.1's proven 4096/1536/512 (glm_glue_norm.cuh: block_sum, div_full, sqrt_approx, the double bf16
// round) — only D/NW change, so their math is r2's `_rmsnorm` (D=6144/2048, 8/4 warps) by construction.
//
// The scorer is r2's `_index_scores` (fused.py) in CUDA, bf16 keys (what 1.1's tf_glm_index_write
// stores), PACK=False:
//   score[r,t] = sum_h (W[r,h]*WSCALE) * relu(dot(Q[r,h,:], K[t,:]) * QSCALE), keys t <= p = POS+R0+r
//   key  = monotone(bits(score)), packed key = (key << 32) | (0x7FFFFFFF - position) so ties go to the
//   lower position (r2's PACK=True); later keys are int64 min + 1 (the radix selection's 0-priority).
// with WSCALE = 1/sqrt(NH), QSCALE = 1/sqrt(D). Accumulation order is D-then-heads in fp32; the GPU
// gate compares it against r2's Triton `_index_scores` (r2 ships its native fp8 sibling, carried in
// glm53_fp8_scores.cu, for the fp8 plane path).
#include "glm_glue_norm.cuh"

// the widths the CSF lane runs and 1.1 does not: hidden 6144 (8 warps), q_lora 2048 (4 warps)
extern "C" __global__ void __launch_bounds__(8 * 32)
    tf_glm53_rmsnorm_6144(const __nv_bfloat16* x, int64_t x_stride, const __nv_bfloat16* w, __nv_bfloat16* out,
                          int64_t o_stride, float* xs, float eps, int sums) {
  glm_glue::rmsnorm<6144, 8192, 8, 8>(x, x_stride, w, out, o_stride, xs, eps, sums);
}

extern "C" __global__ void __launch_bounds__(4 * 32)
    tf_glm53_rmsnorm_2048(const __nv_bfloat16* x, int64_t x_stride, const __nv_bfloat16* w, __nv_bfloat16* out,
                          int64_t o_stride, float* xs, float eps, int sums) {
  glm_glue::rmsnorm<2048, 2048, 4, 8>(x, x_stride, w, out, o_stride, xs, eps, sums);
}

namespace {

constexpr int KS = 128;  // keys a block scores (one thread each)

// the monotone int of an fp32 (fused.py: bits >= 0 ? bits : bits ^ 0x7FFFFFFF), then the unsigned
// order word (_index_scores PACK=False: key ^ -2147483648); a later key is the 0 sentinel.
__device__ __forceinline__ long long packed_key(float score, int position) {
  const int bits = __float_as_int(score);
  const int key = (bits >= 0) ? bits : (bits ^ 0x7FFFFFFF);   // monotone in the fp32's order
  const unsigned lo = 0x7FFFFFFFu - static_cast<unsigned>(position);
  return (static_cast<long long>(key) << 32) | static_cast<long long>(lo);
}

}  // namespace

extern "C" __global__ void __launch_bounds__(KS)
    tf_glm53_index_score(const __nv_bfloat16* Q, const float* W, const __nv_bfloat16* KC, long long* OUT,
                         int T, int R0, int POS, int NH, int D, float QSCALE, float WSCALE) {
  __shared__ __nv_bfloat16 q[32 * 128];   // one row's queries: NH x D (NH 32, D 128 in this model)
  __shared__ __nv_bfloat16 k[KS * 128];   // the block's key tile
  const int r = blockIdx.x;               // window row
  const int t0 = blockIdx.y * KS;         // this block's first key
  const int p = POS + R0 + r;             // the row's position: keys t <= p are visible
  const int tid = threadIdx.x;
  const int qn = NH * D;
  for (int i = tid; i < qn; i += KS) q[i] = Q[(size_t)r * qn + i];
  const int kn = KS * D;
  const bool tile_ok = t0 <= p;
  for (int i = tid; i < kn; i += KS) {
    const int t = t0 + i / D;
    const int d = i % D;
    k[i] = (tile_ok && t < T) ? KC[(size_t)t * D + d] : __float2bfloat16(0.0f);
  }
  __syncthreads();
  const int t = t0 + tid;
  if (t >= T) return;
  long long word = -9223372036854775807LL;   // int64 min + 1: invisible keys, lowest priority
  if (tile_ok && t <= p) {
    const int tn = tid * D;
    float acc = 0.0f;
    for (int h = 0; h < NH; ++h) {
      const int qb = h * D;
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) dot = __fmaf_rn(__bfloat162float(q[qb + d]), __bfloat162float(k[tn + d]), dot);
      const float s = __fmul_rn(dot, QSCALE);
      const float relu = s > 0.0f ? s : 0.0f;
      acc = __fmaf_rn(__fmul_rn(W[(size_t)r * NH + h], WSCALE), relu, acc);
    }
    word = packed_key(acc, t);
  }
  OUT[(size_t)r * T + t] = word;
}
// ---------------------------------------------------------------------------------------------------
// THE ROPE HALF of the latent attention (findings §36). The DeepSeek-shaped MLA this pack carries has
// qk_nope = 192 and qk_rope = 64, and its latent row is kv_lora + rope = 576 (r2: `latentWidth() ==
// kv_lora + rope`, "a 576-wide latent row a token and layer"): the absorbed latent, then the shared rope
// key rotated at the row's TRUE position (r2's `_kv_write`: "rotates the rope half of its row at the
// row's true position (fused.py: ang_p = pg)"). The query is packed the same way (r2's `qpack [rows, H,
// lw + rd]`, `_absorb` writing qlat and qrot), so the attention's score is one dot over 576:
//     score(r,h,j) = qlat(r,h).c(j) + qrot(r,h).c_rope(j)
// and `chunks_wide` at W = 576 needs no separate correction term. The rotation convention is r2's, the
// one this tree's byte-identical `glm53_ik_write8` already carries: interleaved pairs (2i, 2i + 1),
// angle = position * INV[i], (a, b) -> (a*cos - b*sin, b*cos + a*sin), the result rounded to bf16.
// ---------------------------------------------------------------------------------------------------
#include "glm53_glue_attn.cuh"

namespace {

// One interleaved pair's rotation, r2's rounding points: f32 multiply/subtract/add, bf16 out.
__device__ __forceinline__ void rope_pair(float ye, float yo, float ang, __nv_bfloat16* e, __nv_bfloat16* o) {
  const float c = cosf(ang), s = sinf(ang);
  const float ra = __fsub_rn(__fmul_rn(ye, c), __fmul_rn(yo, s));
  const float rb = __fadd_rn(__fmul_rn(yo, c), __fmul_rn(ye, s));
  *e = __float2bfloat16_rn(ra);
  *o = __float2bfloat16_rn(rb);
}

}  // namespace

// _kv_write at width lw + rd: row r of `lat` (bf16, stride lat_stride) into cache slot POS[0] + r, then
// the rope key `krope` (bf16, stride kstride, rd values) rotated at that row's true position.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm53_kv_write(const __nv_bfloat16* __restrict__ lat, int64_t lat_stride,
                      const __nv_bfloat16* __restrict__ krope, int64_t kstride, __nv_bfloat16* __restrict__ cache,
                      const int* __restrict__ pos, int64_t lw, int64_t rd, const float* __restrict__ inv) {
  const int64_t r = blockIdx.x, pg = (int64_t)pos[0] + r, row = pg * (lw + rd);
  for (int64_t k = threadIdx.x; k < lw; k += blockDim.x) cache[row + k] = lat[r * lat_stride + k];
  for (int64_t i = threadIdx.x; i < rd / 2; i += blockDim.x) {
    const float ang = __fmul_rn((float)pg, inv[i]);
    rope_pair(__bfloat162float(krope[r * kstride + 2 * i]), __bfloat162float(krope[r * kstride + 2 * i + 1]), ang,
              cache + row + lw + 2 * i, cache + row + lw + 2 * i + 1);
  }
}

// _absorb's qrot half and the pack: out[r][h][0..lw) = qlat[r][h][:], out[r][h][lw..lw + rd) = the rope
// rotation of q[r][h][dn..dn + rd) at row r's position. One block a (row, head).
extern "C" __global__ void __launch_bounds__(128)
    tf_glm53_qpack(const __nv_bfloat16* __restrict__ qlat, const __nv_bfloat16* __restrict__ q,
                   const float* __restrict__ inv, const int* __restrict__ pos, __nv_bfloat16* __restrict__ out,
                   int64_t rows, int64_t heads, int64_t dn, int64_t lw, int64_t rd) {
  const int64_t i = blockIdx.x;  // r * heads + h
  const int64_t r = i / heads, pg = (int64_t)pos[0] + r;
  __nv_bfloat16* o = out + i * (lw + rd);
  const __nv_bfloat16* l = qlat + i * lw;
  for (int64_t k = threadIdx.x; k < lw; k += blockDim.x) o[k] = l[k];
  const __nv_bfloat16* qr = q + i * (dn + rd) + dn;
  for (int64_t j = threadIdx.x; j < rd / 2; j += blockDim.x) {
    const float ang = __fmul_rn((float)pg, inv[j]);
    rope_pair(__bfloat162float(qr[2 * j]), __bfloat162float(qr[2 * j + 1]), ang, o + lw + 2 * j, o + lw + 2 * j + 1);
  }
}

// latent._dense_chunks/_sparse_chunks at the DeepSeek-shaped pack's width (W = 576): the same body as
// 1.1's 16/32-head entries (glm_glue_attn.cuh), one template argument wider. The output stays lw = 512.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm53_chunks_16(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok, const int* cnt,
                       float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w, float scale, int sparse) {
  glm53_attn::chunks_wide<16, 576>(qa, lc, pos, tok, cnt, po, pm, pl, rows, heads, w, scale, sparse);
}

// The 32-head form at this width needs (32*576 + 2*32*576 + 32*32)*2 + 192*4 = 113,408 B of dynamic
// smem; the GB10's cudaDevAttrMaxSharedMemoryPerBlockOptin is 101,376 B (measured on .111). So the wide
// path is 16 heads a block and a prompt chunk takes two blocks a row - a perf note, not a correctness one.
