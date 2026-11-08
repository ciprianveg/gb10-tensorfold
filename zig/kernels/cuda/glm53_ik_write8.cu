// GLM-5.3's fp8 index-key plane writer: `_ik_write8` / `_ik_write8p` (fused.py) in CUDA — the row the bf16
// cache would hold, quantized to a 132-byte (or 144-byte padded) plane row: 128 e4m3 codes + one fp32 scale
// at word 32, the dual view the scorer reads (codes at IC, scale at ICS).
//
// Numerics (the Triton source's steps, in its order):
//   x  = IK row (fp32); mu = sum(x)/D; v = sum((x-mu)^2)/D; rs = 1/sqrt(v + eps)
//   yv = bf16((x-mu)*rs*NW + NB)                       -> f32        (the non-rotated dims)
//   for the rotated dims (dd < RD), pair i2 = dd or dd-RD/2:
//     ye = bf16((x[2i2]  -mu)*rs*NW[2i2]  +NB[2i2])   -> f32
//     yo = bf16((x[2i2+1]-mu)*rs*NW[2i2+1]+NB[2i2+1]) -> f32
//     ang = float(pg) * INV[i2];  ra = ye*cos - yo*sin;  rb = yo*cos + ye*sin
//     val[dd] = bf16(ra) if dd < RD/2 else bf16(rb)     (f32)
//   amax = max|val|; scale = max(amax/448, 1e-12); codes = e4m3(val/scale), scale stored fp32.
// e4m3 is torch.float8_e4m3fn: __nv_cvt_float_to_fp8 with __NV_SATFINITE (the kv8 convention, r2's own).
//
// DCP: with dcp > 1 only the owner of position pg stores it, at slot pg / dcp (r2's rule). The engine refuses
// dcp > 1 with the fp8 plane on r2's line today (Kv8NeedsDcp1); the guard is kept here so the restriction is
// the caller's switch, not this kernel's.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

constexpr int IW_THREADS = 128;  // one thread a dim (D 128)

// Triton's tl.sum over 128 lanes: a butterfly within the warp, then the four warps' partials in shared.
// The cross-warp total must be broadcast back to every thread: warps 1-3 used to read their own lane-0
// t (0), so their mu and variance were 0 and 3/4 of each row's dims normalized against a zero mean
// (caught by the Q2.5 numeric gate, 2026-10-20).
__device__ __forceinline__ float block_sum(float v, float* smem, int tid) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((tid & 31) == 0) smem[tid >> 5] = v;
  __syncthreads();
  float t = (tid < IW_THREADS / 32) ? smem[tid] : 0.0f;
#pragma unroll
  for (int o = 2; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
  if (tid == 0) smem[0] = t;
  __syncthreads();
  return smem[0];
}

template <int STRIDE_W>  // 33 words (132 bytes) legacy, 36 words (144 bytes) padded: identical values
__device__ void ik_write_body(const float* __restrict__ IK, const __nv_bfloat16* __restrict__ NW,
                              const __nv_bfloat16* __restrict__ NB, const float* __restrict__ INV,
                              __nv_fp8_storage_t* __restrict__ IC, float* __restrict__ ICS,
                              const int* __restrict__ POS, int D, int RD, int DCP, int RANK, float eps) {
  const int r = blockIdx.x;
  const int tid = threadIdx.x;
  const long long pg = (long long)POS[0] + r;
  if (DCP > 1 && (pg % DCP) != RANK) return;
  const long long p = (DCP > 1) ? pg / DCP : pg;

  __shared__ float red[IW_THREADS / 32];
  const float x = IK[(size_t)r * D + tid];
  float mu = block_sum(x, red, tid) / (float)D;
  __syncthreads();
  const float xc = x - mu;
  float v = block_sum(__fmul_rn(xc, xc), red, tid) / (float)D;
  const float rs = 1.0f / sqrtf(v + eps);
  __syncthreads();
  const float yv = __bfloat162float(__float2bfloat16_rn(
      __fadd_rn(__fmul_rn(__fmul_rn((x - mu), rs), __bfloat162float(NW[tid])), __bfloat162float(NB[tid]))));

  float val;
  if (tid < RD) {  // the rotated dims: interleaved pairs, the same pair for both members
    const int i2 = (tid < RD / 2) ? tid : tid - RD / 2;
    const float xe = IK[(size_t)r * D + 2 * i2];
    const float xo = IK[(size_t)r * D + 2 * i2 + 1];
    const float ye = __bfloat162float(__float2bfloat16_rn(__fadd_rn(
        __fmul_rn(__fmul_rn((xe - mu), rs), __bfloat162float(NW[2 * i2])), __bfloat162float(NB[2 * i2]))));
    const float yo = __bfloat162float(__float2bfloat16_rn(__fadd_rn(
        __fmul_rn(__fmul_rn((xo - mu), rs), __bfloat162float(NW[2 * i2 + 1])), __bfloat162float(NB[2 * i2 + 1]))));
    const float ang = __fmul_rn((float)pg, INV[i2]);
    const float c = cosf(ang), s = sinf(ang);
    const float ra = __fsub_rn(__fmul_rn(ye, c), __fmul_rn(yo, s));
    const float rb = __fadd_rn(__fmul_rn(yo, c), __fmul_rn(ye, s));
    val = __bfloat162float(__float2bfloat16_rn(tid < RD / 2 ? ra : rb));
  } else {
    val = yv;
  }

  // amax over the row (max is exact in any order), then the per-token scale and the codes
  __shared__ float amax_red[IW_THREADS / 32];
  const float a = fabsf(val);
  float m = a;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((tid & 31) == 0) amax_red[tid >> 5] = m;
  __syncthreads();
  float amax = (tid < IW_THREADS / 32) ? amax_red[tid] : 0.0f;
#pragma unroll
  for (int o = 2; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  __shared__ float row_scale;
  if (tid == 0) row_scale = fmaxf(amax / 448.0f, 1e-12f);
  __syncthreads();
  const float scale = row_scale;
  IC[p * STRIDE_W * 4 + tid] = __nv_cvt_float_to_fp8(val / scale, __NV_SATFINITE, __NV_E4M3);
  if (tid == 0) ICS[p * STRIDE_W + 32] = scale;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(IW_THREADS)
glm53_ik_write8(const float* __restrict__ IK, const __nv_bfloat16* __restrict__ NW, const __nv_bfloat16* __restrict__ NB,
                const float* __restrict__ INV, __nv_fp8_storage_t* __restrict__ IC, float* __restrict__ ICS,
                const int* __restrict__ POS, int D, int RD, int DCP, int RANK, float eps) {
  ik_write_body<33>(IK, NW, NB, INV, IC, ICS, POS, D, RD, DCP, RANK, eps);
}

extern "C" __global__ void __launch_bounds__(IW_THREADS)
glm53_ik_write8p(const float* __restrict__ IK, const __nv_bfloat16* __restrict__ NW, const __nv_bfloat16* __restrict__ NB,
                 const float* __restrict__ INV, __nv_fp8_storage_t* __restrict__ IC, float* __restrict__ ICS,
                 const int* __restrict__ POS, int D, int RD, int DCP, int RANK, float eps) {
  ik_write_body<36>(IK, NW, NB, INV, IC, ICS, POS, D, RD, DCP, RANK, eps);
}