// GLM-5.3 8-bit KV cache (TF_GLM53_KV8, ours): the MLA latent cache's rows as e4m3 codes with two fp32 scales a
// row (one the latent half, one the shared rope half - capture.py's absmax/448 per-row pattern). The plane's row is
// W = LW + RD bytes of codes followed by the two scales (row stride W + 8 bytes); the indexer's K cache stays bf16
// (the DSA top-k scoring reads all of it every step - quantized scores would move the selected SET, not just the
// values; no lineage precedent covers it, and it is 22 x 256 B of the 94.4 KiB a token takes in bf16).
//
//   glm53_kv8_quant: a window's written rows sit in a bf16 staging buffer as rows 0..R-1 (the unchanged Triton
//     _kv_write put them there: glm53_kv8_stage's tables give row r at staging row r, the rope rotated at the
//     row's TRUE position); one program a row finds its destination row exactly
//     as _kv_write would have (non-ROWS: POS[0] + r; ROWS: BASE[r] + POS[r] - the TRUE tables), takes the two
//     halves' absmax, writes scale = absmax/448 (1.0 when 0) and the e4m3 codes. Deterministic.
//   glm53_kv8_dequant: materializes the bf16 rows the unchanged Triton attention kernels read, into a scratch:
//     mode 0 (prompt / wide windows): rows [0, POS[0] + R) direct - the kernels read keys 0..p with no TOK/BASE,
//     so the scratch mirrors the plane; the launch covers n_max rows and each program leaves when it is past the
//     end (the position lives on the device: graph-safe callers never use this mode).
//     mode 1 (one stream's decode window): program (r, i) of R x K - p = POS[0] + r; while p < K the kernel reads
//     keys 0..p directly, so rows i <= p land at scratch row i (the shared prefix region [0, K)); past K it reads
//     TOK[r*K + i], so the selection lands at scratch row r*K + i (row r's block). The launch passes TOK' = a ramp
//     (r*K + i), which points the kernel at exactly those blocks; rows with p < K never consult TOK'. The prefix
//     region and row 0's block are the same rows [0, K), but a row reads one or the other, never both.
//     mode 2 (--parallel N, a base table): p = POS[r], the row's stream's rows at BASE[r]; every row's keys land in
//     its own block r*K + i (the kernel takes BASE' = a r*K ramp and TOK' = a per-row ramp, so both its p < K and
//     its TOK reads stay inside the block).
//   Several programs may write the same scratch row with the same bytes (mode 1's shared prefix): the writes are
//   identical, so the result does not depend on the order they land.
//
// e4m3 is torch.float8_e4m3fn: __nv_cvt_float_to_fp8 with __NV_SATFINITE (no inf, max 448) and back.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {
constexpr float FP8_MAX = 448.0f;

__device__ __forceinline__ float bf16_at(const unsigned short* s, size_t i) {
    return __bfloat162float(__ushort_as_bfloat16(s[i]));
}

__device__ __forceinline__ void put_bf16(unsigned short* d, size_t i, float v) {
    d[i] = __bfloat16_as_ushort(__float2bfloat16(v));
}

// the plane's row stride in bytes: W codes then the two scales
__device__ __forceinline__ size_t row_bytes(int w) { return (size_t)w + 8; }

// 119.1 (TF_GLM53_KV_PAGES): row r's byte offset in a plane of 512-row pages - direct when PT is null
// (today's arithmetic, bit-identical), or page PT[r >> 9]'s. The same table serves every layer and every
// rank; a shuffled table makes any reader that bypasses it take wrong bytes (the anchors are the proof).
__device__ __forceinline__ size_t row_off(const int* __restrict__ PT, long long r, size_t rb) {
    return PT == nullptr ? (size_t)r * rb : (size_t)PT[r >> 9] * (512 * rb) + (size_t)(r & 511) * rb;
}

// 119.2 (TF_GLM53_KV8_V3): the row's stride in bytes - v3: LW codes + the rope as bf16 + LW/128 scales
// (656 at LW 512, RD 64); v2: W codes + two scales (584).
__device__ __forceinline__ size_t row_stride(int V3, int LW, int W) {
    return V3 ? (size_t)LW + 2 * (size_t)(W - LW) + 16 : row_bytes(W);
}
}  // namespace

// Grid (rows), 128 threads. The staging write's position / base tables (THE ROPE FIX, findings-kv8.md §11):
// _kv_write rotates the rope half of its row at the row's TRUE position (fused.py: ang_p = pg), so the staging
// redirect must NOT zero it. The ROWS variant already decouples the two: the angle comes from POS[r] alone and
// the write row is BASE[r] + POS[r]. This builds P'[r] = the row's true position (rows_mode: POS[r], else
// POS[0] + r) and B'[r] = r - P'[r], so _kv_write ROWS writes staging row r while rotating at the true position -
// the staged row is bit-identical to what the bf16 cache would hold, and the quantizer is unchanged.
extern "C" __global__ void __launch_bounds__(128)
glm53_kv8_stage(const int* __restrict__ POS, int* __restrict__ spos, int* __restrict__ sbase, int rows, int ROWS_MODE, int DCP) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const long long p = ROWS_MODE ? (long long)POS[r] : (long long)POS[0] + r;
    spos[r] = (int)p;
    // DCP8 (findings-dcp8.md section 1): the patched _kv_write ROWS writes its row at B'[r] +
    // P'[r] / DCP (the owner's shard row), so the staging redirect compensates: B' = r - P'/DCP
    // keeps the write at staging row r while the rope still rotates at the TRUE position. The
    // owner guard (P' % DCP == RANK, in the patched kernel) keeps only this rank's rows staged.
    sbase[r] = (int)((long long)r - (DCP > 1 ? p / DCP : p));
}

// Grid (rows), 128 threads. stage: bf16 [rows, W]; plane: rows of W code bytes + 2 fp32 scales.
// ROWS_MODE: dest = BASE[r] + POS[r]; else dest = POS[0] + r (as _kv_write places the row).
// DCP8 (findings-dcp8.md section 1): dest is the RANK'S SHARD row - POS / DCP - and only the
// owned rows are quantized (the staging holds only them under DCP: the patched _kv_write's owner
// guard returned the rest early; the guard here keeps a stale staging row from ever landing).
// DCP 1: the addresses and the coverage are exactly what they were.
extern "C" __global__ void __launch_bounds__(128)
glm53_kv8_quant(const unsigned short* __restrict__ stage, unsigned char* __restrict__ plane,
                const int* __restrict__ POS, const int* __restrict__ BASE, int rows, int ROWS_MODE, int DCP, int RANK, int LW, int W,
                const int* __restrict__ PT, int V3) {
    const int r = blockIdx.x, t = threadIdx.x;
    if (r >= rows) return;
    const long long p = ROWS_MODE ? (long long)POS[r] : (long long)POS[0] + r;
    long long dest;
    if (DCP > 1) {
        if (p % DCP != RANK) return;
        dest = (ROWS_MODE ? (long long)BASE[r] : 0) + p / DCP; // non-ROWS never reads BASE
    } else {
        dest = ROWS_MODE ? (long long)BASE[r] + p : p;
    }
    const unsigned short* s = stage + (size_t)r * (size_t)W;
    unsigned char* d = plane + row_off(PT, dest, row_stride(V3, LW, W));
    if (V3) {
        // v3 (105.1's flashinfer byte map): 512 e4m3 codes [0,LW), 4 fp32 block scales [LW, LW+16), the
        // rope as the staging's own bf16 bits [LW+16, LW+16+2*RD) - exact (v2 quantized it e4m3 with one
        // whole-half scale). The latent half a scale a 128-element block (thread t holds element 128*b + t
        // of block b: LW a multiple of 128, 128 threads - one reduction round a block).
        const int RD = W - LW, NB = LW / 128;
        float* sc3 = (float*)(d + LW);
        float am[8];
        for (int b = 0; b < NB; b++) am[b] = fabsf(bf16_at(s, (size_t)(128 * b + t)));
        __shared__ float red[128];
        for (int b = 0; b < NB; b++) {
            red[t] = am[b];
            __syncthreads();
            if (t == 0) {
                float m = red[0];
                for (int i = 1; i < 128; i++) if (red[i] > m) m = red[i];
                sc3[b] = m > 0.0f ? m / FP8_MAX : 1.0f;
            }
            __syncthreads();
        }
        for (int i = t; i < LW; i += 128)
            d[i] = __nv_cvt_float_to_fp8(bf16_at(s, (size_t)i) / sc3[i / 128], __NV_SATFINITE, __NV_E4M3);
        for (int j = t; j < RD; j += 128)
            *(unsigned short*)(d + LW + 16 + 2 * j) = s[LW + j]; // the rope: the staging's bf16, exact
        return;
    }
    float* sc = (float*)(d + W);
    // the two halves' absmax
    float am0 = 0.0f, am1 = 0.0f;
    for (int i = t; i < W; i += 128) {
        const float v = fabsf(bf16_at(s, (size_t)i));
        if (i < LW) { if (v > am0) am0 = v; } else { if (v > am1) am1 = v; }
    }
    __shared__ float red[128];
    red[t] = am0;
    __syncthreads();
    if (t == 0) {
        for (int i = 1; i < 128; i++) if (red[i] > red[0]) red[0] = red[i];
    }
    __syncthreads();
    am0 = red[0];
    __syncthreads();
    red[t] = am1;
    __syncthreads();
    if (t == 0) {
        for (int i = 1; i < 128; i++) if (red[i] > red[0]) red[0] = red[i];
        const float s0 = am0 > 0.0f ? am0 / FP8_MAX : 1.0f;
        const float s1 = red[0] > 0.0f ? red[0] / FP8_MAX : 1.0f;
        sc[0] = s0;
        sc[1] = s1;
    }
    __syncthreads();
    const float s0 = sc[0], s1 = sc[1];
    for (int i = t; i < W; i += 128)
        d[i] = __nv_cvt_float_to_fp8(bf16_at(s, (size_t)i) / (i < LW ? s0 : s1), __NV_SATFINITE, __NV_E4M3);
}

// Grid (n_max) mode 0 / grid (rows * K) modes 1-2 / a fixed 4096-block grid striding the plane mode 4
// (the per-slot bounds on the device), 128 threads. One program one 576-byte row: 144 words of four
// codes, the half's scale picked per element (LW is a whole number of words: 512 = 128 words).
__device__ __forceinline__ void kv8_dequant_row(const unsigned char* __restrict__ plane, const int* __restrict__ PT,
        unsigned short* __restrict__ out, int W, int LW, long long src, long long dst, int V3) {
    const unsigned char* s = plane + row_off(PT, src, row_stride(V3, LW, W));
    unsigned short* d = out + (size_t)dst * (size_t)W;
    if (V3) {
        // v3 (105.1's byte map): the scales at [LW, LW+16), the rope's bf16 at [LW+16, ...); the latent
        // codes with their block's scale (a word never spans a block: 128 % 4 == 0, block = w / 32)
        const int RD = W - LW;
        const float* sc3 = (const float*)(s + LW);
        const int lwords = LW / 4;
        for (int w = threadIdx.x; w < lwords; w += 128) {
            const unsigned u = *(const unsigned*)(s + 4 * (size_t)w);
            const float sb = sc3[w / 32];
            for (int j = 0; j < 4; j++) {
                __nv_fp8_e4m3 code;
                code.__x = (unsigned char)(u >> (8 * j));
                put_bf16(d, (size_t)(4 * w + j), (float)code * sb);
            }
        }
        const int rwords = (RD * 2) / 4;
        for (int w = threadIdx.x; w < rwords; w += 128) {
            const unsigned u = *(const unsigned*)(s + LW + 16 + 4 * (size_t)w);
            d[LW + 2 * w] = (unsigned short)(u & 0xFFFF);
            d[LW + 2 * w + 1] = (unsigned short)(u >> 16);
        }
        return;
    }
    const float* sc = (const float*)(s + W);
    const float s0 = sc[0], s1 = sc[1];
    const int words = W / 4;
    for (int w = threadIdx.x; w < words; w += 128) {
        const unsigned u = *(const unsigned*)(s + 4 * (size_t)w);
        for (int j = 0; j < 4; j++) {
            const int e = 4 * w + j;
            __nv_fp8_e4m3 code;
            code.__x = (unsigned char)(u >> (8 * j));
            const float v = (float)code * (e < LW ? s0 : s1);
            put_bf16(d, (size_t)e, v);
        }
    }
}

extern "C" __global__ void __launch_bounds__(128)
glm53_kv8_dequant(const unsigned char* __restrict__ plane, unsigned short* __restrict__ out,
                  const int* __restrict__ POS, const int* __restrict__ BASE, const int* __restrict__ TOK,
                  int rows, int K, int LW, int W, int mode, int DCP, long long n_max,
                  const int* __restrict__ PT, int V3) {
    long long src, dst;
    if (mode == 4) {
        // DCP8 + several streams (findings-dcp8.md sections 10 and 14): a multi window's rows read
        // their streams' shards anywhere in the slots plane (BASE[r] rows in), so the mirror is the
        // plane - but ONLY up to each slot's bound: POS points at [slots] absolute row bounds
        // (slot s's rows [s*K, s*K + pos_s/DCP + margin)), which the multi updates per round. The
        // read happens at replay, so captured graphs stay valid; a FIXED grid strides the plane so
        // the launch works at any bound. K here is the slot rows (local), not the selection width.
        // The whole-plane copy this replaces was O(context x slots) a layer a round and
        // bandwidth-bound the decode (section 14).
        for (long long i = blockIdx.x; i < n_max; i += gridDim.x) {
            const int slot = (int)(i / K);
            if (i >= (long long)POS[slot]) continue;
            kv8_dequant_row(plane, PT, out, W, LW, i, i, V3);
        }
        return;
    }
    if (mode == 0) {
        const long long i = blockIdx.x;
        if (i >= n_max || i >= (long long)POS[0] + rows) return;
        src = i;
        dst = i;
    } else if (mode == 3) {
        // DCP8 (findings-dcp8.md section 1): the plane IS this rank's shard (its row i is the
        // global position i * DCP + RANK), and _attn_dcp reads a rank's shard exactly as the
        // bf16 DCP cache it was written for - so the mirror is direct: scratch row i = plane
        // row i, up to the window's last owned position / DCP. The scratch is the pscratch
        // mirror (sized the local capacity - the full shard at parallel == dcp).
        const long long i = blockIdx.x;
        if (i >= n_max || i > ((long long)POS[0] + rows) / DCP) return;
        src = i;
        dst = i;
    } else if (mode == 4) {
        // DCP8 + several streams (findings-dcp8.md section 10, the decode/MTP windows): a multi
        // window's rows read their streams' shards ANYWHERE in the slots plane (BASE[r] rows in,
        // positions a table), so mode 3's POS[0] + rows bound does not cover them - the mirror
        // is the WHOLE plane instead: every row the allocation holds (n_max = local * slots),
        // the position never read.
        const long long i = blockIdx.x;
        if (i >= n_max) return;
        src = i;
        dst = i;
    } else {
        const long long p = blockIdx.x;
        const long long r = p / K, i = p % K;
        if (r >= rows) return;
        if (mode == 2) {
            const long long pr = (long long)POS[r], base = (long long)BASE[r];
            if (pr < K) {
                if (i > pr) return;
                src = base + i;
            } else {
                src = base + (long long)TOK[r * K + i];
            }
            dst = r * K + i;
        } else {
            const long long pr = (long long)POS[0] + r;
            if (pr < K) {
                if (i > pr) return;
                src = i;
                dst = i; // the shared prefix: while p < K the kernel reads key idx flat, TOK' never consulted
            } else {
                src = (long long)TOK[r * K + i];
                dst = r * K + i; // past K the identity ramp (TOK' = r*K + i) points the kernel at row r's block
            }
        }
    }
    kv8_dequant_row(plane, PT, out, W, LW, src, dst, V3);
}
