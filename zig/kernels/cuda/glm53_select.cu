// GLM-5.3 index selection: r2's radix top-k over the indexer's order words (glm53_decode.cu, byte-identical
// bodies; the fmode-2 arm orders packed int64 keys by their high word, ties to the lower column =
// the lower position, so the set is exactly _index_scores PACK=True + torch.topk's).
#include <cstdint>

namespace {
constexpr int SEL_THREADS = 256;
constexpr int SEL_BINS = 2048;

__device__ __forceinline__ unsigned sel_key(unsigned bits, int fmode) {
    if (!fmode) return bits;
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return 0xFFFFFFFFu;     // NaN: first, ties by column
    if (bits == 0x80000000u) bits = 0u;                               // -0 compares equal to +0
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);       // monotone in the float's order
}

// key i of a row at `s` (a row's base: keys + row * ld words, two words a key in fmode 2)
__device__ __forceinline__ unsigned key_at(const unsigned* s, int i, int fmode) {
    if (fmode == 2) return s[2 * (size_t)i + 1] ^ 0x80000000u;    // the packed key's high word, unsigned order
    return sel_key(s[i], fmode);
}

__device__ __forceinline__ const unsigned* row_of(const unsigned* keys, int row, long long ld, int fmode) {
    return keys + (size_t)row * (size_t)ld * (fmode == 2 ? 2 : 1);
}

__device__ __forceinline__ int sel_shift(int pass) { return pass == 0 ? 21 : pass == 1 ? 10 : 0; }
__device__ __forceinline__ int sel_bins(int pass) { return pass == 2 ? 1024 : 2048; }
}  // namespace

// Grid (blocks of `chunk` columns, rows), SEL_THREADS threads. state [rows, 4] uint32: prefix, fixed mask, keys
// still needed at the prefix, arrival count (0 between launches). ghist [rows, SEL_BINS] uint32 (0 between passes).
extern "C" __global__ void __launch_bounds__(SEL_THREADS)
glm53_topk_hist(const unsigned* __restrict__ keys, int nc, long long ld, int chunk, int pass, int K, int fmode,
                unsigned* __restrict__ ghist, unsigned* __restrict__ state) {
    __shared__ unsigned h[SEL_BINS];
    __shared__ unsigned suf[SEL_THREADS];
    __shared__ int last;
    const int row = blockIdx.y, t = threadIdx.x;
    const unsigned* s = row_of(keys, row, ld, fmode);
    unsigned* gh = ghist + (size_t)row * SEL_BINS;
    unsigned* st = state + (size_t)row * 4;
    const int shift = sel_shift(pass), nb = sel_bins(pass);
    // read before any block of this launch can rewrite them (the last block writes only after every block arrived)
    const unsigned prefix = pass ? st[0] : 0u;
    const unsigned fixed = pass ? st[1] : 0u;
    const unsigned need = pass ? st[2] : (unsigned)K;
    for (int i = t; i < nb; i += SEL_THREADS) h[i] = 0u;
    __syncthreads();
    const int c0 = blockIdx.x * chunk;
    const int c1 = min(nc, c0 + chunk);
    for (int i = c0 + t; i < c1; i += SEL_THREADS) {
        const unsigned u = key_at(s, i, fmode);
        if ((u & fixed) == prefix) atomicAdd(&h[(u >> shift) & (unsigned)(nb - 1)], 1u);
    }
    __syncthreads();
    for (int i = t; i < nb; i += SEL_THREADS)
        if (h[i]) atomicAdd(&gh[i], h[i]);
    __threadfence();
    __syncthreads();
    if (t == 0) last = atomicAdd(&st[3], 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = t; i < nb; i += SEL_THREADS) {
        h[i] = __ldcg(&gh[i]);
        gh[i] = 0u;
    }
    __syncthreads();
    const int per = nb / SEL_THREADS;                 // 8 or 4 bins a thread, ascending
    unsigned mine = 0;
    for (int j = 0; j < per; ++j) mine += h[t * per + j];
    suf[t] = mine;
    __syncthreads();
    for (int off = 1; off < SEL_THREADS; off <<= 1) {   // suf[t] = keys in the bins of threads t..
        const unsigned v = t + off < SEL_THREADS ? suf[t + off] : 0u;
        __syncthreads();
        suf[t] += v;
        __syncthreads();
    }
    unsigned acc = suf[t] - mine;                     // keys in the bins above this thread's
    for (int j = per - 1; j >= 0; --j) {              // the digit: keys above it < need <= keys at or above it
        const unsigned c = h[t * per + j];
        if (acc < need && acc + c >= need) {
            st[0] = prefix | ((unsigned)(t * per + j) << shift);
            st[1] = fixed | ((unsigned)(nb - 1) << shift);
            st[2] = need - acc;
        }
        acc += c;
    }
    if (t == 0) st[3] = 0u;
}

// Grid (blocks, rows): cnt[(row * blocks + block) * 2 + {0, 1}] = keys of the block above / equal to the K-th key.
extern "C" __global__ void __launch_bounds__(SEL_THREADS)
glm53_topk_count(const unsigned* __restrict__ keys, int nc, long long ld, int chunk, int fmode,
                 const unsigned* __restrict__ state, unsigned* __restrict__ cnt) {
    __shared__ unsigned sg[SEL_THREADS / 32], se[SEL_THREADS / 32];
    const int row = blockIdx.y, t = threadIdx.x;
    const unsigned* s = row_of(keys, row, ld, fmode);
    const unsigned prefix = state[(size_t)row * 4];
    const int c0 = blockIdx.x * chunk;
    const int c1 = min(nc, c0 + chunk);
    unsigned g = 0, e = 0;
    for (int i = c0 + t; i < c1; i += SEL_THREADS) {
        const unsigned u = key_at(s, i, fmode);
        g += u > prefix;
        e += u == prefix;
    }
    for (int o = 16; o > 0; o >>= 1) {
        g += __shfl_down_sync(0xffffffffu, g, o);
        e += __shfl_down_sync(0xffffffffu, e, o);
    }
    if ((t & 31) == 0) {
        sg[t >> 5] = g;
        se[t >> 5] = e;
    }
    __syncthreads();
    if (t == 0) {
        unsigned a = 0, b = 0;
        for (int w = 0; w < SEL_THREADS / 32; ++w) {
            a += sg[w];
            b += se[w];
        }
        cnt[((size_t)row * gridDim.x + blockIdx.x) * 2] = a;
        cnt[((size_t)row * gridDim.x + blockIdx.x) * 2 + 1] = b;
    }
}

// Grid (blocks, rows): out[row * out_ld + j], j < K = the row's K best columns ascending: every key above the K-th
// and the first `need` keys equal to it (column order). A thread walks chunk / SEL_THREADS consecutive columns.
extern "C" __global__ void __launch_bounds__(SEL_THREADS)
glm53_topk_write(const unsigned* __restrict__ keys, int nc, long long ld, int chunk, int fmode,
                 const unsigned* __restrict__ state, const unsigned* __restrict__ cnt, int* __restrict__ out,
                 long long out_ld) {
    __shared__ unsigned sg[SEL_THREADS], se[SEL_THREADS];
    __shared__ unsigned bg, be;
    const int row = blockIdx.y, t = threadIdx.x;
    const unsigned* s = row_of(keys, row, ld, fmode);
    int* o = out + (size_t)row * out_ld;
    const unsigned prefix = state[(size_t)row * 4];
    const unsigned need = state[(size_t)row * 4 + 2];
    if (t == 0) {
        unsigned a = 0, b = 0;
        for (unsigned i = 0; i < blockIdx.x; ++i) {
            a += cnt[((size_t)row * gridDim.x + i) * 2];
            b += cnt[((size_t)row * gridDim.x + i) * 2 + 1];
        }
        bg = a;
        be = b;
    }
    const int per = chunk / SEL_THREADS;
    const int i0 = blockIdx.x * chunk + t * per;
    const int i1 = min(nc, i0 + per);
    unsigned g = 0, e = 0;
    for (int i = i0; i < i1; ++i) {
        const unsigned u = key_at(s, i, fmode);
        g += u > prefix;
        e += u == prefix;
    }
    sg[t] = g;
    se[t] = e;
    __syncthreads();
    for (int off = 1; off < SEL_THREADS; off <<= 1) {   // inclusive prefix sums over threads
        const unsigned a = t >= off ? sg[t - off] : 0u;
        const unsigned b = t >= off ? se[t - off] : 0u;
        __syncthreads();
        sg[t] += a;
        se[t] += b;
        __syncthreads();
    }
    unsigned gb = bg + sg[t] - g;                     // keys above the K-th before this thread's first column
    unsigned eb = be + se[t] - e;                     // keys equal to it before that column
    for (int i = i0; i < i1; ++i) {
        const unsigned u = key_at(s, i, fmode);
        if (u > prefix) {
            o[gb + min(eb, need)] = i;
            ++gb;
        } else if (u == prefix) {
            if (eb < need) o[gb + eb] = i;
            ++eb;
        }
    }
}


// ROUND 3 (r2 findings-dcp2-build.md §I, ported under SEAM-DCP2 §5): the COMPACT radix select. The five
// launches of the columns select stream the whole keys plane each; past the pass-0 histogram only the
// survivors (the keys at or above the running digit, < K + one bin a row) matter. Here pass 1
// (glm53_topk_hist1c) additionally compacts the at-OR-above-digit-0 positions (cap SEL_CAP a row; past
// it the overflow flag makes hist2c walk the full row instead - rare, exact); pass 2, the final prefix,
// the collect and the emit are ONE block a row (glm53_topk_hist2c) over the compact list. Two full reads
// of the keys instead of five, the same set and tie rule (lower column) as glm53_topk_write.
//
// Local adaptations from r2's bodies (the rest is byte-identical): `fmode` is a parameter (our fp8 plane
// selects over int32 order words, fmode 0; the bf16 fallback over packed int64, fmode 2), and the emit
// writes the COLUMN (int32) our indexer's contract carries, where r2 emits the packed key for dcpCands.
constexpr int SEL_CAP = 8192;

extern "C" __global__ void __launch_bounds__(SEL_THREADS)
glm53_topk_hist1c(const unsigned* __restrict__ keys, int nc, long long ld, int chunk, int K, int fmode,
                  unsigned* __restrict__ ghist, unsigned* __restrict__ state,
                  unsigned* __restrict__ lst, unsigned* __restrict__ lcnt, unsigned* __restrict__ lflg, int cap) {
    __shared__ unsigned h[SEL_BINS];
    __shared__ unsigned suf[SEL_THREADS];
    __shared__ int last;
    const int row = blockIdx.y, t = threadIdx.x;
    const unsigned* s = row_of(keys, row, ld, fmode);
    unsigned* gh = ghist + (size_t)row * SEL_BINS;
    unsigned* st = state + (size_t)row * 4;
    unsigned* lc = lcnt + row;
    unsigned* lf = lflg + row;
    const int shift = sel_shift(1), nb = sel_bins(1);
    const unsigned prefix = st[0], fixed = st[1];
    const unsigned need = st[2];
    for (int i = t; i < nb; i += SEL_THREADS) h[i] = 0u;
    __syncthreads();
    const int c0 = blockIdx.x * chunk;
    const int c1 = min(nc, c0 + chunk);
    for (int i = c0 + t; i < c1; i += SEL_THREADS) {
        const unsigned u = key_at(s, i, fmode);
        const unsigned m = u & fixed;
        if (m >= prefix) {                       // the compact list: at OR above digit 0
            const unsigned idx = atomicAdd(lc, 1u);
            if (idx < (unsigned)cap) lst[(size_t)row * cap + idx] = (unsigned)i;
            else *lf = 1u;
        }
        if (m == prefix) atomicAdd(&h[(u >> shift) & (unsigned)(nb - 1)], 1u);   // the pass-1 histogram: at only
    }
    __syncthreads();
    for (int i = t; i < nb; i += SEL_THREADS)
        if (h[i]) atomicAdd(&gh[i], h[i]);
    __threadfence();
    __syncthreads();
    if (t == 0) last = atomicAdd(&st[3], 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int i = t; i < nb; i += SEL_THREADS) {
        h[i] = __ldcg(&gh[i]);
        gh[i] = 0u;
    }
    __syncthreads();
    const int per = nb / SEL_THREADS;
    unsigned mine = 0;
    for (int j = 0; j < per; ++j) mine += h[t * per + j];
    suf[t] = mine;
    __syncthreads();
    for (int off = 1; off < SEL_THREADS; off <<= 1) {
        const unsigned v = t + off < SEL_THREADS ? suf[t + off] : 0u;
        __syncthreads();
        suf[t] += v;
        __syncthreads();
    }
    unsigned acc = suf[t] - mine;
    for (int j = per - 1; j >= 0; --j) {
        const unsigned c = h[t * per + j];
        if (acc < need && acc + c >= need) {
            st[0] = prefix | ((unsigned)(t * per + j) << shift);
            st[1] = fixed | (unsigned)((nb - 1) << shift);
            st[2] = need - acc;
        }
        acc += c;
    }
    if (t == 0) st[3] = 0u;
}

// One block a row: pass 2 + the finish over the compact list (or, on overflow, the full row) - r2's
// glm53_topk_hist2c, emitting the selected COLUMNS.
extern "C" __global__ void __launch_bounds__(SEL_THREADS)
glm53_topk_hist2c(const unsigned* __restrict__ keys, int nc, long long ld, int K, int fmode,
                  const unsigned* __restrict__ state,
                  const unsigned* __restrict__ lst, const unsigned* __restrict__ lcnt, const unsigned* __restrict__ lflg,
                  int cap, int* __restrict__ out, long long out_ld) {
    __shared__ unsigned pos[SEL_CAP];
    __shared__ unsigned h[1024];
    const int row = blockIdx.x, t = threadIdx.x;
    const unsigned* s = row_of(keys, row, ld, fmode);
    unsigned* st = const_cast<unsigned*>(state) + (size_t)row * 4;
    int* o = out + (size_t)row * out_ld;
    const unsigned prefix1 = st[0], fixed1 = st[1];
    unsigned need = st[2];
    const unsigned ovf = lflg[row];
    unsigned count = 0;
    if (!ovf) {
        count = min(lcnt[row], (unsigned)cap);
        for (unsigned i = t; i < count; i += SEL_THREADS) pos[i] = lst[(size_t)row * cap + i];
        __syncthreads();
    }
    // ---- pass 2: the last digit's histogram over the AT-digit-1 keys (bits 9-0) ----
    for (int i = t; i < 1024; i += SEL_THREADS) h[i] = 0u;
    __syncthreads();
    if (ovf) {
        for (int i = t; i < nc; i += SEL_THREADS) {
            const unsigned u = key_at(s, i, fmode);
            if ((u & fixed1) == prefix1) atomicAdd(&h[u & 1023u], 1u);
        }
    } else {
        for (unsigned i = t; i < count; i += SEL_THREADS) {
            const unsigned u = key_at(s, pos[i], fmode);
            if ((u & fixed1) == prefix1) atomicAdd(&h[u & 1023u], 1u);
        }
    }
    __syncthreads();
    __shared__ unsigned digit, need_at;
    if (t == 0) {
        unsigned acc = 0, tk = need;
        int d = 0;
        for (int i = 1023; i >= 0; --i) {        // the digit: keys above it < need <= keys at or above it
            const unsigned c = h[i];
            if (acc < need && acc + c >= need) {
                d = i;
                tk = need - acc;                 // the count to take AT the final prefix
            }
            acc += c;
        }
        digit = (unsigned)d;
        need_at = tk;
    }
    __syncthreads();
    const unsigned prefixf = prefix1 | digit;    // fixed now covers all 32 bits
    if (t == 0) {
        st[0] = prefixf;
        st[1] = 0xFFFFFFFFu;
        st[2] = need_at;
    }
    // ---- OVERFLOW (mass ties: the at-or-above set beyond the cap): NO collect, NO sort - the raw row
    // walked in column order IS the sorted order (r2's exact fallback; this also avoids the shared
    // pos[] overflow the first version shipped with).
    if (ovf) {
        __shared__ unsigned abv[SEL_THREADS], atr[SEL_THREADS];
        const unsigned cstride = (nc + SEL_THREADS - 1) / SEL_THREADS;
        const unsigned c0 = t * cstride;
        const unsigned c1 = min((unsigned)nc, c0 + cstride);
        unsigned mya = 0, myt = 0;
        for (unsigned i = c0; i < c1; ++i) {
            const unsigned u = key_at(s, i, fmode);
            mya += u > prefixf;
            myt += u == prefixf;
        }
        abv[t] = mya;
        atr[t] = myt;
        __syncthreads();
        if (t == 0) {
            unsigned ra = 0, rt = 0;
            for (int w = 0; w < SEL_THREADS; ++w) {
                const unsigned va = abv[w], vt = atr[w];
                abv[w] = ra;
                atr[w] = rt;
                ra += va;
                rt += vt;
            }
        }
        __syncthreads();
        unsigned a_run = abv[t];
        unsigned t_run = atr[t];
        for (unsigned i = c0; i < c1; ++i) {
            const unsigned u = key_at(s, i, fmode);
            if (u > prefixf) {
                o[a_run + (t_run > need_at ? need_at : t_run)] = (int)i;
                ++a_run;
            } else if (u == prefixf) {
                if (t_run < need_at) o[a_run + t_run] = (int)i;
                ++t_run;
            }
        }
        return;
    }
    // ---- the compact path: sort the list ascending (bitonic over the next power of two) ----
    unsigned m = count;
    unsigned pow2 = 1;
    while (pow2 < m) pow2 <<= 1;
    for (unsigned i = m + t; i < pow2; i += SEL_THREADS) pos[i] = 0xFFFFFFFFu;
    __syncthreads();
    for (unsigned k = 2; k <= pow2; k <<= 1) {
        for (unsigned j = k >> 1; j > 0; j >>= 1) {
            for (unsigned i = t; i < pow2; i += SEL_THREADS) {
                const unsigned l = i ^ j;
                if (l > i) {
                    const unsigned a = pos[i], b = pos[l];
                    const bool up = (i & k) == 0;
                    if ((a > b) == up) { pos[i] = b; pos[l] = a; }
                }
            }
            __syncthreads();
        }
    }
    // ---- emit in column order over the sorted list, skipping keys below the final prefix ----
    __shared__ unsigned abv[SEL_THREADS], atr[SEL_THREADS];
    const unsigned stride = (m + SEL_THREADS - 1) / SEL_THREADS;
    const unsigned i0 = t * stride;
    const unsigned i1 = min(m, i0 + stride);
    unsigned mya = 0, myt = 0;
    for (unsigned i = i0; i < i1; ++i) {
        const unsigned u = key_at(s, pos[i], fmode);
        mya += u > prefixf;
        myt += u == prefixf;
    }
    abv[t] = mya;
    atr[t] = myt;
    __syncthreads();
    if (t == 0) {
        unsigned ra = 0, rt = 0;
        for (int w = 0; w < SEL_THREADS; ++w) {
            const unsigned va = abv[w], vt = atr[w];
            abv[w] = ra;
            atr[w] = rt;
            ra += va;
            rt += vt;
        }
    }
    __syncthreads();
    unsigned a_run = abv[t];
    unsigned t_run = atr[t];
    for (unsigned i = i0; i < i1; ++i) {
        const unsigned u = key_at(s, pos[i], fmode);
        if (u > prefixf) {
            o[a_run + (t_run > need_at ? need_at : t_run)] = (int)pos[i];
            ++a_run;
        } else if (u == prefixf) {
            if (t_run < need_at) o[a_run + t_run] = (int)pos[i];
            ++t_run;
        }
    }
}
