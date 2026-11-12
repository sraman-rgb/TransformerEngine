// libce1p.so (overlay_ce1p, MLPERF_CE1P=1). RECONSTRUCTED on rhea 2026-11-11: the original ce1p.cu was lost
// (gitignored lib/ dir). Kernel body ported from TE branch sraman/ce-mxfp8-grad-cuda-swizzled
// (transformer_engine/common/cross_entropy/cross_entropy_mxfp8.cu, which is the upstreamed form of this kernel),
// wrapped in the ctypes ABI that overlay_ce1p/te/triton_cross_entropy.py expects:
//
//   int ce1p_bwd_mxfp8(logits bf16*, target i64*, stats f32*, n_non_ignore i64* (or 0), grad_output f32*,
//                      int64 go_stride, row_data u8*, row_scale u8*, col_data u8*, col_scale u8*,
//                      int64 n_rows, int64 n_cols, int64 ignore_idx, float label_smoothing (must be 0),
//                      float one_minus_ls (1.0), int reduce_loss, int rcp_bf16, int swizzle, cudaStream_t)
//   int ce1p_cat2(whole u8*, a u8*, b u8*, int64 n_rows, int64 bytes_a, int64 bytes_b, int split, int sm_count,
//                 cudaStream_t)   split=0: whole = cat(a, b) along the last dim; split=1: a, b = split(whole)
//
// Gradient: bf16((exp(x - m) / d - (c == y)) * go) (0 for ignored rows) with the FP32 instruction sequence of the
// Triton kernel (sub, mul log2e, ex2.approx, div.full via rcp.approx, sub 1, mul go, cvt.rn.bf16), then MXFP8 E4M3
// row-wise (1x32) and column-wise (32x1) as TE's cast: E8M0 = ceil_pow2(amax * rcp(448)), where rcp(448) is the BF16
// value 0x3b12 (rcp_bf16=1, the _ENABLE_MXFMA TE build / MLPERF_CE_MXFP8_RCP=bf16) or FP32 1/448.
// One CTA = 128 tokens x 128 vocab columns, 256 threads; scales compact or GEMM-swizzled (128x4 tiles of 512 B).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int kRows = 128, kCols = 128, kThreads = 256, kSteps = 8, kWarp = 32;

__device__ __forceinline__ float ex2_approx(float x) { float y; asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float ex2_approx_ftz(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ float div_full(float a, float b) { float y; asm("div.full.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float fsub(float a, float b) { float y; asm("sub.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float fmul(float a, float b) { float y; asm("mul.rn.f32 %0, %1, %2;" : "=f"(y) : "f"(a), "f"(b)); return y; }
__device__ __forceinline__ float rcp_approx_ftz(float x) { float y; asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

__device__ __forceinline__ uint32_t to_bf16x2(float lo, float hi) {
  uint32_t r; asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo)); return r;
}
__device__ __forceinline__ void from_bf16x2(uint32_t v, float &lo, float &hi) {
  lo = __uint_as_float(v << 16); hi = __uint_as_float(v & 0xFFFF0000u);
}
__device__ __forceinline__ uint16_t to_e4m3x2(float lo, float hi) {
  uint16_t r; asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo)); return r;
}
template <bool kRcpBf16>
__device__ __forceinline__ uint32_t scale_exponent(float amax) {
  const float k = kRcpBf16 ? 0.0022277832031250f : 1.0f / 448.0f;
  const uint32_t bits = __float_as_uint(fmul(amax, k));
  const uint32_t e = (bits >> 23) & 0xFF, mnt = bits & 0x7FFFFF;
  const bool up = (mnt > 0) && (e != 0xFE) && !((e == 0) && (mnt <= 0x400000));
  return up ? e + 1 : e;
}
__device__ __forceinline__ float scale_reciprocal(uint32_t e) {
  return __uint_as_float(e == 254 ? 0x00400000u : ((254u - e) << 23));
}
__device__ __forceinline__ int swizzled_offset(int r, int c) { return (r % 32) * 16 + ((r % 128) / 32) * 4 + (c % 4); }

struct Params {
  const __nv_bfloat16 *logits; const int64_t *target; const float *stats; const int64_t *n_non_ignore;
  const float *grad_output; int64_t go_stride;
  uint8_t *row_data, *row_scale, *col_data, *col_scale;
  int64_t n_rows, n_cols, ignore_idx; bool swizzled;
};

template <bool kReduceLoss, bool kRcpBf16>
__global__ void __launch_bounds__(kThreads, 4) ce1p_bwd_kernel(const Params p) {
  __shared__ float s_col_amax[kThreads / kWarp][kCols];
  __shared__ __align__(16) uint8_t s_rs[512];
  __shared__ __align__(16) uint8_t s_cs[512];
  __shared__ __align__(16) uint32_t s_tile[kRows * kCols / 2];
  const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
  const int64_t row_base = (int64_t)blockIdx.y * kRows, col_base = (int64_t)blockIdx.x * kCols;
  const int c0 = 8 * (lane % 16);
  const int64_t col = col_base + c0, n_cols = p.n_cols;
#pragma unroll
  for (int i = 0; i < kSteps; ++i) {
    const int rl = 16 * warp + 2 * i + lane / 16;
    const uint32_t dst = (uint32_t)__cvta_generic_to_shared(&s_tile[(rl * kCols + c0) / 2]);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(p.logits + (row_base + rl) * n_cols + col));
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;" ::: "memory");

  float nni = 1.f;
  if (kReduceLoss) nni = (float)(*p.n_non_ignore);
  float col_amax[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) col_amax[j] = 0.f;

#pragma unroll
  for (int i = 0; i < kSteps; ++i) {
    const int rl = 16 * warp + 2 * i + lane / 16;
    const int64_t r = row_base + rl;
    const int64_t y = p.target[r];
    const float m = p.stats[2 * r], d = p.stats[2 * r + 1];
    float go = p.grad_output[r * p.go_stride];
    if (kReduceLoss) go = div_full(go, nni);
    const bool ignored = (y == p.ignore_idx);
    const uint64_t toff = (uint64_t)(y - col);
    const float ds = fabsf(d) > 8.50705917302346158658e+37f ? 0.25f : (fabsf(d) < 1.175494350822287508e-38f ? 16777216.f : 1.f);
    const float d_rcp = rcp_approx_ftz(fmul(d, ds));
    const uint4 raw = *reinterpret_cast<const uint4 *>(&s_tile[(rl * kCols + c0) / 2]);
    const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
    float t[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float lo, hi; from_bf16x2(w[j], lo, hi);
      t[2 * j] = fmul(fsub(lo, m), 1.4426950408889634f);
      t[2 * j + 1] = fmul(fsub(hi, m), 1.4426950408889634f);
    }
    float t_min = t[0];
#pragma unroll
    for (int j = 1; j < 8; ++j) t_min = fminf(t_min, t[j]);
    uint32_t g[4];
    const bool fast = (ds == 1.f) && !ignored && (toff >= 8) && (t_min >= -126.f);
    if (__all_sync(0xffffffffu, fast)) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float a = fmul(fmul(d_rcp, ex2_approx_ftz(t[2 * j])), go);
        const float b = fmul(fmul(d_rcp, ex2_approx_ftz(t[2 * j + 1])), go);
        g[j] = to_bf16x2(a, b);
      }
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        float a = ex2_approx(t[2 * j]), b = ex2_approx(t[2 * j + 1]);
        if (ds != 1.f) { a = fmul(a, ds); b = fmul(b, ds); }
        a = fmul(d_rcp, a); b = fmul(d_rcp, b);
        if (toff == (uint64_t)(2 * j)) a = fsub(a, 1.f);
        if (toff == (uint64_t)(2 * j + 1)) b = fsub(b, 1.f);
        a = fmul(a, go); b = fmul(b, go);
        if (ignored) { a = 0.f; b = 0.f; }
        g[j] = to_bf16x2(a, b);
      }
    }
    float row_amax = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float lo, hi; from_bf16x2(g[j], lo, hi);
      const float a0 = fabsf(lo), a1 = fabsf(hi);
      row_amax = fmaxf(row_amax, fmaxf(a0, a1));
      col_amax[2 * j] = fmaxf(col_amax[2 * j], a0);
      col_amax[2 * j + 1] = fmaxf(col_amax[2 * j + 1], a1);
    }
    *reinterpret_cast<uint4 *>(&s_tile[(rl * kCols + c0) / 2]) = make_uint4(g[0], g[1], g[2], g[3]);
    row_amax = fmaxf(row_amax, __shfl_xor_sync(0xffffffffu, row_amax, 1));
    row_amax = fmaxf(row_amax, __shfl_xor_sync(0xffffffffu, row_amax, 2));
    const uint32_t e = scale_exponent<kRcpBf16>(row_amax);
    const float rcp = scale_reciprocal(e);
    uint16_t q[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) { float lo, hi; from_bf16x2(g[j], lo, hi); q[j] = to_e4m3x2(fmul(lo, rcp), fmul(hi, rcp)); }
    uint2 out; out.x = q[0] | ((uint32_t)q[1] << 16); out.y = q[2] | ((uint32_t)q[3] << 16);
    *reinterpret_cast<uint2 *>(p.row_data + r * n_cols + col) = out;
    if ((lane % 4) == 0) {
      const int blk = (lane % 16) / 4;
      if (p.swizzled) s_rs[swizzled_offset(rl, blk)] = (uint8_t)e;
      else p.row_scale[r * (n_cols / 32) + col_base / 32 + blk] = (uint8_t)e;
    }
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    col_amax[j] = fmaxf(col_amax[j], __shfl_xor_sync(0xffffffffu, col_amax[j], 16));
    if (lane < 16) s_col_amax[warp][c0 + j] = col_amax[j];
  }
  __syncthreads();
  {
    const int tb = warp / 2;
    float rcp[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float amax = fmaxf(s_col_amax[2 * tb][c0 + j], s_col_amax[2 * tb + 1][c0 + j]);
      const uint32_t e = scale_exponent<kRcpBf16>(amax);
      rcp[j] = scale_reciprocal(e);
      if ((warp % 2) == 0 && lane < 16) {
        if (p.swizzled) s_cs[swizzled_offset(c0 + j, tb)] = (uint8_t)e;
        else p.col_scale[(row_base / 32 + tb) * n_cols + col + j] = (uint8_t)e;
      }
    }
#pragma unroll
    for (int i = 0; i < kSteps; ++i) {
      const int rl = 16 * warp + 2 * i + lane / 16;
      const uint4 v = *reinterpret_cast<const uint4 *>(&s_tile[(rl * kCols + c0) / 2]);
      const uint32_t w[4] = {v.x, v.y, v.z, v.w};
      uint16_t q[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) { float lo, hi; from_bf16x2(w[j], lo, hi); q[j] = to_e4m3x2(fmul(lo, rcp[2 * j]), fmul(hi, rcp[2 * j + 1])); }
      uint2 out; out.x = q[0] | ((uint32_t)q[1] << 16); out.y = q[2] | ((uint32_t)q[3] << 16);
      *reinterpret_cast<uint2 *>(p.col_data + (row_base + rl) * n_cols + col) = out;
    }
  }
  if (p.swizzled) {
    __syncthreads();
    if (threadIdx.x < 32) {
      const int64_t tile = (int64_t)blockIdx.y * (n_cols / 128) + blockIdx.x;
      reinterpret_cast<uint4 *>(p.row_scale + tile * 512)[threadIdx.x] = reinterpret_cast<const uint4 *>(s_rs)[threadIdx.x];
    } else if (threadIdx.x < 64) {
      const int64_t tile = (int64_t)blockIdx.x * (p.n_rows / 128) + blockIdx.y;
      reinterpret_cast<uint4 *>(p.col_scale + tile * 512)[threadIdx.x - 32] = reinterpret_cast<const uint4 *>(s_cs)[threadIdx.x - 32];
    }
  }
}

// whole[r] = a[r] ++ b[r] (split=0) or the reverse (split=1); 16-byte vectors, grid-stride.
__global__ void cat2_kernel(uint4 *whole, uint4 *a, uint4 *b, int64_t n_rows, int64_t va, int64_t vb, int split) {
  const int64_t vw = va + vb, total = n_rows * vw;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
    const int64_t r = i / vw, c = i - r * vw;
    uint4 *side = c < va ? a + r * va + c : b + r * vb + (c - va);
    if (split) *side = whole[i]; else whole[i] = *side;
  }
}
}  // namespace

extern "C" int ce1p_bwd_mxfp8(const void *logits, const void *target, const void *stats, const void *n_non_ignore,
                              const void *grad_output, int64_t go_stride, void *row_data, void *row_scale,
                              void *col_data, void *col_scale, int64_t n_rows, int64_t n_cols, int64_t ignore_idx,
                              float label_smoothing, float one_minus_ls, int reduce_loss, int rcp_bf16, int swizzle,
                              void *stream) {
  if (label_smoothing != 0.f || one_minus_ls != 1.f) return 1;
  if (n_rows % kRows || n_cols % kCols) return 2;
  if (n_rows / kRows > 65535) return 3;
  if (reduce_loss && !n_non_ignore) return 4;
  if (((uintptr_t)logits | (uintptr_t)row_scale | (uintptr_t)col_scale) % 16 || ((uintptr_t)row_data | (uintptr_t)col_data) % 8) return 5;
  if (n_rows == 0 || n_cols == 0) return 0;
  Params p{(const __nv_bfloat16 *)logits, (const int64_t *)target, (const float *)stats, (const int64_t *)n_non_ignore,
           (const float *)grad_output, go_stride, (uint8_t *)row_data, (uint8_t *)row_scale, (uint8_t *)col_data,
           (uint8_t *)col_scale, n_rows, n_cols, ignore_idx, swizzle != 0};
  const dim3 grid((unsigned)(n_cols / kCols), (unsigned)(n_rows / kRows));
  cudaStream_t s = (cudaStream_t)stream;
  if (reduce_loss) {
    if (rcp_bf16) ce1p_bwd_kernel<true, true><<<grid, kThreads, 0, s>>>(p); else ce1p_bwd_kernel<true, false><<<grid, kThreads, 0, s>>>(p);
  } else {
    if (rcp_bf16) ce1p_bwd_kernel<false, true><<<grid, kThreads, 0, s>>>(p); else ce1p_bwd_kernel<false, false><<<grid, kThreads, 0, s>>>(p);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : 10;
}

extern "C" int ce1p_cat2(void *whole, void *a, void *b, int64_t n_rows, int64_t bytes_a, int64_t bytes_b, int split,
                         int sm_count, void *stream) {
  if (bytes_a % 16 || bytes_b % 16 || ((uintptr_t)whole | (uintptr_t)a | (uintptr_t)b) % 16) return 1;
  const int64_t total = n_rows * ((bytes_a + bytes_b) / 16);
  if (total == 0) return 0;
  int64_t blocks = (total + 255) / 256;
  const int64_t cap = (int64_t)(sm_count > 0 ? sm_count : 148) * 8;
  if (blocks > cap) blocks = cap;
  cat2_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>((uint4 *)whole, (uint4 *)a, (uint4 *)b, n_rows,
                                                                  bytes_a / 16, bytes_b / 16, split);
  return cudaGetLastError() == cudaSuccess ? 0 : 10;
}
