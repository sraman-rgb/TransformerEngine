// NVFP4 EP dispatch "direct receive": build the NVFP4 GroupedTensor the fused grouped MLP
// consumes (rowwise data + swizzled scales, columnwise RHT data + swizzled scales, per-group
// amaxes) straight from packed NVFP4 token rows, without a BF16 round trip.
//
// Packed row (row_bytes, 16 B aligned): [H/2 E2M1 pairs | H/16 E4M3 block scales | FP32 token
// decode scale]. The sender picks power-of-two token scales, so the rowwise operand reuses the
// sender's E2M1 codes unchanged: each block scale is only re-expressed against the per-group
// scale G = max token scale (an exact power-of-two shift while it stays in E4M3's normal range).
// The columnwise operand (16-point RHT along tokens) is decoded and quantized in one pass. Its
// per-group second-level scale is the bound 4 * G instead of a measured post-RHT amax: |RHT(x)|
// <= 4 max|x| and max|x| <= 6 * 448 * G, so no amax pass over the data is needed.
//
// Everything is bounded on device by the per-expert counts (CUDA-graph safe). Experts must be
// 128-token aligned (the fused grouped MLP requires 256) and H % 256 == 0.
//
// The same receive serves the NVFP4 combine backward (the grad of the expert output, FC2's
// grad_output): the sender packs with stochastic rounding (pack_tokens below), and the receiver
// quantizes the columnwise RHT operand with stochastic rounding too, as the NVFP4 recipe does for
// gradients. Random bits come from Philox4x32-10 keyed by a device seed drawn per call, so CUDA
// graph replays draw fresh bits.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

namespace {

constexpr int kTileM = 128;   // tokens per tile (8 RHT blocks of 16)
constexpr int kTileK = 256;   // hidden columns per tile (16 scale blocks of 16)
constexpr int kThreads = 256;
constexpr int kMaxGroups = 128;
constexpr float kNvfp4Max = 6.0f * 448.0f;

__device__ __forceinline__ float e4m3_to_float(uint8_t b) {
  __nv_fp8_e4m3 v;
  v.__x = b;
  return static_cast<float>(v);
}

__device__ __forceinline__ uint8_t float_to_e4m3_satfinite(float x) {
  return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ uint8_t float2_to_e2m1x2(float lo, float hi) {
  // Low nibble = lo, high nibble = hi; RN, satfinite (hardware conversion on SM100+).
  return static_cast<uint8_t>(
      __nv_cvt_float2_to_fp4x2(make_float2(lo, hi), __NV_E2M1, cudaRoundNearest));
}

__device__ __forceinline__ uint8_t half2_to_e2m1x2(__half2 v) {
  // Low nibble = v.x; RN, satfinite (cvt.rn.satfinite.e2m1x2.f16x2).
  return static_cast<uint8_t>(
      __nv_cvt_halfraw2_to_fp4x2(static_cast<__half2_raw>(v), __NV_E2M1, cudaRoundNearest));
}

// Stochastic rounding of 4 floats to E2M1 (satfinite); x[0] lands in the low nibble.
__device__ __forceinline__ uint16_t float4_to_e2m1x4_sr(float x0, float x1, float x2, float x3,
                                                        uint32_t rbits) {
  uint16_t out;
  asm volatile("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;\n"
               : "=h"(out)
               : "f"(x3), "f"(x2), "f"(x1), "f"(x0), "r"(rbits));
  return out;
}

// Philox4x32-10 (Salmon et al.), the generator curand uses.
__device__ __forceinline__ uint4 philox4x32_10(uint4 c, uint2 k) {
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
    c = make_uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    k.x += 0x9E3779B9u;
    k.y += 0xBB67AE85u;
  }
  return c;
}

__device__ __forceinline__ uint2 load_seed(const int64_t* seed) {
  const uint64_t v = static_cast<uint64_t>(*seed);
  return make_uint2(static_cast<uint32_t>(v), static_cast<uint32_t>(v >> 32));
}

template <bool kFastMath>
__device__ __forceinline__ float reciprocal(float x) {
  float r;
  if constexpr (kFastMath) {
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  } else {
    r = 1.0f / x;
  }
  return fminf(r, 3.4028234663852886e38f);
}

// Offset of scale (row r, scale col c) in the GEMM-swizzled 128x4 tile layout.
__device__ __forceinline__ int64_t swizzled_scale_offset(int64_t r, int64_t c,
                                                         int64_t num_scale_cols) {
  return ((r / 128) * ((num_scale_cols + 3) / 4) + c / 4) * 512 + (r % 32) * 16 +
         ((r / 32) % 4) * 4 + (c % 4);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

struct Groups {
  int num;
  int64_t ends[kMaxGroups];  // inclusive prefix sums of counts
};

__device__ __forceinline__ void load_groups(const int64_t* counts, int num_groups, Groups& s) {
  if (threadIdx.x == 0) {
    int64_t acc = 0;
    for (int g = 0; g < num_groups; ++g) {
      acc += counts[g];
      s.ends[g] = acc;
    }
    s.num = num_groups;
  }
  __syncthreads();
}

__device__ __forceinline__ int group_of_row(const Groups& s, int64_t r) {
  int g = 0;
  while (g < s.num - 1 && s.ends[g] <= r) ++g;
  return g;
}

// Pass 1: per-group max of the (non-negative) token decode scales, as float bits.
__global__ void group_max_kernel(const uint8_t* __restrict__ packed, const int64_t* counts,
                                 int num_groups, int64_t capacity, int64_t row_bytes,
                                 int64_t scale_offset, int* __restrict__ gmax_bits) {
  __shared__ Groups s;
  __shared__ int local_max[kMaxGroups];
  load_groups(counts, num_groups, s);
  for (int g = threadIdx.x; g < num_groups; g += blockDim.x) local_max[g] = 0;
  __syncthreads();
  const int64_t total = min(s.ends[num_groups - 1], capacity);
  for (int64_t r = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; r < total;
       r += (int64_t)gridDim.x * blockDim.x) {
    const float ts = *reinterpret_cast<const float*>(packed + r * row_bytes + scale_offset);
    atomicMax(&local_max[group_of_row(s, r)], __float_as_int(fmaxf(ts, 0.0f)));
  }
  __syncthreads();
  for (int g = threadIdx.x; g < num_groups; g += blockDim.x) {
    if (local_max[g] != 0) atomicMax(&gmax_bits[g], local_max[g]);
  }
}

__device__ __forceinline__ float group_scale(const int* gmax_bits, int g) {
  const float G = __int_as_float(gmax_bits[g]);
  return G > 0.0f ? G : 1.0f;
}

// Pass 2: rowwise copy + scale re-expression, and columnwise RHT quantization, per 128x256 tile.
// Values are handled relative to the expert scale G (x / G): the codes times re-expressed block
// scales. The columnwise math runs in half2 (two adjacent hidden columns share every E2M1 byte,
// block scale and token scale), with the RHT's 1/4 and random signs folded into the per-token
// scale. Relative values are <= 4 * 6 * 448, well inside FP16.
template <bool kFastMath, bool kStochastic>
__global__ void __launch_bounds__(kThreads)
    direct_receive_kernel(const uint8_t* __restrict__ packed, const int64_t* counts,
                          int num_groups, int64_t capacity, int H, int64_t row_bytes,
                          const int* __restrict__ gmax_bits, uint32_t sign_mask,
                          const int64_t* __restrict__ seed,
                          uint8_t* __restrict__ row_data, uint8_t* __restrict__ row_scale,
                          uint8_t* __restrict__ col_data, uint8_t* __restrict__ col_scale,
                          float* __restrict__ row_amax, float* __restrict__ col_amax) {
  __shared__ Groups s;
  // Double-buffered tile inputs, filled with cp.async one tile ahead.
  __shared__ __align__(16) uint8_t s_codes_buf[2][kTileM][kTileK / 2];
  __shared__ __align__(16) uint8_t s_bscale_buf[2][kTileM][kTileK / 16];
  __shared__ __align__(16) float s_tscale_buf[2][kTileM];
  __shared__ __align__(16) __half2 s_hscale[kTileM][kTileK / 16];  // b * ts / G * sign / 4, x2
  load_groups(counts, num_groups, s);
  uint2 key = make_uint2(0, 0);
  if constexpr (kStochastic) key = load_seed(seed);
  if (blockIdx.x == 0) {
    for (int g = threadIdx.x; g < num_groups; g += blockDim.x) {
      const float G = group_scale(gmax_bits, g);
      row_amax[g] = kNvfp4Max * G;
      col_amax[g] = 4.0f * kNvfp4Max * G;
    }
  }

  const int64_t total = min(s.ends[num_groups - 1], capacity);
  const int col_tiles = H / kTileK;
  const int64_t num_tiles = (total / kTileM) * col_tiles;
  const int64_t num_scale_cols_row = H / 16;
  const int64_t tscale_offset = H / 2 + H / 16;

  auto issue_loads = [&](int64_t tile, int buf) {
    const int64_t lr0 = (tile / col_tiles) * kTileM;
    const int lc0 = static_cast<int>(tile % col_tiles) * kTileK;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = threadIdx.x + i * kThreads;  // 0..1023
      const int rr = idx >> 3;                     // 8 x 16 B per row
      const int cb = (idx & 7) * 16;
      cp_async16(&s_codes_buf[buf][rr][cb], packed + (lr0 + rr) * row_bytes + lc0 / 2 + cb);
    }
    if (threadIdx.x < kTileM) {
      const int rr = threadIdx.x;
      cp_async16(&s_bscale_buf[buf][rr][0], packed + (lr0 + rr) * row_bytes + H / 2 + lc0 / 16);
    } else {
      const int rr = threadIdx.x - kTileM;
      cp_async4(&s_tscale_buf[buf][rr], packed + (lr0 + rr) * row_bytes + tscale_offset);
    }
    cp_async_commit();
  };

  int buf = 0;
  if (blockIdx.x < num_tiles) issue_loads(blockIdx.x, 0);
  for (int64_t t = blockIdx.x; t < num_tiles; t += gridDim.x, buf ^= 1) {
    if (t + gridDim.x < num_tiles) {
      issue_loads(t + gridDim.x, buf ^ 1);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    auto& s_codes = s_codes_buf[buf];
    auto& s_bscale = s_bscale_buf[buf];
    auto& s_tscale = s_tscale_buf[buf];
    const int64_t r0 = (t / col_tiles) * kTileM;
    const int c0 = static_cast<int>(t % col_tiles) * kTileK;
    const int g = group_of_row(s, r0);
    const int64_t g_end = s.ends[g];
    const int64_t g_rows = g == 0 ? g_end : g_end - s.ends[g - 1];
    const int64_t g_off = g_end - g_rows;
    const float inv_G = 1.0f / group_scale(gmax_bits, g);  // exact: G is a power of two

    // The rowwise data is the sender's codes, unchanged.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int idx = threadIdx.x + i * kThreads;
      const int rr = idx >> 3;
      const int cb = (idx & 7) * 16;
      *reinterpret_cast<uint4*>(row_data + (r0 + rr) * (H / 2) + c0 / 2 + cb) =
          *reinterpret_cast<const uint4*>(&s_codes[rr][cb]);
    }

    // Re-expressed block scales: rowwise E4M3 (swizzled, one 4-byte word per item) and the
    // FP16 columnwise factors.
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = threadIdx.x + i * kThreads;  // 0..511: 128 rows x 4 words
      const int rr = idx & (kTileM - 1);
      const int m = idx >> 7;
      const float ratio = s_tscale[rr] * inv_G;
      const float col_factor = (((sign_mask >> (rr & 15)) & 1) ? -0.25f : 0.25f);
      uint32_t word = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float b = e4m3_to_float(s_bscale[rr][m * 4 + j]) * ratio;
        word |= static_cast<uint32_t>(float_to_e4m3_satfinite(b)) << (8 * j);
        s_hscale[rr][m * 4 + j] = __float2half2_rn(b * col_factor);
      }
      const int64_t off = swizzled_scale_offset(r0 + rr, c0 / 16 + m * 4, num_scale_cols_row);
      *reinterpret_cast<uint32_t*>(row_scale + off) = word;
    }
    __syncthreads();

    // Columnwise: thread = 2 adjacent hidden columns x 4 RHT blocks of 16 tokens.
    {
      const int cp = threadIdx.x & 127;
      const int th = threadIdx.x >> 7;
      const int sc = cp >> 3;
      uint32_t out_x[8], out_y[8];
      uint8_t scl_x[4], scl_y[4];
#pragma unroll
      for (int b4 = 0; b4 < 4; ++b4) {
        const int rbase = (th * 4 + b4) * 16;
        __half2 v[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const int rr = rbase + i;
          const __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(
              static_cast<__nv_fp4x2_storage_t>(s_codes[rr][cp]), __NV_E2M1);
          v[i] = __hmul2(__half2(raw), s_hscale[rr][sc]);
        }
#pragma unroll
        for (int h = 1; h < 16; h <<= 1) {
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            if ((i & h) == 0) {
              const __half2 a = v[i], b = v[i + h];
              v[i] = __hadd2(a, b);
              v[i + h] = __hsub2(a, b);
            }
          }
        }
        __half2 m2 = __habs2(v[0]);
#pragma unroll
        for (int i = 1; i < 16; ++i) m2 = __hmax2(m2, __habs2(v[i]));
        const float2 bmax = __half22float2(m2);
        // Relative to G: sb = bmax_rel / 24, q = y_rel / (4 sb).
        const uint8_t sbx = float_to_e4m3_satfinite(fminf(bmax.x * (1.0f / 24.0f), 448.0f));
        const uint8_t sby = float_to_e4m3_satfinite(fminf(bmax.y * (1.0f / 24.0f), 448.0f));
        scl_x[b4] = sbx;
        scl_y[b4] = sby;
        const __half2 rx = __float2half2_rn(reciprocal<kFastMath>(4.0f * e4m3_to_float(sbx)));
        const __half2 ry = __float2half2_rn(reciprocal<kFastMath>(4.0f * e4m3_to_float(sby)));
        if constexpr (kStochastic) {
          // 4 random words per 8 tokens of both columns: one Philox call per (block, half).
          const __half2 r2 = __halves2half2(__low2half(rx), __low2half(ry));
#pragma unroll
          for (int w = 0; w < 2; ++w) {
            const uint4 rb = philox4x32_10(
                make_uint4(threadIdx.x | (static_cast<uint32_t>(b4 * 2 + w) << 16),
                           static_cast<uint32_t>(t), static_cast<uint32_t>(t >> 32), 0x5EEDu),
                key);
            float fx[8], fy[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              const float2 f = __half22float2(__hmul2(v[w * 8 + j], r2));
              fx[j] = f.x;
              fy[j] = f.y;
            }
            out_x[b4 * 2 + w] =
                static_cast<uint32_t>(float4_to_e2m1x4_sr(fx[0], fx[1], fx[2], fx[3], rb.x)) |
                (static_cast<uint32_t>(float4_to_e2m1x4_sr(fx[4], fx[5], fx[6], fx[7], rb.y))
                 << 16);
            out_y[b4 * 2 + w] =
                static_cast<uint32_t>(float4_to_e2m1x4_sr(fy[0], fy[1], fy[2], fy[3], rb.z)) |
                (static_cast<uint32_t>(float4_to_e2m1x4_sr(fy[4], fy[5], fy[6], fy[7], rb.w))
                 << 16);
          }
        } else {
#pragma unroll
          for (int w = 0; w < 2; ++w) {
            uint32_t wx = 0, wy = 0;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
              const int i = w * 8 + p * 2;
              // Regroup (col c, col c+1) pairs into (token i, token i+1) pairs per column.
              const __half2 qx = __hmul2(__lows2half2(v[i], v[i + 1]), rx);
              const __half2 qy = __hmul2(__highs2half2(v[i], v[i + 1]), ry);
              wx |= static_cast<uint32_t>(half2_to_e2m1x2(qx)) << (8 * p);
              wy |= static_cast<uint32_t>(half2_to_e2m1x2(qy)) << (8 * p);
            }
            out_x[b4 * 2 + w] = wx;
            out_y[b4 * 2 + w] = wy;
          }
        }
      }
      const int64_t local = r0 - g_off;
      const int64_t k = c0 + 2 * cp;
      uint8_t* dst = col_data + g_off * (H / 2) + k * (g_rows / 2) + local / 2 + th * 32;
      *reinterpret_cast<uint4*>(dst) = make_uint4(out_x[0], out_x[1], out_x[2], out_x[3]);
      *reinterpret_cast<uint4*>(dst + 16) = make_uint4(out_x[4], out_x[5], out_x[6], out_x[7]);
      dst += g_rows / 2;
      *reinterpret_cast<uint4*>(dst) = make_uint4(out_y[0], out_y[1], out_y[2], out_y[3]);
      *reinterpret_cast<uint4*>(dst + 16) = make_uint4(out_y[4], out_y[5], out_y[6], out_y[7]);
      const int64_t blk = local / 16 + th * 4;
      const int64_t ncols = g_rows / 16;
      const uint32_t sx = scl_x[0] | (scl_x[1] << 8) | (scl_x[2] << 16) |
                          (static_cast<uint32_t>(scl_x[3]) << 24);
      const uint32_t sy = scl_y[0] | (scl_y[1] << 8) | (scl_y[2] << 16) |
                          (static_cast<uint32_t>(scl_y[3]) << 24);
      *reinterpret_cast<uint32_t*>(col_scale + g_off * (H / 16) +
                                   swizzled_scale_offset(k, blk, ncols)) = sx;
      *reinterpret_cast<uint32_t*>(col_scale + g_off * (H / 16) +
                                   swizzled_scale_offset(k + 1, blk, ncols)) = sy;
    }
    __syncthreads();
  }
}


// Sender pack: one block per token row, one thread per 16-element scale block (blockDim = H/16).
// Same arithmetic as the Triton pack in ep.py: power-of-two token decode scale, E4M3 block scales
// from the block amax, E2M1 codes (RN, or stochastic rounding with Philox bits).
template <bool kFastMath, bool kStochastic>
__global__ void pack_kernel(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ out,
                            int H, int64_t row_bytes, const int64_t* __restrict__ seed) {
  __shared__ float s_warp_max[32];
  const int64_t row = blockIdx.x;
  const int b = threadIdx.x;
  const __nv_bfloat16* xr = x + row * H + b * 16;
  uint8_t* orow = out + row * row_bytes;
  float v[16];
  {
    const uint4 a = *reinterpret_cast<const uint4*>(xr);
    const uint4 c = *reinterpret_cast<const uint4*>(xr + 8);
    const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* pc = reinterpret_cast<const __nv_bfloat162*>(&c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 fa = __bfloat1622float2(pa[i]);
      const float2 fc = __bfloat1622float2(pc[i]);
      v[2 * i] = fa.x;
      v[2 * i + 1] = fa.y;
      v[8 + 2 * i] = fc.x;
      v[8 + 2 * i + 1] = fc.y;
    }
  }
  float bmax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) bmax = fmaxf(bmax, fabsf(v[i]));
  float m = bmax;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) s_warp_max[warp] = m;
  __syncthreads();
  if (warp == 0) {
    const int nw = (blockDim.x + 31) >> 5;
    float mm = lane < nw ? s_warp_max[lane] : 0.0f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, o));
    if (lane == 0) s_warp_max[0] = mm;
  }
  __syncthreads();
  const float row_amax = s_warp_max[0];
  // Power-of-two token scale: the smallest 2^e >= amax / (6 * 448).
  const uint32_t ratio_bits = __float_as_uint(__fdiv_rn(row_amax, 448.0f * 6.0f));
  const uint32_t exp_bits = ratio_bits & 0x7F800000u;
  uint32_t dec_bits = (ratio_bits & 0x7FFFFFu) == 0 ? exp_bits : exp_bits + 0x800000u;
  dec_bits = max(dec_bits, 0x800000u);
  const float global_decode = row_amax == 0.0f ? 0.0f : __uint_as_float(dec_bits);
  const float global_encode = row_amax == 0.0f ? 1.0f : __fdiv_rn(1.0f, global_decode);
  const float bs_f = fminf(bmax * (global_encode * (1.0f / 6.0f)), 448.0f);
  const uint8_t bs = float_to_e4m3_satfinite(bs_f);
  const float rcp = reciprocal<kFastMath>(e4m3_to_float(bs) * global_decode);
  uint32_t lo, hi;
  if constexpr (kStochastic) {
    const uint4 rb = philox4x32_10(
        make_uint4(static_cast<uint32_t>(b), static_cast<uint32_t>(row),
                   static_cast<uint32_t>(row >> 32), 0xC0B1Du),
        load_seed(seed));
    lo = static_cast<uint32_t>(float4_to_e2m1x4_sr(v[0] * rcp, v[1] * rcp, v[2] * rcp, v[3] * rcp, rb.x)) |
         (static_cast<uint32_t>(float4_to_e2m1x4_sr(v[4] * rcp, v[5] * rcp, v[6] * rcp, v[7] * rcp, rb.y)) << 16);
    hi = static_cast<uint32_t>(float4_to_e2m1x4_sr(v[8] * rcp, v[9] * rcp, v[10] * rcp, v[11] * rcp, rb.z)) |
         (static_cast<uint32_t>(float4_to_e2m1x4_sr(v[12] * rcp, v[13] * rcp, v[14] * rcp, v[15] * rcp, rb.w)) << 16);
  } else {
    lo = 0;
    hi = 0;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      lo |= static_cast<uint32_t>(float2_to_e2m1x2(v[2 * p] * rcp, v[2 * p + 1] * rcp)) << (8 * p);
      hi |= static_cast<uint32_t>(float2_to_e2m1x2(v[8 + 2 * p] * rcp, v[8 + 2 * p + 1] * rcp))
            << (8 * p);
    }
  }
  *reinterpret_cast<uint2*>(orow + b * 8) = make_uint2(lo, hi);
  orow[H / 2 + b] = bs;
  const int64_t tail = H / 2 + H / 16;
  if (b == 0) *reinterpret_cast<float*>(orow + tail) = global_decode;
  for (int64_t i = tail + 4 + b; i < row_bytes; i += blockDim.x) orow[i] = 0;
}

int num_programs(const at::Tensor& t) {
  return 2 * at::cuda::getDeviceProperties(t.get_device())->multiProcessorCount;
}

}  // namespace

// Fills row/col data + scales + amaxes. gmax (int32 [G]) is scratch, zeroed here.
void direct_receive(at::Tensor packed, at::Tensor counts, int64_t hidden, at::Tensor gmax,
                    int64_t sign_mask, at::Tensor row_data, at::Tensor row_scale,
                    at::Tensor col_data, at::Tensor col_scale, at::Tensor row_amax,
                    at::Tensor col_amax, bool fast_math, c10::optional<at::Tensor> seed) {
  const bool stochastic = seed.has_value();
  if (stochastic) TORCH_CHECK(seed->is_cuda() && seed->scalar_type() == at::kLong);
  TORCH_CHECK(packed.is_cuda() && packed.scalar_type() == at::kByte && packed.dim() == 2);
  TORCH_CHECK(counts.scalar_type() == at::kLong && counts.is_contiguous());
  const int num_groups = static_cast<int>(counts.numel());
  TORCH_CHECK(num_groups >= 1 && num_groups <= kMaxGroups, "1..", kMaxGroups, " groups");
  TORCH_CHECK(hidden % kTileK == 0, "hidden must be a multiple of ", kTileK);
  TORCH_CHECK(packed.size(0) % kTileM == 0, "capacity must be a multiple of ", kTileM);
  const int64_t row_bytes = packed.size(1);
  TORCH_CHECK(row_bytes % 16 == 0 && row_bytes >= hidden / 2 + hidden / 16 + 4);
  TORCH_CHECK(packed.is_contiguous() && row_data.is_contiguous() && col_data.is_contiguous());
  const c10::cuda::CUDAGuard guard(packed.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  gmax.zero_();
  const int nprog = num_programs(packed);
  group_max_kernel<<<nprog, 256, 0, stream>>>(packed.data_ptr<uint8_t>(), counts.data_ptr<int64_t>(),
                                              num_groups, packed.size(0), row_bytes,
                                              hidden / 2 + hidden / 16, gmax.data_ptr<int>());
  auto launch = [&](auto kernel) {
    int per_sm = 1;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, 0));
    const int grid =
        std::max(1, per_sm) * at::cuda::getDeviceProperties(packed.get_device())->multiProcessorCount;
    kernel<<<grid, kThreads, 0, stream>>>(
        packed.data_ptr<uint8_t>(), counts.data_ptr<int64_t>(), num_groups, packed.size(0),
        static_cast<int>(hidden), row_bytes, gmax.data_ptr<int>(),
        static_cast<uint32_t>(sign_mask),
        stochastic ? seed->data_ptr<int64_t>() : nullptr, row_data.data_ptr<uint8_t>(),
        row_scale.data_ptr<uint8_t>(), col_data.data_ptr<uint8_t>(),
        col_scale.data_ptr<uint8_t>(), row_amax.data_ptr<float>(), col_amax.data_ptr<float>());
  };
  if (fast_math) {
    stochastic ? launch(direct_receive_kernel<true, true>) : launch(direct_receive_kernel<true, false>);
  } else {
    stochastic ? launch(direct_receive_kernel<false, true>)
               : launch(direct_receive_kernel<false, false>);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Packs BF16 rows [T, H] into NVFP4 rows [T, row_bytes] with power-of-two token scales; stochastic
// rounding when ``seed`` (int64 [1], device) is given.
void pack_tokens(at::Tensor x, at::Tensor out, bool fast_math, c10::optional<at::Tensor> seed) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous());
  TORCH_CHECK(out.scalar_type() == at::kByte && out.dim() == 2 && out.is_contiguous());
  const int64_t T = x.size(0);
  const int H = static_cast<int>(x.size(1));
  TORCH_CHECK(H % 16 == 0 && H / 16 <= 1024 && (H / 16) % 32 == 0, "hidden must be 512*k <= 16384");
  TORCH_CHECK(out.size(0) == T && out.size(1) >= H / 2 + H / 16 + 4);
  const bool stochastic = seed.has_value();
  if (stochastic) TORCH_CHECK(seed->is_cuda() && seed->scalar_type() == at::kLong);
  if (T == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto kernel) {
    kernel<<<static_cast<unsigned>(T), H / 16, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), out.data_ptr<uint8_t>(),
        H, out.size(1), stochastic ? seed->data_ptr<int64_t>() : nullptr);
  };
  if (fast_math) {
    stochastic ? launch(pack_kernel<true, true>) : launch(pack_kernel<true, false>);
  } else {
    stochastic ? launch(pack_kernel<false, true>) : launch(pack_kernel<false, false>);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("direct_receive", &direct_receive, "NVFP4 EP direct receive", py::arg("packed"),
        py::arg("counts"), py::arg("hidden"), py::arg("gmax"), py::arg("sign_mask"),
        py::arg("row_data"), py::arg("row_scale"), py::arg("col_data"), py::arg("col_scale"),
        py::arg("row_amax"), py::arg("col_amax"), py::arg("fast_math"),
        py::arg("seed") = py::none());
  m.def("pack_tokens", &pack_tokens, "NVFP4 EP pack (power-of-two token scales)", py::arg("x"),
        py::arg("out"), py::arg("fast_math"), py::arg("seed") = py::none());
}
