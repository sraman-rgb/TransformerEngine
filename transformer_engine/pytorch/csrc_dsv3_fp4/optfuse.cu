// overlay_optfuse kernels (liboptfuse.so), ctypes ABI used by overlay_optfuse/te/utils.py
// (_optfuse_build / _optfuse_nvfp4_2d) and overlay_optfuse/mcore/distrib_optimizer.py
// (_optfuse_bf16_to_f32).
//
// RECONSTRUCTED on rhea (2026-11-11): the original optfuse.cu was lost (gitignored lib/). The
// NVFP4 kernels follow the upstream port of the same kernels, TE branch
// sraman-rgb/TransformerEngine:sraman/nvfp4-partial-cast-fused
// (transformer_engine/common/recipe/nvfp4_multi_tensor.cu, 9a8bb10), with the per-tensor
// metadata read from the device meta array that utils.py packs (19 int64 per tensor):
//   0 master (fp32 shard)   1 length   2 start   3 h   4 w   5 tile_rows   6 tile_cols
//   7 first tile row        8 tile_base (prefix of touched tiles)
//   9 block amax [th*tw]    10 per-block decode scale [th*tw]   11 global amax [1]
//   12 global encode scale [1]   13 out (rowwise data + start/2)   14 target scale_inv (uint8)
//   15 target amax [1]      16 rows_padded   17 scale_base (prefix of 16-row groups x tw)
//   18 flags: 1 vector loads, 2 8-byte stores, 4 round master to bf16 (model dtype bf16)
// Arithmetic is that of TE's nvfp4_multi_tensor_{compute_partial_amax, fused_scale,
// 2d_partial_cast} + nvfp4_compute_global_scale (E4M3 scales), so the bytes are identical.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kTile = 16;
constexpr int kThreads = 256;
constexpr int kNF = 19;
constexpr float kFp4Max = 6.0f;
constexpr float kScaleMax = 448.0f;  // E4M3
constexpr float kFltMax = 3.402823466e+38f;
constexpr float kTiny = 1.17549435e-38f;

struct Meta {
  const float *input;
  int64_t len, start, h, w, th, tw, tr0, tile_base;
  float *block_amax;
  float *decode_scale;
  float *gamax;
  float *gscale;
  uint8_t *out;
  uint8_t *target_scale;
  float *target_amax;
  int64_t rows_padded, scale_base, flags;
};

__device__ __forceinline__ Meta load_meta(const int64_t *meta, int i) {
  const int64_t *p = meta + (int64_t)i * kNF;
  Meta m;
  m.input = reinterpret_cast<const float *>(p[0]);
  m.len = p[1];
  m.start = p[2];
  m.h = p[3];
  m.w = p[4];
  m.th = p[5];
  m.tw = p[6];
  m.tr0 = p[7];
  m.tile_base = p[8];
  m.block_amax = reinterpret_cast<float *>(p[9]);
  m.decode_scale = reinterpret_cast<float *>(p[10]);
  m.gamax = reinterpret_cast<float *>(p[11]);
  m.gscale = reinterpret_cast<float *>(p[12]);
  m.out = reinterpret_cast<uint8_t *>(p[13]);
  m.target_scale = reinterpret_cast<uint8_t *>(p[14]);
  m.target_amax = reinterpret_cast<float *>(p[15]);
  m.rows_padded = p[16];
  m.scale_base = p[17];
  m.flags = p[18];
  return m;
}

// Largest i with meta[i].field <= t (field = 8 tile_base or 17 scale_base).
__device__ __forceinline__ int find_tensor(const int64_t *meta, int n, int field, int64_t t) {
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (meta[(int64_t)mid * kNF + field] <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

__device__ __forceinline__ float rnd(float x, bool bf) {
  return bf ? __bfloat162float(__float2bfloat16_rn(x)) : x;
}

__device__ __forceinline__ void load4(const float *p, float (&v)[4]) {
  const float4 t = __ldg(reinterpret_cast<const float4 *>(p));
  v[0] = t.x; v[1] = t.y; v[2] = t.z; v[3] = t.w;
}

__device__ __forceinline__ bool tile_in_shard(const Meta &m, int64_t r0, int64_t c0) {
  return (r0 + kTile <= m.h) && (c0 + kTile <= m.w) && (r0 * m.w + c0 >= m.start) &&
         ((r0 + kTile - 1) * m.w + c0 + kTile - 1 < m.start + m.len);
}

__global__ void __launch_bounds__(kThreads)
    amax_kernel(const int64_t *__restrict__ meta, int n, int64_t total) {
  const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= total) return;
  const int ti = find_tensor(meta, n, 8, t);
  const Meta m = load_meta(meta, ti);
  const int64_t local = t - m.tile_base;
  const int64_t r0 = (m.tr0 + local / m.tw) * kTile;
  const int64_t c0 = (local % m.tw) * kTile;
  const int64_t end = m.start + m.len;
  const float *in = m.input - m.start;
  const bool bf = (m.flags & 4) != 0;
  float amax = 0.f;
  if ((m.flags & 1) && tile_in_shard(m, r0, c0)) {
#pragma unroll 4
    for (int r = 0; r < kTile; ++r) {
      const float *row = in + (r0 + r) * m.w + c0;
#pragma unroll
      for (int k = 0; k < kTile / 4; ++k) {
        float v[4];
        load4(row + 4 * k, v);
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
      }
    }
  } else {
    bool any = false;
    for (int r = 0; r < kTile && r0 + r < m.h; ++r) {
      for (int c = 0; c < kTile && c0 + c < m.w; ++c) {
        const int64_t idx = (r0 + r) * m.w + c0 + c;
        if (idx >= m.start && idx < end) {
          amax = fmaxf(amax, fabsf(in[idx]));
          any = true;
        }
      }
    }
    if (!any) return;  // other ranks' tiles stay 0 (caller zeroes the buffer)
  }
  // bf16(max |x|) == max |bf16(x)| (RNE is monotonic)
  m.block_amax[(r0 / kTile) * m.tw + c0 / kTile] = rnd(amax, bf);
}

// One block per tensor: global amax = max of the all-reduced block amaxes; global encode scale.
__global__ void __launch_bounds__(kThreads) gscale_kernel(const int64_t *__restrict__ meta) {
  const Meta m = load_meta(meta, blockIdx.x);
  const int64_t nb = m.th * m.tw;
  float a = 0.f;
  for (int64_t i = threadIdx.x; i < nb; i += blockDim.x) a = fmaxf(a, m.block_amax[i]);
  for (int d = 16; d > 0; d >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, d));
  __shared__ float smem[kThreads / 32];
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = a;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 0; i < kThreads / 32; ++i) a = fmaxf(a, smem[i]);
    *m.gamax = a;
    const float safe = fmaxf(a, kTiny);
    *m.gscale = (a > 0.0f) ? fminf((kScaleMax * kFp4Max) / safe, kFltMax) : 1.0f;
  }
}

__device__ __forceinline__ uint8_t float_to_e4m3_bits(float x) {
  // static_cast<fp8e4m3>(float): __nv_fp8_e4m3 conversion = cvt.rn.satfinite.e4m3x2.f32
  __nv_fp8_e4m3 q(x);
  return reinterpret_cast<const uint8_t &>(q);
}

__device__ __forceinline__ float e4m3_roundtrip(float x) {
  __nv_fp8_e4m3 q(x);
  return static_cast<float>(q);
}

__global__ void __launch_bounds__(kThreads)
    scale_kernel(const int64_t *__restrict__ meta, int n, int64_t total) {
  const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= total) return;
  const int ti = find_tensor(meta, n, 17, t);
  const Meta m = load_meta(meta, ti);
  const int64_t local = t - m.scale_base;
  const int64_t tile_row = local / m.tw;
  const int64_t col = local % m.tw;
  const float g_amax = *m.gamax;
  if (local == 0) *m.target_amax = g_amax;
  float scale_val = 0.0f;
  if (tile_row < m.th) {
    const float safe_global_amax = fmaxf(g_amax, kTiny);
    const float global_scale =
        (g_amax > 0.0f) ? fminf((kScaleMax * kFp4Max) / safe_global_amax, kFltMax) : 1.0f;
    constexpr float fp4_max_inv = 1.0f / kFp4Max;
    const float mult = global_scale * fp4_max_inv;
    const float amax_val = m.block_amax[tile_row * m.tw + col];
    scale_val = fminf(amax_val * mult, kFltMax);
    m.decode_scale[tile_row * m.tw + col] = scale_val;
  }
  const uint8_t q = float_to_e4m3_bits(scale_val);
  const int64_t row_end = min((tile_row + 1) * kTile, m.rows_padded);
  for (int64_t r = tile_row * kTile; r < row_end; ++r) m.target_scale[r * m.tw + col] = q;
}

// = TE ptx::mul_cvt_fp32_to_fp4_4x_with_rn (sm_10x)
__device__ __forceinline__ uint16_t cvt_4x(const float (&v)[4], const float s) {
  uint32_t out_4x = 0;
  const float2 in01 = make_float2(v[0], v[1]);
  const float2 in23 = make_float2(v[2], v[3]);
  const float2 scale = make_float2(s, s);
  asm volatile(
      "{\n"
      ".reg.b64 v01; \n\t"
      ".reg.b64 v23; \n\t"
      ".reg.b32 v0; \n\t"
      ".reg.b32 v1; \n\t"
      ".reg.b32 v2; \n\t"
      ".reg.b32 v3; \n\t"
      ".reg.b8 f0; \n\t"
      ".reg.b8 f1; \n\t"
      "mov.b64 {v0, v1} , %1; \n\t"
      "mov.b64 {v2, v3} , %2; \n\t"
      "mov.b64 v01, {v0, v1}; \n\t"
      "mov.b64 v23, {v2, v3}; \n\t"
      "mul.f32x2 v01, v01, %3; \n\t"
      "mul.f32x2 v23, v23, %3; \n\t"
      "mov.b64 {v1, v0}, v01; \n\t"
      "mov.b64 {v3, v2}, v23; \n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
      "mov.b32 %0, {f0, f1, f0, f1};\n\t"
      "}"
      : "=r"(out_4x)
      : "l"(reinterpret_cast<const uint64_t &>(in01)),
        "l"(reinterpret_cast<const uint64_t &>(in23)),
        "l"(reinterpret_cast<const uint64_t &>(scale)));
  return static_cast<uint16_t>(out_4x & 0xFFFFu);
}

__device__ __forceinline__ void store_nibble(uint8_t *byte, int64_t elem_idx, uint8_t nib) {
  *byte = ((elem_idx & 1) == 0) ? static_cast<uint8_t>((*byte & 0xF0u) | nib)
                                : static_cast<uint8_t>((*byte & 0x0Fu) | (nib << 4));
}

__global__ void __launch_bounds__(kThreads)
    cast_kernel(const int64_t *__restrict__ meta, int n, int64_t total) {
  const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= total) return;
  const int ti = find_tensor(meta, n, 8, t);
  const Meta m = load_meta(meta, ti);
  const int64_t local = t - m.tile_base;
  const int64_t r0 = (m.tr0 + local / m.tw) * kTile;
  const int64_t c0 = (local % m.tw) * kTile;
  const int64_t end = m.start + m.len;
  const float *in = m.input - m.start;
  const bool bf = (m.flags & 4) != 0;

  float global_encode_scale = *m.gscale;
  if (global_encode_scale <= 0.f) global_encode_scale = 1.f;
  const float global_decode_scale = 1.0f / global_encode_scale;
  float tile_decode_scale = m.decode_scale[(r0 / kTile) * m.tw + c0 / kTile];
  tile_decode_scale = e4m3_roundtrip(tile_decode_scale);
  float s = (tile_decode_scale > 0.f) ? 1.0f / (tile_decode_scale * global_decode_scale) : kFltMax;
  s = fminf(s, kFltMax);

  if ((m.flags & 1) && tile_in_shard(m, r0, c0)) {
#pragma unroll 2
    for (int r = 0; r < kTile; ++r) {
      const int64_t idx0 = (r0 + r) * m.w + c0;
      uint64_t bits = 0;
#pragma unroll
      for (int k = 0; k < kTile / 4; ++k) {
        float v[4];
        load4(in + idx0 + 4 * k, v);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = rnd(v[j], bf);
        bits |= static_cast<uint64_t>(cvt_4x(v, s)) << (16 * k);
      }
      uint8_t *out = m.out + ((idx0 - m.start) >> 1);
      if (m.flags & 2) {
        *reinterpret_cast<uint2 *>(out) =
            make_uint2(static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32));
      } else {
#pragma unroll
        for (int b = 0; b < 8; ++b) out[b] = static_cast<uint8_t>(bits >> (8 * b));
      }
    }
    return;
  }
  for (int r = 0; r < kTile && r0 + r < m.h; ++r) {
    for (int g = 0; g < kTile / 4; ++g) {
      float v[4];
      bool mask[4];
      int64_t eidx[4];
      bool any = false;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int64_t col = c0 + 4 * g + j;
        const int64_t idx = (r0 + r) * m.w + col;
        eidx[j] = idx;
        mask[j] = col < m.w && idx >= m.start && idx < end;
        v[j] = mask[j] ? rnd(in[idx], bf) : 0.0f;
        any |= mask[j];
      }
      if (!any) continue;
      const uint16_t pb = cvt_4x(v, s);
      for (int pair = 0; pair < 2; ++pair) {
        const int f = 2 * pair, sc = f + 1;
        if (!mask[f] && !mask[sc]) continue;
        const int64_t ref = mask[f] ? eidx[f] : eidx[sc];
        uint8_t byte = m.out[(ref - m.start) >> 1];
        if (mask[f]) store_nibble(&byte, eidx[f], static_cast<uint8_t>((pb >> (4 * f)) & 0xF));
        if (mask[sc]) store_nibble(&byte, eidx[sc], static_cast<uint8_t>((pb >> (4 * sc)) & 0xF));
        m.out[(ref - m.start) >> 1] = byte;
      }
    }
  }
}

// bf16 -> fp32, 8 elements (16 B in, 32 B out) per item, grid-stride.
__global__ void __launch_bounds__(kThreads)
    bf16_to_f32_kernel(const uint4 *__restrict__ src, float4 *__restrict__ dst, int64_t n8) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n8;
       i += (int64_t)gridDim.x * blockDim.x) {
    const uint4 v = __ldg(src + i);
    float4 a, b;
    a.x = __uint_as_float(v.x << 16); a.y = __uint_as_float(v.x & 0xFFFF0000u);
    a.z = __uint_as_float(v.y << 16); a.w = __uint_as_float(v.y & 0xFFFF0000u);
    b.x = __uint_as_float(v.z << 16); b.y = __uint_as_float(v.z & 0xFFFF0000u);
    b.z = __uint_as_float(v.w << 16); b.w = __uint_as_float(v.w & 0xFFFF0000u);
    dst[2 * i] = a;
    dst[2 * i + 1] = b;
  }
}

inline unsigned grid_for(int64_t total) { return (unsigned)((total + kThreads - 1) / kThreads); }

}  // namespace

extern "C" {

int optfuse_meta_size() { return 8 * kNF; }

int optfuse_amax(const void *meta, int n, int64_t total_tiles, void *stream) {
  if (n <= 0 || total_tiles <= 0) return 0;
  amax_kernel<<<grid_for(total_tiles), kThreads, 0, (cudaStream_t)stream>>>(
      (const int64_t *)meta, n, total_tiles);
  return (int)cudaGetLastError();
}

int optfuse_gscale(const void *meta, int n, void *stream) {
  if (n <= 0) return 0;
  gscale_kernel<<<n, kThreads, 0, (cudaStream_t)stream>>>((const int64_t *)meta);
  return (int)cudaGetLastError();
}

int optfuse_scale(const void *meta, int n, int64_t total_scale, void *stream) {
  if (n <= 0 || total_scale <= 0) return 0;
  scale_kernel<<<grid_for(total_scale), kThreads, 0, (cudaStream_t)stream>>>(
      (const int64_t *)meta, n, total_scale);
  return (int)cudaGetLastError();
}

int optfuse_cast(const void *meta, int n, int64_t total_tiles, void *stream) {
  if (n <= 0 || total_tiles <= 0) return 0;
  cast_kernel<<<grid_for(total_tiles), kThreads, 0, (cudaStream_t)stream>>>(
      (const int64_t *)meta, n, total_tiles);
  return (int)cudaGetLastError();
}

int optfuse_bf16_to_f32(const void *src, void *dst, int64_t n8, int sms, void *stream) {
  if (n8 <= 0) return 0;
  if (((uintptr_t)src & 15) || ((uintptr_t)dst & 15)) return -1;
  int64_t blocks = (n8 + kThreads - 1) / kThreads;
  const int64_t cap = (int64_t)(sms > 0 ? sms : 1) * 8;
  if (blocks > cap) blocks = cap;
  bf16_to_f32_kernel<<<(unsigned)blocks, kThreads, 0, (cudaStream_t)stream>>>(
      (const uint4 *)src, (float4 *)dst, n8);
  return (int)cudaGetLastError();
}

}  // extern "C"
