# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Fused BHSD permute + GEMM swizzle of MXFP8 attention scales.

``mxfp8_transpose_swizzle`` in ``utils.py`` permutes each tensor's scale_inv from SBHD/BSHD to
BHSD with ``multi_tensor_transpose_to_bhsd`` and then swizzles it for the GEMM with
``multi_tensor_swizzle_scales_for_gemm_unchecked_`` (two kernels per direction). Here one Triton
kernel per direction reads the source layout and writes the swizzled BHSD layout directly.
"""

import torch
import triton
import triton.language as tl

_MAX_TENSORS = 3


@triton.jit
def _row_tile(src, dst, B, H, S, DR, KP, ss, sb, sh, pid, HB: tl.constexpr, BLOCK_KT: tl.constexpr):
    # Rowwise scales: matrix [B*H*S, KP] (row = (b*H + h)*S + s), swizzled in 128x4 tiles.
    # A program covers 128 sequence positions x HB heads; lanes run along (head, k), which is
    # contiguous in the SBHD/BSHD source. Each int32 word packs scales k = 4*kt .. 4*kt+3 of
    # one row; row si of a tile lands at word (si % 32) * 4 + si // 32 of that tile.
    n_hb = tl.cdiv(H, HB)
    n_st = S // 128
    hb = pid % n_hb
    st = (pid // n_hb) % n_st
    b = pid // (n_hb * n_st)
    si = tl.arange(0, 128)
    j = tl.arange(0, HB * BLOCK_KT)
    h = hb * HB + j // BLOCK_KT
    kt = j % BLOCK_KT
    s = st * 128 + si
    base = s[:, None] * ss + b * sb + (h * sh)[None, :]
    ok = ((h < H) & (kt < KP // 4))[None, :]
    word = tl.zeros((128, HB * BLOCK_KT), dtype=tl.int32)
    for c in tl.static_range(4):
        k = kt * 4 + c
        v = tl.load(src + base + k[None, :], mask=ok & (k < DR)[None, :], other=0)
        word = word | (v.to(tl.int32) << (8 * c))
    tile = (b * H + h) * n_st + st
    dst_w = dst.to(tl.pointer_type(tl.int32), bitcast=True)
    off = (tile * (KP // 4) + kt)[None, :] * 128 + ((si % 32) * 4 + si // 32)[:, None]
    tl.store(dst_w + off, word, mask=ok)


@triton.jit
def _col_tile(src, dst, H, SC, D, MC, ss, sb, sh, nt, mb, MB: tl.constexpr):
    # Columnwise scales: matrix [B*H*SC, NP] (row mc = (b*H + h)*SC + sc), swizzled as the
    # transposed [NP, MC] operand in 128x4 tiles. Word w of tile (nt, q) holds rows
    # mc = 4*q .. 4*q+3 at column n = nt*128 + (w % 4) * 32 + w // 4.
    w = tl.arange(0, 128)
    n = nt * 128 + (w % 4) * 32 + w // 4
    q = mb * MB + tl.arange(0, MB)
    word = tl.zeros((MB, 128), dtype=tl.int32)
    for c in tl.static_range(4):
        mc = q * 4 + c
        bh = mc // SC
        sc = mc % SC
        b = bh // H
        h = bh % H
        row = sc * ss + b * sb + h * sh
        m = (mc < MC)[:, None] & (n < D)[None, :]
        v = tl.load(src + row[:, None] + n[None, :], mask=m, other=0)
        word = word | (v.to(tl.int32) << (8 * c))
    dst_w = dst.to(tl.pointer_type(tl.int32), bitcast=True)
    off = (nt * (MC // 4) + q)[:, None] * 128 + w[None, :]
    tl.store(dst_w + off, word, mask=(q < MC // 4)[:, None])


@triton.jit
def _row_kernel(
    s0, d0, b0, h0, sq0, dr0, kp0, ss0, sb0, sh0, t0,
    s1, d1, b1, h1, sq1, dr1, kp1, ss1, sb1, sh1, t1,
    s2, d2, b2, h2, sq2, dr2, kp2, ss2, sb2, sh2, t2,
    HB: tl.constexpr, BLOCK_KT: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    if tid == 0:
        if pid < t0:
            _row_tile(s0, d0, b0, h0, sq0, dr0, kp0, ss0, sb0, sh0, pid, HB, BLOCK_KT)
    elif tid == 1:
        if pid < t1:
            _row_tile(s1, d1, b1, h1, sq1, dr1, kp1, ss1, sb1, sh1, pid, HB, BLOCK_KT)
    else:
        if pid < t2:
            _row_tile(s2, d2, b2, h2, sq2, dr2, kp2, ss2, sb2, sh2, pid, HB, BLOCK_KT)


@triton.jit
def _col_kernel(
    s0, d0, h0, sc0, dd0, mc0, ss0, sb0, sh0, nt0, t0,
    s1, d1, h1, sc1, dd1, mc1, ss1, sb1, sh1, nt1, t1,
    s2, d2, h2, sc2, dd2, mc2, ss2, sb2, sh2, nt2, t2,
    MB: tl.constexpr,
):
    pid = tl.program_id(0)
    tid = tl.program_id(1)
    if tid == 0:
        if pid < t0:
            _col_tile(s0, d0, h0, sc0, dd0, mc0, ss0, sb0, sh0, pid % nt0, pid // nt0, MB)
    elif tid == 1:
        if pid < t1:
            _col_tile(s1, d1, h1, sc1, dd1, mc1, ss1, sb1, sh1, pid % nt1, pid // nt1, MB)
    else:
        if pid < t2:
            _col_tile(s2, d2, h2, sc2, dd2, mc2, ss2, sb2, sh2, pid % nt2, pid // nt2, MB)


def _align(x, a):
    return (x + a - 1) // a * a


def _bshd_dims(t4d, src_format):
    """(B, S, H, D, stride_s, stride_b, stride_h) of a 4D scale view."""
    if src_format == "sbhd":
        S, B, H, D = t4d.shape
        ss, sb, sh, sd = t4d.stride()
    else:
        B, S, H, D = t4d.shape
        sb, ss, sh, sd = t4d.stride()
    assert sd == 1, "scale_inv must be contiguous in the last dim"
    return B, S, H, D, ss, sb, sh


_COL_MB = 8
_ROW_HB = 16


def mxfp8_transpose_swizzle_fused(fp8_tensors, src_format):
    """Drop-in for ``mxfp8_transpose_swizzle``: BHSD-permute + GEMM-swizzle in one pass."""
    if not fp8_tensors:
        return
    assert src_format in ("bshd", "sbhd")
    assert len(fp8_tensors) <= _MAX_TENSORS
    rows, cols, total = [], [], 0
    for t in fp8_tensors:
        rs = t._rowwise_scale_inv
        if rs is not None:
            B, S, H, DR, ss, sb, sh = _bshd_dims(rs, src_format)
            KP = _align(DR, 4)
            assert S % 128 == 0
            rows.append((t, rs, total, B, S, H, DR, KP, ss, sb, sh))
            total += B * H * S * KP
        else:
            rows.append(None)
        cs = t._columnwise_scale_inv
        if cs is not None:
            B, SC, H, D, ss, sb, sh = _bshd_dims(cs, src_format)
            NP = _align(D, 128)
            MC = B * H * SC
            assert MC % 4 == 0
            cols.append((t, cs, total, B, SC, H, D, NP, MC, ss, sb, sh))
            total += MC * NP
        else:
            cols.append(None)
    device = next(t._rowwise_scale_inv if t._rowwise_scale_inv is not None
                  else t._columnwise_scale_inv for t in fp8_tensors).device
    buf = torch.empty(total, dtype=torch.uint8, device=device)

    row_args, max_row, max_kp = [], 0, 4
    for e in rows:
        if e is None:
            row_args.append((buf, buf, 1, 1, 128, 0, 4, 0, 0, 0, 0))
            continue
        t, rs, off, B, S, H, DR, KP, ss, sb, sh = e
        out = buf[off : off + B * H * S * KP]
        tiles = B * triton.cdiv(H, _ROW_HB) * (S // 128)
        row_args.append((rs.view(torch.uint8), out, B, H, S, DR, KP, ss, sb, sh, tiles))
        max_row, max_kp = max(max_row, tiles), max(max_kp, KP)
        t._rowwise_scale_inv = out.view(B * H * S, KP)
    while len(row_args) < _MAX_TENSORS:
        row_args.append((buf, buf, 1, 1, 128, 0, 4, 0, 0, 0, 0))
    if max_row:
        _row_kernel[(max_row, len(fp8_tensors))](
            *[a for args in row_args for a in args], HB=_ROW_HB, BLOCK_KT=triton.next_power_of_2(max_kp // 4)
        )

    col_args, max_col = [], 0
    for e in cols:
        if e is None:
            col_args.append((buf, buf, 1, 1, 0, 4, 0, 0, 0, 1, 0))
            continue
        t, cs, off, B, SC, H, D, NP, MC, ss, sb, sh = e
        out = buf[off : off + MC * NP]
        n_nt = NP // 128
        progs = n_nt * triton.cdiv(MC // 4, _COL_MB)
        col_args.append((cs.view(torch.uint8), out, H, SC, D, MC, ss, sb, sh, n_nt, progs))
        max_col = max(max_col, progs)
        t._columnwise_scale_inv = out.view(MC, NP)
    while len(col_args) < _MAX_TENSORS:
        col_args.append((buf, buf, 1, 1, 0, 4, 0, 0, 0, 1, 0))
    if max_col:
        _col_kernel[(max_col, len(fp8_tensors))](*[a for args in col_args for a in args], MB=_COL_MB)

    for t in fp8_tensors:
        t._with_gemm_swizzled_scales = True
