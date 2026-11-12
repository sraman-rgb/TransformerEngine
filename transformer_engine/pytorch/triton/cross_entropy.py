# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch wrapper functions for Cross Entropy Triton kernels."""

import os
from typing import Optional, Union
import torch
import torch.distributed as dist

import triton
import triton.language as tl

from transformer_engine.common.triton.cross_entropy import (
    cross_entropy_forward_kernel,
    cross_entropy_tp_pre_kernel,
    cross_entropy_tp_post_kernel,
    cross_entropy_backward_kernel,
)

# The optimal maximum block size depends on your hardware, your kernel, and your dtype
MAX_FUSED_SIZE = 65536 // 2

# MLPERF_CE_TUNE=1: launch the single-rank forward and the backward kernels with smaller
# blocks and fewer warps (the defaults run one 1024-thread program per row and reach well
# under half of HBM bandwidth at a 129k vocabulary). Only the reduction order of the
# softmax statistics changes.
_CE_TUNE = os.getenv("MLPERF_CE_TUNE", "0") == "1"
_CE_FWD_BLOCK = int(os.getenv("MLPERF_CE_FWD_BLOCK", "8192"))
_CE_FWD_WARPS = int(os.getenv("MLPERF_CE_FWD_WARPS", "4"))
_CE_FWDA_BLOCK = int(os.getenv("MLPERF_CE_FWDA_BLOCK", "4096"))
_CE_FWDA_WARPS = int(os.getenv("MLPERF_CE_FWDA_WARPS", "4"))
_CE_BWD_BLOCK = int(os.getenv("MLPERF_CE_BWD_BLOCK", "4096"))
_CE_BWD_WARPS = int(os.getenv("MLPERF_CE_BWD_WARPS", "8"))
# MLPERF_CE_BWD_MXFP8=1: the backward writes the gradient directly as an MXFP8 (E4M3) tensor
# with row-wise and column-wise data, bit-identical to quantizing the BF16 gradient with
# MXFP8Quantizer, so the LM-head linear skips its own grad-output quantize pass.
_CE_BWD_MXFP8 = os.getenv("MLPERF_CE_BWD_MXFP8", "0") == "1"
# Scale arithmetic of TE's MXFP8 cast: "bf16" uses the BF16 reciprocal of 448 as an exact
# FP32 product (the _ENABLE_MXFMA build), "fp32" an FP32 multiply by 1/448.
_CE_MXFP8_RCP = os.getenv("MLPERF_CE_MXFP8_RCP", "bf16")
# MLPERF_CE1P=1 (overlay_ce1p): with MLPERF_CE_TUNE / MLPERF_CE_BWD_MXFP8, (a) the argmax forward keeps
# per-lane running maxima and reduces them once per row instead of once per block (the softmax
# statistics use the same blocks and reductions, so loss and argmax are bit-identical); (b) the MXFP8
# backward is a CUDA kernel (libce1p.so, overlay_ce1p/lib/ce1p.cu: same FP32 instruction sequence as
# the Triton kernel, 128x128 tiles) that writes the scales directly in the GEMM-swizzled layout (same
# E4M3 bytes; the scales equal TE's swizzle of the old ones), so the LM-head dgrad / wgrad GEMMs skip
# their scale-swizzle kernels; (c) ce1p_cat2 / ce1p_split2 serve the MTP eh_proj concat (overlay_ce1p
# multi_token_prediction.py).
_CE1P = os.getenv("MLPERF_CE1P", "0") == "1"
_CE1P_FOLD = int(os.getenv("MLPERF_CE1P_FOLD", "4"))
_CE1P_FOLD_C = tl.constexpr(_CE1P_FOLD)
# MLPERF_CE1P_ARGMAX: 2 (default) = compare chain over 4 adjacent elements per lane, rows whose softmax sum is
# NaN (a NaN or +inf logit, or all -inf) re-scanned; 1 = fold with per-lane NaN tracking.
_CE1P_ARGMAX = int(os.getenv("MLPERF_CE1P_ARGMAX", "2"))
_CE1P_SWIZZLE = os.getenv("MLPERF_CE1P_SWIZZLE", "1") == "1"


def cross_entropy_forward(
    _input: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    dist_process_group: Union[dist.ProcessGroup, None],
    ignore_idx: int,
    overwrite_input: bool,
    compute_argmax: bool = False,
):
    """Forward implementation that saves the input and compact softmax statistics.

    With ``compute_argmax``, also returns the index of each row's first maximum in the
    full vocabulary (otherwise None).
    """

    B, SQ, V = _input.shape
    n_rows = B * SQ
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    target = target.contiguous()
    saved_input = (
        _input
        if overwrite_input
        else torch.empty(_input.shape, dtype=_input.dtype, device=_input.device)
    )
    loss_1d = torch.empty(n_rows, dtype=torch.float32, device=_input.device)
    stats = torch.empty((n_rows, 2), dtype=torch.float32, device=_input.device)
    n_non_ignore = torch.zeros(1, dtype=torch.int64, device=_input.device) if reduce_loss else None
    argmax = (
        torch.empty(n_rows, dtype=torch.int64, device=_input.device) if compute_argmax else None
    )

    rank = 0 if dist_process_group is None else dist.get_rank(dist_process_group)
    world_size = 1 if dist_process_group is None else dist.get_world_size(dist_process_group)

    if world_size == 1:
        fwd_kernel = (
            (
                (_ce_forward_kernel_chain_argmax if _CE1P_ARGMAX == 2 else _ce_forward_kernel_lane_argmax)
                if _CE1P
                else _ce_forward_kernel_fast_argmax
            )
            if (_CE_TUNE and compute_argmax)
            else cross_entropy_forward_kernel
        )
        fwd_kernel[(n_rows,)](
            X_ptr=_input,
            X_stride_0=_input.stride(0),
            X_stride_1=_input.stride(1),
            X_stride_2=_input.stride(2),
            saved_input_ptr=saved_input,
            Y_ptr=target,
            loss_ptr=loss_1d,
            stats_ptr=stats,
            argmax_ptr=argmax if argmax is not None else stats,
            n_non_ignore=n_non_ignore if n_non_ignore is not None else stats,
            n_cols=V,
            n_rows_1=SQ,
            ignore_idx=ignore_idx,
            label_smoothing=label_smoothing,
            COUNT_NON_IGNORE=reduce_loss,
            COPY_INPUT=not overwrite_input,
            COMPUTE_ARGMAX=compute_argmax,
            BLOCK_SIZE=(
                min(_CE_FWDA_BLOCK if compute_argmax else _CE_FWD_BLOCK, BLOCK_SIZE)
                if _CE_TUNE
                else BLOCK_SIZE
            ),
            num_warps=(_CE_FWDA_WARPS if compute_argmax else _CE_FWD_WARPS) if _CE_TUNE else 32,
        )
    else:
        # Per row and rank: max, sum of exponentials, target logit, logit sum, and with
        # compute_argmax the local argmax (as FP32 bits) and its value (NaN if the shard has one)
        local_data_size = 6 if compute_argmax else 4
        local_data = torch.empty(
            (n_rows, local_data_size), dtype=torch.float32, device=_input.device
        )
        cross_entropy_tp_pre_kernel[(n_rows,)](
            X_ptr=_input,
            X_stride_0=_input.stride(0),
            X_stride_1=_input.stride(1),
            X_stride_2=_input.stride(2),
            saved_input_ptr=saved_input,
            Y_ptr=target,
            local_data_ptr=local_data,
            n_non_ignore=n_non_ignore if n_non_ignore is not None else stats,
            rank=rank,
            n_cols=V,
            n_rows_1=SQ,
            ignore_idx=ignore_idx,
            COUNT_NON_IGNORE=reduce_loss,
            COMPUTE_X_SUM=label_smoothing > 0,
            COPY_INPUT=not overwrite_input,
            COMPUTE_ARGMAX=compute_argmax,
            LOCAL_DATA_SIZE=local_data_size,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )
        gathered_data = torch.empty(
            (world_size * n_rows, local_data_size), dtype=torch.float32, device=_input.device
        )
        dist.all_gather_into_tensor(gathered_data, local_data, group=dist_process_group)
        cross_entropy_tp_post_kernel[(n_rows,)](
            gathered_data_ptr=gathered_data,
            Y_ptr=target,
            loss_ptr=loss_1d,
            stats_ptr=stats,
            argmax_ptr=argmax if argmax is not None else stats,
            world_size=world_size,
            n_rows=n_rows,
            n_cols=V,
            ignore_idx=ignore_idx,
            label_smoothing=label_smoothing,
            COMPUTE_ARGMAX=compute_argmax,
            LOCAL_DATA_SIZE=local_data_size,
            num_warps=1,
        )

    loss = loss_1d.reshape(B, SQ)
    if reduce_loss:
        n_non_ignore.clamp_(min=1)
        loss = loss_1d.sum() / n_non_ignore
    if argmax is not None:
        argmax = argmax.reshape(B, SQ)

    return loss, saved_input, stats, target, n_non_ignore, rank, world_size, argmax


def cross_entropy_backward(
    saved_input: torch.Tensor,
    stats: torch.Tensor,
    target: torch.Tensor,
    n_non_ignore: Optional[torch.Tensor],
    grad_output: torch.Tensor,
    label_smoothing: float,
    reduce_loss: bool,
    rank: int,
    world_size: int,
    ignore_idx: int,
):
    """Reconstruct the derivative in FP32 and overwrite the saved input buffer."""

    B, SQ, V = saved_input.shape
    n_rows = B * SQ
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))
    grad_output = grad_output.contiguous()

    if (
        _CE1P
        and _CE_BWD_MXFP8
        and _ce1p_bwd_supported(saved_input, world_size, grad_output, label_smoothing, target, stats)
    ):
        return _cross_entropy_backward_mxfp8_ce1p(
            saved_input,
            stats,
            target,
            n_non_ignore,
            grad_output,
            label_smoothing,
            reduce_loss,
            ignore_idx,
        )
    if _CE_BWD_MXFP8 and _mxfp8_bwd_supported(saved_input, world_size):
        return _cross_entropy_backward_mxfp8(
            saved_input,
            stats,
            target,
            n_non_ignore,
            grad_output,
            label_smoothing,
            reduce_loss,
            ignore_idx,
        )

    cross_entropy_backward_kernel[(n_rows,)](
        saved_input_ptr=saved_input,
        Y_ptr=target,
        stats_ptr=stats,
        n_non_ignore_ptr=n_non_ignore if n_non_ignore is not None else stats,
        grad_output_ptr=grad_output,
        grad_output_stride=0 if reduce_loss else 1,
        rank=rank,
        world_size=world_size,
        n_cols=V,
        ignore_idx=ignore_idx,
        reduce_loss=reduce_loss,
        label_smoothing=label_smoothing,
        BLOCK_SIZE=min(_CE_BWD_BLOCK, BLOCK_SIZE) if _CE_TUNE else BLOCK_SIZE,
        num_warps=_CE_BWD_WARPS if _CE_TUNE else 32,
    )
    return saved_input


@triton.jit
def _to_e8m0(amax, RCP_BF16: tl.constexpr):
    """E8M0 block exponent as in TE's MXFP8 cast: round amax/448 up to a power of two."""
    if RCP_BF16:
        val = amax * 0.0022277832031250  # BF16 0x3b12, exact FP32 product with a BF16 amax
    else:
        val = amax * (1.0 / 448.0)
    bits = val.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    round_up = (mantissa > 0) & (exponent != 0xFE) & ~((exponent == 0) & (mantissa <= 0x400000))
    return tl.where(round_up, exponent + 1, exponent)


@triton.jit
def _e8m0_rcp(e):
    """2^-(e-127) as FP32, matching TE's exp2f_rcp<float>."""
    bits = tl.where(e == 254, 0x00400000, (254 - e) << 23)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _ce_backward_mxfp8_kernel(
    saved_input_ptr,
    Y_ptr,
    stats_ptr,
    n_non_ignore_ptr,
    grad_output_ptr,
    grad_output_stride,
    row_data_ptr,
    row_scale_ptr,
    row_scale_stride,
    col_data_ptr,
    col_scale_ptr,
    col_scale_stride,
    n_cols,
    ignore_idx,
    reduce_loss: tl.constexpr,
    label_smoothing: tl.constexpr,
    RCP_BF16: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Derivative of a 32-row x BLOCK_C tile, rounded to BF16 as the unfused backward stores
    it, then MXFP8-quantized row-wise (1x32 blocks) and column-wise (32x1 blocks)."""
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * 32 + tl.arange(0, 32)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    rows64 = rows.to(tl.int64)
    offs = rows64[:, None] * n_cols + cols[None, :]

    y = tl.load(Y_ptr + rows)
    m = tl.load(stats_ptr + rows64 * 2)
    d = tl.load(stats_ptr + rows64 * 2 + 1)
    go = tl.load(grad_output_ptr + rows64 * grad_output_stride).to(tl.float32)
    if reduce_loss:
        go = go / tl.load(n_non_ignore_ptr)

    eps = label_smoothing / n_cols
    x = tl.load(saved_input_ptr + offs).to(tl.float32)
    grad = tl.exp(x - m[:, None]) / d[:, None] - eps
    is_target = cols[None, :] == y[:, None]
    grad -= tl.where(is_target, 1 - label_smoothing, 0.0)
    grad = grad * go[:, None]
    grad = tl.where((y == ignore_idx)[:, None], 0.0, grad)
    g = grad.to(tl.bfloat16).to(tl.float32)
    ag = tl.abs(g)

    # Row-wise: 1x32 blocks along the vocabulary.
    NB: tl.constexpr = BLOCK_C // 32
    row_amax = tl.max(tl.reshape(ag, (32, NB, 32)), axis=2)
    row_e = _to_e8m0(row_amax, RCP_BF16)
    row_rcp = _e8m0_rcp(row_e)
    gq = tl.reshape(tl.reshape(g, (32, NB, 32)) * row_rcp[:, :, None], (32, BLOCK_C))
    tl.store(row_data_ptr + offs, gq.to(tl.float8e4nv).to(tl.uint8, bitcast=True))
    sb = pid_c * NB + tl.arange(0, NB)
    tl.store(row_scale_ptr + rows64[:, None] * row_scale_stride + sb[None, :], row_e.to(tl.uint8))

    # Column-wise: 32x1 blocks along the tokens.
    col_amax = tl.max(ag, axis=0)
    col_e = _to_e8m0(col_amax, RCP_BF16)
    col_rcp = _e8m0_rcp(col_e)
    tl.store(
        col_data_ptr + offs, (g * col_rcp[None, :]).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    )
    tl.store(col_scale_ptr + pid_r.to(tl.int64) * col_scale_stride + cols, col_e.to(tl.uint8))


_MXFP8_BWD_LOGGED = False
_MXFP8_BLOCK_C = int(os.getenv("MLPERF_CE_MXFP8_BLOCK_C", "256"))
_MXFP8_WARPS = int(os.getenv("MLPERF_CE_MXFP8_WARPS", "4"))


def _mxfp8_bwd_supported(saved_input: torch.Tensor, world_size: int) -> bool:
    B, SQ, V = saved_input.shape
    n_rows = B * SQ
    return (
        world_size == 1
        and saved_input.dtype == torch.bfloat16
        and saved_input.is_contiguous()
        and n_rows % 128 == 0
        and V % _MXFP8_BLOCK_C == 0
        and (V // 32) % 4 == 0
    )


def _cross_entropy_backward_mxfp8(
    saved_input,
    stats,
    target,
    n_non_ignore,
    grad_output,
    label_smoothing,
    reduce_loss,
    ignore_idx,
):
    """Backward that returns the gradient as an MXFP8Tensor (row-wise and column-wise)."""
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    import transformer_engine_torch as tex

    global _MXFP8_BWD_LOGGED
    if not _MXFP8_BWD_LOGGED:
        _MXFP8_BWD_LOGGED = True
        print("MLPERF_CE_BWD_MXFP8: cross-entropy backward writes the MXFP8 gradient directly",
              flush=True)
    B, SQ, V = saved_input.shape
    n_rows = B * SQ
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    out = quantizer.make_empty(saved_input.shape, dtype=torch.bfloat16, device=saved_input.device)
    row_scale = out._rowwise_scale_inv
    col_scale = out._columnwise_scale_inv
    _ce_backward_mxfp8_kernel[(n_rows // 32, V // _MXFP8_BLOCK_C)](
        saved_input_ptr=saved_input,
        Y_ptr=target,
        stats_ptr=stats,
        n_non_ignore_ptr=n_non_ignore if n_non_ignore is not None else stats,
        grad_output_ptr=grad_output,
        grad_output_stride=0 if reduce_loss else 1,
        row_data_ptr=out._rowwise_data,
        row_scale_ptr=row_scale,
        row_scale_stride=row_scale.shape[-1],
        col_data_ptr=out._columnwise_data,
        col_scale_ptr=col_scale,
        col_scale_stride=col_scale.shape[-1],
        n_cols=V,
        ignore_idx=ignore_idx,
        reduce_loss=reduce_loss,
        label_smoothing=label_smoothing,
        RCP_BF16=_CE_MXFP8_RCP == "bf16",
        BLOCK_C=_MXFP8_BLOCK_C,
        num_warps=_MXFP8_WARPS,
    )
    return out


@triton.jit
def _ce_forward_kernel_fast_argmax(
    X_ptr,
    X_stride_0,
    X_stride_1,
    X_stride_2,
    saved_input_ptr,
    Y_ptr,
    loss_ptr,
    stats_ptr,
    argmax_ptr,
    n_non_ignore,
    n_cols,
    n_rows_1,
    ignore_idx,
    label_smoothing: tl.constexpr,
    COUNT_NON_IGNORE: tl.constexpr,
    COPY_INPUT: tl.constexpr,
    COMPUTE_ARGMAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """cross_entropy_forward_kernel with a cheaper argmax fold that keeps torch.argmax
    semantics (first maximum wins; the first NaN wins over any number): the block maximum of
    the softmax pass is reused, the first index of that maximum is one min-reduction, and the
    NaN position is only searched for in a block that has one."""

    row = tl.program_id(0).to(tl.int64)
    row_0 = row // n_rows_1
    row_1 = row - row_0 * n_rows_1
    X_ptr += row_0 * X_stride_0 + row_1 * X_stride_1
    saved_input_ptr += row * n_cols

    y = tl.load(Y_ptr + row)
    if COUNT_NON_IGNORE:
        if y != ignore_idx:
            tl.atomic_add(n_non_ignore, 1)

    m = float("-inf")
    d = 0.0
    x_sum = 0.0
    run_idx = 0
    run_max = float("-inf")
    run_nan = 0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(X_ptr + offsets * X_stride_2, mask=mask, other=float("-inf"))
        if COPY_INPUT:
            tl.store(saved_input_ptr + offsets, x, mask=mask)
        x = x.to(tl.float32)
        is_nan = x != x
        x_nonan = tl.where(is_nan, float("-inf"), x)
        block_max = tl.max(x_nonan)
        if COMPUTE_ARGMAX:
            if run_nan == 0:
                block_has_nan = tl.max(is_nan.to(tl.int32)) > 0
                if block_has_nan:
                    run_idx = tl.min(tl.where(is_nan, offsets, n_cols))
                    run_nan = 1
                elif block_max > run_max:
                    run_idx = tl.min(tl.where(x_nonan == block_max, offsets, n_cols))
                    run_max = block_max
        bm = tl.max(x)
        m_new = tl.maximum(m, bm)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new
        if label_smoothing > 0:
            x_sum += tl.sum(tl.where(mask, x, 0.0))

    tl.store(stats_ptr + row * 2, m)
    tl.store(stats_ptr + row * 2 + 1, d)
    if COMPUTE_ARGMAX:
        tl.store(argmax_ptr + row, run_idx.to(tl.int64))

    if y == ignore_idx:
        tl.store(loss_ptr + row, 0.0)
        return

    x_y = float("-inf")
    if y >= 0:
        if y < n_cols:
            x_y = tl.load(X_ptr + y * X_stride_2).to(tl.float32)

    loss = -(x_y - m - tl.log(d))
    if label_smoothing > 0:
        eps = label_smoothing / n_cols
        smooth_loss = -eps * x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss
    tl.store(loss_ptr + row, loss)



# ---------------------------------------------------------------------------------------------
# MLPERF_CE1P kernels (overlay_ce1p)
# ---------------------------------------------------------------------------------------------


@triton.jit
def _ce_forward_kernel_lane_argmax(
    X_ptr,
    X_stride_0,
    X_stride_1,
    X_stride_2,
    saved_input_ptr,
    Y_ptr,
    loss_ptr,
    stats_ptr,
    argmax_ptr,
    n_non_ignore,
    n_cols,
    n_rows_1,
    ignore_idx,
    label_smoothing: tl.constexpr,
    COUNT_NON_IGNORE: tl.constexpr,
    COPY_INPUT: tl.constexpr,
    COMPUTE_ARGMAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """_ce_forward_kernel_fast_argmax with the argmax kept per lane: lane i holds the first
    maximum among positions i, i + BLOCK, ... (a NaN locks the lane), and one reduction at the end
    picks the first NaN, else the first position of the row maximum (torch.argmax semantics).
    The softmax statistics and loss are computed exactly as in the other forward kernels."""

    row = tl.program_id(0).to(tl.int64)
    row_0 = row // n_rows_1
    row_1 = row - row_0 * n_rows_1
    X_ptr += row_0 * X_stride_0 + row_1 * X_stride_1
    saved_input_ptr += row * n_cols

    y = tl.load(Y_ptr + row)
    if COUNT_NON_IGNORE:
        if y != ignore_idx:
            tl.atomic_add(n_non_ignore, 1)

    m = float("-inf")
    d = 0.0
    x_sum = 0.0
    LANES: tl.constexpr = BLOCK_SIZE // _CE1P_FOLD_C
    lane_v = tl.full([LANES], float("-inf"), tl.float32)
    lane_i = tl.zeros([LANES], dtype=tl.int32)
    fold_r = tl.arange(0, _CE1P_FOLD_C)[:, None]
    fold_c = tl.arange(0, LANES)[None, :]
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(X_ptr + offsets * X_stride_2, mask=mask, other=float("-inf"))
        if COPY_INPUT:
            tl.store(saved_input_ptr + offsets, x, mask=mask)
        x = x.to(tl.float32)
        if COMPUTE_ARGMAX:
            # Fold the block to [FOLD, LANES] (element f * LANES + c in lane c, in-thread for the
            # default layout), take each lane's first NaN, else its first maximum, then merge into
            # the running per-lane candidates (earlier positions win ties; a NaN locks the lane).
            x2 = tl.reshape(x, (_CE1P_FOLD_C, LANES))
            n2 = x2 != x2
            nan_r = tl.min(tl.where(n2, fold_r, _CE1P_FOLD_C), axis=0)
            cmax = tl.max(tl.where(n2, float("-inf"), x2), axis=0)
            max_r = tl.min(tl.where(x2 == cmax[None, :], fold_r, _CE1P_FOLD_C), axis=0)
            has_nan = nan_r < _CE1P_FOLD_C
            cand_v = tl.where(has_nan, float("nan"), cmax)
            cand_i = i + tl.where(has_nan, nan_r, max_r) * LANES + tl.arange(0, LANES)
            upd = (lane_v == lane_v) & ((cand_v > lane_v) | has_nan)
            lane_v = tl.where(upd, cand_v, lane_v)
            lane_i = tl.where(upd, cand_i, lane_i)
        bm = tl.max(x)
        m_new = tl.maximum(m, bm)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new
        if label_smoothing > 0:
            x_sum += tl.sum(tl.where(mask, x, 0.0))

    tl.store(stats_ptr + row * 2, m)
    tl.store(stats_ptr + row * 2 + 1, d)
    if COMPUTE_ARGMAX:
        lane_nan = lane_v != lane_v
        big = 2147483647
        nan_idx = tl.min(tl.where(lane_nan, lane_i, big))
        vmax = tl.max(tl.where(lane_nan, float("-inf"), lane_v))
        max_idx = tl.min(tl.where((lane_v == vmax) & ~lane_nan, lane_i, big))
        res = tl.where(nan_idx < big, nan_idx, max_idx)
        tl.store(argmax_ptr + row, res.to(tl.int64))

    if y == ignore_idx:
        tl.store(loss_ptr + row, 0.0)
        return

    x_y = float("-inf")
    if y >= 0:
        if y < n_cols:
            x_y = tl.load(X_ptr + y * X_stride_2).to(tl.float32)

    loss = -(x_y - m - tl.log(d))
    if label_smoothing > 0:
        eps = label_smoothing / n_cols
        smooth_loss = -eps * x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss
    tl.store(loss_ptr + row, loss)


_CE1P_LIB = None


def _ce1p_lib():
    """libce1p.so (overlay_ce1p/lib/ce1p.cu), mounted next to this file."""
    global _CE1P_LIB
    if _CE1P_LIB is None:
        import ctypes

        lib = ctypes.CDLL(
            os.getenv(
                "CE1P_LIB",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "libce1p.so"),
            )
        )
        vp, i64, f32, i32 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_float, ctypes.c_int
        lib.ce1p_bwd_mxfp8.argtypes = [vp, vp, vp, vp, vp, i64, vp, vp, vp, vp, i64, i64, i64, f32, f32,
                                       i32, i32, i32, vp]
        lib.ce1p_bwd_mxfp8.restype = i32
        lib.ce1p_cat2.argtypes = [vp, vp, vp, i64, i64, i64, i32, i32, vp]
        lib.ce1p_cat2.restype = i32
        _CE1P_LIB = lib
    return _CE1P_LIB


def _ce1p_bwd_supported(saved_input, world_size, grad_output, label_smoothing, target, stats):
    B, SQ, V = saved_input.shape
    n_rows = B * SQ
    return (
        world_size == 1
        and saved_input.dtype == torch.bfloat16
        and saved_input.is_contiguous()
        and grad_output.dtype == torch.float32
        and target.dtype == torch.int64
        and target.is_contiguous()
        and stats.dtype == torch.float32
        and stats.is_contiguous()
        and label_smoothing == 0.0
        and n_rows % 128 == 0
        and V % 128 == 0
        and saved_input.data_ptr() % 16 == 0
    )


def _cross_entropy_backward_mxfp8_ce1p(
    saved_input,
    stats,
    target,
    n_non_ignore,
    grad_output,
    label_smoothing,
    reduce_loss,
    ignore_idx,
):
    """MXFP8 gradient (row-wise and column-wise), scales in the GEMM-swizzled layout (CUDA kernel)."""
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
    import transformer_engine_torch as tex

    global _MXFP8_BWD_LOGGED
    if not _MXFP8_BWD_LOGGED:
        _MXFP8_BWD_LOGGED = True
        print(
            "MLPERF_CE1P: cross-entropy backward writes the MXFP8 gradient with GEMM-swizzled"
            f" scales directly (swizzle={_CE1P_SWIZZLE})",
            flush=True,
        )
    B, SQ, V = saved_input.shape
    n_rows = B * SQ
    dev = saved_input.device
    row_data = torch.empty(saved_input.shape, dtype=torch.uint8, device=dev)
    col_data = torch.empty(saved_input.shape, dtype=torch.uint8, device=dev)
    # No padding: n_rows % 128 == 0 and V % 128 == 0.
    row_scale = torch.empty((n_rows, V // 32), dtype=torch.uint8, device=dev)
    col_scale = torch.empty((n_rows // 32, V), dtype=torch.uint8, device=dev)
    err = _ce1p_lib().ce1p_bwd_mxfp8(
        saved_input.data_ptr(),
        target.data_ptr(),
        stats.data_ptr(),
        n_non_ignore.data_ptr() if (reduce_loss and n_non_ignore is not None) else 0,
        grad_output.data_ptr(),
        0 if reduce_loss else 1,
        row_data.data_ptr(),
        row_scale.data_ptr(),
        col_data.data_ptr(),
        col_scale.data_ptr(),
        n_rows,
        V,
        ignore_idx,
        0.0,
        1.0,
        1 if reduce_loss else 0,
        1 if _CE_MXFP8_RCP == "bf16" else 0,
        1 if _CE1P_SWIZZLE else 0,
        torch.cuda.current_stream(dev).cuda_stream,
    )
    if err != 0:
        raise RuntimeError(f"ce1p_bwd_mxfp8 failed: {err}")
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    quantizer.optimize_for_gemm = _CE1P_SWIZZLE
    return MXFP8Tensor(
        shape=saved_input.shape,
        dtype=torch.bfloat16,
        rowwise_data=row_data,
        rowwise_scale_inv=row_scale,
        columnwise_data=col_data,
        columnwise_scale_inv=col_scale,
        quantizer=quantizer,
        requires_grad=False,
        fp8_dtype=tex.DType.kFloat8E4M3,
        with_gemm_swizzled_scales=_CE1P_SWIZZLE,
    )


def ce1p_cat2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """torch.cat((a, b), -1) for contiguous same-shaped-prefix tensors (one 16-byte-vector copy kernel)."""
    out = torch.empty(a.shape[:-1] + (a.shape[-1] + b.shape[-1],), dtype=a.dtype, device=a.device)
    n_rows = a.numel() // a.shape[-1]
    err = _ce1p_lib().ce1p_cat2(out.data_ptr(), a.data_ptr(), b.data_ptr(), n_rows,
                                a.shape[-1] * a.element_size(), b.shape[-1] * b.element_size(), 0,
                                _sm_count(a.device), torch.cuda.current_stream(a.device).cuda_stream)
    if err != 0:
        raise RuntimeError(f"ce1p_cat2 failed: {err}")
    return out


def ce1p_split2(whole: torch.Tensor, n_a: int):
    """Contiguous copies of whole[..., :n_a] and whole[..., n_a:] (one kernel)."""
    whole = whole.contiguous()
    n_b = whole.shape[-1] - n_a
    a = torch.empty(whole.shape[:-1] + (n_a,), dtype=whole.dtype, device=whole.device)
    b = torch.empty(whole.shape[:-1] + (n_b,), dtype=whole.dtype, device=whole.device)
    n_rows = whole.numel() // whole.shape[-1]
    err = _ce1p_lib().ce1p_cat2(whole.data_ptr(), a.data_ptr(), b.data_ptr(), n_rows,
                                n_a * whole.element_size(), n_b * whole.element_size(), 1,
                                _sm_count(whole.device), torch.cuda.current_stream(whole.device).cuda_stream)
    if err != 0:
        raise RuntimeError(f"ce1p_split2 failed: {err}")
    return a, b


_SM_COUNT = {}


def _sm_count(device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


def ce1p_cat2_supported(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.is_cuda
        and a.dtype == b.dtype
        and a.is_contiguous()
        and b.is_contiguous()
        and a.shape[:-1] == b.shape[:-1]
        and (a.shape[-1] * a.element_size()) % 16 == 0
        and (b.shape[-1] * b.element_size()) % 16 == 0
        and a.data_ptr() % 16 == 0
        and b.data_ptr() % 16 == 0
    )


@triton.jit
def _ce_forward_kernel_chain_argmax(
    X_ptr,
    X_stride_0,
    X_stride_1,
    X_stride_2,
    saved_input_ptr,
    Y_ptr,
    loss_ptr,
    stats_ptr,
    argmax_ptr,
    n_non_ignore,
    n_cols,
    n_rows_1,
    ignore_idx,
    label_smoothing: tl.constexpr,
    COUNT_NON_IGNORE: tl.constexpr,
    COPY_INPUT: tl.constexpr,
    COMPUTE_ARGMAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Softmax statistics and loss exactly as the other forward kernels. Argmax: lane c of a block holds
    positions 4c..4c+3 (in-thread for the default layout) and keeps its first maximum with a strict-greater
    compare chain; one reduction per row picks the first position of the row maximum. NaN never wins a
    compare, so rows whose sum of exponentials is NaN (a NaN or +inf logit, or all -inf) are re-scanned
    with torch.argmax semantics (first NaN, else first maximum)."""

    row = tl.program_id(0).to(tl.int64)
    row_0 = row // n_rows_1
    row_1 = row - row_0 * n_rows_1
    X_ptr += row_0 * X_stride_0 + row_1 * X_stride_1
    saved_input_ptr += row * n_cols

    y = tl.load(Y_ptr + row)
    if COUNT_NON_IGNORE:
        if y != ignore_idx:
            tl.atomic_add(n_non_ignore, 1)

    m = float("-inf")
    d = 0.0
    x_sum = 0.0
    LANES: tl.constexpr = BLOCK_SIZE // 4
    lane_v = tl.full([LANES], float("-inf"), tl.float32)
    lane_i = tl.zeros([LANES], dtype=tl.int32)
    base4 = tl.arange(0, LANES) * 4
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(X_ptr + offsets * X_stride_2, mask=mask, other=float("-inf"))
        if COPY_INPUT:
            tl.store(saved_input_ptr + offsets, x, mask=mask)
        x = x.to(tl.float32)
        if COMPUTE_ARGMAX:
            xa, xb = tl.split(tl.reshape(x, (LANES, 2, 2)))  # [c, a, 0], [c, a, 1]
            x0, x2 = tl.split(xa)  # positions 4c + 0, 4c + 2
            x1, x3 = tl.split(xb)  # positions 4c + 1, 4c + 3
            b = i + base4
            u = x0 > lane_v
            lane_v = tl.where(u, x0, lane_v)
            lane_i = tl.where(u, b, lane_i)
            u = x1 > lane_v
            lane_v = tl.where(u, x1, lane_v)
            lane_i = tl.where(u, b + 1, lane_i)
            u = x2 > lane_v
            lane_v = tl.where(u, x2, lane_v)
            lane_i = tl.where(u, b + 2, lane_i)
            u = x3 > lane_v
            lane_v = tl.where(u, x3, lane_v)
            lane_i = tl.where(u, b + 3, lane_i)
        bm = tl.max(x)
        m_new = tl.maximum(m, bm)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new
        if label_smoothing > 0:
            x_sum += tl.sum(tl.where(mask, x, 0.0))

    tl.store(stats_ptr + row * 2, m)
    tl.store(stats_ptr + row * 2 + 1, d)
    if COMPUTE_ARGMAX:
        big = 2147483647
        vmax = tl.max(lane_v)
        res = tl.min(tl.where(lane_v == vmax, lane_i, big))
        if d != d:
            # Rare: re-scan with NaN tracking (first NaN wins, else first maximum).
            run_idx = 0
            run_max = float("-inf")
            run_nan = 0
            for i2 in range(0, n_cols, BLOCK_SIZE):
                offs2 = i2 + tl.arange(0, BLOCK_SIZE)
                x2r = tl.load(X_ptr + offs2 * X_stride_2, mask=offs2 < n_cols, other=float("-inf")).to(
                    tl.float32
                )
                isn = x2r != x2r
                xnn = tl.where(isn, float("-inf"), x2r)
                bmax2 = tl.max(xnn)
                if run_nan == 0:
                    if tl.max(isn.to(tl.int32)) > 0:
                        run_idx = tl.min(tl.where(isn, offs2, n_cols))
                        run_nan = 1
                    elif bmax2 > run_max:
                        run_idx = tl.min(tl.where(xnn == bmax2, offs2, n_cols))
                        run_max = bmax2
            res = run_idx
        tl.store(argmax_ptr + row, res.to(tl.int64))

    if y == ignore_idx:
        tl.store(loss_ptr + row, 0.0)
        return

    x_y = float("-inf")
    if y >= 0:
        if y < n_cols:
            x_y = tl.load(X_ptr + y * X_stride_2).to(tl.float32)

    loss = -(x_y - m - tl.log(d))
    if label_smoothing > 0:
        eps = label_smoothing / n_cols
        smooth_loss = -eps * x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss
    tl.store(loss_ptr + row, loss)
