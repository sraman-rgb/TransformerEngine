# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for MoE grouped MLP."""

from __future__ import annotations
from collections.abc import Callable
import functools
import inspect
import math
import os
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...module.base import get_dummy_wgrad
from ...quantization import Recipe
from ...tensor import NVFP4Quantizer, NVFP4Tensor
from ...tensor.grouped_tensor import GroupedTensor
from ...tensor.mxfp8_tensor import MXFP8Quantizer
from ...utils import clear_tensor_data, get_cached_ones_tensor, get_device_compute_capability
from ...constants import MXFP8_BLOCK_SCALING_SIZE, NVFP4_BLOCK_SCALING_SIZE
from ..basic import GroupedLinear, ScaledSReLU, ScaledClampedQGeGLU, ScaledSwiGLU
from ..fuser import register_backward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import (
    _nvidia_cudnn_frontend_supports_wgrad,
    fuse_grouped_mlp_ops,
    maybe_dequantize,
    validate_grouped_mlp_dims,
)
from ...cpp_extensions import (
    general_gemm,
    general_grouped_gemm_for_grouped_tensor,
)
from ...module.base import _2X_ACC_WGRAD
from ...triton.grouped_dbias_dscales import compute_grouped_dbias_dscales


def _pack_nvfp4_amax_list(tensors: list) -> None:
    """Ensure discrete NVFP4 weight list uses contiguous per-group amax buffers.

    The discrete-input grouped GEMM kernels expect a single contiguous device
    buffer for amax pointers across groups. This rebinds each tensor's
    ``_amax_rowwise`` / ``_amax_columnwise`` to a 1-element view into a packed
    buffer so that the resulting pointer array is contiguous in device memory.
    """
    if not tensors:
        return
    row_amaxes = [getattr(tensor, "_amax_rowwise", None) for tensor in tensors]
    if all(amax is not None for amax in row_amaxes):
        packed_row_amax = torch.cat([amax.view(-1) for amax in row_amaxes], dim=0).contiguous()
        for idx, tensor in enumerate(tensors):
            tensor._amax_rowwise = packed_row_amax[idx : idx + 1]
    col_amaxes = [getattr(tensor, "_amax_columnwise", None) for tensor in tensors]
    if all(amax is not None for amax in col_amaxes):
        packed_col_amax = torch.cat([amax.view(-1) for amax in col_amaxes], dim=0).contiguous()
        for idx, tensor in enumerate(tensors):
            tensor._amax_columnwise = packed_col_amax[idx : idx + 1]


def _mark_with_gemm_swizzled_scales(tensors: Any) -> None:
    """Mark tensors whose scale buffers are already in GEMM-swizzled layout."""
    if tensors is None:
        return
    if hasattr(tensors, "with_gemm_swizzled_scales"):
        tensors.with_gemm_swizzled_scales = True
    if hasattr(tensors, "_with_gemm_swizzled_scales"):
        tensors._with_gemm_swizzled_scales = True


def _enable_nvfp4_rht_for_group_quantize(quantizer) -> None:
    """Use the graph-safe NVFP4 grouped quantization path."""
    if isinstance(quantizer, NVFP4Quantizer):
        quantizer.with_rht = True
        quantizer.with_post_rht_amax = True


def _group_quantize_for_grouped_mlp(
    tensor: torch.Tensor,
    quantizer: Any,
    num_groups: int,
    split_sizes: Optional[torch.Tensor],
    *,
    tensor_offsets: Optional[torch.Tensor] = None,
) -> GroupedTensor:
    """Quantize into grouped storage, using regular quantize for one-group NVFP4."""
    if num_groups != 1 or not isinstance(quantizer, NVFP4Quantizer):
        return tex.group_quantize(tensor, quantizer, num_groups, split_sizes)

    quantized = tex.quantize(tensor, quantizer)
    with_gemm_swizzled_scales = getattr(quantized, "_with_gemm_swizzled_scales", False)
    if getattr(quantizer, "optimize_for_gemm", False):
        tex.swizzle_scales_for_gemm_(quantized)
        with_gemm_swizzled_scales = True
    rowwise_data = getattr(quantized, "_rowwise_data", None)
    rowwise_scale = getattr(quantized, "_rowwise_scale_inv", None)
    columnwise_data = getattr(quantized, "_columnwise_data", None)
    columnwise_scale = getattr(quantized, "_columnwise_scale_inv", None)
    amax = getattr(quantized, "_amax_rowwise", None)
    columnwise_amax = getattr(quantized, "_amax_columnwise", None)

    if split_sizes is None:
        split_sizes = torch.full(
            (1,),
            tensor.shape[0],
            dtype=torch.int64,
            device=tensor.device,
        )
    else:
        split_sizes = split_sizes.to(dtype=torch.int64, device=tensor.device)

    m_dim = tensor.shape[0]
    if rowwise_data is not None:
        k_dim = rowwise_data.shape[-1] * 2
    elif columnwise_data is not None:
        k_dim = columnwise_data.shape[0]
    else:
        k_dim = tensor.shape[-1]

    if tensor_offsets is None:
        tensor_offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=tensor.device),
                torch.cumsum(split_sizes * k_dim, dim=0),
            ],
        )

    grouped = GroupedTensor(
        shape=(m_dim, k_dim),
        dtype=tensor.dtype,
        quantizer=quantizer,
        num_tensors=1,
        data=rowwise_data.reshape(-1) if rowwise_data is not None else None,
        columnwise_data=columnwise_data.reshape(-1) if columnwise_data is not None else None,
        scale_inv=rowwise_scale.reshape(-1) if rowwise_scale is not None else None,
        columnwise_scale_inv=columnwise_scale.reshape(-1)
        if columnwise_scale is not None
        else None,
        amax=amax,
        columnwise_amax=columnwise_amax,
        first_dims=split_sizes,
        tensor_offsets=tensor_offsets,
        with_gemm_swizzled_scales=with_gemm_swizzled_scales,
    )
    return grouped


def _nvfp4_logical_data_view(data: torch.Tensor) -> torch.Tensor:
    """View packed NVFP4 data with its logical K dimension for scale swizzling."""
    return data.as_strided(
        (data.shape[0], data.shape[1] * 2),
        (data.stride(0), 0),
    )


def _nvfp4_amax(tensors: Any, *, columnwise: bool) -> torch.Tensor:
    """Get one NVFP4 amax value per group."""
    grouped_attr = "columnwise_amax" if columnwise else "amax"
    tensor_attr = "_amax_columnwise" if columnwise else "_amax_rowwise"

    if hasattr(tensors, grouped_attr):
        amax = getattr(tensors, grouped_attr)
        if amax is None:
            raise RuntimeError(f"NVFP4 GroupedTensor is missing {grouped_attr}.")
        return amax.view(-1)

    amaxes = [getattr(tensor, tensor_attr, None) for tensor in tensors]
    if any(amax is None for amax in amaxes):
        raise RuntimeError(f"NVFP4 tensor list is missing {tensor_attr}.")
    return torch.cat([amax.view(-1) for amax in amaxes], dim=0)


def _get_first_grad_output_quantizer(ctx: OperationContext):
    """Return grad-output quantizer from plural or singular context state."""
    quantizers = getattr(ctx, "grad_output_quantizers", None)
    if quantizers is not None:
        return quantizers[0]
    return getattr(ctx, "grad_output_quantizer", None)


def _nvfp4_single_tensor_from_grouped(grouped: Any) -> NVFP4Tensor:
    """Build a single NVFP4Tensor view over a one-member grouped storage."""
    quantizer = grouped.quantizer
    if not isinstance(quantizer, NVFP4Quantizer):
        raise TypeError("Expected an NVFP4 GroupedTensor.")
    dtype = grouped.dtype if isinstance(grouped, torch.Tensor) else grouped.get_dtype()

    shape = tuple(grouped.logical_shape)
    rowwise_data = None
    if grouped.rowwise_data is not None:
        rowwise_data = grouped.rowwise_data.view(quantizer.convert_shape_for_fp4(shape))

    rowwise_scale_inv = None
    if grouped.scale_inv is not None:
        rowwise_scale_inv = grouped.scale_inv.view(quantizer.get_scale_shape(shape, False))

    columnwise_data = None
    if grouped.columnwise_data is not None:
        columnwise_shape = quantizer.get_columnwise_shape(shape)
        columnwise_data = grouped.columnwise_data.view(
            quantizer.convert_shape_for_fp4(columnwise_shape)
        )

    columnwise_scale_inv = None
    if grouped.columnwise_scale_inv is not None:
        columnwise_scale_inv = grouped.columnwise_scale_inv.view(
            quantizer.get_scale_shape(shape, True)
        )

    return NVFP4Tensor(
        shape=shape,
        dtype=dtype,
        rowwise_data=rowwise_data,
        rowwise_scale_inv=rowwise_scale_inv,
        columnwise_data=columnwise_data,
        columnwise_scale_inv=columnwise_scale_inv,
        amax_rowwise=grouped.amax,
        amax_columnwise=grouped.columnwise_amax,
        fp4_dtype=quantizer.dtype,
        quantizer=quantizer,
        requires_grad=False,
        with_gemm_swizzled_scales=getattr(
            grouped,
            "_with_gemm_swizzled_scales",
            getattr(grouped, "with_gemm_swizzled_scales", quantizer.optimize_for_gemm),
        ),
    )


def _nvfp4_single_group_wgrad_gemm(
    grouped_x: GroupedTensor,
    grouped_dy: GroupedTensor,
    wgrad_output,
    *,
    weight_shape: tuple[int, int],
    accumulate: bool,
) -> None:
    """Run one-group NVFP4 wgrad with regular GEMM instead of grouped GEMM."""
    x_single = _nvfp4_single_tensor_from_grouped(grouped_x)
    dy_single = _nvfp4_single_tensor_from_grouped(grouped_dy)
    if isinstance(wgrad_output, GroupedTensor):
        out = wgrad_output.rowwise_data.view(1, *weight_shape)[0]
    else:
        out = wgrad_output[0]

    general_gemm(
        x_single,
        dy_single,
        out_dtype=out.dtype,
        out=out,
        layout="NT",
        accumulate=accumulate,
        use_split_accumulator=_2X_ACC_WGRAD,
    )


def _cudnn_compute_wgrad(
    grouped_x: GroupedTensor,
    grouped_dy: GroupedTensor,
    wgrad_output,
    weight_shape: tuple,
    offsets: torch.Tensor,
    accumulate: bool,
    wgrad_kernel_fn,
    single_grouped_weight: bool,
    current_stream=None,
):
    """Compute wgrad using the cuDNN CuTe DSL grouped GEMM wgrad kernel.

    The cuDNN wgrad kernel computes:
        wgrad[e] = a[:, tok_start:tok_end] @ b[tok_start:tok_end, :]
    where a = DY^T = (out_features, total_tokens) row-major and
          b = X  = (total_tokens, in_features) column-major.
    """
    out_features, in_features = weight_shape
    total_tokens = grouped_dy.logical_shape[0]

    use_nvfp4 = isinstance(getattr(grouped_x, "quantizer", None), NVFP4Quantizer) or isinstance(
        getattr(grouped_dy, "quantizer", None), NVFP4Quantizer
    )
    data_dtype = torch.float4_e2m1fn_x2 if use_nvfp4 else torch.float8_e4m3fn
    scale_view_dtype = torch.float8_e4m3fn if use_nvfp4 else torch.float8_e8m0fnu
    sf_vec_size = NVFP4_BLOCK_SCALING_SIZE if use_nvfp4 else MXFP8_BLOCK_SCALING_SIZE

    if use_nvfp4:
        # NVFP4 columnwise data is stored expert-major as per-expert
        # (logical_K, group_M / 2) chunks. cuDNN consumes that layout directly
        # with input_order="tensor_ragged".
        a_tensor = grouped_dy.columnwise_data.view(dtype=data_dtype).view(
            out_features, total_tokens // 2
        )
        b_tensor = grouped_x.columnwise_data.view(dtype=data_dtype).view(
            in_features, total_tokens // 2
        ).T
    else:
        # a_tensor = DY^T = (out_features, total_tokens) row-major
        a_tensor = grouped_dy.columnwise_data.view(dtype=data_dtype).view(
            total_tokens, out_features
        ).T
        # b_tensor = X = (total_tokens, in_features) column-major
        b_tensor = grouped_x.columnwise_data.view(dtype=data_dtype).view(
            total_tokens, in_features
        )

    sfa_leading_dim = ((out_features + 127) // 128) * 128
    sfb_leading_dim = ((in_features + 127) // 128) * 128
    sfa_tensor = grouped_dy.columnwise_scale_inv.view(sfa_leading_dim, -1).view(
        dtype=scale_view_dtype
    )
    sfb_tensor = grouped_x.columnwise_scale_inv.view(sfb_leading_dim, -1).view(
        dtype=scale_view_dtype
    )

    global_scale_a = None
    global_scale_b = None
    if use_nvfp4:
        global_scale_denom = 448.0 * 6.0
        global_scale_a = (
            grouped_dy.columnwise_amax.view(-1).to(torch.float32) / global_scale_denom
        )
        global_scale_b = (
            grouped_x.columnwise_amax.view(-1).to(torch.float32) / global_scale_denom
        )

    common_wgrad_kwargs = {
        "a_tensor": a_tensor,
        "b_tensor": b_tensor,
        "sfa_tensor": sfa_tensor,
        "sfb_tensor": sfb_tensor,
        "offsets_tensor": offsets,
        "global_scale_a": global_scale_a,
        "global_scale_b": global_scale_b,
        "acc_dtype": torch.float32,
        "sf_vec_size": sf_vec_size,
        "accumulate_on_output": accumulate,
        "current_stream": current_stream,
    }
    if use_nvfp4:
        common_wgrad_kwargs["input_order"] = "tensor_ragged"

    # Prepare wgrad output
    if single_grouped_weight:
        # Dense mode: single (num_groups, out_features, in_features) tensor
        wgrad_tensor = wgrad_output.rowwise_data.view(offsets.shape[0], out_features, in_features)
        wgrad_kernel_fn(
            **common_wgrad_kwargs,
            output_mode="dense",
            wgrad_tensor=wgrad_tensor,
            wgrad_dtype=wgrad_tensor.dtype,
        )
    else:
        # Discrete mode: per-expert wgrad device pointers
        (wgrad_ptrs,) = tex.convert_host_pointers_to_tensor([wgrad_output])
        wgrad_kernel_fn(
            **common_wgrad_kwargs,
            output_mode="discrete",
            wgrad_ptrs=wgrad_ptrs,
            wgrad_dtype=wgrad_output[0].dtype,
        )


@functools.lru_cache(maxsize=None)
def _wrapper_has_arg(wrapper_name: str, arg_name: str) -> bool:
    """True if a cudnn-frontend SM100 wrapper accepts an argument."""
    try:
        import cudnn  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False
    try:
        wrapper = getattr(cudnn, wrapper_name)
        params = inspect.signature(wrapper).parameters
    except (AttributeError, TypeError, ValueError):
        return False
    return arg_name in params


def _wrapper_has_generate_dbias_arg(wrapper_name: str) -> bool:
    """True if a cudnn-frontend SM100 wrapper accepts ``generate_dbias``."""
    return _wrapper_has_arg(wrapper_name, "generate_dbias")


def _dglu_wrapper_has_generate_dbias_arg() -> bool:
    """True if cudnn-frontend SM100 dGLU wrapper accepts ``generate_dbias``."""
    return _wrapper_has_generate_dbias_arg("grouped_gemm_dglu_wrapper_sm100")


def _dsrelu_wrapper_has_generate_dbias_arg() -> bool:
    """True if cudnn-frontend SM100 dSReLU wrapper accepts ``generate_dbias``."""
    return _wrapper_has_generate_dbias_arg("grouped_gemm_dsrelu_wrapper_sm100")


def _dsrelu_wrapper_has_reuse_arg() -> bool:
    """True if cudnn-frontend SM100 dSReLU wrapper accepts ``use_dsrelu_reuse``."""
    return _wrapper_has_arg("grouped_gemm_dsrelu_wrapper_sm100", "use_dsrelu_reuse")


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _compute_grad_params(
    fc_op,
    ctx,
    num_groups,
    weight_shape,
    grouped_x,
    grouped_dy,
    dtype,
    device,
    bias_grads,
    bias_grad_packed,
    label="",
    *,
    cudnn_wgrad_kernel_fn,
    offsets,
):
    """Compute weight gradients and build grad_params for a GroupedLinear layer.
    Returns the grad_params list in parameter registration order.
    """

    # Allocate grad buffers, determine accumulate flag
    accumulate_into_main_grad = False
    grouped_wgrad = None
    wgrad_output = None
    if fc_op.single_grouped_weight:
        w_list = [None]
        if ctx.weight_requires_grad:
            weight_param = fc_op.weight
            if fc_op._accumulate_into_main_grad:
                if hasattr(weight_param, "__fsdp_param__"):
                    weight_param.main_grad = weight_param.get_main_grad()
                main_grad = weight_param.main_grad
                grouped_shape = (num_groups, *weight_shape)
                if main_grad.shape != grouped_shape:
                    if main_grad.numel() != math.prod(grouped_shape):
                        raise RuntimeError(
                            f"Grouped MLP fused backward expected {label} main_grad to have "
                            f"shape {grouped_shape} or matching numel, "
                            f"but got shape {tuple(main_grad.shape)}"
                        )
                    try:
                        main_grad = main_grad.view(grouped_shape)
                    except RuntimeError as e:
                        raise RuntimeError(
                            f"Grouped MLP fused backward requires {label} main_grad to be "
                            f"viewable as {grouped_shape} without copy, but got shape"
                            f" {tuple(main_grad.shape)} and stride"
                            f" {tuple(main_grad.stride())}"
                        ) from e
                accumulate_into_main_grad = not getattr(weight_param, "overwrite_main_grad", False)
                if accumulate_into_main_grad:
                    grouped_wgrad = GroupedTensor.make_grouped_tensor_from_rowwise_data(
                        num_tensors=num_groups,
                        tensor_shape=weight_shape,
                        rowwise_data=main_grad,
                        dtype=main_grad.dtype,
                    )

            if grouped_wgrad is None:
                grouped_wgrad = GroupedTensor.make_grouped_tensor_with_shapes(
                    num_tensors=num_groups,
                    shapes=[weight_shape] * num_groups,
                    quantizer=None,
                    device=device,
                    dtype=dtype,
                )
            wgrad_output = grouped_wgrad
    else:
        w_list = [None] * num_groups
        if ctx.weight_requires_grad:
            if fc_op._accumulate_into_main_grad:
                for idx in range(num_groups):
                    wp = getattr(fc_op, f"weight{idx}")
                    if hasattr(wp, "__fsdp_param__"):
                        wp.main_grad = wp.get_main_grad()
                    w_list[idx] = wp.main_grad
                accumulate_into_main_grad = not getattr(fc_op.weight0, "overwrite_main_grad", False)
            else:
                for idx in range(num_groups):
                    w_list[idx] = torch.empty(weight_shape, dtype=dtype, device=device)
            wgrad_output = w_list

    if ctx.weight_requires_grad:
        # Launch or defer the GEMM
        delay_wgrad = fc_op.wgrad_store is not None and fc_op.wgrad_store.delay_wgrad_compute()
        if cudnn_wgrad_kernel_fn is not None:
            offsets = offsets if offsets.dtype == torch.int32 else offsets.to(dtype=torch.int32)
            gemm_fn = functools.partial(
                _cudnn_compute_wgrad,
                weight_shape=weight_shape,
                offsets=offsets,
                accumulate=accumulate_into_main_grad,
                wgrad_kernel_fn=cudnn_wgrad_kernel_fn,
                single_grouped_weight=fc_op.single_grouped_weight,
                current_stream=torch.cuda.current_stream().cuda_stream,
            )
        elif (
            num_groups == 1
            and isinstance(grouped_x, GroupedTensor)
            and isinstance(grouped_dy, GroupedTensor)
            and isinstance(getattr(grouped_x, "quantizer", None), NVFP4Quantizer)
            and isinstance(getattr(grouped_dy, "quantizer", None), NVFP4Quantizer)
        ):
            gemm_fn = functools.partial(
                _nvfp4_single_group_wgrad_gemm,
                weight_shape=weight_shape,
                accumulate=accumulate_into_main_grad,
            )
        else:
            gemm_fn = functools.partial(
                general_grouped_gemm_for_grouped_tensor,
                layout="NT",
                accumulate=accumulate_into_main_grad,
                use_split_accumulator=_2X_ACC_WGRAD,
            )

        if delay_wgrad:
            fc_op.wgrad_store.put([grouped_x, grouped_dy, wgrad_output], gemm_fn)
        else:
            gemm_fn(grouped_x, grouped_dy, wgrad_output)

        # Extract results, mark accumulated if needed
        if fc_op.single_grouped_weight:
            packed_wgrad = None
            if not delay_wgrad:
                packed_wgrad = grouped_wgrad.rowwise_data.view(num_groups, *weight_shape)
            if accumulate_into_main_grad and hasattr(weight_param, "grad_added_to_main_grad"):
                weight_param.grad_added_to_main_grad = True
                packed_wgrad = get_dummy_wgrad(
                    list(weight_param.size()),
                    weight_param.dtype,
                    zero=getattr(weight_param, "zero_out_wgrad", False),
                )
            w_list = [packed_wgrad]
        else:
            if delay_wgrad or accumulate_into_main_grad:
                w_list = [None] * num_groups
            if accumulate_into_main_grad:
                for idx in range(num_groups):
                    wp = getattr(fc_op, f"weight{idx}")
                    if hasattr(wp, "grad_added_to_main_grad"):
                        wp.grad_added_to_main_grad = True
                        w_list[idx] = get_dummy_wgrad(
                            list(wp.size()),
                            wp.dtype,
                            zero=getattr(wp, "zero_out_wgrad", False),
                        )

    # Assemble grad_params in parameter registration order.
    if not fc_op.has_bias:
        return w_list

    if fc_op.single_grouped_bias:
        return w_list + [bias_grad_packed]

    bias_list = bias_grads if bias_grads is not None else [None] * num_groups
    if fc_op.single_grouped_weight:
        return bias_list + w_list
    return w_list + bias_list


class BackwardGroupedMLP_CuTeGEMMDSwiGLU_MXFP8(FusedOperation):
    """Fused op for MXFP8 GroupedLinear + ScaledSwiGLU or ScaledClampedQGeGLU + GroupedLinear

    Uses experimental CuTe DSL kernel from cuDNN front-end.

    """

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_dglu_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM, GLU activation backward, and scale grad."""
        from cudnn import grouped_gemm_dglu_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_dglu_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_quant_kernel(cls) -> Callable:
        """Grouped GEMM quant kernel for block-scaled inputs."""
        from cudnn import grouped_gemm_quant_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_quant_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_wgrad_kernel(cls) -> Optional[Callable]:
        """CuTe DSL kernel for grouped GEMM wgrad on SM100+.
        Returns ``None`` when the cuDNN front-end package is older than
        1.23.0.
        """
        if not _nvidia_cudnn_frontend_supports_wgrad():
            return None
        from cudnn import grouped_gemm_wgrad_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_wgrad_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether this fused operation is supported on the current system."""
        if int(os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "0")) <= 0:
            return False
        if get_device_compute_capability()[0] != 10:
            return False
        try:
            cls.grouped_gemm_dglu_kernel()
            cls.grouped_gemm_quant_kernel()
        except ImportError:
            return False
        return True

    @classmethod
    def is_fc1_bias_supported(cls) -> bool:
        """Whether cudnn-frontend exposes ``generate_dbias`` on the dGLU SM100 wrapper (FC1 bias grad only)."""
        if not cls.is_supported():
            return False
        return _dglu_wrapper_has_generate_dbias_arg()

    def __init__(
        self,
        *,
        fc1: GroupedLinear,
        swiglu: Optional[ScaledSwiGLU | ScaledClampedQGeGLU] = None,
        srelu: Optional[ScaledSReLU] = None,
        fc2: GroupedLinear,
    ) -> None:
        if swiglu is not None and srelu is not None:
            raise TypeError(
                "Expected exactly one activation op, but both swiglu and srelu were provided."
            )
        activation = swiglu if swiglu is not None else srelu
        if activation is None:
            raise TypeError("Expected a grouped MLP activation op.")
        super().__init__((fc1, activation, fc2))
        if not self.is_supported():
            self.grouped_gemm_dglu_kernel()  # Try triggering import error
            raise RuntimeError(f"{self.__class__.__name__} is not supported on this system.")
        validate_grouped_mlp_dims(fc1, activation, fc2)
        if isinstance(activation, ScaledSReLU):
            # grouped_gemm_dsrelu_wrapper_sm100 is dSReLU-specific and does not
            # take the GLU ``act_func`` selector.
            self._cudnn_dact_func: Optional[str] = None
        else:
            # The cuDNN dgeglu implementation corresponds to ScaledClampedQGeGLU.
            # The act_func string should be fixed on the cuDNN FE side.
            self._cudnn_dact_func = (
                "dgeglu" if isinstance(activation, ScaledClampedQGeGLU) else "dswiglu"
            )

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        **unused,  # pylint: disable=unused-argument
    ) -> tuple[
        torch.Tensor,
        list[tuple[Optional[torch.Tensor], ...]],
        list[tuple[()]],
    ]:

        # Get basic operations
        fc1_op, _, fc2_op = self.basic_ops
        fc1_ctx, activation_ctx, fc2_ctx = basic_op_ctxs

        # Tensor properties
        fc1_weight_shape = (fc1_op.out_features, fc1_op.in_features)
        fc2_weight_shape = (fc2_op.out_features, fc2_op.in_features)
        grad_output = grad_output.reshape(-1, fc2_weight_shape[0])
        out_shape = list(grad_output.size())
        num_groups = fc1_op.num_groups
        fc1_weight_param = fc1_op.weight if fc1_op.single_grouped_weight else fc1_op.weight0
        fc2_weight_param = fc2_op.weight if fc2_op.single_grouped_weight else fc2_op.weight0
        device = fc1_weight_param.device
        dtype = fc1_ctx.dtype

        # Saved tensors from FC1 forward
        saved_tensors = fc1_ctx.saved_tensors
        split_sizes, split_points, saved_tensors = (
            saved_tensors[0],
            saved_tensors[1],
            saved_tensors[2:],
        )

        if fc1_op.single_grouped_weight:
            grouped_fc1_weight, saved_tensors = saved_tensors[0], saved_tensors[1:]
        else:
            grouped_fc1_weight, saved_tensors = (
                saved_tensors[:num_groups],
                saved_tensors[num_groups:],
            )

        (
            fc1_x_data,
            fc1_x_col_data,
            fc1_x_scale,
            fc1_x_col_scale,
            fc1_x_tensor_offsets,
            fc1_x_amax,
            fc1_x_col_amax,
        ), saved_tensors = (
            saved_tensors[:7],
            saved_tensors[7:],
        )

        # Saved tensors from activation forward
        swiglu_in, scales = activation_ctx.saved_tensors
        recompute_fc2_x_from_dsrelu = bool(
            getattr(fc2_ctx, "recompute_input_from_dsrelu", False)
        ) and bool(fc2_ctx.weight_requires_grad)

        # Saved tensors from FC2 forward
        saved_tensors = fc2_ctx.saved_tensors
        _, saved_tensors = saved_tensors[0], saved_tensors[1:]  # Assume same split sizes as FC1
        if fc2_op.single_grouped_weight:
            grouped_fc2_weight, saved_tensors = saved_tensors[0], saved_tensors[1:]
        else:
            grouped_fc2_weight, saved_tensors = (
                saved_tensors[:num_groups],
                saved_tensors[num_groups:],
            )

        (
            fc2_x_data,
            fc2_x_col_data,
            fc2_x_scale,
            fc2_x_col_scale,
            fc2_x_tensor_offsets,
            fc2_x_amax,
            fc2_x_col_amax,
        ), saved_tensors = (
            saved_tensors[:7],
            saved_tensors[7:],
        )

        # Group splits
        if int(split_sizes.numel()) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {int(split_sizes.numel())}.")
        scale_bias = fc2_op._scale_bias and fc2_op.has_bias

        grouped_fc1_x = None
        if fc1_ctx.weight_requires_grad:
            grouped_fc1_x = GroupedTensor(
                shape=(out_shape[0], fc1_weight_shape[1]),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=fc1_ctx.input_quantizer,
                data=fc1_x_data,
                columnwise_data=fc1_x_col_data,
                scale_inv=fc1_x_scale,
                columnwise_scale_inv=fc1_x_col_scale,
                amax=fc1_x_amax,
                columnwise_amax=fc1_x_col_amax,
                first_dims=split_sizes,
                tensor_offsets=fc1_x_tensor_offsets,
                with_gemm_swizzled_scales=True,
            )

        grouped_fc2_x = None
        if fc2_ctx.weight_requires_grad and not recompute_fc2_x_from_dsrelu:
            grouped_fc2_x = GroupedTensor(
                shape=(out_shape[0], fc2_weight_shape[1]),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=fc2_ctx.input_quantizer,
                data=fc2_x_data,
                columnwise_data=fc2_x_col_data,
                scale_inv=fc2_x_scale,
                columnwise_scale_inv=fc2_x_col_scale,
                amax=fc2_x_amax,
                columnwise_amax=fc2_x_col_amax,
                first_dims=split_sizes,
                tensor_offsets=fc2_x_tensor_offsets,
                with_gemm_swizzled_scales=True,
            )

        # Split grad output tensor and convert dtypes if needed
        fc2_grad_output_quantizer = _get_first_grad_output_quantizer(fc2_ctx)
        fc2_grad_output_quantizer.set_usage(rowwise=True, columnwise=fc2_ctx.weight_requires_grad)
        fc2_grad_output_quantizer.optimize_for_gemm = True
        _enable_nvfp4_rht_for_group_quantize(fc2_grad_output_quantizer)
        output_fc2_dbias = fc2_op.has_bias
        fc2_dbias_packed = None
        fc2_dy = None
        if (
            not output_fc2_dbias
            and isinstance(grad_output, GroupedTensor)
            and isinstance(getattr(grad_output, "quantizer", None), MXFP8Quantizer)
        ):
            grouped_fc2_dy = grad_output
        else:
            fc2_dy = maybe_dequantize(grad_output, dtype)
            if output_fc2_dbias and not scale_bias:
                grouped_fc2_dy, fc2_dbias_packed = tex.bgrad_group_quantize(
                    fc2_dy,
                    fc2_grad_output_quantizer,
                    num_groups,
                    split_sizes,
                )
            else:
                grouped_fc2_dy = _group_quantize_for_grouped_mlp(
                    fc2_dy,
                    fc2_grad_output_quantizer,
                    num_groups,
                    split_sizes,
                    tensor_offsets=fc1_ctx.base_split_offsets * fc2_weight_shape[0],
                )

        # NVFP4 vs MXFP8 data layout constants
        use_nvfp4 = (
            isinstance(fc2_grad_output_quantizer, NVFP4Quantizer)
            or type(fc1_weight_param).__name__ == "NVFP4Tensor"
            or type(fc2_weight_param).__name__ == "NVFP4Tensor"
        )
        data_dtype = torch.float4_e2m1fn_x2 if use_nvfp4 else torch.float8_e4m3fn
        scale_view_dtype = torch.float8_e4m3fn if use_nvfp4 else torch.float8_e8m0fnu
        sf_vec_size = NVFP4_BLOCK_SCALING_SIZE if use_nvfp4 else MXFP8_BLOCK_SCALING_SIZE
        # NVFP4 byte-packs the K dimension (two FP4 values per byte).
        data_k = out_shape[1] // 2 if use_nvfp4 else out_shape[1]
        fc2_weight_k = fc2_weight_shape[1] // 2 if use_nvfp4 else fc2_weight_shape[1]
        # Number of FP4/FP8 values represented by one block scale along K.
        # For MXFP8: 4 * 32 = 128 (matches the 128-block tiling).
        # For NVFP4: 2 * 16 = 32 logical values = 16 byte-packed columns.
        k_sf_divisor = 2 * sf_vec_size if use_nvfp4 else 4 * sf_vec_size
        # Pack data tensors
        # Note: Fused kernel expects tensor with non-contiguous
        # logical dims.
        # Data actual shape: (1, sum(m), k)
        # Scale actual shape: (1, sum(m)/128, k/128, 32 (block row),
        #  4 (block row), 4 (block col))
        # Data logical shape: (sum(m), k, 1)
        # Scale logical shape: (32 (block row), 4 (block row),
        #   sum(m)/128, 4 (block col), k/128, 1)
        # For NVFP4, rowwise_data is byte-packed along K (K/2 storage).
        fc2_dy_data = grouped_fc2_dy.rowwise_data.view(dtype=data_dtype)
        fc2_dy_data = fc2_dy_data.view(out_shape[0], data_k)
        fc2_dy_data = fc2_dy_data.unsqueeze(0).permute(1, 2, 0)
        fc2_dy_scales = grouped_fc2_dy.scale_inv
        fc2_dy_scales = fc2_dy_scales.view(dtype=scale_view_dtype)
        with_gemm_swizzled_scales = getattr(
            grouped_fc2_dy,
            "_with_gemm_swizzled_scales",
            getattr(grouped_fc2_dy, "with_gemm_swizzled_scales", False),
        )
        if use_nvfp4 and with_gemm_swizzled_scales:
            fc2_dy_scales = fc2_dy_scales.view(
                1,
                out_shape[0] // 128,
                data_k // k_sf_divisor,
                32,
                4,
                4,
            )
            fc2_dy_scales = fc2_dy_scales.permute(3, 4, 1, 5, 2, 0)
        elif use_nvfp4 and not with_gemm_swizzled_scales:
            fc2_dy_scales = fc2_dy_scales.view(
                1,
                out_shape[0] // 128,
                4,
                32,
                data_k // k_sf_divisor,
                4,
            )
            fc2_dy_scales = fc2_dy_scales.permute(3, 2, 1, 5, 4, 0)
        else:
            fc2_dy_scales = fc2_dy_scales.view(
                1,
                (out_shape[0] + 127) // 128,
                (out_shape[1] + k_sf_divisor - 1) // k_sf_divisor,
                32,
                4,
                4,
            )
            fc2_dy_scales = fc2_dy_scales.permute(3, 4, 1, 5, 2, 0)

        # Kernel scaling factors
        alpha_tensor = get_cached_ones_tensor(num_groups, torch.float32, device)
        norm_const_tensor = get_cached_ones_tensor(1, torch.float32, device)
        current_stream = torch.cuda.current_stream().cuda_stream

        if scales is None:
            scales_f32 = None
            scales_tensor = torch.ones(
                (out_shape[0], 1, 1),
                dtype=torch.float32,
                device=device,
            )
            dscales_tensor = None
        else:
            scales_f32 = scales.detach().to(dtype=torch.float32)
            scales_tensor = scales_f32.reshape(-1, 1, 1)
            dscales_tensor = torch.zeros_like(scales_tensor)

        fc2_d_dtype = torch.bfloat16 if use_nvfp4 else torch.float8_e4m3fn
        if use_nvfp4:
            # Dequantize the GEMM accumulator (dy_fp4 @ W_fp4^T).
            _amax_fc2_dy = _nvfp4_amax(grouped_fc2_dy, columnwise=False)
            _amax_fc2_w_col = _nvfp4_amax(grouped_fc2_weight, columnwise=True)
            _nvfp4_fp4_max = 6.0
            _nvfp4_fp8_max = 448.0
            fc2_alpha_tensor = (
                torch.sqrt(_amax_fc2_dy * _amax_fc2_w_col) / (_nvfp4_fp8_max * _nvfp4_fp4_max)
            ).expand(num_groups)
            fc2_beta_tensor = get_cached_ones_tensor(num_groups, torch.float32, device)
            fc2_norm_const_tensor = None
        else:
            fc2_alpha_tensor = alpha_tensor
            fc2_beta_tensor = fc2_alpha_tensor
            fc2_norm_const_tensor = norm_const_tensor

        fc2_dglu_kwargs = {
            "a_tensor": fc2_dy_data,
            "c_tensor": swiglu_in.unsqueeze(0).permute(1, 2, 0),
            "sfa_tensor": fc2_dy_scales,
            "padded_offsets": split_points,
            "alpha_tensor": fc2_alpha_tensor,
            "prob_tensor": scales_tensor,
            "dprob_tensor": dscales_tensor,
            "generate_dbias": fc1_op.has_bias,
            "norm_const_tensor": fc2_norm_const_tensor,
            "d_dtype": fc2_d_dtype,
            "cd_major": "n",
            "sf_vec_size": sf_vec_size,
            "current_stream": current_stream,
            "discrete_col_sfd": not use_nvfp4,
            "use_dynamic_sched": True,
        }
        if self._cudnn_dact_func is not None:
            fc2_dglu_kwargs["beta_tensor"] = fc2_beta_tensor
            fc2_dglu_kwargs["act_func"] = self._cudnn_dact_func
        elif _dsrelu_wrapper_has_reuse_arg():
            fc2_dglu_kwargs["use_dsrelu_reuse"] = _env_flag(
                "NVTE_CUTEDSL_FUSED_GROUPED_MLP_DSRELU_REUSE",
            )

        if fc2_op.single_grouped_weight:
            # Clone and swizzle scales for GEMM
            fc2_weight_for_gemm = grouped_fc2_weight.copy()
            tex.grouped_swizzle_for_gemm(fc2_weight_for_gemm, rowwise=False, columnwise=True)
            # Pack weight tensors for stacked kernel
            # Data actual shape: (num_groups, k, n)
            # Data logical shape: (n, k, num_groups)
            fc2_w_data = fc2_weight_for_gemm.columnwise_data
            fc2_w_data = fc2_w_data.view(dtype=data_dtype)
            fc2_w_data = fc2_w_data.view(num_groups, fc2_weight_shape[0], fc2_weight_k)
            # NVFP4 columnwise data is stored already transposed relative to
            # MXFP8, so we swap the M / K logical axes for the wrapper.
            fc2_w_data = fc2_w_data.permute(1, 2, 0) if use_nvfp4 else fc2_w_data.permute(2, 1, 0)
            fc2_w_scales = fc2_weight_for_gemm.columnwise_scale_inv.view(dtype=scale_view_dtype)
            fc2_w_scales = fc2_w_scales.view(
                num_groups,
                (fc2_weight_shape[1] + k_sf_divisor - 1) // k_sf_divisor,
                (fc2_weight_shape[0] + 127) // 128,
                32,
                4,
                4,
            )
            # NVFP4 columnwise SFB is laid out with M and N axes swapped
            # relative to MXFP8.
            fc2_w_scales = (
                fc2_w_scales.permute(3, 4, 2, 5, 1, 0)
                if use_nvfp4
                else fc2_w_scales.permute(3, 4, 1, 5, 2, 0)
            )

            fc2_dglu_kwargs["b_tensor"] = fc2_w_data
            fc2_dglu_kwargs["sfb_tensor"] = fc2_w_scales
        else:
            fc2_weight_data_for_ptrs = [w._columnwise_data for w in grouped_fc2_weight]
            if use_nvfp4:
                fc2_weight_data_for_ptrs = [
                    _nvfp4_logical_data_view(data) for data in fc2_weight_data_for_ptrs
                ]
            fc2_b_ptrs, fc2_sfb_ptrs, _fc2_sw = tex.get_device_pointer_for_data_and_scales(
                fc2_weight_data_for_ptrs,
                [w._columnwise_scale_inv for w in grouped_fc2_weight],
                swizzle=True,
                rowwise=False,
                data_dtype=(
                    grouped_fc2_weight[0]._fp4_dtype
                    if use_nvfp4
                    else grouped_fc2_weight[0]._fp8_dtype
                ),
            )
            fc2_dglu_kwargs["b_ptrs"] = fc2_b_ptrs
            fc2_dglu_kwargs["sfb_ptrs"] = fc2_sfb_ptrs
            fc2_dglu_kwargs["n"] = fc2_weight_shape[1]
            fc2_dglu_kwargs["b_dtype"] = data_dtype
            fc2_dglu_kwargs["b_major"] = "k" if use_nvfp4 else "n"

        fc2_dgrad_kernel_out = self.grouped_gemm_dglu_kernel()(**fc2_dglu_kwargs)

        if use_nvfp4:
            # NVFP4 path: kernel outputs unquantized BF16. Re-quantize to NVFP4
            # to obtain a properly-formatted GroupedTensor for both FC1 dgrad
            # and FC1 wgrad.
            fc1_dy_bf16 = fc2_dgrad_kernel_out["d_row_tensor"]
            fc1_dy_bf16 = fc1_dy_bf16.view(out_shape[0], fc1_weight_shape[0]).contiguous()
            fc1_dy_row_data = None
            fc1_dy_row_scale = None
            fc1_dy_col_data = None
            fc1_dy_col_scale = None
        else:
            fc1_dy_bf16 = None
            fc1_dy_row_data = fc2_dgrad_kernel_out["d_row_tensor"]
            fc1_dy_row_data = fc1_dy_row_data.view(out_shape[0], fc1_weight_shape[0])
            # View scale in their actual swizzled shape
            fc1_dy_row_scale = (
                fc2_dgrad_kernel_out["sfd_row_tensor"].permute(5, 2, 4, 0, 1, 3).view(-1)
            )
            fc1_dy_col_data = fc2_dgrad_kernel_out["d_col_tensor"]
            fc1_dy_col_data = fc1_dy_col_data.view(out_shape[0], fc1_weight_shape[0])
            # View scale in their actual swizzled shape
            fc1_dy_col_scale = (
                fc2_dgrad_kernel_out["sfd_col_tensor"].permute(5, 2, 4, 0, 1, 3).view(-1)
            )
        grad_scales = (
            fc2_dgrad_kernel_out["dprob_tensor"].view(-1) if scales is not None else None
        )

        if recompute_fc2_x_from_dsrelu:
            d_srelu_tensor = fc2_dgrad_kernel_out.get("d_srelu_tensor")
            if d_srelu_tensor is None:
                raise RuntimeError(
                    "SReLU recompute is enabled, but the DSReLU kernel did not return "
                    "the recomputed FC2 input tensor."
                )

            recomputed_fc2_x_offsets = fc2_x_tensor_offsets
            if recomputed_fc2_x_offsets is None:
                recomputed_fc2_x_offsets = fc1_ctx.base_split_offsets * fc2_weight_shape[1]

            sfd_col_d_srelu_tensor = fc2_dgrad_kernel_out.get("sfd_col_d_srelu_tensor")
            if sfd_col_d_srelu_tensor is None:
                raise RuntimeError(
                    "SReLU recompute is enabled, but the DSReLU kernel did not return "
                    "the recomputed FC2 input column scale tensor."
                )

            fc2_x_col_data = d_srelu_tensor.view(out_shape[0], fc2_weight_shape[1])
            fc2_x_col_scale = sfd_col_d_srelu_tensor.permute(5, 2, 4, 0, 1, 3)
            grouped_fc2_x = GroupedTensor(
                shape=(out_shape[0], fc2_weight_shape[1]),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=fc2_ctx.input_quantizer,
                data=None,
                columnwise_data=fc2_x_col_data.reshape(-1),
                scale_inv=None,
                columnwise_scale_inv=fc2_x_col_scale.reshape(-1),
                first_dims=split_sizes,
                tensor_offsets=recomputed_fc2_x_offsets,
                with_gemm_swizzled_scales=True,
            )

        fc2_bias_grads: Optional[list[Optional[torch.Tensor]]] = None
        fc2_bias_grad_packed: Optional[torch.Tensor] = None
        if scale_bias:
            if scales_f32 is None:
                raise RuntimeError("FC2 scale bias requires activation scales.")
            fc2_biases = fc2_op._get_bias_tensors(dtype)
            bias_packed = torch.stack(fc2_biases)
            fc2_dbias_packed_result, grad_scales = compute_grouped_dbias_dscales(
                fc2_dy,
                scales_f32,
                bias_packed,
                offsets=fc1_ctx.base_split_offsets,
                dscales=grad_scales,
            )
            fc2_dbias_packed_result = fc2_dbias_packed_result.to(dtype=dtype)
            if fc2_op.single_grouped_bias:
                fc2_bias_grad_packed = fc2_dbias_packed_result
            else:
                fc2_bias_grads = [fc2_dbias_packed_result[idx] for idx in range(num_groups)]
        elif fc2_dbias_packed is not None:
            fc2_dbias_packed = fc2_dbias_packed.to(dtype=dtype)
            if fc2_op.single_grouped_bias:
                fc2_bias_grad_packed = fc2_dbias_packed
            else:
                fc2_bias_grads = [fc2_dbias_packed[idx] for idx in range(num_groups)]

        if grad_scales is not None:
            grad_scales = grad_scales.to(dtype=dtype)

        fc1_bias_grads: Optional[list[Optional[torch.Tensor]]] = None
        fc1_bias_grad_packed: Optional[torch.Tensor] = None
        if fc1_op.has_bias:
            dbias_t = fc2_dgrad_kernel_out["dbias_tensor"]
            if dbias_t is not None:
                dbias_2d = dbias_t.squeeze(-1).to(dtype=dtype)
                if fc1_op.single_grouped_bias:
                    fc1_bias_grad_packed = dbias_2d
                else:
                    fc1_bias_grads = [dbias_2d[group_idx] for group_idx in range(num_groups)]

        # FC1 grad output for dgrad and wgrad GEMMs
        fc1_dy_tensor_offsets = fc1_ctx.base_split_offsets * fc1_weight_shape[0]
        if use_nvfp4:
            # Re-quantize the BF16 dGLU output into a GroupedTensor with both
            # rowwise (for dgrad) and columnwise (for wgrad) NVFP4 tiles.
            fc1_ctx.grad_output_quantizer.set_usage(
                rowwise=True, columnwise=fc1_ctx.weight_requires_grad
            )
            fc1_ctx.grad_output_quantizer.optimize_for_gemm = True
            _enable_nvfp4_rht_for_group_quantize(fc1_ctx.grad_output_quantizer)
            grouped_fc1_dy = _group_quantize_for_grouped_mlp(
                fc1_dy_bf16,
                fc1_ctx.grad_output_quantizer,
                num_groups,
                split_sizes,
                tensor_offsets=fc1_dy_tensor_offsets,
            )
            _mark_with_gemm_swizzled_scales(grouped_fc1_dy)
        else:
            grouped_fc1_dy = GroupedTensor(
                shape=(out_shape[0], fc1_weight_shape[0]),
                dtype=dtype,
                num_tensors=num_groups,
                quantizer=fc1_ctx.grad_output_quantizer,
                data=fc1_dy_row_data,
                columnwise_data=fc1_dy_col_data,
                scale_inv=fc1_dy_row_scale,
                columnwise_scale_inv=fc1_dy_col_scale,
                first_dims=split_sizes,
                tensor_offsets=fc1_dy_tensor_offsets,
                with_gemm_swizzled_scales=True,
            )

        # FC2/FC1 wgrad GEMM. NVFP4 cuDNN wgrad is opt-in through
        # NVTE_CUTEDSL_FUSED_GROUPED_MLP_NVFP4_WGRAD.
        nvfp4_cudnn_wgrad_mode = os.environ.get(
            "NVTE_CUTEDSL_FUSED_GROUPED_MLP_NVFP4_WGRAD",
            "0",
        ).lower()
        enable_nvfp4_fc2_wgrad = nvfp4_cudnn_wgrad_mode in (
            "1",
            "true",
            "all",
            "fc2",
        )
        enable_nvfp4_fc1_wgrad = nvfp4_cudnn_wgrad_mode in (
            "1",
            "true",
            "all",
            "fc1",
        )
        fc2_wgrad_kernel_fn = (
            self.grouped_gemm_wgrad_kernel()
            if (not use_nvfp4 or enable_nvfp4_fc2_wgrad)
            else None
        )
        fc2_grad_params = _compute_grad_params(
            fc_op=fc2_op,
            ctx=fc2_ctx,
            num_groups=num_groups,
            weight_shape=fc2_weight_shape,
            grouped_x=grouped_fc2_x,
            grouped_dy=grouped_fc2_dy,
            dtype=dtype,
            device=device,
            bias_grads=fc2_bias_grads,
            bias_grad_packed=fc2_bias_grad_packed,
            label="FC2",
            cudnn_wgrad_kernel_fn=fc2_wgrad_kernel_fn,
            offsets=split_points,
        )

        # Clear FC2 input tensor if possible
        if grouped_fc2_x is not None and not (
            fc2_ctx.weight_requires_grad
            and fc2_op.wgrad_store is not None
            and fc2_op.wgrad_store.delay_wgrad_compute()
        ):
            clear_tensor_data(
                grouped_fc2_x.data,
                grouped_fc2_x.columnwise_data,
                grouped_fc2_x.scale_inv,
                grouped_fc2_x.columnwise_scale_inv,
            )

        # FC1 dgrad GEMM
        grad_input = None
        if fc1_ctx.input_requires_grad:
            in_shape = out_shape[:-1] + [fc1_weight_shape[1]]

            if use_nvfp4:
                # NVFP4 dgrad uses the generic grouped GEMM wrapper since the
                # cuDNN quant wrapper expects FP8 inputs and packed scales.
                if not fc1_op.single_grouped_weight:
                    _pack_nvfp4_amax_list(grouped_fc1_weight)
                _mark_with_gemm_swizzled_scales(grouped_fc1_weight)
                _mark_with_gemm_swizzled_scales(grouped_fc1_dy)
                grad_input = torch.empty(in_shape, dtype=dtype, device=device)
                # NN layout: weight (n, k) @ dy (m, n) -> dx (m, k).
                if num_groups == 1:
                    if fc1_op.single_grouped_weight:
                        fc1_w_single = grouped_fc1_weight.split_into_quantized_tensors()[0]
                    else:
                        fc1_w_single = grouped_fc1_weight[0]
                    fc1_dy_single = _nvfp4_single_tensor_from_grouped(grouped_fc1_dy)
                    general_gemm(
                        fc1_w_single,
                        fc1_dy_single,
                        out_dtype=dtype,
                        out=grad_input,
                        layout="NN",
                    )
                else:
                    grouped_grad_input = GroupedTensor(
                        shape=(out_shape[0], fc1_weight_shape[1]),
                        dtype=dtype,
                        num_tensors=num_groups,
                        quantizer=None,
                        data=grad_input.view(-1),
                        first_dims=split_sizes,
                        tensor_offsets=fc1_x_tensor_offsets,
                    )
                    general_grouped_gemm_for_grouped_tensor(
                        grouped_fc1_weight,
                        grouped_fc1_dy,
                        grouped_grad_input,
                        layout="NN",
                    )
            else:
                fc1_dgrad_a_data = fc2_dgrad_kernel_out["d_row_tensor"]
                fc1_dgrad_a_scales = fc2_dgrad_kernel_out["sfd_row_tensor"]

                fc1_dgrad_kwargs = {
                    "a_tensor": fc1_dgrad_a_data,
                    "sfa_tensor": fc1_dgrad_a_scales,
                    "padded_offsets": split_points,
                    "alpha_tensor": alpha_tensor.float(),
                    "norm_const_tensor": None,
                    "prob_tensor": torch.ones(
                        (out_shape[0], 1, 1), dtype=torch.float32, device=device
                    ),
                    "acc_dtype": torch.float32,
                    "d_dtype": dtype,
                    "cd_major": "n",
                    "sf_vec_size": MXFP8_BLOCK_SCALING_SIZE,
                    "current_stream": current_stream,
                    "discrete_col_sfd": True,
                    "use_dynamic_sched": True,
                }

                if fc1_op.single_grouped_weight:
                    # Clone and swizzle scales for GEMM
                    fc1_weight_for_gemm = grouped_fc1_weight.copy()
                    tex.grouped_swizzle_for_gemm(
                        fc1_weight_for_gemm, rowwise=False, columnwise=True
                    )

                    fc1_w_data = fc1_weight_for_gemm.columnwise_data
                    fc1_w_data = fc1_w_data.view(dtype=torch.float8_e4m3fn)
                    fc1_w_data = fc1_w_data.view(
                        num_groups, fc1_weight_shape[0], fc1_weight_shape[1]
                    )
                    fc1_w_data = fc1_w_data.permute(2, 1, 0)
                    fc1_w_scales = fc1_weight_for_gemm.columnwise_scale_inv.view(
                        dtype=torch.float8_e8m0fnu
                    )
                    fc1_w_scales = fc1_w_scales.view(
                        num_groups,
                        (fc1_weight_shape[1] + 127) // 128,
                        (fc1_weight_shape[0] + 127) // 128,
                        MXFP8_BLOCK_SCALING_SIZE,
                        4,
                        4,
                    )
                    fc1_w_scales = fc1_w_scales.permute(3, 4, 1, 5, 2, 0)

                    fc1_dgrad_kwargs["b_tensor"] = fc1_w_data
                    fc1_dgrad_kwargs["sfb_tensor"] = fc1_w_scales
                else:
                    fc1_b_ptrs, fc1_sfb_ptrs, _ = tex.get_device_pointer_for_data_and_scales(
                        [w._columnwise_data for w in grouped_fc1_weight],
                        [w._columnwise_scale_inv for w in grouped_fc1_weight],
                        swizzle=True,
                        rowwise=False,
                        data_dtype=grouped_fc1_weight[0]._fp8_dtype,
                    )

                    fc1_dgrad_kwargs["b_ptrs"] = fc1_b_ptrs
                    fc1_dgrad_kwargs["sfb_ptrs"] = fc1_sfb_ptrs
                    fc1_dgrad_kwargs["n"] = fc1_weight_shape[1]
                    fc1_dgrad_kwargs["b_dtype"] = torch.float8_e4m3fn
                    fc1_dgrad_kwargs["b_major"] = "n"

                fc1_dgrad_kernel_out = self.grouped_gemm_quant_kernel()(**fc1_dgrad_kwargs)
                grad_input = fc1_dgrad_kernel_out["d_tensor"].view(in_shape)

        fc1_wgrad_kernel_fn = (
            self.grouped_gemm_wgrad_kernel()
            if (not use_nvfp4 or enable_nvfp4_fc1_wgrad)
            else None
        )
        fc1_grad_params = _compute_grad_params(
            fc_op=fc1_op,
            ctx=fc1_ctx,
            num_groups=num_groups,
            weight_shape=fc1_weight_shape,
            grouped_x=grouped_fc1_x,
            grouped_dy=grouped_fc1_dy,
            dtype=dtype,
            device=device,
            bias_grads=fc1_bias_grads,
            bias_grad_packed=fc1_bias_grad_packed,
            label="FC1",
            cudnn_wgrad_kernel_fn=fc1_wgrad_kernel_fn,
            offsets=split_points,
        )

        # Clear FC1 input tensor if possible
        if grouped_fc1_x is not None and not (
            fc1_ctx.weight_requires_grad
            and fc1_op.wgrad_store is not None
            and fc1_op.wgrad_store.delay_wgrad_compute()
        ):
            clear_tensor_data(
                grouped_fc1_x.data,
                grouped_fc1_x.columnwise_data,
                grouped_fc1_x.scale_inv,
                grouped_fc1_x.columnwise_scale_inv,
            )

        fc2_grad_extra = (None, None) if fc2_op._scale_bias else (None,)
        activation_grad_extra = (grad_scales,) if grad_scales is not None else ()
        return (
            grad_input,
            [fc1_grad_params, (), fc2_grad_params],
            [(None,), activation_grad_extra, fc2_grad_extra],
        )


class BackwardGroupedMLP_CuTeGEMMDSReLU_MXFP8(BackwardGroupedMLP_CuTeGEMMDSwiGLU_MXFP8):
    """Fused backward op for GroupedLinear + ScaledSReLU + GroupedLinear."""

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_dglu_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM and dSReLU activation backward."""
        from cudnn import grouped_gemm_dsrelu_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_dsrelu_wrapper_sm100

    @classmethod
    def is_fc1_bias_supported(cls) -> bool:
        """Whether cudnn-frontend exposes ``generate_dbias`` on the dSReLU SM100 wrapper."""
        if not cls.is_supported():
            return False
        return _dsrelu_wrapper_has_generate_dbias_arg()


def fuse_backward_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply operation fusion for backward pass.

    Parameters
    ----------
    ops : list of FusibleOperation
        Forward pass operations.
    recipe : Recipe, optional
        Quantization recipe.

    Returns
    -------
    ops : list of FusibleOperation
        Updated backward pass operations

    """

    return fuse_grouped_mlp_ops(
        ops,
        recipe=recipe,
        fused_op_cls=BackwardGroupedMLP_CuTeGEMMDSwiGLU_MXFP8,
    )


def fuse_backward_srelu_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply GroupedLinear + ScaledSReLU + GroupedLinear fusion for backward pass."""

    return fuse_grouped_mlp_ops(
        ops,
        recipe=recipe,
        fused_op_cls=BackwardGroupedMLP_CuTeGEMMDSReLU_MXFP8,
        activation_op_types=(ScaledSReLU,),
        activation_kwarg="srelu",
    )


# Register fusion if available
if BackwardGroupedMLP_CuTeGEMMDSwiGLU_MXFP8.is_supported():
    register_backward_fusion(fuse_backward_ops, prepend=True)
if BackwardGroupedMLP_CuTeGEMMDSReLU_MXFP8.is_supported():
    register_backward_fusion(fuse_backward_srelu_ops, prepend=True)
