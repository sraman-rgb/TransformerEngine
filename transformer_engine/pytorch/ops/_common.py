# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions used in fusible operations."""

from __future__ import annotations
import functools
from importlib.metadata import PackageNotFoundError, version as get_pkg_version
from typing import Optional

import torch
from packaging.version import Version as PkgVersion

from transformer_engine_torch import FP8TensorMeta
from ..torch_version import torch_version
from ..quantization import FP8GlobalStateManager
from ..tensor.float8_tensor import Float8Tensor
from ..quantized_tensor import QuantizedTensorStorage
from ..utils import canonicalize_dtype


@functools.lru_cache(maxsize=1)
def _nvidia_cudnn_frontend_supports_scaled_clamped_qgeglu() -> bool:
    """Check cuDNN FE min version with fixed numerics for qgeglu."""
    try:
        return PkgVersion(get_pkg_version("nvidia-cudnn-frontend")) >= PkgVersion("1.23.0")
    except PackageNotFoundError:
        return False


@functools.lru_cache(maxsize=1)
def _nvidia_cudnn_frontend_supports_wgrad() -> bool:
    """Check cuDNN FE min version for grouped GEMM wgrad kernel."""
    try:
        return PkgVersion(get_pkg_version("nvidia-cudnn-frontend")) >= PkgVersion("1.23.0")
    except PackageNotFoundError:
        return False


def is_quantized_tensor(tensor: torch.Tensor | QuantizedTensorStorage) -> bool:
    """Check if tensor is a quantized tensor"""
    return isinstance(tensor, QuantizedTensorStorage)


def maybe_dequantize(
    tensor: torch.Tensor | QuantizedTensorStorage, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize tensor to given dtype or just convert if not a quantized tensor"""
    if is_quantized_tensor(tensor):
        return tensor.dequantize(dtype=dtype)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def maybe_autocast_dtype(
    *,
    device_type: str = "cuda",
    default_dtype: Optional[torch.dtype] = None,
) -> torch.dtype:
    """Get autocast dtype if enabled"""

    if torch_version() >= (2, 4, 3):
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    else:
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return canonicalize_dtype(default_dtype)


def get_fp8_meta_from_fp8_tensor(tensor: Float8Tensor) -> tuple[FP8TensorMeta, int]:
    """Get FP8TensorMeta object and index corresponding to Float8Tensor

    Constructs FP8TensorMeta if needed.

    """

    # Check if tensor already has FP8 metadata
    if tensor._fp8_meta is not None:
        key = FP8GlobalStateManager.get_meta_tensor_key(
            forward=tensor._fp8_meta_forward,
        )
        return tensor._fp8_meta[key], tensor._fp8_meta_index

    # Create FP8TensorMeta class
    fp8_meta = FP8TensorMeta()
    fp8_meta.scale = tensor._scale_inv.reciprocal()
    fp8_meta.amax_history = torch.empty(1, 1, dtype=torch.float32, device=tensor.device)
    fp8_meta.scale_inv = tensor._scale_inv
    return fp8_meta, 0


def validate_grouped_mlp_dims(fc1, activation_op, fc2) -> None:
    """Validate FC1 / activation / FC2 dimensions for fused grouped MLP."""
    from .basic import (  # pylint: disable=import-outside-toplevel
        ScaledSReLU,
        ScaledClampedQGeGLU,
        ScaledSwiGLU,
    )

    if fc1.in_features % 64 != 0 or fc1.out_features % 64 != 0:
        raise ValueError(
            f"Unsupported dims for FC1 (num_groups={fc1.num_groups}, "
            f"in_features={fc1.in_features}, out_features={fc1.out_features})."
        )
    if fc2.in_features % 64 != 0 or fc2.out_features % 64 != 0:
        raise ValueError(
            f"Unsupported dims for FC2 (num_groups={fc2.num_groups}, "
            f"in_features={fc2.in_features}, out_features={fc2.out_features})."
        )
    if isinstance(activation_op, (ScaledSwiGLU, ScaledClampedQGeGLU)):
        expected_fc1_out_features = 2 * fc2.in_features
    elif isinstance(activation_op, ScaledSReLU):
        expected_fc1_out_features = fc2.in_features
    else:
        raise TypeError(
            f"Unsupported grouped MLP activation ({activation_op.__class__.__name__})."
        )

    if fc1.out_features != expected_fc1_out_features or fc1.num_groups != fc2.num_groups:
        raise ValueError(
            f"FC1 (num_groups={fc1.num_groups}, in_features={fc1.in_features}, "
            f"out_features={fc1.out_features}) "
            f"and FC2 (num_groups={fc2.num_groups}, in_features={fc2.in_features}, "
            f"out_features={fc2.out_features}) do not match."
        )
    if (
        isinstance(activation_op, (ScaledSwiGLU, ScaledClampedQGeGLU))
        and activation_op.glu_interleave_size != 32
    ):
        raise ValueError(
            "Fused kernel requires 32-wide GLU interleaving, "
            f"but got glu_interleave_size={activation_op.glu_interleave_size}."
        )


def fuse_grouped_mlp_ops(
    ops,
    *,
    recipe,
    fused_op_cls,
    activation_op_types=None,
    activation_kwarg: str = "swiglu",
):
    """Sliding-window fusion for GroupedLinear + activation + GroupedLinear.

    Parameters
    ----------
    ops : list of FusibleOperation
        Operations to scan.
    recipe : Recipe or None
        Quantization recipe.
    fused_op_cls : type
        Fused operation class with ``is_supported()`` classmethod and
        constructor accepting ``fc1``, activation op, and ``fc2`` keyword args.
        May also expose ``is_fc1_bias_supported()`` and/or
        ``is_fc2_bias_supported()`` classmethods for bias eligibility.

    Returns
    -------
    list of FusibleOperation
        Updated operations with matched triples replaced by fused ops.
    """
    from .basic import (  # pylint: disable=import-outside-toplevel
        GroupedLinear,
        ScaledSReLU,
        ScaledClampedQGeGLU,
        ScaledSwiGLU,
    )

    if not fused_op_cls.is_supported():
        return ops
    if recipe is None or not (recipe.mxfp8() or recipe.nvfp4()):
        return ops
    if activation_op_types is None:
        activation_op_types = (ScaledSwiGLU, ScaledClampedQGeGLU)

    fc1_bias_ok = (
        not hasattr(fused_op_cls, "is_fc1_bias_supported") or fused_op_cls.is_fc1_bias_supported()
    )
    fc2_bias_ok = (
        not hasattr(fused_op_cls, "is_fc2_bias_supported") or fused_op_cls.is_fc2_bias_supported()
    )

    out = []
    window, ops = ops[:3], ops[3:]
    while len(window) == 3:

        matches_pattern = True
        if not (
            isinstance(window[0], GroupedLinear)
            and isinstance(window[1], activation_op_types)
            and isinstance(window[2], GroupedLinear)
        ):
            matches_pattern = False
        elif isinstance(window[1], ScaledClampedQGeGLU) and (
            abs(window[1]._clamped.alpha - 1.702) > 0.001
            or not _nvidia_cudnn_frontend_supports_scaled_clamped_qgeglu()
        ):
            matches_pattern = False
        elif window[0].has_bias and not fc1_bias_ok:
            matches_pattern = False
        elif window[2].has_bias and not fc2_bias_ok:
            matches_pattern = False
        else:
            try:
                validate_grouped_mlp_dims(window[0], window[1], window[2])
            except (TypeError, ValueError):
                matches_pattern = False

        if matches_pattern:
            op = fused_op_cls(
                fc1=window[0],
                fc2=window[2],
                **{activation_kwarg: window[1]},
            )
            window = [op]
        else:
            out.extend(window[:-2])
            window = window[-2:]

        out.extend(window[:-3])
        window = window[-3:]
        while ops and len(window) < 3:
            window.append(ops[0])
            ops = ops[1:]

    out.extend(window)
    return out
