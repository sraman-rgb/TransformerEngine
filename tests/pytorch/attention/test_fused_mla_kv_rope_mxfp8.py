# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for fused_mla_kv_rope_mxfp8: MLA key/value split, key RoPE and MXFP8 quantization.

The reference applies Megatron-LM's MLA key/value RoPE (the Triton kernel in mla_rope_utils)
and quantizes key and value with mxfp8_quantize_only; the fused kernel must produce the same
bytes.

Run:
    python3 -m pytest tests/pytorch/attention/test_fused_mla_kv_rope_mxfp8.py -v
"""

import pathlib
import sys

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    fused_mla_kv_rope_mxfp8,
    mxfp8_quantize_only,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent)] + sys.path
from mla_rope_utils import HAVE_TRITON, apply_mla_rope_kv, build_rope_tables

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
sm100 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10

pytestmark = [
    pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8),
    pytest.mark.skipif(not sm100, reason="The fused kernel requires SM 10.0+."),
    pytest.mark.skipif(not HAVE_TRITON, reason="The reference MLA RoPE needs Triton."),
]

HEAD_DIM_NOPE = 128
HEAD_DIM_ROPE = 64
HEAD_DIM_V = 128
KV_LORA_RANK = 512
MXFP8_BUFFERS = ("_rowwise_data", "_rowwise_scale_inv", "_columnwise_data", "_columnwise_scale_inv")


def _quantizer():
    return MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)


def _inputs(s, b, h, cos_dtype, seed=1234):
    torch.manual_seed(seed)
    kv = (torch.randn(s, b, h, HEAD_DIM_NOPE + HEAD_DIM_V, device="cuda") * 4).to(torch.bfloat16)
    # An all-zero scaling block and a value that saturates the scale of its blocks.
    kv[5, 0, 0, :32] = 0
    kv[7, 0, h - 1, 3] = 3.0e4
    kv[9, b - 1, 1, HEAD_DIM_NOPE + 17] = -2.5e4
    # k_pos_emb is a slice of the key/value down-projection output, as in MLA.
    kv_down = (torch.randn(s, b, KV_LORA_RANK + HEAD_DIM_ROPE, device="cuda") * 4).to(
        torch.bfloat16
    )
    k_pos_emb = kv_down[..., KV_LORA_RANK:].unsqueeze(-2)
    cos, sin = build_rope_tables(s, emb_dim=HEAD_DIM_ROPE, device="cuda")
    # A magnitude scale as in YaRN, so that the products are not exact.
    cos, sin = (cos * 1.2).to(cos_dtype), (sin * 1.2).to(cos_dtype)
    return kv, k_pos_emb, cos, sin


def _reference(kv, k_pos_emb, cos, sin):
    key, value = apply_mla_rope_kv(kv, k_pos_emb, cos_table=cos, sin_table=sin)
    return mxfp8_quantize_only(
        [(key.contiguous(), _quantizer()), (value.contiguous(), _quantizer())], "sbhd"
    )


def _assert_identical(actual, expected, name):
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} vs {expected.shape}"
    for buffer in MXFP8_BUFFERS:
        a, e = getattr(actual, buffer), getattr(expected, buffer)
        assert a.shape == e.shape, f"{name}.{buffer}: shape {tuple(a.shape)} vs {tuple(e.shape)}"
        a, e = a.view(torch.uint8), e.view(torch.uint8)
        mismatches = int((a != e).sum())
        assert mismatches == 0, f"{name}.{buffer}: {mismatches} of {e.numel()} bytes differ"


@pytest.mark.parametrize(
    "s, b, h", [(4096, 1, 128), (2048, 2, 64), (512, 1, 16), (256, 3, 8), (128, 2, 2)]
)
@pytest.mark.parametrize("cos_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_matches_rope_then_quantize(s, b, h, cos_dtype):
    """Same bytes as the RoPE / split kernel followed by MXFP8 quantization."""
    kv, k_pos_emb, cos, sin = _inputs(s, b, h, cos_dtype)
    ref_key, ref_value = _reference(kv, k_pos_emb, cos, sin)
    cos4, sin4 = cos.view(s, 1, 1, HEAD_DIM_ROPE), sin.view(s, 1, 1, HEAD_DIM_ROPE)
    key, value = fused_mla_kv_rope_mxfp8(kv, k_pos_emb, cos4, sin4, HEAD_DIM_V)
    _assert_identical(key, ref_key, "key")
    _assert_identical(value, ref_value, "value")


def test_contiguous_k_pos_emb_and_longer_rope_tables():
    """A contiguous [s, b, emb_dim] k_pos_emb and cos / sin with more rows than tokens."""
    s, b, h = 256, 2, 4
    kv, k_pos_emb, _, _ = _inputs(s, b, h, torch.bfloat16)
    cos, sin = build_rope_tables(2 * s, emb_dim=HEAD_DIM_ROPE, device="cuda")
    cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
    ref_key, ref_value = _reference(kv, k_pos_emb, cos[:s], sin[:s])
    key, value = fused_mla_kv_rope_mxfp8(kv, k_pos_emb.squeeze(-2).contiguous(), cos, sin, 128)
    _assert_identical(key, ref_key, "key")
    _assert_identical(value, ref_value, "value")


def test_non_finite_values():
    """Inf and NaN inputs give the quantizer's scaling factors and data."""
    s, b, h = 128, 1, 4
    kv, k_pos_emb, cos, sin = _inputs(s, b, h, torch.bfloat16)
    kv[3, 0, 0, 40] = float("inf")
    kv[40, 0, 1, HEAD_DIM_NOPE + 5] = float("-inf")
    kv[64, 0, 2, 100] = float("nan")
    kv[96:128, 0, 3, HEAD_DIM_NOPE + 32 : HEAD_DIM_NOPE + 64] = float("nan")
    ref_key, ref_value = _reference(kv, k_pos_emb, cos, sin)
    key, value = fused_mla_kv_rope_mxfp8(kv, k_pos_emb, cos, sin, HEAD_DIM_V)
    _assert_identical(key, ref_key, "key")
    _assert_identical(value, ref_value, "value")


@pytest.mark.parametrize(
    "s, h, message", [(96, 4, "multiple of 128"), (128, 3, "must be even")], ids=["s", "h"]
)
def test_unsupported_shapes(s, h, message):
    kv, k_pos_emb, cos, sin = _inputs(s, 1, h, torch.bfloat16)
    with pytest.raises((RuntimeError, ValueError), match=message):
        fused_mla_kv_rope_mxfp8(kv, k_pos_emb, cos, sin, HEAD_DIM_V)
