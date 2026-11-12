# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""PyTorch Expert Parallelism (EP) API."""

from __future__ import annotations

import atexit
import functools
import logging
import os
import warnings
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import transformer_engine_torch as tex

from .constants import DType
from .cpu_offload import mark_not_offload
from .distributed import symm_mem_alloc, release_symm_mem_pool
from .quantized_tensor import QuantizedTensor, QuantizedTensorStorage
from .tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage

# Type-hint-only import; keeps the ``Recipe`` annotation without a runtime import of
# common.recipe (the concrete recipe classes are imported lazily where used).
if TYPE_CHECKING:
    from ..common.recipe import Recipe
    from .quantized_tensor import Quantizer
    from .tensor.mxfp8_tensor import MXFP8Quantizer

__all__ = [
    "EpConfig",
    "EpBuffer",
    "ep_bootstrap",
    "get_ep_drop_on_overflow",
    "get_ep_group",
    "is_ep_bootstrapped",
    "ep_finalize",
    "ep_dispatch",
    "ep_combine",
    "symm_mem_alloc",
    "release_symm_mem_pool",
    "is_symm_backed",
]


# ``symm_mem_alloc`` (imported from .distributed) allocates the symm-mem buffers
# used by the zero-copy IO path. Set ``ep_bootstrap(zero_copy=True)`` to opt in;
# the C++ backend then operates the EP group in zero-copy mode.


# Bootstrap


# NCCL EP requires NCCL >= 2.30.4 (matches the C++ backend's runtime check).
_MIN_NCCL_VERSION = (2, 30, 4)


def _check_nccl_runtime_version() -> None:
    """Raise with a clear message if the loaded libnccl is too old for NCCL EP."""
    import ctypes

    try:
        lib = ctypes.CDLL("libnccl.so.2", mode=ctypes.RTLD_GLOBAL)
        v = ctypes.c_int(0)
        if lib.ncclGetVersion(ctypes.byref(v)) != 0:
            warnings.warn("ncclGetVersion failed; skipping NCCL EP version check.")
            return
    except OSError:  # libnccl not findable; let the C++ side error
        return
    n = v.value
    # NCCL packs as (major*10000 + minor*100 + patch) up to ~2.x; newer
    # builds use the same scheme. Decode defensively.
    major, minor, patch = n // 10000, (n // 100) % 100, n % 100
    if (major, minor, patch) < _MIN_NCCL_VERSION:
        min_str = ".".join(str(x) for x in _MIN_NCCL_VERSION)
        raise RuntimeError(
            f"NCCL EP requires NCCL >= {min_str}, found {major}.{minor}.{patch} at runtime. "
            "Set LD_LIBRARY_PATH to a newer libnccl.so before launching."
        )


_BOOTSTRAPPED = False
_ATEXIT_REGISTERED = False
# EP group captured at bootstrap; used by the zero-copy symm-mem pool allocator.
_EP_GROUP: Optional[dist.ProcessGroup] = None
# Eager-mode toggle captured at bootstrap (set when recv_capacity_per_rank is
# omitted); ep_dispatch reads it to size the recv outputs from the per-step
# recv-token total instead of a fixed recv_capacity_per_rank.
_EAGER = False
_BOOTSTRAP_SETTINGS: Optional[dict[str, object]] = None


@dataclass(frozen=True, slots=True)
class EpConfig:
    """Immutable configuration shared by EP MoE operations.

    The configuration describes the routing that expert parallelism performs and
    the format of the data it moves. It describes no particular transport: the
    ops in ``transformer_engine.pytorch.ops.basic`` accept it whichever backend
    carries the tokens, and a backend that needs transport settings of its own
    (an ``EpBuffer``'s receive capacity, alignment, payload dtype, zero-copy and
    overflow policy) is configured with them directly.
    """

    top_k: int
    hidden_dim: int
    num_local_experts: int
    max_tokens_per_rank: int
    recv_capacity_per_rank: Optional[int]
    ep_group: dist.ProcessGroup
    alignment: int = 0
    payload_dtype: torch.dtype = torch.bfloat16
    zero_copy: bool = False
    drop_on_overflow: bool = False
    # Quantization recipes used to move the dispatched tokens forward and the
    # combined gradient backward; ``None`` moves BF16 both ways.
    dispatch_fwd_quant_recipe: Optional["Recipe"] = None
    combine_bwd_quant_recipe: Optional["Recipe"] = None


def _atexit_finalize() -> None:
    """Best-effort teardown at interpreter shutdown; swallows errors."""
    global _BOOTSTRAPPED, _EP_GROUP, _EAGER, _BOOTSTRAP_SETTINGS
    if _BOOTSTRAPPED:
        try:
            tex.ep_finalize()
        except Exception:  # pylint: disable=broad-exception-caught
            import traceback

            traceback.print_exc()
        finally:
            _BOOTSTRAPPED = False
            _EP_GROUP = None
            _EAGER = False
            _BOOTSTRAP_SETTINGS = None


def ep_bootstrap(
    ep_group: dist.ProcessGroup,
    num_experts: int,
    max_tokens_per_rank: int,
    hidden_dim: int,
    num_topk: int,
    recv_capacity_per_rank: Optional[int] = None,
    max_num_sms: int = 0,
    zero_copy: bool = False,
    drop_on_overflow: bool = False,
    max_token_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Initialize EP by borrowing ep_group's NCCL comm. Call once per process.

    max_token_dtype sets the widest token dtype this EP group will dispatch;
    it sizes NCCL EP staging buffers.

    ``recv_capacity_per_rank`` bounds the tokens one rank receives per step and
    sizes the recv outputs. Omit it (``None``) for eager mode, which sizes recv
    outputs from the per-step recv total instead; eager needs a host sync each
    step and is not CUDA-graph capturable.

    ``zero_copy`` opts the EP group into the symm-mem zero-copy IO path; pass
    ``True`` only when payload tensors are allocated via ``symm_mem_alloc``.
    Requires ``recv_capacity_per_rank``. To capture a CUDA graph, supply
    persistent recv_tokens / grad_out buffers to dispatch/combine; the pool-based
    auto-allocation used when they are omitted is not CUDA-graph capturable.

    ``num_topk`` is the per-token top-k; it sizes NCCL EP internal buffers.

    ``drop_on_overflow`` drops tokens exceeding ``recv_capacity_per_rank`` instead
    of trapping. Requires ``recv_capacity_per_rank``.
    """
    global _BOOTSTRAPPED, _ATEXIT_REGISTERED, _EP_GROUP, _EAGER, _BOOTSTRAP_SETTINGS
    eager = recv_capacity_per_rank is None
    if _BOOTSTRAPPED:
        raise RuntimeError("ep_bootstrap was already called in this process")
    if ep_group.size() < 2:
        raise ValueError(f"ep_bootstrap requires ep_group.size() >= 2 (got {ep_group.size()}).")
    if num_topk < 1:
        raise ValueError(f"ep_bootstrap requires num_topk >= 1 (got {num_topk}).")
    if zero_copy and eager:
        raise ValueError("ep_bootstrap: zero_copy requires recv_capacity_per_rank")
    if drop_on_overflow and eager:
        raise ValueError("ep_bootstrap: drop_on_overflow requires recv_capacity_per_rank")
    _check_nccl_runtime_version()
    if zero_copy:
        if not tex.ep_zero_copy_supported():
            raise RuntimeError(
                "ep_bootstrap: zero_copy=True requires the Transformer Engine torch extension "
                "built with NCCL symm-mem support (torch >= 2.11 with USE_NCCL)."
            )
        warnings.warn(
            "ep_bootstrap(zero_copy=True) is experimental; the symm-mem IO path "
            "and its alias contracts on EpBuffer slots are subject to change.",
            stacklevel=2,
        )

    # Materialize the PG's NCCL comm before borrowing its raw handle.
    dist.barrier(group=ep_group, device_ids=[torch.cuda.current_device()])
    comm_ptr = tex.get_nccl_comm_ptr(ep_group)

    tex.ep_initialize(
        int(comm_ptr),
        str(ep_group.group_name),
        int(num_experts),
        int(max_tokens_per_rank),
        # Eager mode (recv_capacity_per_rank=None) sizes recv buffers per routing,
        # so the group uses the library-derived bound (0 = NCCL_EP_AUTO).
        int(recv_capacity_per_rank or 0),
        int(hidden_dim),
        int(max_num_sms),
        max_token_dtype,
        bool(zero_copy),
        int(num_topk),
        bool(drop_on_overflow),
    )
    _BOOTSTRAPPED = True
    _EP_GROUP = ep_group
    _EAGER = bool(eager)
    _BOOTSTRAP_SETTINGS = {
        "top_k": int(num_topk),
        "hidden_dim": int(hidden_dim),
        "num_local_experts": int(num_experts) // ep_group.size(),
        "max_tokens_per_rank": int(max_tokens_per_rank),
        "recv_capacity_per_rank": (
            None if recv_capacity_per_rank is None else int(recv_capacity_per_rank)
        ),
        "payload_dtype": max_token_dtype,
        "zero_copy": bool(zero_copy),
        "drop_on_overflow": bool(drop_on_overflow),
    }
    if not _ATEXIT_REGISTERED:
        atexit.register(_atexit_finalize)
        _ATEXIT_REGISTERED = True


def is_ep_bootstrapped() -> bool:
    """Whether EP has been initialized in this process."""
    return _BOOTSTRAPPED


def get_ep_group() -> Optional[dist.ProcessGroup]:
    """Return the process group registered by :func:`ep_bootstrap`."""
    return _EP_GROUP


def get_ep_drop_on_overflow() -> Optional[bool]:
    """Return the overflow policy registered by :func:`ep_bootstrap`."""
    if _BOOTSTRAP_SETTINGS is None:
        return None
    return bool(_BOOTSTRAP_SETTINGS["drop_on_overflow"])


def _ep_is_eager() -> bool:
    """Return whether the bootstrapped EP group uses variable-size eager buffers."""
    return _EAGER


def ep_finalize() -> None:
    """Optional explicit EP teardown; idempotent.

    An atexit handler covers normal interpreter shutdown, so most users do not
    need to call this. Call it explicitly only before
    ``dist.destroy_process_group()``, since the borrowed NCCL comm becomes
    invalid once the PG is destroyed. This also releases the symm-mem pool, so
    a caller that used ``symm_mem_alloc(use_pool=True)`` does not need a separate
    ``release_symm_mem_pool()`` before destroying the PG.
    """
    global _BOOTSTRAPPED, _EP_GROUP, _EAGER, _BOOTSTRAP_SETTINGS
    if not _BOOTSTRAPPED:
        return
    try:
        # Deregister pooled symm-mem windows while the group's comm is still valid.
        release_symm_mem_pool()
        tex.ep_finalize()
    finally:
        _BOOTSTRAPPED = False
        _EP_GROUP = None
        _EAGER = False
        _BOOTSTRAP_SETTINGS = None


def is_symm_backed(t: torch.Tensor) -> bool:
    """Whether ``t`` is symm-mem-backed on the EP group. Prefer torch's local ``is_symm_mem_tensor``
    when the build provides it (no collective, no exception); otherwise fall back to the rendezvous
    probe the C++ ep kernel uses (``maybe_make_window``): cached for an already-registered tensor,
    raises for a plain one."""
    from torch.distributed import _symmetric_memory as _symm

    if hasattr(_symm, "is_symm_mem_tensor"):
        return bool(_symm.is_symm_mem_tensor(t))
    if _EP_GROUP is None:
        raise RuntimeError(
            "is_symm_backed called before ensure_nccl_ep_bootstrapped(); no EP group registered."
        )
    try:
        _symm.rendezvous(t, _EP_GROUP.group_name)
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False


# Buffer


class EpBuffer:
    """Per-microbatch EP layer state: handle_mem, tokens_per_expert, and shape/dtype config.

    Use one EpBuffer per concurrently-in-flight call (e.g. per PP-1F1B microbatch).
    """

    __slots__ = (
        "handle_mem",
        "top_k",
        "alignment",
        "max_tokens_per_rank",
        "recv_capacity_per_rank",
        "hidden_dim",
        "num_local_experts",
        "payload_dtype",
        "device",
        "tokens_per_expert",
        "num_local_tokens",
        "zero_copy",
        "eager",
        "total_recv_tokens",
        "dispatch_fwd_quant_recipe",
        "combine_bwd_quant_recipe",
    )

    def __init__(
        self,
        top_k: int,
        max_tokens_per_rank: int,
        hidden_dim: int,
        num_local_experts: int,
        recv_capacity_per_rank: Optional[int] = None,
        alignment: int = 0,
        payload_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        dispatch_fwd_quant_recipe: Optional["Recipe"] = None,
        combine_bwd_quant_recipe: Optional["Recipe"] = None,
    ) -> None:
        if not _BOOTSTRAPPED:
            raise RuntimeError("EpBuffer requires ep_bootstrap() to be called first.")
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        alignment = int(alignment)
        if alignment > 1 and (alignment & (alignment - 1)) != 0:
            raise ValueError(f"alignment must be 0, 1, or a power of two (got {alignment}).")
        self.eager = _EAGER
        if not self.eager and recv_capacity_per_rank is None:
            raise ValueError(
                "EpBuffer requires recv_capacity_per_rank unless the EP group was "
                "bootstrapped in eager mode (recv_capacity_per_rank omitted)."
            )
        self.top_k = int(top_k)
        self.alignment = alignment
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.recv_capacity_per_rank = (
            None if recv_capacity_per_rank is None else int(recv_capacity_per_rank)
        )
        self.hidden_dim = int(hidden_dim)
        self.num_local_experts = int(num_local_experts)
        self.payload_dtype = payload_dtype
        self.device = device
        self.zero_copy = bool(tex.ep_get_zero_copy())
        self.dispatch_fwd_quant_recipe = dispatch_fwd_quant_recipe
        self.combine_bwd_quant_recipe = combine_bwd_quant_recipe

        size_bytes = tex.ep_handle_mem_size(self.top_k, self.alignment)
        self.handle_mem = torch.empty(int(size_bytes), dtype=torch.uint8, device=device)
        self.tokens_per_expert = torch.empty(
            self.num_local_experts, dtype=torch.int64, device=device
        )
        self.num_local_tokens = self.max_tokens_per_rank
        # Persistent tensor; keep resident if activation CPU offloading is on.
        mark_not_offload(self.handle_mem)
        # Per-step recv-token total (int64 [1]), written by ep_prepare. Eager reads it
        # host-side to size the recv outputs, so it lives in pinned host memory the prepare
        # kernel writes directly (UVA) — no D2H copy, just a stream sync. Graph mode keeps
        # it on device for the backend's post-replay overflow check. The eager tensor is host
        # memory, so activation CPU offloading (which targets device tensors) never touches it
        # and no mark_not_offload guard is needed.
        if self.eager:
            self.total_recv_tokens = torch.empty(1, dtype=torch.int64, pin_memory=True)
        else:
            self.total_recv_tokens = torch.empty(1, dtype=torch.int64, device=device)
            mark_not_offload(self.total_recv_tokens)


# torch.library custom ops (so they don't graph-break under torch.compile)

_LIB = "transformer_engine_ep"


@torch.library.custom_op(
    f"{_LIB}::prepare",
    mutates_args=("handle_mem", "tokens_per_expert", "total_recv_tokens"),
    device_types="cuda",
)
def _prepare_op(
    handle_mem: torch.Tensor,
    top_k: int,
    topk_idx: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    alignment: int,
    total_recv_tokens: torch.Tensor,
) -> None:
    tex.ep_prepare(handle_mem, topk_idx, tokens_per_expert, top_k, alignment, total_recv_tokens)


@_prepare_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::dispatch",
    mutates_args=("recv_tokens", "recv_topk_weights", "recv_scale_inv"),
    device_types="cuda",
)
def _dispatch_op(
    handle_mem: torch.Tensor,
    topk_idx: torch.Tensor,
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    recv_tokens: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    tokens_scale_inv: Optional[torch.Tensor] = None,
    recv_scale_inv: Optional[torch.Tensor] = None,
) -> None:
    tex.ep_dispatch(
        handle_mem,
        topk_idx,
        tokens,
        topk_weights,
        recv_tokens,
        recv_topk_weights,
        tokens_scale_inv,
        recv_scale_inv,
    )


@_dispatch_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::prepare_and_dispatch",
    mutates_args=(
        "handle_mem",
        "tokens_per_expert",
        "total_recv_tokens",
        "recv_tokens",
        "recv_topk_weights",
        "recv_scale_inv",
    ),
    device_types="cuda",
)
def _prepare_and_dispatch_op(
    handle_mem: torch.Tensor,
    topk_idx: torch.Tensor,
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_recv_tokens: torch.Tensor,
    top_k: int,
    alignment: int,
    recv_tokens: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    recv_scale_inv: Optional[torch.Tensor] = None,
    tokens_scale_inv: Optional[torch.Tensor] = None,
) -> None:
    tex.ep_prepare_and_dispatch(
        handle_mem,
        topk_idx,
        tokens,
        topk_weights,
        tokens_per_expert,
        total_recv_tokens,
        top_k,
        alignment,
        recv_tokens,
        recv_topk_weights,
        recv_scale_inv,
        tokens_scale_inv,
    )


@_prepare_and_dispatch_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::combine",
    mutates_args=("result",),
    device_types="cuda",
)
def _combine_op(
    handle_mem: torch.Tensor,
    expert_out: torch.Tensor,
    result: torch.Tensor,
) -> None:
    tex.ep_combine(handle_mem, expert_out, result)


@_combine_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::dispatch_bwd",
    mutates_args=("grad_tokens", "grad_topk_weights"),
    device_types="cuda",
)
def _dispatch_bwd_op(
    handle_mem: torch.Tensor,
    grad: torch.Tensor,
    g_recv_topk_weights: torch.Tensor,
    grad_tokens: torch.Tensor,
    grad_topk_weights: torch.Tensor,
) -> None:
    tex.ep_dispatch_bwd(handle_mem, grad, g_recv_topk_weights, grad_tokens, grad_topk_weights)


@_dispatch_bwd_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::combine_bwd",
    mutates_args=("grad_expert_out", "grad_expert_out_scale_inv"),
    device_types="cuda",
)
def _combine_bwd_op(
    handle_mem: torch.Tensor,
    grad: torch.Tensor,
    grad_expert_out: torch.Tensor,
    grad_scale_inv: Optional[torch.Tensor] = None,
    grad_expert_out_scale_inv: Optional[torch.Tensor] = None,
) -> None:
    tex.ep_combine_bwd(handle_mem, grad, grad_expert_out, grad_scale_inv, grad_expert_out_scale_inv)


@_combine_bwd_op.register_fake
def _(*_args, **_kw):
    return None


# Non-autograd primitives


def ep_prepare(buffer: "EpBuffer", topk_idx: torch.Tensor) -> torch.Tensor:
    """AllGather the routing map; fills ``buffer.handle_mem`` and returns
    ``buffer.tokens_per_expert`` (int64, shape [num_local_experts]). topk_idx must
    be int32 or int64.

    Also fills ``buffer.total_recv_tokens`` (int64 [1]; pinned host in eager mode,
    device otherwise) with the per-step recv total; graph mode reads it device-side
    to detect overflow.
    """
    torch.ops.transformer_engine_ep.prepare(
        buffer.handle_mem,
        buffer.top_k,
        topk_idx,
        buffer.tokens_per_expert,
        buffer.alignment,
        buffer.total_recv_tokens,
    )
    return buffer.tokens_per_expert


def _ep_dispatch_raw(
    buffer: "EpBuffer",
    topk_idx: torch.Tensor,
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    recv_tokens: torch.Tensor,
    recv_topk_weights: torch.Tensor,
) -> None:
    """Raw dispatch; no autograd, no prepare. Caller must run ep_prepare first."""
    tex.ep_dispatch(
        buffer.handle_mem, topk_idx, tokens, topk_weights, recv_tokens, recv_topk_weights
    )


def _ep_combine_raw(buffer: "EpBuffer", expert_out: torch.Tensor, result: torch.Tensor) -> None:
    """Raw combine; no autograd. Caller pre-weights expert_out."""
    tex.ep_combine(buffer.handle_mem, expert_out, result)


# NVFP4 token packing for EP dispatch


# One packed row per token: [H/2 bytes of E2M1 pairs | H/16 E4M3 block scales | FP32 token
# scale], zero-padded to 16 bytes. Elements use the standard NVFP4 layout (E2M1 values, one E4M3
# scale per 16 elements); the second-level FP32 scale is per token instead of per tensor, so
# every token decodes on its own and the receiver needs no cross-rank scale agreement.
_NVFP4_TILE = 512  # elements per kernel tile: 32 blocks of 16
# Return the NVFP4 GroupedTensor built straight from the packed rows instead of BF16 recv tokens.
_NVFP4_FUSED_RECEIVE = os.environ.get("NVTE_EP_NVFP4_FUSED_RECEIVE", "0") == "1"
# Build that GroupedTensor with the CUDA direct receive: the rowwise operand reuses the sender's
# E2M1 codes (power-of-two token scales), the columnwise RHT operand is quantized in one pass.
_NVFP4_DIRECT_RECEIVE = os.environ.get("NVTE_EP_NVFP4_DIRECT_RECEIVE", "0") == "1"
# NVFP4 combine backward with the direct receive: the sender packs the result-grad with stochastic
# rounding and power-of-two token scales (CUDA pack), the receiver builds FC2's grad_output NVFP4
# GroupedTensor (rowwise codes reused, columnwise RHT quantized with stochastic rounding).
_NVFP4_DIRECT_CBWD = os.environ.get("NVTE_EP_NVFP4_DIRECT_CBWD", "0") == "1"
_NVFP4_TILE_BLOCKS = 32
_NVFP4_BLOCK = 16
_E2M1_MAX = 6.0
_E4M3_MAX = 448.0


_NVFP4_RECEIVE_PATHS_LOGGED = set()


def _log_nvfp4_receive_path_once(fused: bool) -> None:
    """Log once per process which NVFP4 receive path runs (fused GroupedTensor or BF16 decode)."""
    if not (_NVFP4_FUSED_RECEIVE or _NVFP4_DIRECT_RECEIVE) or fused in _NVFP4_RECEIVE_PATHS_LOGGED:
        return
    _NVFP4_RECEIVE_PATHS_LOGGED.add(fused)
    kind = "direct" if _NVFP4_DIRECT_RECEIVE else "fused"
    path = f"{kind} NVFP4 GroupedTensor" if fused else "BF16 decode (NVFP4 compute inactive)"
    # Logged rather than warned: training jobs commonly filter UserWarnings.
    logging.getLogger(__name__).warning("NVFP4 EP dispatch receive path: %s", path)


_NVFP4_CBWD_PATH_LOGGED = []


def _log_nvfp4_cbwd_path_once() -> None:
    """Log once per process that the NVFP4 combine backward uses the direct receive."""
    if _NVFP4_CBWD_PATH_LOGGED:
        return
    _NVFP4_CBWD_PATH_LOGGED.append(True)
    logging.getLogger(__name__).warning(
        "NVFP4 EP combine backward path: direct NVFP4 GroupedTensor (stochastic rounding)"
    )


def _nvfp4_compute_active() -> bool:
    """Whether the expert GEMMs will run in NVFP4, so they can consume the fused-receive
    GroupedTensor. Outside NVFP4 autocast (e.g. BF16 evaluation) the receiver decodes to BF16."""
    from .quantization import FP8GlobalStateManager

    if not FP8GlobalStateManager.is_fp8_enabled():
        return False
    return FP8GlobalStateManager.get_fp8_recipe().nvfp4()


def _nvfp4_fast_math() -> bool:
    """Whether TE's NVFP4 casts use the approximate reciprocal (NVTE_USE_FAST_MATH)."""
    return os.environ.get("NVTE_USE_FAST_MATH", "0") == "1"


def _nvfp4_row_bytes(hidden: int) -> int:
    """Bytes of one packed NVFP4 token row, padded to the 16-byte EP row alignment."""
    raw = hidden // 2 + hidden // _NVFP4_BLOCK + 4
    return (raw + 15) // 16 * 16


def _is_nvfp4_recipe(recipe) -> bool:
    if recipe is None:
        return False
    from ..common.recipe import NVFP4BlockScaling

    return isinstance(recipe, NVFP4BlockScaling)


@triton.jit
def _nvfp4_reciprocal(x, FAST_MATH: tl.constexpr):
    """1 / x as TE's NVFP4 cast computes it: rcp.approx.ftz under NVTE_USE_FAST_MATH, else IEEE
    division; capped at FLT_MAX like TE (x == 0 gives FLT_MAX)."""
    if FAST_MATH:
        r = tl.inline_asm_elementwise(
            "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
        )
    else:
        r = tl.div_rn(tl.full(x.shape, 1.0, tl.float32), x)
    return tl.minimum(r, 3.4028234663852886e38)


@triton.jit
def _e2m1_encode(v):
    """Round to the nearest E2M1 value (ties to even) and return the 4-bit code."""
    a = tl.abs(v)
    code = tl.where(
        a <= 0.25,
        0,
        tl.where(
            a < 0.75,
            1,
            tl.where(
                a <= 1.25,
                2,
                tl.where(
                    a < 1.75,
                    3,
                    tl.where(a <= 2.5, 4, tl.where(a < 3.5, 5, tl.where(a <= 5.0, 6, 7))),
                ),
            ),
        ),
    )
    # Sign from the float sign bit, so -0.0 encodes as -0 like the hardware conversion.
    return code + tl.where(v.to(tl.int32, bitcast=True) < 0, 8, 0)


@triton.jit
def _e2m1_decode(code):
    """Value of a 4-bit E2M1 code."""
    m = code & 7
    mag = tl.where(
        m == 0,
        0.0,
        tl.where(
            m == 1,
            0.5,
            tl.where(
                m == 2,
                1.0,
                tl.where(
                    m == 3,
                    1.5,
                    tl.where(m == 4, 2.0, tl.where(m == 5, 3.0, tl.where(m == 6, 4.0, 6.0))),
                ),
            ),
        ),
    )
    return tl.where((code & 8) != 0, -mag, mag)


@triton.jit
def _e2m1_sr_code(v, u):
    """Stochastically round to an E2M1 value (saturating at 6) with uniform ``u`` in [0, 1) and
    return the 4-bit code: rounds up to the next grid point with probability equal to the
    fractional distance, so the rounding is unbiased."""
    a = tl.minimum(tl.abs(v), 6.0)
    lo = tl.where(a < 2.0, tl.floor(a * 2.0) * 0.5, tl.where(a < 4.0, tl.floor(a), 4.0))
    step = tl.where(a < 2.0, 0.5, tl.where(a < 4.0, 1.0, 2.0))
    r = tl.minimum(tl.where(u * step < a - lo, lo + step, lo), 6.0)
    code = tl.where(
        r < 2.0, (r * 2.0).to(tl.int32), tl.where(r < 4.0, r.to(tl.int32) + 2, tl.where(r < 6.0, 6, 7))
    )
    return code + tl.where(v.to(tl.int32, bitcast=True) < 0, 8, 0)


@triton.jit
def _nvfp4_pack_kernel(
    x_ptr,
    out_ptr,
    H: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    NUM_TILES: tl.constexpr,
    FAST_MATH: tl.constexpr,
    seed_ptr,
    STOCHASTIC: tl.constexpr,
    POW2: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + row * H
    out_row = out_ptr + row * ROW_BYTES
    blk = tl.arange(0, 32)[:, None]
    pair = tl.arange(0, 8)[None, :]
    elem = blk * 16 + 2 * pair

    amax = tl.zeros([32, 8], dtype=tl.float32)
    for t in range(NUM_TILES):
        xe = tl.load(x_row + t * 512 + elem).to(tl.float32)
        xo = tl.load(x_row + t * 512 + elem + 1).to(tl.float32)
        amax = tl.maximum(amax, tl.maximum(tl.abs(xe), tl.abs(xo)))
    row_amax = tl.max(tl.max(amax, axis=1), axis=0)
    # Same arithmetic as TE's row-scaled NVFP4 quantizer (core_nvfp4.cuh), so the codes and
    # scales match it bit for bit.
    if POW2:
        # Power-of-two token scale: the smallest 2^e >= amax / (6 * 448). The direct receive
        # re-expresses block scales against a per-expert power of two, which is then exact.
        ratio_bits = tl.div_rn(row_amax, 448.0 * 6.0).to(tl.int32, bitcast=True)
        exp_bits = ratio_bits & 0x7F800000
        dec_bits = tl.where((ratio_bits & 0x7FFFFF) == 0, exp_bits, exp_bits + 0x800000)
        dec_bits = tl.maximum(dec_bits, 0x800000)
        global_decode = tl.where(row_amax == 0.0, 0.0, dec_bits.to(tl.float32, bitcast=True))
        global_encode = tl.where(row_amax == 0.0, 1.0, tl.div_rn(1.0, global_decode))
    else:
        global_encode = tl.minimum(tl.div_rn(448.0 * 6.0, row_amax), 3.4028234663852886e38)
        global_encode = tl.where((row_amax == 0.0) | (global_encode == 0.0), 1.0, global_encode)
        global_decode = tl.div_rn(1.0, global_encode)
    tl.store((out_row + H // 2 + H // 16).to(tl.pointer_type(tl.float32)), global_decode)

    for t in range(NUM_TILES):
        xe = tl.load(x_row + t * 512 + elem).to(tl.float32)
        xo = tl.load(x_row + t * 512 + elem + 1).to(tl.float32)
        block_amax = tl.max(tl.maximum(tl.abs(xe), tl.abs(xo)), axis=1)
        block_scale = tl.minimum(block_amax * (global_encode * (1.0 / 6.0)), 448.0)
        block_scale = block_scale.to(tl.float8e4nv)
        tl.store(
            out_row + H // 2 + t * 32 + tl.arange(0, 32), block_scale.to(tl.uint8, bitcast=True)
        )
        denom = block_scale.to(tl.float32) * global_decode
        rcp = _nvfp4_reciprocal(denom, FAST_MATH)[:, None]
        if STOCHASTIC:
            seed = tl.load(seed_ptr)
            offs = (row * H + t * 512 + elem).to(tl.int32)
            code_e = _e2m1_sr_code(xe * rcp, tl.rand(seed, offs))
            code_o = _e2m1_sr_code(xo * rcp, tl.rand(seed, offs + 1))
            packed = code_e | (code_o << 4)
        else:
            packed = _e2m1x2_cvt(xe * rcp, xo * rcp)
        tl.store(out_row + t * 256 + blk * 8 + pair, packed.to(tl.uint8))


@triton.jit
def _nvfp4_unpack_kernel(
    in_ptr,
    out_ptr,
    counts_ptr,
    num_rows,
    NUM_COUNTS: tl.constexpr,
    COUNTS_P2: tl.constexpr,
    H: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    TM: tl.constexpr,
    TK: tl.constexpr,
    NPROG: tl.constexpr,
):
    bound = num_rows
    if NUM_COUNTS > 0:
        idx = tl.arange(0, COUNTS_P2)
        counts = tl.load(counts_ptr + idx, mask=idx < NUM_COUNTS, other=0)
        bound = tl.minimum(tl.sum(counts).to(tl.int32), num_rows)
    num_tiles = tl.cdiv(bound, TM) * (H // TK)
    for t in range(tl.program_id(0), num_tiles, NPROG):
        r = (t // (H // TK)) * TM + tl.arange(0, TM)
        c = (t % (H // TK)) * TK + tl.arange(0, TK)[None, :]
        mask = (r < bound)[:, None]
        row = in_ptr + r.to(tl.int64)[:, None] * ROW_BYTES
        code = (tl.load(row + c // 2, mask=mask, other=0).to(tl.int32) >> (4 * (c % 2))) & 15
        sb = tl.load(row + H // 2 + c // 16, mask=mask, other=0)
        token_scale = tl.load(
            (row + H // 2 + H // 16).to(tl.pointer_type(tl.float32)), mask=mask, other=0.0
        )
        scale = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32) * token_scale
        out = _e2m1_decode_bits(code) * scale
        tl.store(out_ptr + r.to(tl.int64)[:, None] * H + c, out.to(tl.bfloat16), mask=mask)


def _nvfp4_pack_tokens(
    tokens: torch.Tensor,
    stochastic: bool = False,
    seed: Optional[torch.Tensor] = None,
    pow2_scale: bool = False,
) -> torch.Tensor:
    """Quantize BF16 tokens [T, H] to packed NVFP4 rows, returned as uint8 [T, row_bytes].

    ``stochastic`` rounds the E2M1 values stochastically (for gradients, as the NVFP4 recipe's
    grad quantizer does). The Philox seed is a device tensor (``seed``, int64 [1]); by default it
    is drawn from torch's CUDA generator per call, which is CUDA-graph safe, so every replay draws
    fresh bits. ``pow2_scale`` rounds each token's decode scale up to a power of two (needed by
    the direct receive); zero tokens then carry a zero decode scale."""
    tokens = tokens.contiguous()
    num_tokens, hidden = tokens.shape
    if hidden % _NVFP4_TILE != 0:
        raise ValueError(f"NVFP4 EP dispatch requires hidden % {_NVFP4_TILE} == 0, got {hidden}.")
    row_bytes = _nvfp4_row_bytes(hidden)
    packed = torch.zeros(num_tokens, row_bytes, dtype=torch.uint8, device=tokens.device)
    if stochastic and seed is None:
        seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64, device=tokens.device)
    if num_tokens > 0:
        _nvfp4_pack_kernel[(num_tokens,)](
            tokens,
            packed,
            H=hidden,
            ROW_BYTES=row_bytes,
            NUM_TILES=hidden // _NVFP4_TILE,
            FAST_MATH=_nvfp4_fast_math(),
            seed_ptr=seed if seed is not None else packed,
            STOCHASTIC=stochastic,
            POW2=pow2_scale,
        )
    return packed


def _nvfp4_unpack_tokens(
    packed: torch.Tensor,
    hidden: int,
    counts: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantize packed NVFP4 rows (uint8 [R, row_bytes]) to BF16 [R, H]. With ``counts`` only
    the first ``sum(counts)`` rows are decoded, read on device so CUDA graph replays see the live
    count; later rows are left untouched, as a BF16 dispatch leaves them."""
    num_rows = packed.shape[0]
    row_bytes = _nvfp4_row_bytes(hidden)
    if packed.shape[1] != row_bytes:
        raise ValueError(f"packed NVFP4 rows must be {row_bytes} bytes, got {packed.shape[1]}.")
    if out is None:
        out = torch.empty(num_rows, hidden, dtype=torch.bfloat16, device=packed.device)
    num_counts = 0 if counts is None else counts.numel()
    if num_rows > 0:
        nprog = _nvfp4_num_programs(packed.device.index)
        _nvfp4_unpack_kernel[(nprog,)](
            packed,
            out,
            packed if counts is None else counts,
            num_rows,
            NUM_COUNTS=num_counts,
            COUNTS_P2=max(1, triton.next_power_of_2(max(num_counts, 1))),
            H=hidden,
            ROW_BYTES=row_bytes,
            TM=32,
            TK=256,
            NPROG=nprog,
            num_warps=4,
        )
    return out


# NVFP4 fused receive: packed rows -> the NVFP4 GroupedTensor the fused grouped MLP consumes


# Tile of the fused receive: 128 tokens (8 RHT blocks of 16) x 64 hidden (4 scale blocks of 16).
# Experts are 256-token aligned for the fused grouped MLP, so a tile never crosses an expert.
_FUSED_TILE_M = 128
_FUSED_TILE_K = 64


@triton.jit
def _e2m1_decode_bits(code):
    """FP32 value of 4-bit E2M1 codes via BF16 bit assembly (exact; -0 stays -0)."""
    m = code & 1
    e = (code >> 1) & 3
    bits = tl.where(e == 0, m * 0x3F00, ((e + 126) << 7) | (m << 6)) | ((code & 8) << 12)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)


@triton.jit
def _e2m1x2_cvt(lo, hi):
    """Pack two FP32 tensors into E2M1 pairs (low nibble = lo) with the hardware RN-satfinite
    conversion TE's casts use."""
    return tl.inline_asm_elementwise(
        "{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t; }",
        "=r,f,f",
        [lo, hi],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _nvfp4_recv_tile(in_row, c0, H: tl.constexpr):
    """Decode a [128, 64] tile of packed rows (pointers ``in_row`` [128]) at hidden offset ``c0``
    to FP32 values rounded through BF16, exactly as the BF16 decode path stores them."""
    pair = tl.arange(0, 32)[None, :]
    packed = tl.load(in_row[:, None] + c0 // 2 + pair).to(tl.int32)
    vals = tl.reshape(
        tl.join(_e2m1_decode_bits(packed & 15), _e2m1_decode_bits(packed >> 4)), (128, 64)
    )
    sb = tl.load(in_row[:, None] + H // 2 + c0 // 16 + tl.arange(0, 4)[None, :])
    token_scale = tl.load((in_row + H // 2 + H // 16).to(tl.pointer_type(tl.float32)))
    scale = sb.to(tl.float8e4nv, bitcast=True).to(tl.float32) * token_scale[:, None]
    vals = tl.reshape(tl.reshape(vals, (128, 4, 16)) * scale[:, :, None], (128, 64))
    return vals.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _nvfp4_counts(counts_ptr, NUM_GROUPS: tl.constexpr, GROUPS_P2: tl.constexpr):
    idx = tl.arange(0, GROUPS_P2)
    counts = tl.load(counts_ptr + idx, mask=idx < NUM_GROUPS, other=0).to(tl.int32)
    return idx, counts, tl.cumsum(counts, axis=0)


@triton.jit
def _nvfp4_group_of_row(idx, counts, ends, r0):
    """(group, group row offset, group rows) of the row r0 (groups are tile aligned)."""
    g = tl.sum((ends <= r0).to(tl.int32))
    g_end = tl.sum(tl.where(idx == g, ends, 0))
    g_rows = tl.sum(tl.where(idx == g, counts, 0))
    return g, g_end - g_rows, g_rows


@triton.jit
def _nvfp4_rht_tile(vals, bd, ROUND_BF16: tl.constexpr):
    """RHT along tokens of a [128, 64] tile as one block-diagonal [128, 128] x [128, 64] GEMM
    (``bd`` = I_8 (x) H^T in BF16, FP32 accumulate). TE rounds the result to BF16 except under
    NVTE_USE_FAST_MATH, where it quantizes the FP32 accumulator."""
    y = tl.dot(bd, vals.to(tl.bfloat16))
    if ROUND_BF16:
        y = y.to(tl.bfloat16).to(tl.float32)
    return y


@triton.jit
def _nvfp4_load_block_diag(bd_ptr):
    i = tl.arange(0, 128)
    return tl.load(bd_ptr + i[:, None] * 128 + i[None, :])


@triton.jit
def _nvfp4_recv_amax_kernel(
    in_ptr,
    counts_ptr,
    bd_ptr,
    row_amax_ptr,
    col_amax_ptr,
    NUM_GROUPS: tl.constexpr,
    GROUPS_P2: tl.constexpr,
    H: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    NPROG: tl.constexpr,
):
    idx, counts, ends = _nvfp4_counts(counts_ptr, NUM_GROUPS, GROUPS_P2)
    num_tiles = tl.cdiv(tl.sum(counts), 128) * (H // 64)
    bd = _nvfp4_load_block_diag(bd_ptr)
    for t in range(tl.program_id(0), num_tiles, NPROG):
        r0 = (t // (H // 64)) * 128
        c0 = (t % (H // 64)) * 64
        g, g_off, g_rows = _nvfp4_group_of_row(idx, counts, ends, r0)
        rows = (r0 + tl.arange(0, 128)).to(tl.int64)
        vals = _nvfp4_recv_tile(in_ptr + rows * ROW_BYTES, c0, H)
        tl.atomic_max(row_amax_ptr + g, tl.max(tl.max(tl.abs(vals), axis=1), axis=0))
        y = _nvfp4_rht_tile(vals, bd, True)
        tl.atomic_max(col_amax_ptr + g, tl.max(tl.max(tl.abs(y), axis=1), axis=0))


@triton.jit
def _nvfp4_global_encode(amax):
    enc = tl.minimum(tl.div_rn(448.0 * 6.0, amax), 3.4028234663852886e38)
    return tl.where((amax == 0.0) | (enc == 0.0), 1.0, enc)


@triton.jit
def _nvfp4_swizzled_scale_offset(r, c, num_scale_cols):
    """Offset of scale (row r, scale col c) in the GEMM-swizzled 128x4 tile layout."""
    return (
        ((r // 128) * ((num_scale_cols + 3) // 4) + c // 4) * 512
        + (r % 32) * 16
        + ((r // 32) % 4) * 4
        + (c % 4)
    )


@triton.jit
def _nvfp4_recv_encode_kernel(
    in_ptr,
    counts_ptr,
    bd_ptr,
    row_amax_ptr,
    col_amax_ptr,
    row_data_ptr,
    row_scale_ptr,
    col_data_ptr,
    col_scale_ptr,
    NUM_GROUPS: tl.constexpr,
    GROUPS_P2: tl.constexpr,
    H: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    FAST_MATH: tl.constexpr,
    NPROG: tl.constexpr,
):
    idx, counts, ends = _nvfp4_counts(counts_ptr, NUM_GROUPS, GROUPS_P2)
    num_tiles = tl.cdiv(tl.sum(counts), 128) * (H // 64)
    bd = _nvfp4_load_block_diag(bd_ptr)
    for t in range(tl.program_id(0), num_tiles, NPROG):
        r0 = (t // (H // 64)) * 128
        c0 = (t % (H // 64)) * 64
        g, g_off, g_rows = _nvfp4_group_of_row(idx, counts, ends, r0)
        rows = (r0 + tl.arange(0, 128)).to(tl.int64)
        vals = _nvfp4_recv_tile(in_ptr + rows * ROW_BYTES, c0, H)

        # Rowwise: 16-element blocks along hidden, group second-level scale.
        enc = _nvfp4_global_encode(tl.load(row_amax_ptr + g))
        dec = tl.div_rn(1.0, enc)
        v = tl.reshape(vals, (128, 4, 16))
        bmax = tl.max(tl.abs(v), axis=2)
        sb = tl.minimum(bmax * (enc * (1.0 / 6.0)), 448.0).to(tl.float8e4nv)
        rcp = _nvfp4_reciprocal(sb.to(tl.float32) * dec, FAST_MATH)
        q_lo, q_hi = tl.split(tl.reshape(v * rcp[:, :, None], (128, 32, 2)))
        pair = tl.arange(0, 32)[None, :]
        tl.store(
            row_data_ptr + rows[:, None] * (H // 2) + c0 // 2 + pair,
            _e2m1x2_cvt(q_lo, q_hi).to(tl.uint8),
        )
        sc = c0 // 16 + tl.arange(0, 4)[None, :]
        tl.store(
            row_scale_ptr + _nvfp4_swizzled_scale_offset(rows[:, None], sc, H // 16),
            sb.to(tl.uint8, bitcast=True),
        )

        # Columnwise: RHT along tokens, 16-token blocks, group post-RHT second-level scale.
        # Per-group layout: data [H, g_rows / 2], scales [H, g_rows / 16] (swizzled, per group).
        enc_c = _nvfp4_global_encode(tl.load(col_amax_ptr + g))
        dec_c = tl.div_rn(1.0, enc_c)
        y = tl.reshape(_nvfp4_rht_tile(vals, bd, not FAST_MATH), (8, 16, 64))
        bmax_c = tl.max(tl.abs(y), axis=1)  # [8 token blocks, 64 hidden]
        sb_c = tl.minimum(bmax_c * (enc_c * (1.0 / 6.0)), 448.0).to(tl.float8e4nv)
        rcp_c = _nvfp4_reciprocal(sb_c.to(tl.float32) * dec_c, FAST_MATH)
        qc = tl.reshape(tl.permute(y * rcp_c[:, None, :], (2, 0, 1)), (64, 64, 2))
        qc_lo, qc_hi = tl.split(qc)  # [hidden, token pairs]
        k = c0 + tl.arange(0, 64)
        local = r0 - g_off
        g_off64 = g_off.to(tl.int64)
        tok_pair = local // 2 + tl.arange(0, 64)[None, :]
        tl.store(
            col_data_ptr + g_off64 * (H // 2) + k[:, None].to(tl.int64) * (g_rows // 2) + tok_pair,
            _e2m1x2_cvt(qc_lo, qc_hi).to(tl.uint8),
        )
        blk = local // 16 + tl.arange(0, 8)[:, None]
        tl.store(
            col_scale_ptr
            + g_off64 * (H // 16)
            + _nvfp4_swizzled_scale_offset(k[None, :], blk, g_rows // 16),
            sb_c.to(tl.uint8, bitcast=True),
        )


@functools.lru_cache(maxsize=None)
def _nvfp4_rht_block_diag(device: torch.device) -> torch.Tensor:
    """I_8 (x) H^T in BF16: the 16x16 RHT applied to each 16-token block of a 128-token tile."""
    from .tensor.nvfp4_tensor import get_rht_matrix

    rht_t = get_rht_matrix(True, device.index).t().float()
    return torch.block_diag(*([rht_t] * 8)).to(torch.bfloat16).contiguous()


@functools.lru_cache(maxsize=None)
def _nvfp4_num_programs(device_index: int) -> int:
    return 4 * torch.cuda.get_device_properties(device_index).multi_processor_count


def _nvfp4_fused_receive(
    recv_packed: torch.Tensor,
    hidden: int,
    counts: torch.Tensor,
    fake_dtype: torch.dtype = torch.bfloat16,
):
    """Build the NVFP4 GroupedTensor the fused grouped MLP consumes (rowwise data + swizzled
    scales + per-group amax; columnwise RHT data + swizzled scales + per-group post-RHT amax)
    straight from packed NVFP4 rows. Matches ``group_quantize`` of the decoded BF16 rows up to
    rounding ties, without materializing them. Bounded on device by ``counts`` (CUDA-graph safe).
    """
    from .tensor.grouped_tensor import GroupedTensor
    from .tensor.nvfp4_tensor import NVFP4Quantizer

    capacity = recv_packed.shape[0]
    if capacity % _FUSED_TILE_M != 0 or hidden % _NVFP4_TILE != 0:
        raise ValueError(
            f"NVFP4 fused receive needs capacity % {_FUSED_TILE_M} == 0 and hidden % "
            f"{_NVFP4_TILE} == 0, got capacity={capacity}, hidden={hidden}."
        )
    device = recv_packed.device
    num_groups = counts.numel()
    block_diag = _nvfp4_rht_block_diag(device)
    row_amax = torch.zeros(num_groups, dtype=torch.float32, device=device)
    col_amax = torch.zeros(num_groups, dtype=torch.float32, device=device)
    # Rows past sum(counts) are never read by the grouped GEMMs, so no zero fill.
    row_data = torch.empty(capacity * hidden // 2, dtype=torch.uint8, device=device)
    row_scale = torch.empty(capacity * hidden // 16, dtype=torch.uint8, device=device)
    col_data = torch.empty(capacity * hidden // 2, dtype=torch.uint8, device=device)
    col_scale = torch.empty(capacity * hidden // 16, dtype=torch.uint8, device=device)
    nprog = 2 * torch.cuda.get_device_properties(device.index).multi_processor_count
    num_warps = 8
    meta = dict(
        NUM_GROUPS=num_groups,
        GROUPS_P2=triton.next_power_of_2(max(num_groups, 1)),
        H=hidden,
        ROW_BYTES=_nvfp4_row_bytes(hidden),
        NPROG=nprog,
    )
    _nvfp4_recv_amax_kernel[(nprog,)](
        recv_packed, counts, block_diag, row_amax, col_amax, num_warps=num_warps, **meta
    )
    _nvfp4_recv_encode_kernel[(nprog,)](
        recv_packed,
        counts,
        block_diag,
        row_amax,
        col_amax,
        row_data,
        row_scale,
        col_data,
        col_scale,
        FAST_MATH=_nvfp4_fast_math(),
        num_warps=num_warps,
        **meta,
    )
    quantizer = NVFP4Quantizer(
        rowwise=True, columnwise=True, with_rht=True, with_post_rht_amax=True
    )
    quantizer.optimize_for_gemm = True
    quantizer.internal = True
    return GroupedTensor(
        shape=(capacity, hidden),
        dtype=fake_dtype,
        num_tensors=num_groups,
        quantizer=quantizer,
        data=row_data,
        columnwise_data=col_data,
        scale_inv=row_scale,
        columnwise_scale_inv=col_scale,
        amax=row_amax,
        columnwise_amax=col_amax,
        first_dims=counts,
        tensor_offsets=tex.splits_to_offsets(counts, hidden),
        with_gemm_swizzled_scales=True,
    )


@functools.lru_cache(maxsize=None)
def _nvfp4_rht_sign_mask() -> int:
    """Bitmask of TE's wgrad RHT random signs (bit i set: sign i is -1), computed on the host."""
    from .tensor.nvfp4_tensor import get_sign_from_vector, get_wgrad_sign_vector

    return int(get_sign_from_vector(get_wgrad_sign_vector("cpu")))


@functools.lru_cache(maxsize=None)
def _nvfp4_direct_ext():
    """The prebuilt direct-receive CUDA extension (mounted next to this module)."""
    import importlib.util

    path = os.environ.get(
        "NVTE_EP_NVFP4_DIRECT_EXT",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_nvfp4_direct_ext.so"),
    )
    spec = importlib.util.spec_from_file_location("_nvfp4_direct_ext", path)
    if spec is None:
        raise RuntimeError(f"NVFP4 direct receive extension not found at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nvfp4_pack_tokens_cuda(tokens: torch.Tensor, stochastic: bool) -> torch.Tensor:
    """CUDA pack with power-of-two token scales (``_nvfp4_pack_tokens(pow2_scale=True)`` layout);
    stochastic rounding uses Philox bits from a per-call device seed (CUDA-graph safe)."""
    tokens = tokens.contiguous()
    num_tokens, hidden = tokens.shape
    packed = torch.empty(
        num_tokens, _nvfp4_row_bytes(hidden), dtype=torch.uint8, device=tokens.device
    )
    seed = None
    if stochastic:
        seed = torch.randint(0, 2**62, (1,), dtype=torch.int64, device=tokens.device)
    _nvfp4_direct_ext().pack_tokens(tokens, packed, _nvfp4_fast_math(), seed)
    return packed


def _nvfp4_direct_receive(
    recv_packed: torch.Tensor,
    hidden: int,
    counts: torch.Tensor,
    fake_dtype: torch.dtype = torch.bfloat16,
    stochastic: bool = False,
):
    """Like ``_nvfp4_fused_receive`` but in one CUDA pass and without requantizing the rowwise
    operand. Needs rows packed with ``pow2_scale=True``. Second-level scales are per expert:
    rowwise ``6 * 448 * G`` and columnwise ``4 * 6 * 448 * G`` with ``G`` the largest token scale
    of the expert (an upper bound of the post-RHT amax, so no amax pass is needed)."""
    from .tensor.grouped_tensor import GroupedTensor
    from .tensor.nvfp4_tensor import NVFP4Quantizer

    capacity = recv_packed.shape[0]
    device = recv_packed.device
    num_groups = counts.numel()
    counts = counts.to(torch.int64)
    row_data = torch.empty(capacity * hidden // 2, dtype=torch.uint8, device=device)
    row_scale = torch.empty(capacity * hidden // 16, dtype=torch.uint8, device=device)
    col_data = torch.empty(capacity * hidden // 2, dtype=torch.uint8, device=device)
    col_scale = torch.empty(capacity * hidden // 16, dtype=torch.uint8, device=device)
    row_amax = torch.empty(num_groups, dtype=torch.float32, device=device)
    col_amax = torch.empty(num_groups, dtype=torch.float32, device=device)
    gmax = torch.empty(num_groups, dtype=torch.int32, device=device)
    seed = None
    if stochastic:
        seed = torch.randint(0, 2**62, (1,), dtype=torch.int64, device=device)
    _nvfp4_direct_ext().direct_receive(
        recv_packed,
        counts,
        hidden,
        gmax,
        _nvfp4_rht_sign_mask(),
        row_data,
        row_scale,
        col_data,
        col_scale,
        row_amax,
        col_amax,
        _nvfp4_fast_math(),
        seed,
    )
    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_rht=True,
        with_post_rht_amax=True,
        stochastic_rounding=stochastic,
    )
    quantizer.optimize_for_gemm = True
    quantizer.internal = True
    return GroupedTensor(
        shape=(capacity, hidden),
        dtype=fake_dtype,
        num_tensors=num_groups,
        quantizer=quantizer,
        data=row_data,
        columnwise_data=col_data,
        scale_inv=row_scale,
        columnwise_scale_inv=col_scale,
        amax=row_amax,
        columnwise_amax=col_amax,
        first_dims=counts,
        tensor_offsets=tex.splits_to_offsets(counts, hidden),
        with_gemm_swizzled_scales=True,
    )


# autograd.Function wrappers


@dataclass(slots=True)
class _DispatchState:
    """Backward state + output-shaping flags for the prepare-and-dispatch op. ``handle_mem`` is
    buffer-owned and mutated in place, so it rides here as a plain attribute rather than through
    save_for_backward (which would version-track it). The autograd wrapper stashes this on ctx;
    other callers keep it however they manage backward state."""

    handle_mem: torch.Tensor
    tokens_shape: torch.Size
    topk_weights_shape: torch.Size
    tokens_T_flat: int
    topk_T_flat: int
    top_k: int
    hidden_dim: int
    eager: bool
    is_scaled: bool


def _ep_prepare_and_dispatch_fwd(
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_idx: torch.Tensor,
    buffer: "EpBuffer",
    recv_tokens: Optional[torch.Tensor],
    recv_topk_weights: Optional[torch.Tensor],
):
    """Validate inputs, size/allocate the recv outputs, and run prepare+dispatch. Eager sizes recv
    from the host count; otherwise recv uses the buffer static recv capacity. Returns the recv
    output (a per-expert GroupedTensor for MXFP8, else the raw recv tokens), the recv topk weights,
    and a _DispatchState for backward. No autograd; the caller owns context handling."""
    _require_bf16("dispatch input", tokens)
    if _is_nvfp4_recipe(buffer.dispatch_fwd_quant_recipe):
        return _ep_prepare_and_dispatch_nvfp4_fwd(
            tokens, topk_weights, topk_idx, buffer, recv_tokens, recv_topk_weights
        )
    tokens_scale_inv = None
    if buffer.dispatch_fwd_quant_recipe is not None:
        from .tensor.mxfp8_tensor import MXFP8Quantizer

        quantizer = MXFP8Quantizer(
            DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
        )
        quantizer.internal = True
        tokens, tokens_scale_inv = quantize_for_ep(tokens, quantizer)

    handle_mem = buffer.handle_mem
    tokens_per_expert = buffer.tokens_per_expert
    total_recv_tokens = buffer.total_recv_tokens
    top_k = buffer.top_k
    alignment = buffer.alignment
    eager = buffer.eager
    num_recv_tokens = buffer.recv_capacity_per_rank
    payload_dtype = buffer.payload_dtype
    is_scaled = tokens_scale_inv is not None
    tokens_data = tokens._rowwise_data if isinstance(tokens, MXFP8TensorStorage) else tokens
    assert tokens_data.dim() == 2, "EP dispatch tokens must be 2D [num_tokens, hidden]"
    hidden = tokens_data.shape[-1]
    if is_scaled and tokens._fp8_dtype != DType.kFloat8E4M3:
        raise NotImplementedError("EP dispatch supports only E4M3 MXFP8 tokens for now.")
    # Reinterpret byte-backed FP8 data as the fp8 dtype so the backend sees a scaled tensor.
    dispatch_tokens = tokens_data.view(torch.float8_e4m3fn) if is_scaled else tokens_data

    recv_scale_inv = None
    if eager:
        # The C++ op sizes and allocates the recv outputs from the host recv-count and returns
        # them (called directly, not via torch.library: eager forbids graph capture).
        outs = tex.ep_prepare_and_dispatch(
            handle_mem,
            topk_idx,
            dispatch_tokens,
            topk_weights,
            tokens_per_expert,
            total_recv_tokens,
            top_k,
            alignment,
            tokens_scale_inv=tokens_scale_inv,
        )
        recv_tokens, recv_topk_weights = outs[0], outs[1]
        recv_scale_inv = outs[2] if is_scaled else None
    else:
        device = tokens_data.device
        zero_copy = buffer.zero_copy
        if is_scaled:
            # recv data + scales share one buffer (data then scales); carve or allocate here.
            recv_tokens, recv_scale_inv = _scale_alloc_io(
                recv_tokens,
                num_recv_tokens,
                hidden,
                tokens_scale_inv.shape[-1],
                tokens_data.dtype,
                tokens_scale_inv.dtype,
                device,
                zero_copy,
            )
        elif recv_tokens is None:
            recv_tokens = _alloc_io((num_recv_tokens, hidden), payload_dtype, device, zero_copy)
        if recv_topk_weights is None:
            recv_topk_weights = _alloc_io((num_recv_tokens,), torch.float32, device, zero_copy)
        dispatch_recv = recv_tokens.view(torch.float8_e4m3fn) if is_scaled else recv_tokens
        torch.ops.transformer_engine_ep.prepare_and_dispatch(
            handle_mem,
            topk_idx,
            dispatch_tokens,
            topk_weights,
            tokens_per_expert,
            total_recv_tokens,
            top_k,
            alignment,
            dispatch_recv,
            recv_topk_weights,
            recv_scale_inv,
            tokens_scale_inv,
        )
    state = _DispatchState(
        handle_mem=handle_mem,
        tokens_shape=tokens.shape,
        topk_weights_shape=topk_weights.shape,
        tokens_T_flat=tokens_data.shape[0],
        topk_T_flat=topk_weights.numel() // topk_weights.shape[-1],
        top_k=topk_weights.shape[-1],
        hidden_dim=hidden,
        eager=eager,
        is_scaled=is_scaled,
    )
    # For scaled inputs the expert-major recv data + scales are wrapped into a per-expert
    # GroupedTensor so downstream grouped GEMM and autograd see a proper quantized grouped tensor.
    if is_scaled:
        recv_out = _make_grouped_mxfp8(
            recv_tokens.view(tokens._rowwise_data.dtype),
            recv_scale_inv,
            tokens_per_expert,
            tokens._fp8_dtype,
            tokens._dtype,
        )
        return recv_out, recv_topk_weights, state
    return recv_tokens, recv_topk_weights, state


def _ep_prepare_and_dispatch_nvfp4_fwd(
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_idx: torch.Tensor,
    buffer: "EpBuffer",
    recv_tokens: Optional[torch.Tensor],
    recv_topk_weights: Optional[torch.Tensor],
):
    """NVFP4 dispatch: pack each token to an NVFP4 row, move the rows as an unscaled payload, and
    decode them back to BF16 on the receiver. The packed row travels as a BF16 view because the
    forward dispatch only copies bytes. Returns BF16 recv tokens, so the expert path and the
    dispatch backward are unchanged; the backward is a straight-through BF16 gradient."""
    tokens_2d = tokens.reshape(-1, tokens.shape[-1])
    hidden = tokens_2d.shape[-1]
    packed = _nvfp4_pack_tokens(tokens_2d, pow2_scale=_NVFP4_DIRECT_RECEIVE).view(torch.bfloat16)
    packed_cols = packed.shape[-1]
    handle_mem = buffer.handle_mem
    tokens_per_expert = buffer.tokens_per_expert
    total_recv_tokens = buffer.total_recv_tokens
    eager = buffer.eager
    if eager:
        if recv_tokens is not None or recv_topk_weights is not None:
            raise ValueError("eager mode cannot use caller-supplied recv buffers")
        recv_packed, recv_topk_weights = tex.ep_prepare_and_dispatch(
            handle_mem,
            topk_idx,
            packed,
            topk_weights,
            tokens_per_expert,
            total_recv_tokens,
            buffer.top_k,
            buffer.alignment,
        )
        recv_out = _nvfp4_unpack_tokens(recv_packed.view(torch.uint8), hidden)
        fused = False
    else:
        device = tokens_2d.device
        num_recv_tokens = buffer.recv_capacity_per_rank
        recv_packed = _alloc_io(
            (num_recv_tokens, packed_cols), torch.bfloat16, device, buffer.zero_copy
        )
        if recv_topk_weights is None:
            recv_topk_weights = _alloc_io(
                (num_recv_tokens,), torch.float32, device, buffer.zero_copy
            )
        torch.ops.transformer_engine_ep.prepare_and_dispatch(
            handle_mem,
            topk_idx,
            packed,
            topk_weights,
            tokens_per_expert,
            total_recv_tokens,
            buffer.top_k,
            buffer.alignment,
            recv_packed,
            recv_topk_weights,
            None,
            None,
        )
        fused = (
            (_NVFP4_FUSED_RECEIVE or _NVFP4_DIRECT_RECEIVE)
            and recv_tokens is None
            and _nvfp4_compute_active()
        )
        _log_nvfp4_receive_path_once(fused)
        if fused and _NVFP4_DIRECT_RECEIVE:
            recv_out = _nvfp4_direct_receive(
                recv_packed.view(torch.uint8), hidden, tokens_per_expert, tokens.dtype
            )
        elif fused:
            recv_out = _nvfp4_fused_receive(
                recv_packed.view(torch.uint8), hidden, tokens_per_expert, tokens.dtype
            )
        else:
            if recv_tokens is None:
                recv_tokens = torch.empty(
                    num_recv_tokens, hidden, dtype=torch.bfloat16, device=device
                )
            recv_out = _nvfp4_unpack_tokens(
                recv_packed.view(torch.uint8), hidden, counts=tokens_per_expert, out=recv_tokens
            )
    state = _DispatchState(
        handle_mem=handle_mem,
        tokens_shape=tokens.shape,
        topk_weights_shape=topk_weights.shape,
        tokens_T_flat=tokens_2d.shape[0],
        topk_T_flat=topk_weights.numel() // topk_weights.shape[-1],
        top_k=topk_weights.shape[-1],
        hidden_dim=hidden,
        eager=eager,
        # A fused-receive GroupedTensor is returned like the MXFP8 one (not detached).
        is_scaled=not eager and fused,
    )
    return recv_out, recv_topk_weights, state


def _ep_dispatch_bwd(
    state: "_DispatchState",
    g_recv_tokens: torch.Tensor,
    g_recv_topk_weights: torch.Tensor,
):
    """Run dispatch_bwd and reshape the grads to the forward input layout. Returns
    ``(grad_tokens, grad_topk_weights)``. No autograd; the caller owns context handling."""
    handle_mem = state.handle_mem
    device = handle_mem.device
    g_recv_tokens = g_recv_tokens.contiguous()
    g_recv_topk_weights = g_recv_topk_weights.contiguous()
    # Dispatch grad follows the recv grad's (high-precision) dtype; the quantizer's STE
    # owns the fp8 boundary for scaled inputs.
    grad_tokens = torch.empty(
        state.tokens_T_flat, state.hidden_dim, dtype=g_recv_tokens.dtype, device=device
    )
    grad_topk_weights = torch.empty(
        state.topk_T_flat, state.top_k, dtype=torch.float32, device=device
    )
    # Eager is not graph-capturable, so call the backend op directly and skip torch.library.
    if state.eager:
        tex.ep_dispatch_bwd(
            handle_mem, g_recv_tokens, g_recv_topk_weights, grad_tokens, grad_topk_weights
        )
    else:
        torch.ops.transformer_engine_ep.dispatch_bwd(
            handle_mem,
            g_recv_tokens,
            g_recv_topk_weights,
            grad_tokens,
            grad_topk_weights,
        )
    return (
        grad_tokens.view(state.tokens_shape),
        grad_topk_weights.view(state.topk_weights_shape),
    )


class _EpPrepareAndDispatch(torch.autograd.Function):
    """Autograd prepare and dispatch grouped into one C++ op (two kernel launches, not a fused
    kernel). In eager mode the op sizes and allocates the recv outputs from the per-step host
    recv-count, so no Python runs between the count read and the dispatch launch; caller-supplied
    buffers and zero-copy are then forbidden. Otherwise the recv outputs are allocated here to the
    static recv capacity (caller-supplied or symm-mem-backed under zero-copy) and passed in. When
    MXFP8 is configured, forward quantizes ``tokens`` to lightweight storage locally while
    keeping the high-precision tensor as the autograd operand. The compute lives in
    ``_ep_prepare_and_dispatch_fwd`` / ``_ep_dispatch_bwd``; this wrapper only bridges autograd
    context handling."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_idx: torch.Tensor,
        buffer: "EpBuffer",
        recv_tokens: Optional[torch.Tensor] = None,
        recv_topk_weights: Optional[torch.Tensor] = None,
    ):
        """Only tokens and topk_weights are differentiable, so the non-diff buffer tensors ride on
        the buffer object to keep the autograd operand list short."""
        recv_out, recv_topk_weights, state = _ep_prepare_and_dispatch_fwd(
            tokens,
            topk_weights,
            topk_idx,
            buffer,
            recv_tokens,
            recv_topk_weights,
        )
        ctx.state = state
        # Detach so the long-lived buffers aren't tracked as differentiable outputs; autograd
        # re-attaches grad_fn pointing back at this Function. Eager recv tensors are freshly built
        # in C++ (requires_grad=False, no input alias), so skip detach and its Python->C++ hop. The
        # scaled recv is a freshly built GroupedTensor, so only its topk weights need detaching.
        if state.is_scaled:
            return recv_out, recv_topk_weights if state.eager else recv_topk_weights.detach()
        if state.eager:
            return recv_out, recv_topk_weights
        return recv_out.detach(), recv_topk_weights.detach()

    @staticmethod
    def backward(ctx, g_recv_tokens, g_recv_topk_weights):  # type: ignore[override]
        """Dispatch bwd: run dispatch_bwd and reshape grads to the fwd input layout."""
        grad_tokens, grad_topk_weights = _ep_dispatch_bwd(
            ctx.state, g_recv_tokens, g_recv_topk_weights
        )
        return (
            grad_tokens,
            grad_topk_weights,
            None,  # topk_idx
            None,  # buffer
            None,  # recv_tokens
            None,  # recv_topk_weights
        )


@dataclass(slots=True)
class _CombineState:
    """Backward state for the combine op. ``grad_out`` and ``handle_mem`` ride here as plain
    attributes rather than through save_for_backward, which would version-track tensors we mutate.
    The autograd wrapper stashes this on ctx; other callers keep it however they manage state."""

    handle_mem: torch.Tensor
    grad_out: Optional[torch.Tensor]
    bwd_quant_recipe: object
    token_counts: torch.Tensor
    expert_out_shape: torch.Size
    expert_out_dtype: torch.dtype
    device: torch.device
    eager: bool
    zero_copy: bool
    # NVFP4 combine backward returns FC2's grad_output as a direct-receive GroupedTensor. Decided
    # in the forward, where the NVFP4 autocast state is visible.
    nvfp4_direct: bool = False


def _ep_combine_fwd(
    expert_out: torch.Tensor,
    grad_out: Optional[torch.Tensor],
    buffer: "EpBuffer",
    num_local_tokens: int,
    bwd_quant_recipe,
):
    """Run combine and return ``(result, _CombineState)``. Eager mode is not graph-capturable, so
    it calls the backend op directly and skips the torch.library dispatch. No autograd; the caller
    owns context handling."""

    handle_mem = buffer.handle_mem
    eager = buffer.eager
    device = expert_out.device
    result = torch.empty(num_local_tokens, buffer.hidden_dim, dtype=expert_out.dtype, device=device)
    if eager:
        tex.ep_combine(handle_mem, expert_out, result)
    else:
        torch.ops.transformer_engine_ep.combine(handle_mem, expert_out, result)
    state = _CombineState(
        handle_mem=handle_mem,
        grad_out=grad_out,
        bwd_quant_recipe=bwd_quant_recipe,
        token_counts=buffer.tokens_per_expert,
        expert_out_shape=expert_out.shape,
        expert_out_dtype=expert_out.dtype,
        device=device,
        eager=eager,
        zero_copy=buffer.zero_copy,
        nvfp4_direct=(
            _NVFP4_DIRECT_CBWD
            and not eager
            and _is_nvfp4_recipe(bwd_quant_recipe)
            and _nvfp4_compute_active()
        ),
    )
    return result, state


def _ep_combine_bwd(state: "_CombineState", g_result: torch.Tensor):
    """Scatter the result-grad to expert positions and return the expert_out grad. High-precision
    sends the grad as-is; a quantized recipe (MXFP8 today) quantizes it and returns a per-expert
    GroupedTensor. No autograd; the caller owns context handling."""
    _require_bf16("combine backward grad_output", g_result)
    if not g_result.is_contiguous():
        g_result = g_result.contiguous()
    handle_mem = state.handle_mem

    if _is_nvfp4_recipe(state.bwd_quant_recipe):
        grad_expert_out = _ep_combine_bwd_nvfp4(state, g_result)
    elif state.bwd_quant_recipe is None:
        grad_expert_out = state.grad_out
        if grad_expert_out is None:
            grad_expert_out = _alloc_io(
                state.expert_out_shape, state.expert_out_dtype, state.device, state.zero_copy
            )
        if state.eager:
            tex.ep_combine_bwd(handle_mem, g_result, grad_expert_out, None, None)
        else:
            torch.ops.transformer_engine_ep.combine_bwd(handle_mem, g_result, grad_expert_out)
    else:
        from .tensor.mxfp8_tensor import MXFP8Quantizer

        quantizer = MXFP8Quantizer(
            DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
        )
        quantizer.internal = True
        mx, grad_scale_inv = quantize_for_ep(g_result, quantizer)
        g_data = mx._rowwise_data
        recv_pr, hidden = state.expert_out_shape[0], state.expert_out_shape[-1]
        ge_data, ge_scale_inv = _scale_alloc_io(
            state.grad_out,
            recv_pr,
            hidden,
            grad_scale_inv.shape[-1],
            g_data.dtype,
            grad_scale_inv.dtype,
            state.device,
            state.zero_copy,
        )
        # The backend keys on the fp8 scaling mode; reinterpret the byte-backed data as fp8.
        g_data_fp8 = g_data.view(torch.float8_e4m3fn)
        ge_data_fp8 = ge_data.view(torch.float8_e4m3fn)
        if state.eager:
            tex.ep_combine_bwd(handle_mem, g_data_fp8, ge_data_fp8, grad_scale_inv, ge_scale_inv)
        else:
            torch.ops.transformer_engine_ep.combine_bwd(
                handle_mem, g_data_fp8, ge_data_fp8, grad_scale_inv, ge_scale_inv
            )
        grad_expert_out = _make_grouped_mxfp8(
            ge_data, ge_scale_inv, state.token_counts, mx._fp8_dtype, state.expert_out_dtype
        )
    return grad_expert_out


def _ep_combine_bwd_nvfp4(state: "_CombineState", g_result: torch.Tensor) -> torch.Tensor:
    """NVFP4 combine backward: pack the result-grad rows to NVFP4 with stochastic rounding, scatter
    the packed rows as an unscaled payload (the combine backward only copies bytes), and decode
    them to BF16 in the expert_out grad. Decoding into ``state.grad_out`` when the caller supplies
    it, bounded by the device-side per-expert counts so CUDA graph replays see the live count."""
    recv_rows, hidden = state.expert_out_shape[0], state.expert_out_shape[-1]
    if state.nvfp4_direct:
        packed = _nvfp4_pack_tokens_cuda(g_result.reshape(-1, hidden), stochastic=True)
        ge_packed = _alloc_io(
            (recv_rows, packed.shape[-1] // 2), torch.bfloat16, state.device, state.zero_copy
        )
        torch.ops.transformer_engine_ep.combine_bwd(
            state.handle_mem, packed.view(torch.bfloat16), ge_packed
        )
        _log_nvfp4_cbwd_path_once()
        return _nvfp4_direct_receive(
            ge_packed.view(torch.uint8),
            hidden,
            state.token_counts,
            state.expert_out_dtype,
            stochastic=True,
        )
    packed = _nvfp4_pack_tokens(g_result.reshape(-1, hidden), stochastic=True).view(torch.bfloat16)
    ge_packed = _alloc_io(
        (recv_rows, packed.shape[-1]), torch.bfloat16, state.device, state.zero_copy
    )
    if state.eager:
        tex.ep_combine_bwd(state.handle_mem, packed, ge_packed, None, None)
    else:
        torch.ops.transformer_engine_ep.combine_bwd(state.handle_mem, packed, ge_packed)
    grad_expert_out = state.grad_out
    if grad_expert_out is None:
        grad_expert_out = torch.empty(
            state.expert_out_shape, dtype=state.expert_out_dtype, device=state.device
        )
    return _nvfp4_unpack_tokens(
        ge_packed.view(torch.uint8),
        hidden,
        counts=None if state.eager else state.token_counts,
        out=grad_expert_out.view(-1, hidden),
    ).view(state.expert_out_shape)


class _EpCombine(torch.autograd.Function):
    """Autograd combine; bwd scatters the expert_out grad into ``grad_out``. When the caller
    supplies it that buffer is used as-is; otherwise it is allocated in the backward from the
    symm-mem pool in zero-copy mode (one-sided target) or a plain tensor in normal mode (keeps
    allocation torch.compile / CUDA-graph safe and lets autograd own the grad's lifetime).

    Only ``expert_out`` is differentiable; the non-diff buffer tensors ride on ``buffer`` to keep
    the operand list short. The compute lives in ``_ep_combine_fwd`` / ``_ep_combine_bwd``; this
    wrapper only bridges autograd context handling."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        expert_out: torch.Tensor,
        grad_out: Optional[torch.Tensor],
        buffer: "EpBuffer",
        num_local_tokens: int,
        bwd_quant_recipe=None,
    ):
        """Combine fwd; stashes the backward state on ctx. When ``bwd_quant_recipe`` is set, the
        backward sends the result-grad as MXFP8."""
        result, ctx.state = _ep_combine_fwd(
            expert_out, grad_out, buffer, num_local_tokens, bwd_quant_recipe
        )
        return result

    @staticmethod
    def backward(ctx, g_result):  # type: ignore[override]
        """Combine bwd; scatters the result-grad to expert positions."""
        grad_expert_out = _ep_combine_bwd(ctx.state, g_result)
        return (
            grad_expert_out,
            None,  # grad_out
            None,  # buffer
            None,  # num_local_tokens
            None,  # bwd_quant_recipe
        )


# Public high-level wrappers


# NCCL EP inputs are bfloat16; MXFP8 is applied internally via the buffer's dispatch_fwd_quant_recipe.
def _require_bf16(name: str, t: torch.Tensor) -> None:
    if (
        not isinstance(t, torch.Tensor)
        or isinstance(t, QuantizedTensorStorage)
        or t.dtype is not torch.bfloat16
    ):
        raise TypeError(
            f"NCCL EP requires {name} to be a plain BF16 tensor, "
            f"got {type(t).__name__} with dtype={getattr(t, 'dtype', None)}."
        )


def _alloc_io(shape, dtype: torch.dtype, device, zero_copy: bool) -> torch.Tensor:
    """Allocate a dispatch/combine IO tensor the caller did not supply: from the symm-mem pool in
    zero-copy mode (auto-registered segment, lifecycle managed by torch refcount), else plain.

    The zero-copy pool path is not CUDA-graph capturable; supply persistent recv_tokens / grad_out
    buffers to capture a graph."""
    if zero_copy:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "EP zero-copy pool allocation is not CUDA-graph capturable; supply persistent "
                "recv_tokens / grad_out buffers (allocated once via symm_mem_alloc) before capture."
            )
        t = symm_mem_alloc(shape, dtype, _EP_GROUP, device=device, use_pool=True)
        # symm-mem storage is non-resizable; exempt it from CPU activation offloading (which
        # releases via storage.resize_(0)). Matters for bf16 recv_tokens (the saved activation).
        mark_not_offload(t)
        return t
    return torch.empty(*shape, dtype=dtype, device=device)


def quantize_for_ep(
    input_: torch.Tensor | QuantizedTensorStorage,
    quantizer: Optional["Quantizer"],
) -> tuple[MXFP8TensorStorage, torch.Tensor]:
    """Quantize an EP input with a supported quantizer."""
    from .tensor.mxfp8_tensor import MXFP8Quantizer

    if quantizer is None:
        raise ValueError("EP quantization requires a quantizer.")
    if isinstance(quantizer, MXFP8Quantizer):
        return _quantize_mxfp8(input_, quantizer)
    raise NotImplementedError(
        "EP dispatch/combine quantization currently supports only MXFP8; "
        f"got {type(quantizer).__name__}."
    )


def _quantize_mxfp8(
    input_: torch.Tensor | QuantizedTensorStorage,
    quantizer: "MXFP8Quantizer",
) -> tuple[MXFP8TensorStorage, torch.Tensor]:
    """Return E4M3 MXFP8 storage and compact rowwise scales for EP."""
    from .constants import MXFP8_BLOCK_SCALING_SIZE

    if quantizer.dtype != DType.kFloat8E4M3:
        raise NotImplementedError("EP supports only E4M3 MXFP8 data.")
    if isinstance(input_, MXFP8TensorStorage):
        quantized = input_
    elif isinstance(input_, QuantizedTensorStorage):
        raise TypeError(f"EP requires an MXFP8 input, got {type(input_).__name__}.")
    else:
        if not quantizer.internal:
            quantizer = quantizer.copy()
            quantizer.internal = True
        quantized = quantizer(input_)

    if quantized._fp8_dtype != DType.kFloat8E4M3:
        raise NotImplementedError("EP supports only E4M3 MXFP8 data.")
    if quantized._with_gemm_swizzled_scales:
        raise ValueError("EP requires unswizzled MXFP8 scales.")
    data = quantized._rowwise_data
    scale_inv = quantized._rowwise_scale_inv
    if data is None or scale_inv is None:
        raise ValueError("EP requires rowwise MXFP8 data and scales.")
    t_flat, hidden = input_.shape
    scale_cols = hidden // MXFP8_BLOCK_SCALING_SIZE
    # The backend forwards each token's scale row with a 16-byte-aligned store, so the row
    # (cols * dtype bytes) must be a multiple of 16.
    scale_row_bytes = scale_cols * scale_inv.element_size()
    if scale_row_bytes % 16 != 0:
        raise ValueError(
            f"MXFP8 dispatch requires a 16-byte-aligned scale row; hidden={hidden} gives "
            f"{scale_row_bytes} bytes. Use a hidden size that is a multiple of "
            f"{16 * MXFP8_BLOCK_SCALING_SIZE}."
        )
    # scale_inv is 2D [round_up(T, 128), cols]; drop the row padding to the logical [T, H/block]
    # the backend expects. cols is a multiple of 4 (16-byte row), so no column padding and the
    # slice stays contiguous; assert rather than force a copy.
    scale_inv = scale_inv[:t_flat, :scale_cols]
    if not scale_inv.is_contiguous():
        raise ValueError("EP requires compact contiguous MXFP8 scales.")
    return quantized, scale_inv


def _scale_alloc_io(buf, rows, data_cols, scale_cols, data_dtype, scale_dtype, device, zero_copy):
    """Block-scaled output data + scale buffers, each ``rows`` tall, laid out back-to-back
    (``[rows, data_cols]`` data of ``data_dtype`` then ``[rows, scale_cols]`` scales of
    ``scale_dtype``). Carve both from a single caller ``buf`` when it is large enough, so one
    symm-mem window backs both views; else allocate them (symm-mem pool under zero-copy, else
    plain). Recipe-agnostic: byte sizes come from the element sizes."""
    data_bytes = rows * data_cols * data_dtype.itemsize
    scale_bytes = rows * scale_cols * scale_dtype.itemsize
    if buf is not None:
        # Reinterpret in place; a non-contiguous buf would force a copy and leave the caller's
        # buffer unwritten, so require contiguous and view rather than reshape.
        if not buf.is_contiguous():
            raise ValueError("scaled output buffer must be contiguous.")
        flat = buf.view(-1).view(torch.uint8)
        if flat.numel() < data_bytes + scale_bytes:
            raise ValueError(
                f"scaled output buffer too small: need {data_bytes + scale_bytes} bytes "
                f"(data + scales), got {flat.numel()}."
            )
        data = flat[:data_bytes].view(data_dtype).reshape(rows, data_cols)
        scale_inv = (
            flat[data_bytes : data_bytes + scale_bytes].view(scale_dtype).reshape(rows, scale_cols)
        )
        return data, scale_inv
    data = _alloc_io((rows, data_cols), data_dtype, device, zero_copy)
    scale_inv = _alloc_io((rows, scale_cols), scale_dtype, device, zero_copy)
    return data, scale_inv


def _make_grouped_mxfp8(data, scale_inv, token_counts, fp8_dtype, fake_dtype):
    """Wrap expert-major MXFP8 recv data + compact e8m0 scales as a per-expert ``GroupedTensor``.

    ``token_counts`` (int64 [num_local_experts]) is the padded per-expert row counts (128-aligned),
    used as the group sizes. Grouping is device-side (first_dims/tensor_offsets), so the counts never
    sync to host; the outer shape is the static recv capacity, bounded per expert by first_dims.
    """
    from .tensor.grouped_tensor import GroupedTensor
    from .tensor.mxfp8_tensor import MXFP8Quantizer

    assert data.dim() == 2, "recv data must be 2D [capacity_rows, hidden]"
    capacity_rows, hidden = data.shape
    quantizer = MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=False)
    return GroupedTensor(
        shape=(capacity_rows, hidden),
        dtype=fake_dtype,
        num_tensors=token_counts.numel(),
        quantizer=quantizer,
        data=data.reshape(-1).detach(),
        scale_inv=scale_inv.reshape(-1).detach(),
        first_dims=token_counts,
        tensor_offsets=tex.splits_to_offsets(token_counts, hidden),
    )


def ep_dispatch(
    buffer: EpBuffer,
    tokens: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    recv_tokens: Optional[torch.Tensor] = None,
    recv_topk_weights: Optional[torch.Tensor] = None,
):
    """Prepare + dispatch with autograd. ``tokens`` is bfloat16; ``topk_idx`` is int32 or int64.

    When the buffer's ``dispatch_fwd_quant_recipe`` is set (``MXFP8BlockScaling`` only for now), tokens
    are quantized internally and recv is returned as a per-expert ``GroupedTensor``; otherwise recv
    stays bfloat16. A pre-quantized ``tokens`` is not accepted.

    ``recv_tokens`` / ``recv_topk_weights`` are the recv outputs: pass caller-owned buffers
    (symm-mem-backed under zero-copy) or leave them None to allocate. For MXFP8 the recv data and
    scales share ``recv_tokens`` (data then scales), so size it to at least
    ``recv_capacity_per_rank * (hidden + hidden/block)`` bytes. Eager mode sizes the recv outputs
    per step and forbids caller-supplied buffers. Under zero-copy, leaving them None allocates from
    the symm-mem pool, which is not CUDA-graph capturable; pass persistent buffers to capture a graph.

    Returns (recv_tokens, recv_topk_weights, tokens_per_expert); tokens_per_expert is non-diff. See
    ``buffer.total_recv_tokens`` for the per-step recv total.
    """
    if topk_weights.dtype is not torch.float32:
        raise TypeError(
            f"topk_weights must be float32; got dtype={topk_weights.dtype}. "
            "Cast with topk_weights.float() before calling."
        )
    if isinstance(tokens, QuantizedTensor):
        raise NotImplementedError(
            "NCCL EP dispatch takes a bfloat16 input and quantizes internally when the buffer's "
            "dispatch_fwd_quant_recipe is set; a pre-quantized tensor is not accepted."
        )
    _require_bf16("tokens", tokens)
    if buffer.eager and (recv_tokens is not None or recv_topk_weights is not None):
        raise ValueError(
            "eager mode sizes the recv outputs from the per-step recv-token total "
            "and cannot use caller-supplied recv_tokens / recv_topk_weights"
        )

    if buffer.dispatch_fwd_quant_recipe is not None:
        from ..common.recipe import MXFP8BlockScaling

        if not isinstance(
            buffer.dispatch_fwd_quant_recipe, MXFP8BlockScaling
        ) and not _is_nvfp4_recipe(buffer.dispatch_fwd_quant_recipe):
            raise NotImplementedError(
                "EP block-scaled dispatch supports MXFP8BlockScaling and NVFP4BlockScaling only; got "
                f"{type(buffer.dispatch_fwd_quant_recipe).__name__}."
            )

    # Fused prepare + dispatch in one C++ op. Eager sizes the recv outputs from the per-step host
    # recv-count (allocated in C++, no caller buffers); non-eager sizes them to the static recv
    # capacity here (caller-supplied or symm-mem-backed under zero-copy) and passes them in.
    recv_out, recv_topk_weights = _EpPrepareAndDispatch.apply(
        tokens,
        topk_weights,
        topk_idx,
        buffer,
        recv_tokens,
        recv_topk_weights,
    )
    return recv_out, recv_topk_weights, buffer.tokens_per_expert


def ep_combine(
    buffer: EpBuffer,
    expert_out: torch.Tensor,
    *,
    num_local_tokens: Optional[int] = None,
    grad_out: Optional[torch.Tensor] = None,
):
    """Combine with autograd; caller pre-applies topk weighting.

    ``expert_out`` is the combine input (symm-mem-backed under zero-copy). ``grad_out`` is the
    backward's grad target: pass a caller-owned buffer or leave it None to allocate. For MXFP8 the
    grad data and scales share ``grad_out`` (data then scales), so size it to at least
    ``recv_capacity_per_rank * (hidden + hidden/block)`` bytes (non-zero-copy only). Eager mode sizes
    the grad target per step and forbids a caller-supplied buffer. Under zero-copy, leaving it None
    allocates from the symm-mem pool, which is not CUDA-graph capturable; pass a persistent buffer to
    capture a graph. Result shape is (num_local_tokens, hidden_dim); num_local_tokens defaults to
    buffer.max_tokens_per_rank.
    """
    _require_bf16("expert_out", expert_out)
    if buffer.eager and grad_out is not None:
        raise ValueError(
            "eager mode sizes the combine grad target per step and cannot use a "
            "caller-supplied grad_out"
        )
    if num_local_tokens is None:
        num_local_tokens = buffer.max_tokens_per_rank
    # When combine_bwd_quant_recipe is set the combine backward sends the result-grad over the
    # wire quantized: MXFP8 returns the expert_out grad as a GroupedTensor, NVFP4 packs the rows
    # with stochastic rounding and decodes them back to a BF16 expert_out grad.
    bwd_quant_recipe = None
    if buffer.combine_bwd_quant_recipe is not None:
        from ..common.recipe import MXFP8BlockScaling

        if not isinstance(
            buffer.combine_bwd_quant_recipe, MXFP8BlockScaling
        ) and not _is_nvfp4_recipe(buffer.combine_bwd_quant_recipe):
            raise NotImplementedError(
                "EP combine backward supports MXFP8BlockScaling and NVFP4BlockScaling only; got "
                f"{type(buffer.combine_bwd_quant_recipe).__name__}."
            )
        bwd_quant_recipe = buffer.combine_bwd_quant_recipe
    return _EpCombine.apply(
        expert_out,
        grad_out,
        buffer,
        num_local_tokens,
        bwd_quant_recipe,
    )
