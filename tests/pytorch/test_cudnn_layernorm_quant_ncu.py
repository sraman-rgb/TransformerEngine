# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Nsight Compute check for the cuDNN LayerNorm + MXFP8 quantization fusion."""

import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys

_NCU_TEST_ENV = "NVTE_RUN_NCU_TESTS"
_FUSED_KERNEL_NAME = "ln_tma_mxfp8_fused_kernel"


def _run_layernorm_quant_workload() -> None:
    """Launch the MXFP8 LayerNorm+ColumnLinear path used by fused MLA."""
    # NVTE reads this environment variable into a static on first use, so set it
    # before importing Transformer Engine.
    os.environ["NVTE_NORM_FWD_USE_CUDNN"] = "1"

    import torch
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling

    sequence_length = int(os.getenv("NVTE_TEST_LAYERNORM_ROWS", "4096"))
    batch_size = 1
    hidden_size = int(os.getenv("NVTE_TEST_LAYERNORM_HIDDEN_SIZE", "7168"))
    # DeepSeek-V3's fused MLA down projection concatenates q_lora_rank,
    # kv_lora_rank, and qk_pos_emb_head_dim: 1536 + 512 + 64 = 2112.
    down_proj_size = 2112
    x = torch.randn(
        (sequence_length, batch_size, hidden_size),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    layer = te.LayerNormLinear(
        hidden_size,
        down_proj_size,
        bias=False,
        params_dtype=torch.bfloat16,
        parallel_mode="column",
        tp_size=1,
        return_layernorm_output=False,
        device="cuda",
    )
    recipe = MXFP8BlockScaling()

    def run() -> None:
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            layer(x)

    # Build and cache the cuDNN graph before opening the profiling range.
    run()
    torch.cuda.synchronize()

    torch.cuda.profiler.start()
    run()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()


if __name__ == "__main__":
    _run_layernorm_quant_workload()
    sys.exit(0)


import pytest  # noqa: E402  # The standalone NCU workload does not require pytest.


@pytest.mark.skipif(
    os.getenv(_NCU_TEST_ENV, "0") != "1",
    reason=f"set {_NCU_TEST_ENV}=1 to run Nsight Compute kernel-selection tests",
)
def test_cudnn_layernorm_quant_uses_fused_kernel() -> None:
    """Verify that cuDNN selects one fused kernel for LayerNorm + MXFP8 quantization."""
    ncu = shutil.which("ncu")
    if ncu is None:
        pytest.skip("Nsight Compute (ncu) is not installed")

    command = [
        ncu,
        "--profile-from-start",
        "off",
        "--target-processes",
        "all",
        "--kernel-name",
        _FUSED_KERNEL_NAME,
        "--launch-count",
        "1",
        "--metrics",
        "gpu__time_duration.sum",
        "--kernel-name-base",
        "demangled",
        "--print-kernel-base",
        "demangled",
        "--csv",
        "--page",
        "raw",
        sys.executable,
        str(Path(__file__).resolve()),
    ]
    env = os.environ.copy()
    env["NVTE_NORM_FWD_USE_CUDNN"] = "1"
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    profiler_output = result.stdout + "\n" + result.stderr

    assert result.returncode == 0, (
        f"Nsight Compute failed with exit code {result.returncode}:\n{profiler_output}"
    )
    csv_lines = (line for line in result.stdout.splitlines() if line.startswith('"'))
    kernel_names = [
        row["Kernel Name"]
        for row in csv.DictReader(csv_lines)
        if row.get("Kernel Name")
    ]
    assert kernel_names == [_FUSED_KERNEL_NAME], (
        "Expected LayerNorm and MXFP8 quantization to execute with the cuDNN fused kernel "
        f"{_FUSED_KERNEL_NAME!r}, but NCU reported {kernel_names!r}. "
        f"Profiler output:\n{profiler_output}"
    )
