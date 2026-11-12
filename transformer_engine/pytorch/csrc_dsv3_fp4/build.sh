#!/bin/bash
# Builds the CUDA libraries the DeepSeek-V3 NVFP4 code paths on this branch load at runtime and
# installs them next to their Python modules in the *installed* transformer_engine package:
#   pytorch/triton/libce1p.so          <- ce1p.cu          (MLPERF_CE1P=1: MXFP8 CE backward, MTP concat)
#   pytorch/tensor/liboptfuse.so       <- optfuse.cu       (MLPERF_OPTFUSE_NVFP4=1 / MLPERF_OPTFUSE_GRADCOPY=1)
#   pytorch/_nvfp4_direct_ext.so       <- nvfp4_direct.cu  (NVTE_EP_NVFP4_DIRECT_RECEIVE=1, torch extension)
# Run after `pip install` of this TE tree:  bash transformer_engine/pytorch/csrc_dsv3_fp4/build.sh
set -euo pipefail
SRC=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
# Locate the installed package without importing it (no GPU needed at image build time).
TE_PT=${TE_PT:-$(cd / && python3 -c "import importlib.util, os; s = importlib.util.find_spec('transformer_engine'); print(os.path.join(list(s.submodule_search_locations)[0], 'pytorch'))")}
ARCHS=${ARCHS:-"100a 103a 107a"}
GENCODE=""; for a in ${ARCHS}; do GENCODE="${GENCODE} -gencode=arch=compute_${a},code=sm_${a}"; done
NVCC_FLAGS="-O3 -std=c++17 ${GENCODE} -lineinfo -shared -Xcompiler -fPIC"
nvcc ${NVCC_FLAGS} -o "${TE_PT}/triton/libce1p.so" "${SRC}/ce1p.cu"
nvcc ${NVCC_FLAGS} -o "${TE_PT}/tensor/liboptfuse.so" "${SRC}/optfuse.cu"
B=$(mktemp -d)
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"10.0a;10.3a;10.7a"}
GENCODE="${GENCODE}" SRC="${SRC}" B="${B}" OUT="${TE_PT}/_nvfp4_direct_ext.so" python3 - <<'PY'
import glob, os, shutil
from torch.utils.cpp_extension import load
load(
    name="_nvfp4_direct_ext",
    sources=[os.path.join(os.environ["SRC"], "nvfp4_direct.cu")],
    extra_cuda_cflags=["-O3", "-lineinfo"] + os.environ["GENCODE"].split(),
    build_directory=os.environ["B"],
    verbose=False,
)
shutil.copy(glob.glob(os.path.join(os.environ["B"], "_nvfp4_direct_ext*.so"))[0], os.environ["OUT"])
PY
rm -rf "${B}"
ls -la "${TE_PT}/triton/libce1p.so" "${TE_PT}/tensor/liboptfuse.so" "${TE_PT}/_nvfp4_direct_ext.so"
