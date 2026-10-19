#!/usr/bin/env bash
# Build libb70_moe_decode.so in the compile image (oneAPI 2026.0 + torch 2.13.0+xpu, the runtime's versions).
#   kernels/b70-moe-decode/build.sh            -> kernels/b70-moe-decode/build/libb70_moe_decode.so
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
IMAGE=${COMPILE_IMAGE:-k8v4-compile-2026:latest}
mkdir -p "$here/build"
docker run --rm -v "$here:/src" -w /src --entrypoint bash "$IMAGE" -c '
set -e
source /opt/intel-2026/oneapi/setvars.sh --force >/dev/null 2>&1
T=$(/opt/venv/bin/python -c "import torch,os;print(os.path.dirname(torch.__file__))")
icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg" \
  -O3 -std=c++17 -shared -fPIC -ffast-math -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I$T/include -I$T/include/torch/csrc/api/include \
  -x c++ b70_moe_decode.sycl -o build/libb70_moe_decode.so \
  -L$T/lib -lc10 -lc10_xpu -ltorch -ltorch_cpu -ltorch_xpu -Wl,-rpath,$T/lib
ls -la build/libb70_moe_decode.so
'
