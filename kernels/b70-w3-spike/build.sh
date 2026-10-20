#!/usr/bin/env bash
# Build the 3-bit spike op in the compile container (oneAPI 2026.0 + torch 2.13.0+xpu, as kernels/b70-moe-decode).
#   kernels/b70-w3-spike/build.sh  -> kernels/b70-w3-spike/build/libb70_w3_spike.so
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
IMAGE=${COMPILE_IMAGE:-k8v4-compile-2026:latest}
SETVARS=${COMPILE_SETVARS:-/opt/intel-2026/oneapi/setvars.sh}
EXTRA=${EXTRA_BACKEND:-}
mkdir -p "$here/build"
docker run --rm -v "$here:/src" -w /src --entrypoint bash "$IMAGE" -c "
set -e
source $SETVARS --force >/dev/null 2>&1
T=\$(/opt/venv/bin/python -c 'import torch, os; print(os.path.dirname(torch.__file__))' 2>/dev/null || python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen \"-device bmg $EXTRA\" \
  -O3 -std=c++17 -shared -fPIC -ffast-math -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I\$T/include -I\$T/include/torch/csrc/api/include \
  -x c++ /src/b70_w3_spike.sycl -o /src/build/libb70_w3_spike.so \
  -L\$T/lib -lc10 -lc10_xpu -ltorch -ltorch_cpu -ltorch_xpu -Wl,-rpath,\$T/lib
ls -la /src/build/libb70_w3_spike.so
"
