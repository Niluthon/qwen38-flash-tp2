#!/usr/bin/env bash
# Build libb70_step_trace.so in the oneAPI 2026.0 compile container (same toolchain as kernels/b70-moe-decode).
#   kernels/b70-step-trace/build.sh   -> kernels/b70-step-trace/build/libb70_step_trace.so
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
IMAGE=${COMPILE_IMAGE:-k8v4-compile-2026:latest}
SETVARS=${COMPILE_SETVARS:-/opt/intel-2026/oneapi/setvars.sh}
mkdir -p "$here/build"
docker run --rm -v "$here:/src" -w /src --entrypoint bash "$IMAGE" -c "
set -e
source $SETVARS --force >/dev/null 2>&1
py=\${PY:-/opt/venv/bin/python}; command -v \$py >/dev/null || py=python3
T=\$(\$py -c 'import torch, os; print(os.path.dirname(torch.__file__))')
icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen '-device bmg' \
  -O2 -std=c++17 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I\$T/include -I\$T/include/torch/csrc/api/include \
  -x c++ /src/b70_step_trace.sycl -o /src/build/libb70_step_trace.so \
  -L\$T/lib -lc10 -lc10_xpu -ltorch -ltorch_cpu -ltorch_xpu -Wl,-rpath,\$T/lib
ls -la /src/build/libb70_step_trace.so
"
