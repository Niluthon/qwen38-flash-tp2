#!/usr/bin/env bash
# Compile libb70_moe_decode.so where icpx (oneAPI 2026.0, = the runtime's SYCL) and the runtime's torch are present:
#   compile.sh <source dir> <output .so> [python with torch]
# Used by build.sh (in a compile container) and by the b70moe stage of image/Dockerfile.tp2.
set -euo pipefail
src=$1 out=$2 py=${3:-/opt/venv/bin/python}
T=$("$py" -c "import torch, os; print(os.path.dirname(torch.__file__))")
mkdir -p "$(dirname "$out")"
icpx -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg" \
  -O3 -std=c++17 -shared -fPIC -ffast-math -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I"$T/include" -I"$T/include/torch/csrc/api/include" \
  -x c++ "$src/b70_moe_decode.sycl" -o "$out" \
  -L"$T/lib" -lc10 -lc10_xpu -ltorch -ltorch_cpu -ltorch_xpu -Wl,-rpath,"$T/lib"
ls -la "$out"
