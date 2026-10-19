#!/usr/bin/env bash
# Build libb70_moe_decode.so outside the image, in a compile container with oneAPI 2026.0 + torch 2.13.0+xpu (the
# runtime's versions). The engine image builds it itself (image/Dockerfile.tp2, stage b70moe); this is for development.
#   kernels/b70-moe-decode/build.sh            -> kernels/b70-moe-decode/build/libb70_moe_decode.so
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
IMAGE=${COMPILE_IMAGE:-k8v4-compile-2026:latest}
SETVARS=${COMPILE_SETVARS:-/opt/intel-2026/oneapi/setvars.sh}
mkdir -p "$here/build"
docker run --rm -v "$here:/src" -w /src --entrypoint bash "$IMAGE" -c "
set -e
source $SETVARS --force >/dev/null 2>&1
bash /src/compile.sh /src /src/build/libb70_moe_decode.so
"
