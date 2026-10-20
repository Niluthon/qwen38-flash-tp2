#!/usr/bin/env bash
# Build libb70_hc.so in the oneAPI 2026.0 compile container (same toolchain as kernels/b70-moe-decode).
#   kernels/b70-hc/build.sh   -> kernels/b70-hc/build/libb70_hc.so
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
IMAGE=${COMPILE_IMAGE:-k8v4-compile-2026:latest}
SETVARS=${COMPILE_SETVARS:-/opt/intel-2026/oneapi/setvars.sh}
mkdir -p "$here/build"
docker run --rm -v "$here:/src" -w /src --entrypoint bash "$IMAGE" -c "
set -e
source $SETVARS --force >/dev/null 2>&1
bash /src/compile.sh /src /src/build/libb70_hc.so
"
