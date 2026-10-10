#!/usr/bin/env bash
# run-tp2.sh [extra vllm serve flags...] — the tp2-hotcold engine (2x B70, ~64 GB RAM) in a container.
#
#   MODELS=~/models WORK=~/b70-work engines/run-tp2.sh
#
# Image: the official vLLM v0.30.0 XPU image (by digest) with vllm-xpu-kernels 0.1.14.1+b70.3 installed
# (image/Dockerfile.tp2). The code runs from the source tree on WORK (PYTHONPATH=/work/vllm-b70, branch
# b70/tp2-hotcold); the residency shim is read from WORK/opt. Environment: engines/tp2-hotcold.env; flags:
# engines/serve-tp2.args (one flag and its value per line). Extra arguments are appended to `vllm serve`;
# DOCKER_ARGS adds `docker run` options (e.g. DOCKER_ARGS='-e VLLM_LOGGING_LEVEL=DEBUG').
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
IMAGE=${IMAGE:-b70-flash-next:tp2}
MODELS=${MODELS:-$HOME/models}; WORK=${WORK:-$HOME/b70-work}
MODEL_DIR=${MODEL_DIR:-/models/qwen38-w4a16-snapshot}
NAME=${NAME:-b70-tp2}
mkdir -p "$WORK"/cache/{hf,tmp,triton,vllm,xdg,ple-nvme-native}
mapfile -t args < <(grep -v '^#' "$here/serve-tp2.args" | sed '/^$/d' | sed 's/^\(--[^ ]*\) \(.*\)$/\1\n\2/')
exec docker run --rm --name "$NAME" \
  --device /dev/dri --group-add render --group-add video --ipc host --network host --shm-size 16g \
  --ulimit memlock=-1 -w /opt/venv \
  --env-file "$here/tp2-hotcold.env" ${DOCKER_ARGS:-} \
  -v /dev/dri/by-path:/dev/dri/by-path:ro -v "$MODELS:/models:ro" -v "$WORK:/work" \
  --entrypoint vllm "$IMAGE" serve "$MODEL_DIR" "${args[@]}" "$@"
