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
#
# Warm-up (WARMUP=1, the default): once /health answers, one short chat request runs inside the container. Known
# issue (docs/known-issues.md item 13): with MTP, if the FIRST request after start needs more than one prefill chunk
# (> max-num-batched-tokens prompt tokens), the MTP draft's graph replay fails ("Cannot prepare for replay during
# capturing stage") and the engine stops. Any short request first prevents it. WARMUP=0 skips it.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
IMAGE=${IMAGE:-b70-flash-next:tp2}
MODELS=${MODELS:-$HOME/models}; WORK=${WORK:-$HOME/b70-work}
MODEL_DIR=${MODEL_DIR:-/models/qwen38-w4a16-snapshot}
NAME=${NAME:-b70-tp2}
mkdir -p "$WORK"/cache/{hf,tmp,triton,vllm,xdg,ple-nvme-native}
port=$(sed -n 's/^--port //p' "$here/serve-tp2.args"); port=${port:-8000}
if [ "${WARMUP:-1}" = 1 ]; then
  ( for _ in $(seq 1 180); do
      sleep 5
      docker exec "$NAME" /opt/venv/bin/python -I -c '
import json, sys, urllib.request
base = "http://localhost:" + sys.argv[1]
urllib.request.urlopen(base + "/health", timeout=5)
q = {"model": "qwen38-flash-next", "max_tokens": 16, "temperature": 0,
     "messages": [{"role": "user", "content": "Say OK."}],
     "chat_template_kwargs": {"enable_thinking": False}}
r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions",
    json.dumps(q).encode(), {"Content-Type": "application/json"}), timeout=600))
print("[run-tp2] warm-up request OK:", repr(r["choices"][0]["message"]["content"]), flush=True)
' "$port" 2>/dev/null && exit 0
    done
    echo "[run-tp2] warm-up request did not succeed within 15 min" ) &
fi
mapfile -t args < <(grep -v '^#' "$here/serve-tp2.args" | sed '/^$/d' | sed 's/^\(--[^ ]*\) \(.*\)$/\1\n\2/')
exec docker run --rm --name "$NAME" \
  --device /dev/dri --group-add render --group-add video --ipc host --network host --shm-size 16g \
  --ulimit memlock=-1 -w /opt/venv \
  --env-file "$here/tp2-hotcold.env" ${DOCKER_ARGS:-} \
  -v /dev/dri/by-path:/dev/dri/by-path:ro -v "$MODELS:/models:ro" -v "$WORK:/work" \
  --entrypoint vllm "$IMAGE" serve "$MODEL_DIR" "${args[@]}" "$@"
