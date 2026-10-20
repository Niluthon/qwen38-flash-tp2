#!/usr/bin/env bash
# compose-serve.sh serve|health|warmup — entrypoint of the compose services (compose.yaml).
#
#   serve   vllm serve $MODEL_DIR with the flags of $SERVE_ARGS_FILE (one flag and its value per line) plus
#           $EXTRA_ARGS (shell-quoted; appended, so the last value of a flag wins); log also to $ENGINE_LOG
#   health  exit 0 when /health answers on the serving port
#   warmup  one short chat request (known issues item 13), then exit
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-/models/qwen38-w4a16-snapshot}
SERVE_ARGS_FILE=${SERVE_ARGS_FILE:-/b70-engines/two-cards/args/serve-tp2.args}
ENGINE_LOG=${ENGINE_LOG:-/work/engine.log}
PY=/opt/venv/bin/python

args() {  # the serve flags, one word per line: file first, then EXTRA_ARGS
  grep -v '^#' "$SERVE_ARGS_FILE" | sed '/^$/d' | sed 's/^\(--[^ ]*\) \(.*\)$/\1\n\2/'
  if [ -n "${EXTRA_ARGS:-}" ]; then eval "printf '%s\n' $EXTRA_ARGS"; fi
}

port() {  # the last --port given (default 8000)
  args | awk 'p {v=$0; p=0} $0=="--port" {p=1} END {print (v ? v : 8000)}'
}

case "${1:-serve}" in
  serve)
    # B70_MOE_CPU from the host (.env / shell, via compose.yaml) wins over the env files
    if [ -n "${B70_MOE_CPU_FROM_HOST:-}" ]; then export B70_MOE_CPU=$B70_MOE_CPU_FROM_HOST; fi
    mkdir -p /work/cache/{hf,tmp,triton,vllm,xdg,ple-nvme-native}
    mapfile -t a < <(args)
    : > "$ENGINE_LOG"
    exec > >(tee -a "$ENGINE_LOG") 2>&1
    echo "[compose-serve] vllm serve $MODEL_DIR ${a[*]}"
    exec vllm serve "$MODEL_DIR" "${a[@]}"
    ;;
  health)
    exec "$PY" -I -c 'import sys, urllib.request; urllib.request.urlopen(f"http://localhost:{sys.argv[1]}/health", timeout=5)' "$(port)"
    ;;
  warmup)
    exec "$PY" -I -c '
import json, sys, urllib.request
base = "http://localhost:" + sys.argv[1]
q = {"model": sys.argv[2], "max_tokens": 16, "temperature": 0,
     "messages": [{"role": "user", "content": "Say OK."}], "chat_template_kwargs": {"enable_thinking": False}}
r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", json.dumps(q).encode(),
    {"Content-Type": "application/json"}), timeout=600))
print("[compose-serve] warm-up request OK:", repr(r["choices"][0]["message"]["content"]), flush=True)
' "$(port)" "$(args | awk 'p {v=$0; p=0} $0=="--served-model-name" {p=1} END {print (v ? v : "qwen38-flash-next")}')"
    ;;
  *) echo "usage: $0 serve|health|warmup" >&2; exit 2 ;;
esac
