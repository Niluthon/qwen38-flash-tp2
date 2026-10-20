#!/usr/bin/env bash
# prepare-models-tp2.sh — download and prepare the weights of the two-card engine. Runs inside the tp2 image:
#
#   docker compose run --rm prepare
#
# Every step is skipped when its result is already there, so it is safe to run again (an interrupted download
# resumes). Needs ~130 GB free on the models volume (~185 GB with BUILD_INT8_PLE=1) and an internet connection;
# ~1-3 h depending on the link.
#
#   1. albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE @ $REVISION   -> /models/qwen38-w4a16              (~129 GB)
#   2. only with BUILD_INT8_PLE=1: INT8 PLE table built from its FP8 shards, then verified
#                                                               -> /models/qwen38-w4a16-int8ple       (~52.5 GB)
#      The engine serves the FP8 shards in place (patch 0050) and does not need it; it is the fallback table
#      (B70_PLE_INT8=1 + B70_PLE_INT8_NVME=1, docs/ple-nvme.md).
#   3. serve snapshots (symlinks + config.json) for the target -> /models/qwen38-w4a16-snapshot
#      and for the MTP draft                                    -> /models/qwen38-w4a16-mtp-snapshot
#
# Already downloaded? Put the files under the names above (HOW-TO-RUN.md, "Weights you already have"):
#   SKIP_DOWNLOAD=1   use $MODELS/qwen38-w4a16 as it is (no network, no revision check)
#   an INT8 table without its .ok marker is verified (not rebuilt) and then marked
# New files are given the owner of the models directory. A private or gated mirror: set HF_TOKEN / HF_ENDPOINT.
set -euo pipefail
REPO=${REPO:-albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE}
REVISION=${REVISION:-c6af9fee7ad2cb8ef44b534a99b8c1b81fb0f6da}
TOOLS=${TOOLS:-/b70-tools}
CONFIG=${CONFIG:-/b70-engines/serve-configs/serve-config-intel-autoround.json}
M=/models
CKPT=$M/qwen38-w4a16
PLE=$M/qwen38-w4a16-int8ple
PY=/opt/venv/bin/python
export HF_HUB_OFFLINE=0 HF_HOME=${HF_HOME:-/work/cache/hf}
step() { echo; echo "[prepare] $*"; }

[ -w "$M" ] || { echo "[prepare] $M is not writable (MODELS in .env)" >&2; exit 1; }
mkdir -p "$HF_HOME"

step "1/3 checkpoint $REPO @ ${REVISION:0:8} -> $CKPT"
if [ "${SKIP_DOWNLOAD:-0}" = 1 ]; then
  [ -f "$CKPT/config.json" ] && [ -d "$CKPT/runtime/mtp-int4-g32" ] \
    || { echo "[prepare] SKIP_DOWNLOAD=1 but $CKPT has no config.json or runtime/mtp-int4-g32" >&2; exit 1; }
  echo "SKIP_DOWNLOAD=1: using $CKPT as it is"
elif [ -f "$CKPT/.b70-download-ok" ] && [ "$(cat "$CKPT/.b70-download-ok")" = "$REVISION" ]; then
  echo "already downloaded"
else
  "$PY" -I -c '
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], max_workers=8)
' "$REPO" "$REVISION" "$CKPT"
  echo "$REVISION" > "$CKPT/.b70-download-ok"
fi

step "2/3 INT8 PLE table -> $PLE"
T=$PLE/ple_ngram_int8_rowscale.safetensors
if [ "${BUILD_INT8_PLE:-0}" != 1 ]; then
  echo "skipped: the engine reads the FP8 shards in place (0050); BUILD_INT8_PLE=1 builds the INT8 fallback table"
elif [ -f "$T.ok" ]; then
  echo "already built"
elif [ -f "$T" ]; then
  echo "found a table without its .ok marker: verifying it instead of rebuilding"
  CKPT=$CKPT OUT_DIR=$PLE "$PY" -I "$TOOLS/build_int8_ple_from_fp8.py" verify
  (cd "$PLE" && sha256sum ple_ngram_int8_rowscale.safetensors > "$T.ok.tmp" && mv "$T.ok.tmp" "$T.ok")
else
  rm -f "$T.part"
  mkdir -p "$PLE"
  CKPT=$CKPT OUT_DIR=$PLE "$PY" -I "$TOOLS/build_int8_ple_from_fp8.py" build
  CKPT=$CKPT OUT_DIR=$PLE "$PY" -I "$TOOLS/build_int8_ple_from_fp8.py" verify
fi

step "3/3 serve snapshots"
[ -e "$M/qwen38-w4a16-snapshot" ] && echo "target snapshot exists" \
  || "$PY" -I "$TOOLS/fp8ple_snapshot.py" "$CKPT" "$CONFIG" "$M/qwen38-w4a16-snapshot"
[ -e "$M/qwen38-w4a16-mtp-snapshot" ] && echo "MTP draft snapshot exists" \
  || "$PY" -I "$TOOLS/fp8ple_snapshot.py" --keep-quant "$CKPT/runtime/mtp-int4-g32" "$CONFIG" "$M/qwen38-w4a16-mtp-snapshot"

owner=$(stat -c %u:%g "$M")
[ -d "$PLE" ] || PLE=
chown -R "$owner" "$CKPT" ${PLE:+"$PLE"} "$M/qwen38-w4a16-snapshot" "$M/qwen38-w4a16-mtp-snapshot" 2>/dev/null || true
step "done; start the engine with: docker compose up -d"
