#!/usr/bin/env bash
# restart-tp2.sh [extra vllm serve flags...] — (re)start the tp2-hotcold engine detached and wait until it serves.
#
#   engines/two-cards/scripts/restart-tp2.sh                 # the configuration in engines/two-cards/args/serve-tp2.args
#   DOCKER_ARGS='-e B70_MOE_HOTCOLD_MEMLOG=1' engines/two-cards/scripts/restart-tp2.sh --enforce-eager
#
# Removes a running b70-tp2 container, starts engines/two-cards/scripts/run-tp2.sh with setsid (so the engine outlives the shell that
# started it), logs to $WORK/engine.log and prints the KV-cache and startup lines. run-tp2.sh then sends its warm-up
# request; wait for "[run-tp2] warm-up request OK" in the log before sending long prompts (known issues item 13).
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)   # engines/two-cards
WORK=${ENGINE_DATA:-${WORK:-$here/../../engine-data}}
NAME=${NAME:-b70-tp2}
log=$WORK/engine.log
docker rm -f "$NAME" >/dev/null 2>&1 || true
sleep 2
setsid nohup "$here/scripts/run-tp2.sh" "$@" > "$log" 2>&1 < /dev/null &
disown
for _ in $(seq 1 90); do
  sleep 10
  grep -q "Application startup complete\|Engine core initialization failed" "$log" && break
done
grep -E "KV cache size|startup complete|initialization failed" "$log" | sed 's/^.*\] //' | cut -c1-120
