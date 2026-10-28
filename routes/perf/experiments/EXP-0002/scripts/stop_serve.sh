#!/usr/bin/env bash
# Stop only this experiment's serve tree. Never pkill -f sglang. Skip orangefs.
set +e
set -u
ROOT="${ROOT:-/data/sgl_pr/runs/perf/EXP-0002}"
PIDF="$ROOT/serve/serve.pid"
PORT="${PORT:-30000}"

kill_tree() {
  local pid="$1"
  [ -n "$pid" ] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  local kids
  kids=$(ps -o pid= --ppid "$pid" 2>/dev/null | tr -d ' ')
  for c in $kids; do
    kill_tree "$c"
  done
  # skip orangefs
  local cmd
  cmd=$(ps -o cmd= -p "$pid" 2>/dev/null || true)
  case "$cmd" in
    *orangefs*|*ofs-client*) echo "skip orangefs pid=$pid"; return 0 ;;
  esac
  kill -TERM "$pid" 2>/dev/null || true
}

if [ -f "$PIDF" ]; then
  PARENT=$(cat "$PIDF")
  echo "stop parent=$PARENT"
  # session leader from setsid
  kill -TERM -- "-$PARENT" 2>/dev/null || kill_tree "$PARENT"
  for i in $(seq 1 40); do
    kill -0 "$PARENT" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$PARENT" 2>/dev/null; then
    kill -KILL -- "-$PARENT" 2>/dev/null || kill -KILL "$PARENT" 2>/dev/null || true
  fi
fi

# leftover python launch_server / scheduler children of this experiment only:
# if port still held by a python in this pid ns, TERM that pid (not orangefs).
for i in $(seq 1 30); do
  holder=$(ss -lntp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p {print}')
  [ -z "$holder" ] && break
  echo "port $PORT still: $holder"
  pid=$(echo "$holder" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
  if [ -n "$pid" ]; then
    cmd=$(ps -o cmd= -p "$pid" 2>/dev/null || true)
    case "$cmd" in
      *orangefs*|*ofs-client*) echo "skip orangefs $pid" ;;
      *sglang*|*python*)
        echo "TERM leftover $pid $cmd"
        kill -TERM "$pid" 2>/dev/null || true
        ;;
    esac
  fi
  sleep 1
done
echo "stop_serve done"
exit 0
