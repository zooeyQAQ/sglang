#!/usr/bin/env bash
# Launch frozen EXP-0001 serving. Does not switch backends/graphs.
# Identity: worktree SHA must be faa56e2b5c5c5dee60eb5dd7b6251944a21a6104.
set +e
set -u

ROOT="${ROOT:-/data/sgl_pr/runs/perf/EXP-0001}"
WT="${WT:-/data/sgl_pr/worktrees/perf}"
PY="${PY:-/opt/sglang/bin/python3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
EXPECT_SHA="faa56e2b5c5c5dee60eb5dd7b6251944a21a6104"

mkdir -p "$ROOT/serve"
LOG="$ROOT/serve/serve.log"
PIDF="$ROOT/serve/serve.pid"
CMDFILE="$ROOT/serve/launch_cmd.txt"
ENVFILE="$ROOT/serve/launch_env.txt"

SHA=$(git -C "$WT" rev-parse HEAD)
if [ "$SHA" != "$EXPECT_SHA" ]; then
  echo "FATAL: worktree SHA $SHA != $EXPECT_SHA" >&2
  exit 2
fi
if git -C "$WT" status --porcelain | grep -q .; then
  echo "FATAL: worktree dirty" >&2
  git -C "$WT" status --porcelain >&2
  exit 3
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset DEEPGEMM_BLACKWELL SGLANG_DSV41_CANONICALIZE_C1_INDICES
unset SGLANG_ALLOW_DSV41_PP SGLANG_PP_LAYER_PARTITION
unset SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT
export PYTHONPATH="$WT/python"
export SGLANG_PROFILE_V2=0
export SGLANG_TORCH_PROFILER_DIR="$ROOT/profiler_default"
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export CUDA_DEVICE_ORDER=PCI_BUS_ID

{
  echo "PYTHONPATH=$PYTHONPATH"
  echo "SGLANG_PROFILE_V2=$SGLANG_PROFILE_V2"
  echo "SGLANG_TORCH_PROFILER_DIR=$SGLANG_TORCH_PROFILER_DIR"
  echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
  echo "NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL"
  echo "HOST=$HOST PORT=$PORT"
  echo "SHA=$SHA"
  date -Iseconds
} > "$ENVFILE"

CMD=(
  "$PY" -m sglang.launch_server
  --trust-remote-code
  --model-path /data1/models/DeepSeek-V4.1-Flash
  --tp-size 8
  --ep-size 8
  --mem-fraction-static 0.80
  --page-size 256
  --chunked-prefill-size 8192
  --max-running-requests 8
  --disable-radix-cache
  --host "$HOST"
  --port "$PORT"
  --watchdog-timeout 1800
)

printf '%q ' "${CMD[@]}" > "$CMDFILE"
echo >> "$CMDFILE"

# Launch in its own session so stop_serve can kill the tree without pkill -f sglang.
setsid "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "started pid=$(cat "$PIDF") log=$LOG"
exit 0
