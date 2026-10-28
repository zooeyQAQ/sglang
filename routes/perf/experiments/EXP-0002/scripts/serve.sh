#!/usr/bin/env bash
# EXP-0002 D-less colocated TP8×EP4. Engram host table on CPU. Decode CUDA graph on.
# Reference envelope: 166:8004 pd_transfer serve_d_mooncake128.sh minus PD/DSpark/DP.
set +e
set -u

ROOT="${ROOT:-/data/sgl_pr/runs/perf/EXP-0002}"
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

ulimit -l unlimited 2>/dev/null || true

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
unset DEEPGEMM_BLACKWELL SGLANG_DSV41_CANONICALIZE_C1_INDICES
unset SGLANG_PP_LAYER_PARTITION
export PATH="/opt/sglang/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${WT}/python"
export SGLANG_PROFILE_V2=0
export SGLANG_TORCH_PROFILER_DIR="$ROOT/profiler_default"
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0
export SGLANG_OPT_USE_TILELANG_INDEXER=0
export SGLANG_SM120_FLASHMLA_BACKEND=flashinfer
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
export SGLANG_ALLOW_DSV41_PP=1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID

{
  echo "PYTHONPATH=$PYTHONPATH"
  echo "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=$SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE"
  echo "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=$SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT"
  echo "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=$SGLANG_FP8_PAGED_MQA_LOGITS_TORCH"
  echo "SGLANG_OPT_USE_TILELANG_INDEXER=$SGLANG_OPT_USE_TILELANG_INDEXER"
  echo "SGLANG_SM120_FLASHMLA_BACKEND=$SGLANG_SM120_FLASHMLA_BACKEND"
  echo "SGLANG_ALLOW_DSV41_PP=$SGLANG_ALLOW_DSV41_PP"
  echo "SGLANG_PROFILE_V2=$SGLANG_PROFILE_V2"
  echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
  echo "NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL"
  echo "HOST=$HOST PORT=$PORT"
  echo "SHA=$SHA"
  echo "NOTE: this SHA has no SGLANG_DSV41_ENGRAM_HOST_TABLE_PIN or DROP_PAGE_CACHE; cudaHostRegister always pins"
  date -Iseconds
} > "$ENVFILE"

CMD=(
  "$PY" -m sglang.launch_server
  --trust-remote-code
  --model-path /data1/models/DeepSeek-V4.1-Flash
  --language-model-only
  --tp-size 8
  --ep-size 4
  --mem-fraction-static 0.75
  --max-running-requests 32
  --chunked-prefill-size 8192
  --swa-prefix-tails 0
  --attention-backend dsv4
  --moe-runner-backend flashinfer_mxfp4
  --fp8-gemm-backend flashinfer_cutlass
  --kv-cache-dtype fp8_e4m3
  --page-size 256
  --disable-prefill-cuda-graph
  --cuda-graph-max-bs-decode 32
  --disable-radix-cache
  --disable-overlap-schedule
  --watchdog-timeout 1800
  --host "$HOST"
  --port "$PORT"
)

printf '%q ' "${CMD[@]}" > "$CMDFILE"
echo >> "$CMDFILE"

setsid "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDF"
echo "started pid=$(cat "$PIDF") log=$LOG"
exit 0
