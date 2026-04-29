#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$ROOT_DIR/.venv-vllm/bin/python"
VLLM_BIN="$ROOT_DIR/.venv-vllm/bin/vllm"

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "vLLM is not installed. Run:"
  echo "  python3.12 -m venv .venv-vllm"
  echo "  .venv-vllm/bin/python -m pip install -r requirements-vllm-cpu.txt"
  exit 1
fi

MODEL="${VLLM_MODEL:-$(grep -E '^LLM_MODEL=' "$ROOT_DIR/.env" | tail -1 | cut -d= -f2-)}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
DTYPE="${VLLM_DTYPE:-float32}"

export VLLM_CPU_KVCACHE_SPACE="${VLLM_CPU_KVCACHE_SPACE:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

exec "$VLLM_BIN" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$MODEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --dtype "$DTYPE" \
  --enforce-eager
