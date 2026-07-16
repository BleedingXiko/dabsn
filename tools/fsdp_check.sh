#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
base="$PWD/.release-check"
venv="$base/fsdp-venv"
mkdir -p "$base"
dist=$(mktemp -d "$base/fsdp-dist.XXXXXX")

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "DABSN TWO-GPU FSDP CHECK: FAIL (Linux is required)"
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "DABSN TWO-GPU FSDP CHECK: FAIL (nvidia-smi is unavailable)"
  exit 1
fi
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
if [[ "$gpu_count" -lt 2 ]]; then
  echo "DABSN TWO-GPU FSDP CHECK: FAIL (found $gpu_count GPU; two are required)"
  exit 1
fi
nvidia-smi

python_cmd=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c \
    'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    python_cmd="$candidate"
    break
  fi
done
if [[ -z "$python_cmd" ]]; then
  echo "DABSN TWO-GPU FSDP CHECK: FAIL (Python 3.10 or newer is required)"
  exit 1
fi

"$python_cmd" -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip build setuptools wheel
"$venv/bin/python" -m build --wheel --outdir "$dist"
wheel=$(find "$dist" -maxdepth 1 -name 'dabsn-*.whl' -type f | sort | tail -1)
if [[ -z "$wheel" ]]; then
  echo "DABSN TWO-GPU FSDP CHECK: FAIL (wheel was not built)"
  exit 1
fi
"$venv/bin/python" -m pip install --upgrade \
  "torch==2.6.0" "triton==3.2.0" \
  "numpy>=1.24" "ninja>=1.11" "safetensors>=0.4.5"
"$venv/bin/python" -m pip install --force-reinstall --no-deps "$wheel"
"$venv/bin/torchrun" \
  --nnodes=1 \
  --node-rank=0 \
  --nproc-per-node=2 \
  --master-addr=127.0.0.1 \
  --master-port=29675 \
  tools/fsdp_check.py --output "$base/fsdp-output"
