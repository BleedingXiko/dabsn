#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
base="$PWD/.release-check"
venv="$base/gpu-venv"
mkdir -p "$base"
dist=$(mktemp -d "$base/gpu-dist.XXXXXX")

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "DABSN GPU CHECK: FAIL"
  echo "Run this on Linux. On a Windows laptop, use WSL2 Ubuntu."
  exit 1
fi
nvidia_smi=$(command -v nvidia-smi || true)
if [[ -z "$nvidia_smi" && -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  nvidia_smi=/usr/lib/wsl/lib/nvidia-smi
fi
if [[ -z "$nvidia_smi" ]]; then
  echo "DABSN GPU CHECK: FAIL"
  echo "nvidia-smi is unavailable; install the NVIDIA driver or enable WSL2 GPU passthrough."
  exit 1
fi
"$nvidia_smi"

python_cmd=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c \
    'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    python_cmd="$candidate"
    break
  fi
done
if [[ -z "$python_cmd" ]]; then
  echo "DABSN GPU CHECK: FAIL (Python 3.10 or newer is required)"
  exit 1
fi

"$python_cmd" -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip build setuptools wheel
rm -rf build src/*.egg-info
"$venv/bin/python" -m build --wheel --outdir "$dist"
wheel=$(find "$dist" -maxdepth 1 -name 'dabsn-*.whl' -type f | sort | tail -1)
if [[ -z "$wheel" ]]; then
  echo "DABSN GPU CHECK: FAIL (wheel was not built)"
  exit 1
fi
# Triton 3.3 dropped Turing (sm75). PyTorch 2.6 ships the matching final
# Turing-capable Triton 3.2 release, so this verification environment is pinned
# intentionally instead of following the newest package versions.
"$venv/bin/python" -m pip install --upgrade \
  "torch==2.6.0" "triton==3.2.0" \
  "numpy>=1.24" "ninja>=1.11" "safetensors>=0.4.5" "pytest>=8"
"$venv/bin/python" -m pip install --force-reinstall --no-deps "$wheel"
"$venv/bin/python" tools/release_gate.py \
  --backend cuda \
  --report "$base/gpu-report.json"
