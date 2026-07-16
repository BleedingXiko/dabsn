#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."
base="$PWD/.release-check"
venv="$base/cpu-venv"
mkdir -p "$base"
dist=$(mktemp -d "$base/cpu-dist.XXXXXX")

python_cmd=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c \
    'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    python_cmd="$candidate"
    break
  fi
done
if [[ -z "$python_cmd" ]]; then
  echo "DABSN CPU CHECK: FAIL (Python 3.10 or newer is required)"
  exit 1
fi

"$python_cmd" -m venv "$venv"
"$venv/bin/python" -m pip install --upgrade pip build setuptools wheel
rm -rf build src/*.egg-info
"$venv/bin/python" -m build --wheel --outdir "$dist"
wheel=$(find "$dist" -maxdepth 1 -name 'dabsn-*.whl' -type f | sort | tail -1)
if [[ -z "$wheel" ]]; then
  echo "DABSN CPU CHECK: FAIL (wheel was not built)"
  exit 1
fi
"$venv/bin/python" -m pip install --upgrade \
  "torch>=2.6" "numpy>=1.24" "ninja>=1.11" "safetensors>=0.4.5" "pytest>=8"
"$venv/bin/python" -m pip install --force-reinstall --no-deps "$wheel"
"$venv/bin/python" tools/release_gate.py \
  --backend cpu \
  --report "$base/cpu-report.json"
