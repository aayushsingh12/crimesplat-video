#!/usr/bin/env bash
# One-shot setup for a fresh Linux or WSL2 machine with an NVIDIA GPU.
#   bash scripts/setup.sh
# Needs: NVIDIA driver (nvidia-smi works), ffmpeg, git, internet. No CUDA toolkit needed:
# gsplat and torch come as prebuilt CUDA 12.4 wheels.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "nvidia-smi failed: install the NVIDIA driver first (on WSL2, install it on Windows)"; exit 1; }

echo "== ffmpeg"
command -v ffprobe >/dev/null || {
  echo "ffmpeg missing. Install it: sudo apt install -y ffmpeg   (or ask the machine's admin)"; exit 1; }

echo "== uv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== Python environment (torch 2.4.1+cu124, gsplat, pycolmap)"
(cd pipeline && uv sync)

echo "== Segmentation weights (downloaded once, cached in pipeline/)"
(cd pipeline && uv run python -c "
from ultralytics import YOLO
for w in ['yolo11n-seg.pt', 'yolo11m-seg.pt', 'yolo11x-seg.pt']:
    YOLO(w)
")

if command -v npm >/dev/null; then
  echo "== Viewer"
  (cd web && npm install --no-audit --no-fund)
else
  echo "(npm not found: skipping the viewer; it can run on the laptop instead)"
fi

echo "== Doctor"
(cd pipeline && uv run crimesplat doctor)
