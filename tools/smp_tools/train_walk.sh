#!/usr/bin/env bash
# Usage: bash smp_tools/train_walk.sh [additional train_prior.py arguments]
# Overrides: GPU=0 PYTHON=/path/to/python DEVICE=cpu
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON="${PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    PYTHON=python
fi
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/output/.matplotlib}"
"$PYTHON" -c 'import torch, diffusers, yaml, gymnasium, matplotlib, PIL' || {
    echo "Missing dependencies. Install with: $PYTHON -m pip install -r $SCRIPT_DIR/requirements.txt" >&2
    exit 1
}
mkdir -p output
OUT_DIR="$(mktemp -d "$SCRIPT_DIR/output/walk_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
echo "Output: $OUT_DIR"
"$PYTHON" -u train_prior.py \
    --cfg_path config/tinymdm_multi_clip.yaml \
    --out_dir "$OUT_DIR" \
    --device "${DEVICE:-cuda:0}" "$@" 2>&1 | tee "$OUT_DIR/console.log"
