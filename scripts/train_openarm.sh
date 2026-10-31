#!/usr/bin/env bash
# Train the local OpenArm lift-and-hold task with asymmetric RSL-RL PPO.
# Usage: bash scripts/train_openarm.sh [Isaac Lab options / Hydra overrides]
# Small startup check: bash scripts/train_openarm.sh --num_envs 64 --max_iterations 2
# Full run: bash scripts/train_openarm.sh --num_envs 4096 --max_iterations 3000
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

exec uv run isaaclab train \
    --rl_library rsl_rl \
    --task Qmini-Lift-Cube-OpenArm \
    --num_envs 4096 \
    --max_iterations 3000 \
    --seed 42 \
    --device cuda:0 \
    --viz none \
    --logger tensorboard \
    "$@"
