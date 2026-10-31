#!/usr/bin/env bash
# Default: rough-terrain PPO. Override with --task Qmini-Velocity-Flat-AS2W.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec uv run isaaclab train \
    --rl_library rsl_rl \
    --task Qmini-Velocity-Rough-AS2W \
    --num_envs 1024 \
    --seed 42 \
    --device cuda:0 \
    --viz none \
    --logger tensorboard \
    "$@"
