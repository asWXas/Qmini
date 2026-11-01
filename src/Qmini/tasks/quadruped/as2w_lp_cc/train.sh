#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../../.."
exec uv run isaaclab train \
    --rl_library rsl_rl \
    --task Qmini-Velocity-Rough-AS2W-LP-ACRL \
    --num_envs 1024 \
    --device cuda:0 \
    --viz none \
    --logger tensorboard \
    "$@"
