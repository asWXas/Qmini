#!/usr/bin/env bash
# Usage: bash scripts/play_as2w.sh --checkpoint /path/to/model.pt
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec uv run isaaclab play \
    --rl_library rsl_rl \
    --task Qmini-Velocity-Rough-AS2W \
    --num_envs 16 \
    --checkpoint latest \
    "$@"
