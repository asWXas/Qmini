#!/usr/bin/env bash
# Usage: bash scripts/distill_as2w.sh /absolute/path/to/teacher.pt [options]
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/teacher.pt [Isaac Lab options]" >&2
    exit 2
fi
teacher_checkpoint=$(realpath -- "$1")
if [[ ! -f "$teacher_checkpoint" ]]; then
    echo "Teacher checkpoint not found: $teacher_checkpoint" >&2
    exit 2
fi
shift
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec bash scripts/train_as2w.sh \
    --agent rsl_rl_distillation_cfg_entry_point \
    --checkpoint "$teacher_checkpoint" \
    "$@"
