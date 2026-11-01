# Qmini

An installable downstream Isaac Lab task package with a standard uv `src` layout.

The package registers the Balance, Qmini/GR1T, UMV, TCN, DreamWaQ, HIMLoco, and Multi-Critic task variants.
All task configurations use the concrete Isaac Sim PhysX backend.


## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. Isaac Sim
and RSL-RL are installed by default because every task in this package uses Isaac Sim PhysX:

```bash
uv sync
```

This project's `pyproject.toml` uses editable relative paths to an Isaac Lab source checkout. Update
`[tool.uv.sources]` if either directory moves.

The project forwards every optional extra declared by the Isaac Lab package used to create it, including physics,
rendering, visualizer, RL, teleoperation, and development features. Inspect `pyproject.toml` for the complete list and
pass each required extra to `uv run`:

```bash
uv run isaaclab random_agent --task <TASK_NAME>

# Isaac Sim with the Rerun visualizer
uv run --extra isaacsim --extra rerun isaaclab random_agent --task <TASK_NAME> --viz rerun
```

The `all` extra forwards Isaac Lab's `all` selection. The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes;
you can also select either independently. Commit `pyproject.toml` and `uv.lock` so collaborators use the same
environment.

## Run the tasks

Replace the placeholders below with a task listed above and a selected RL library.

```bash
# List this project's environments and their available presets
uv run isaaclab list_envs --show_presets

# Exercise an environment without a trained policy
uv run isaaclab zero_agent --task <TASK_NAME> --num_envs 16
uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16

# Train and play
uv run isaaclab train --rl_library <RL_LIBRARY> --task <TASK_NAME>
uv run isaaclab play --rl_library <RL_LIBRARY> --task <TASK_NAME> --checkpoint latest

# Distributed training
uv run isaaclab train_multigpu --rl_library <RL_LIBRARY> --task <TASK_NAME> --num_gpus 2

# Benchmark startup, runtime, training, or play
uv run isaaclab benchmark runtime --task <TASK_NAME> --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library <RL_LIBRARY> --task <TASK_NAME> --max_iterations 10
```

The tasks intentionally do not expose a physics preset: `PhysxCfg` is fixed in each environment configuration.

## Project structure

Tasks live in `src/Qmini/tasks`, grouped by robot category:

```text
tasks/
├── humanoid/                  # 人形 / 双足
│   ├── gr1t/
│   ├── gr1t_multi_critic/
│   └── qmini/
├── creative/                  # 创造类
│   └── umv/
├── manipulation/              # 机械臂操作
│   ├── so101/
│   └── openarm/
└── quadruped/                 # 四足（预留，尚无任务）
```

Each robot package keeps its environment configurations, `mdp` terms, agents, and Gym registrations.
Add new robot packages under the appropriate category with an `__init__.py` to enable automatic discovery.
Task IDs and training commands remain unchanged. Python imports use the categorized paths, for example
`Qmini.tasks.humanoid.gr1t` and `Qmini.tasks.manipulation.openarm`.

## Project assets

Keep project-owned USD files and related data under `src/Qmini/assets/data`. The asset module exposes a
stable path that works from an editable checkout and an installed wheel:

```python
from Qmini.assets import QMINI_ASSETS_DIR

robot_usd_path = QMINI_ASSETS_DIR / "robots" / "my_robot.usd"
```

Keep asset configuration in the `Qmini.assets` package and pass `str(robot_usd_path)` to configuration fields that
expect a string. USD files are configured for Git LFS by the project's `.gitattributes`; install Git LFS before adding
large binary assets.

### Unitree AS2W

The original URDF, all 18 STL meshes, upstream license, and pinned source manifest live in
`src/Qmini/assets/data/as2w_description`. Source:
[Unitree unitree_ros](https://github.com/unitreerobotics/unitree_ros/tree/3ae893d905d11795ddf1075825ee5010364a1109/robots/as2w_description).
Mesh paths are relative to the URDF; no ROS package setup is required.

```python
from Qmini.assets.as2w import AS2W_CFG

robot = AS2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

The PhysX configuration uses 12 position-controlled leg joints and 4 velocity-controlled wheels.
Use `JointPositionActionCfg` for `.*_(hip|thigh|calf)` (with `use_default_offset=True`) and
`JointVelocityActionCfg` for `.*_foot`. Wheel position stiffness is zero. Joint effort and velocity
limits match the URDF; PD gains and the initial pose are starting values requiring simulation tuning.
The `JT128` fixed link is preserved. This adds an asset configuration, not a registered training task.

### AS2W training

Registered tasks: `Qmini-Velocity-Rough-AS2W` and `Qmini-Velocity-Flat-AS2W`.
The default configuration uses Isaac Sim PhysX with AS2W explicit ideal PD. The detailed wheel-contact
slip reward requires PhysX. Flat terrain retains all scan observations so the input layout stays consistent.

```bash
# PPO (rough terrain by default)
bash scripts/train_as2w.sh
# Select a physics backend
bash scripts/train_as2w.sh --physics isaacsim_physx
# Small startup/training check
bash scripts/train_as2w.sh --num_envs 16 --max_iterations 2
# Flat-terrain PPO
bash scripts/train_as2w.sh --task Qmini-Velocity-Flat-AS2W
# Distillation: use a teacher trained on the same task/configuration
bash scripts/distill_as2w.sh /absolute/path/to/model.pt
# Playback
bash scripts/play_as2w.sh --checkpoint /absolute/path/to/model.pt
```

For flat-terrain distillation or playback, also pass `--task Qmini-Velocity-Flat-AS2W`.
For student playback, pass `--agent rsl_rl_distillation_cfg_entry_point`.
Scripts forward extra Isaac Lab options and Hydra overrides. PPO logs are under
`logs/rsl_rl/as2w_rough` or `logs/rsl_rl/as2w_flat`; distillation uses the corresponding
`_distillation` experiment directory. No teacher checkpoint is bundled.

#### AS2W rolling and obstacle training

The current rough configuration prioritizes forward/backward commands (65% of moving commands),
retains 20% flat terrain and 50% stairs, and starts at terrain levels 0–1. Wheel scans cast downward
from 1 m above the wheel so that tall step tops remain visible. Local height second differences
distinguish stair edges from planar slopes. Near an edge, action smoothness costs fall to 20% and
posture costs to 25%; clearance applies to swinging wheels or wheels resisting a riser, while a
command-normalized progress penalty discourages staying blocked. These are training heuristics,
not a guaranteed rolling/stepping controller.

Leg position observations now exclude continuous wheel angles. PPO uses a log-parameterized action
standard deviation and an initial learning rate of 3e-4; observation normalization is currently disabled.
Action scales also changed. **Start a fresh run; old checkpoints do not match this observation/action
configuration.** Keep previous runs for comparison.

```bash
# Scoped runtime settings avoid the observed Kit/OpenBLAS startup crash and sourced ROS libraries.
env -u LD_LIBRARY_PATH -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash scripts/train_as2w.sh --experiment_name as2w_roll_step --run_name roll_step_v1

# Evaluate a NEW rough checkpoint separately on flat ground and fixed 10 cm stairs.
env -u LD_LIBRARY_PATH -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python scripts/evaluate_as2w.py --checkpoint /absolute/path/to/model.pt \
  --terrain flat --viz none --output /tmp/as2w_flat.json
env -u LD_LIBRARY_PATH -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python scripts/evaluate_as2w.py --checkpoint /absolute/path/to/model.pt \
  --terrain stairs_up --step_height 0.1 --viz none --output /tmp/as2w_stairs.json
```

The evaluator reports first-episode displacement, failures, wheel contact fraction, leg velocity RMS,
and body roll/pitch RMS. `first_riser_crossed_fraction` requires both forward displacement beyond
the first riser and simultaneous body height gain; it does not measure full-stair completion.
Check flat rolling and repeat stair evaluation at 5, 10, 15 and 20 cm before increasing difficulty.
Aggregate training `success_rate` is a velocity-tracking statistic, not obstacle success. Inspect
value loss for Inf/NaN as well as reward; the built-in NaN check only covers environment outputs.

## Development

Run the registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and writes a git-ignored `pyrightconfig.json`. The resulting
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
