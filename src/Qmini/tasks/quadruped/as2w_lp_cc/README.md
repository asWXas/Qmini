# AS2W LP-ACRL

在 AS2W 轮腿机器人副本中移植 [Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning](https://arxiv.org/html/2601.17428v1) 的课程核心（式 5–7）。保留副本现有的 AS2W 动作、观测、奖励与 PPO 参数；没有替换为 ANYmal 的 12 维动作，也没有复现论文的硬件结果或 LSTM 蒸馏。

## 训练

在仓库根目录执行：

```bash
bash src/Qmini/tasks/quadruped/as2w_lp_cc/train.sh
```

独立任务名，不覆盖原版 AS2W：

- `Qmini-Velocity-Rough-AS2W-LP-ACRL`：600 个任务。
- `Qmini-Velocity-Flat-AS2W-LP-ACRL`：8 个 X 速度任务。
- 上述任务后加 `-MultiCritic`：同一课程、三头 critic。

例如：

```bash
bash src/Qmini/tasks/quadruped/as2w_lp_cc/train.sh \
  --task Qmini-Velocity-Rough-AS2W-LP-ACRL-MultiCritic \
  --num_envs 1024 --max_iterations 3000
```

课程参数位于 `env/as2w_env_cfg.py` 的 `lp_acrl`：

```python
lp_acrl = LPACRLCfg(
    temperature=1.0,                 # beta，与原始回合回报同量纲
    stage_steps=2400,                # 100 次 PPO 更新 × 每次 24 步
    min_episodes_per_task=2,
    terrain_names=("stairs_up", "stairs_down", "rough", "slope_up", "slope_down"),
    terrain_levels=4,
    vx_edges=(0., .5, 1., 1.5, 2., 2.5),
    yaw_edges=(0., .5, 1., 1.5, 2., 2.5, 3.),
    vy_range=(-.5, .5),
)
```

`stage_steps` 是所有并行环境共用的控制步数，不是所有环境的 transition 数。修改地形类型/等级时，同步修改 `mdp/terrains/lp_terrains.py` 的列/行；运行时检查映射。命令的绝对值从所属区间均匀采样，正负号独立均匀采样。Y 速度不划分任务。关闭 heading 控制、分组命令和原有地形升降级课程，回合内不重新抽取命令。

CLI 可覆盖：

```bash
bash src/Qmini/tasks/quadruped/as2w_lp_cc/train.sh \
  env.lp_acrl.temperature=2.0 env.lp_acrl.stage_steps=4800
```

`adaptive=False` 是均匀采样基线，保持同一任务空间。不要用平均训练回报直接比较两个课程，因为它们采样的任务不同。

## 算法与明确的实现选择

- 每个任务在一个阶段内汇总完整回合的**原始累计奖励**，包括终止步及各惩罚项，保留 Isaac Lab 的 dt 加权；不除以回合长度，不把 EPTE 当成奖励。
- `LP = 当前阶段平均回报 - 前一阶段平均回报`，保留正负号；`p = softmax(LP / temperature)`。没有绝对值、正值截断或隐藏的均匀混合。
- 初始均匀采样。第一个阶段建立基线；之后任务需要连续两个阶段各有足够回合才计算 LP。缺样本设 LP=0，不能把缺失回报当成 0，也不延用过期进展。
- 阶段时长、beta 默认值、每任务最低样本数和缺样本策略在论文中没有完整给出，以上是显式工程选择。beta 太小会使采样高度集中，需结合概率和覆盖率调节。
- 终止后先归属旧任务的回报，再选新任务、更新 terrain origin，最后运行重置事件和命令重置。
- 手动 reset、初始随机 episode timer 导致的不完整回合不进入回报估计。策略和仿真状态不随课程状态一起恢复；恢复后从新回合继续。
- 任务由起始地形格定义。格子外可能进入相邻地形，这是现有网格环境的限制。评估记录 `crossed_tile_fraction`；需要严格保持地形任务纯度时，增大地形 `size` 并重新训练/评估，不能把跨格结果解释为单一地形性能。

地形参数对应论文附录 C：四档台阶高 0.05/0.10/0.15/0.20 m，坡度梯度 0–0.4。粗糙地形四档都从 0.02–0.10 m 的噪声范围生成，等级代表不同生成实例。这里的梯度 0.4 不是 0.4 度。对中心出生点，凹金字塔地形对应向外上行。

## 日志、检查点、评估

TensorBoard 的 `LP_ACRL/` 下记录阶段、采样熵、任务覆盖率、最大概率、各地形概率、平均有符号进展、EPTE 和已观测任务成功率。训练统计受采样分布影响，不能当作全任务均匀评估。检查点中的 `lp_acrl` 包含任务表、每任务概率/LP/计数/回报/EPTE、阶段内累积量和课程 RNG 状态；与模型一起存取。

```bash
bash src/Qmini/tasks/quadruped/as2w_lp_cc/train.sh \
  --checkpoint /absolute/path/to/model_1499.pt
```

恢复时保持相同的任务划分、beta、阶段长度等参数；不匹配会报错，避免把旧统计错配到新任务。评估允许关闭自适应采样。检查点不保存仿真器的中途状态。

逐任务评估读取模型旁边 `params/env.yaml`、`params/agent.yaml`，恢复训练时的观测/网络/奖励设置；只评估，不更新 PPO：

```bash
uv run python -m Qmini.tasks.quadruped.as2w_lp_cc.evaluate \
  --checkpoint /absolute/path/to/model_1499.pt \
  --num_envs 32 --episodes 3 --viz none --output /tmp/lp_eval.json
```

默认均匀遍历所有任务，可用 `--task_ids 0 30 120` 仅检查部分任务。报告每任务平均回报、存活时长、X/Y/yaw 的 EPTE-SP 和成功判定。成功要求平均存活步数超过总时长的 90%，X 与 yaw 的 EPTE-SP 均小于 30%；默认 1000 步对应论文的 900 步。

EPTE 按式 8 给跌倒后的剩余步数填 100% 误差。近零命令分母下限是 0.1，作为论文未明确给出的数值处理；超过 100% 的实际误差不裁剪。零 yaw 指令的平地任务仍会记录 yaw 稳定误差。评估阶段保留训练时的随机化与观测噪声。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/test_as2w_lp_acrl.py
```

测试覆盖有符号 LP、softmax、缺样本、任务命令区间、独立注册、回报归属、重置顺序、EPTE、课程状态往返与均匀基线。小规模 CPU 仿真验证运行链路，不代表训练收敛或达到论文指标。

2026-11-01 验证记录：10 项课程测试通过，Ruff F/E9 和启动脚本语法检查通过。CPU/PhysX 下，粗糙地形 PPO、multi critic 各跑 2 次更新；平地跑 4 次更新；模型恢复后课程 stage 从 4 继续到 6；3 个任务各评估 2 回合并导出 JSON。为验证链路，这些短测使用 0.2 秒回合、小网络和 4–8 个环境，不能用于判断性能。平地短测最终概率范围为 0.1093–0.1392（初始均为 0.125）。

仓库通用 `tests/test_registration.py` 仍有两项旧测试失败：已不存在的旧任务名，以及将所有 AS2W 物理配置假定为具体 PhysxCfg 的旧断言；本次没有修改这些通用测试。
