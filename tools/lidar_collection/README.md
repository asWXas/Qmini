# AS2W 雷达数据采集

固定已训练的 AS2W 策略，记录连续轨迹。暂不计算线加速度、角加速度或帧间位姿增量，后处理可根据时间戳和完整位姿计算。

## 工具结构

```text
lidar_collection/
├── collect.py              # 采集入口，命令不变
├── generate_terrain.py     # 离线地形生成入口
├── core/                   # 采集、策略、雷达、真值采样与配置检查
│   ├── collect.py
│   ├── policy.py
│   ├── recorder.py
│   ├── episodes.py
│   ├── lidar_config.py
│   ├── lidar_geometry.py
│   ├── geometry.py
│   └── config_audit.py
├── terrains/               # 地形构建、仿真接入、导出与预览代码
│   ├── terrain.py
│   ├── terrain_config.py
│   ├── research_terrain.py
│   ├── complex_terrain.py
│   ├── generate_terrain.py
│   └── terrain_preview.py
├── training/               # 地形重建网络、训练、评估与推理
├── train.py                # 离线训练入口
├── evaluate.py             # 离线评估入口
├── tests/                  # 测试代码
├── data/
│   ├── runs/               # 采集数据：点云、状态、配置、按 reset 分包
│   ├── terrains/           # 离线地形：PLY、JSON、预览图
│   └── training/           # 重建模型 checkpoint、训练日志、评估结果
└── README.md
```

已有采集目录已移动至 `data/runs/`，离线地形目录已移动至 `data/terrains/`，内容不变。历史 `metadata.json` 中的运行参数保持原记录。新采集和导出请使用下面命令中的 `data/` 路径；`--output` 仍可指定任意新目录，不会自动重定向。数据目录不纳入 Git。

根目录脚本仅作为入口；修改业务逻辑请进入 `core/`、`terrains/` 或 `training/`，测试统一放入 `tests/`。也可在项目根目录使用 `python -m tools.lidar_collection.collect` 或 `python -m tools.lidar_collection.generate_terrain`。

## 训练地形重建网络

持续 GRU 记忆模型使用 `--model sparse4d_v3`：当前观测＋历史预测，五级隐藏特征跨帧保存和对齐，残差更新不限制在观测邻域。新训练加入每帧画布，默认 `--no-observation_memory`，直接输出网络预测，不用缓存补点；支持 `--canvas_mode volume` 全体积候选雕刻。首帧历史仍为平板＋当前观测，支持世界坐标和仅相对变换的部署接口。见 [纯网络训练与缓存对照](training/docs/CANVAS_MEMORY.md) 与 [v3 结构](training/docs/RECURRENT_V3.md)。

已新增 [training 训练包与完整命令](training/README.md)，读取现有采集数据，独立训练地形重建网络，底盘策略保持冻结。根目录入口为 `train.py` 与 `evaluate.py`，输出建议放在 `data/training/`。

当前保留 `--model sparse4d_v1`（两来源直接预测）、`--model sparse4d_v3`（默认，持续 GRU 残差更新）与独立 `--model dense3d` 基线。v2 及旧 `sparse4d` 参数已移除，旧数据和报告保留；v2 检查点不能加载为 v1/v3。新训练默认关闭额外相对位姿编码，位姿仍用于几何对齐。详见 [论文对照](training/docs/PAPER_NOTES.md)。

v3 首帧为固定水平平板＋当前观测，隐藏状态为零；后续连续反馈预测和隐藏特征。训练、评估均按时间连续，不跨 reset。完整单卡/双卡命令见训练包 README。

验证可通过 `evaluate.py --export_frames 12 --export_every 140` 跨轨迹导出样本，再运行 `visualize.py --evaluation 评估目录`，生成离线交互 3D 页面、雷达/预测/真值对照 PNG、误检漏检分布及训练曲线；完整命令见 [可视化说明](training/README.md#可视化验证结果)。

## 复杂采集场景：research

`--terrain research` 创建一块 **24×24 米**的场地，中央出生，采用密集障碍区和连续低起伏路面；仅保留 2.8×2.8 米出生区、约 1.2 米宽连接带和窄外围边界为平整空地。
几何实现位于 [research_terrain.py](terrains/research_terrain.py)。默认一块场地、难度 0.5；现有 `--terrain existing` 的默认行为保持不变。

| 区域（相对出生点） | 几何与限制 |
|---|---|
| 两组楼梯 | 北侧 6 级上/下行、宽 4.4 米；出生区南侧另有 4 级上/下行、宽 2.6 米。每级 **8–15 cm**，踏面 30–45 cm，平台长 1.4 米，两端接地 |
| 两条斜坡 | 东侧宽 3.8 米，西侧内圈宽 2.4 米；均为上坡 → 平台 → 下坡，坡度 8–15° |
| 南侧：碎石路 | 9.2×4.6 米的细碎起伏面，起伏上限随难度从 2 cm 增至 6 cm |
| 西侧：乱石堆 | 9.6×4.6 米，220 个不规则岩块上表面的组合，带低丘；高度设计上界随难度从 19 cm 增至 44 cm |
| 结构化遮挡 | 9 个箱体、7 段墙体、6 根立柱、折角与约 2.42 米净宽走廊，部分延伸到内圈 |
| 原来的空地 | 改为连续的细碎起伏与缓波面；最高难度下起伏上界 6.5 cm，与各区域边缘平滑衔接 |

15 cm 指**相邻台阶的高度差**，不是整段楼梯的总高度；最高难度下平台高 0.9 米。台阶两侧有侧壁，需要沿梯段方向通过。
碎石路和乱石堆均为静态三角网格上表面，不模拟松散颗粒、滚石或真实岩块过悬。粗糙区四边衔接零高度，底下不再叠加一层平地作为真值。

参考论文 [Neural Scene Representation for Locomotion on Structured Terrain，III-D](https://arxiv.org/pdf/2206.08077) 的楼梯、箱体、墙壁、立柱与走廊组成和随机几何。
论文台阶高度为 8–25 cm；此工具按当前要求收紧至 8–15 cm，并额外加入斜坡、碎石和乱石。因此这是**参考论文的扩展采集场景**，不是其训练集的精确复刻。

生成网格、参数清单和预览（无需 GPU）：

```bash
.venv/bin/python tools/lidar_collection/generate_terrain.py \
  --kind research --seed 42 --difficulty 0.7 --preview \
  --output tools/lidar_collection/data/terrains/research_preview_001
```

输出 `terrain.ply`、`terrain.json`、`overview.png`。密集版标记为 `layout_version=2`；seed=42、difficulty=0.7 时，零高度平整地面面积约 50.4 平方米，占 576 平方米场地的 **8.7%**（不把高处台阶平台计作空地）。该统计随导出写入 `terrain.json`。当前已生成的示例见 [预览图](data/terrains/research_dense_seed42_d070/overview.png)。预览图颜色用于区分区域；仿真显示材质可能不同。

加载策略，在同一参数场景中用手柄采集：

```bash
cd /home/online/wx/qmini

.venv/bin/python tools/lidar_collection/collect.py \
  --checkpoint logs/rsl_rl/as2w_rough_gru/2026-11-03_20-14-26/model_7999.pt \
  --agent gru \
  --cmd gamepad --gamepad_path /dev/input/js0 \
  --gamepad_deadzone 0.1 --gamepad_scale 1.5 \
  --terrain research --terrain_rows 1 --terrain_cols 1 \
  --terrain_difficulty 0.7 --seed 42 \
  --lidar_pattern mid360 --lidar_rays 12000 \
  --lidar_offset 0.32 0 0.18 --lidar_rpy 0 60 0 \
  --min_range 0.2 --max_range 10 \
  --map_size 3.2 --gt_density 4000 \
  --num_envs 1 --steps 10000 --scan_every 5 \
  --output tools/lidar_collection/data/runs/research_dense_collect_001 \
  --device cuda:0 --viz kit
```

`--terrain_difficulty` 范围为 `[0, 1]`，0.7 对应每级 12.9 cm、坡度 12.9°。相同代码版本、seed 和 difficulty 会生成相同地形几何；更改 seed 可改变踏面宽度、碎石起伏、岩块和箱体尺寸。
独立导出使用 `[0, 24]` 的局部坐标，仿真会平移至场地中心并附加地面边界，采集目录中的 `terrain.ply` 才是实际世界坐标真值。
固定难度时多块 research 地形会重复同一布局；要采集不同布局，应使用不同 seed 分次运行。通过能力仍需用训练好的策略实测。

## 新增复杂方块场景：research_complex

原有 `--terrain research` 完整保留；新增 `--terrain research_complex`，单独实现于 [complex_terrain.py](terrains/complex_terrain.py)。两者均为 24×24 米，默认 1×1 块、难度 0.5。

新场景保留原场景的两组楼梯、两条斜坡、碎石区、乱石区、墙体、箱体和圆柱；将其间的低起伏背景地面换为平地与密集随机矩形凸起，接近参考图片中的平顶方块布局：

- 新增方块长宽约 0.45–2.1 米，顶面水平、侧面竖直，高度下限 3 cm，上限随难度从 8 cm 增至 15 cm。方块不叠放，与其他障碍至少相隔 10 cm。
- 新增 10 根方柱，高 0.8–1.4 米，用于产生遮挡；属于绕行障碍。
- 中央 2.8×2.8 米出生区留空；新增物体避开宽 1.2 米的十字连接带。原有楼梯和坡道仍位于各自通行方向上。
- 楼梯每级仍不超过 15 cm；原有箱体、乱石堆和楼梯总高度可超过 15 cm。

seed=42、difficulty=0.7 时新增 **141 个方块、10 根方柱**，方块总占地约 **142.3 平方米**。零高度平地约占 46.1%，主要为障碍之间的间隙和连接区；与 `research` 的连续起伏背景不同，不能仅用零高度比例比较场景复杂度。实际数量随 seed 和 difficulty 变化，写入导出的 `terrain.json`。新增方块底面和其下方地面不加入真值表面。

已生成示例：[场景预览](data/terrains/research_complex_seed42_d070/overview.png)。重新导出需使用新目录：

```bash
.venv/bin/python tools/lidar_collection/generate_terrain.py \
  --kind research_complex --seed 42 --difficulty 0.7 --preview \
  --output tools/lidar_collection/data/terrains/research_complex_preview_001
```

完整手柄采集命令（手柄输入倍率 2）：

```bash
cd /home/online/wx/qmini

.venv/bin/python tools/lidar_collection/collect.py \
  --checkpoint logs/rsl_rl/as2w_rough_gru/2026-11-03_20-14-26/model_7999.pt \
  --agent gru \
  --cmd gamepad --gamepad_path /dev/input/js0 \
  --gamepad_deadzone 0.1 --gamepad_scale 2 \
  --terrain research_complex --terrain_rows 1 --terrain_cols 1 \
  --terrain_difficulty 0.7 --seed 42 \
  --lidar_pattern mid360 --lidar_rays 12000 \
  --lidar_offset 0.32 0 0.18 --lidar_rpy 0 60 0 \
  --min_range 0.2 --max_range 10 \
  --map_size 3.2 --gt_density 4000 \
  --num_envs 1 --steps 10000 --scan_every 5 \
  --output tools/lidar_collection/data/runs/research_complex_collect_001 \
  --device cuda:0 --viz kit
```

改回 `--terrain research` 即可运行原场景；每次采集使用新的 `--output` 目录。两个场景使用相同的策略加载、雷达和 reset 分包流程。方块为静态网格，策略通过能力需要在 GPU 仿真中实测。

## 单独生成地形（可选）

无需启动仿真或使用 GPU：

```bash
.venv/bin/python tools/lidar_collection/generate_terrain.py \
  --kind mixed --seed 42 --difficulty 0.5 \
  --output tools/lidar_collection/data/terrains/terrain_001
```

输出 `terrain.ply` 和 `terrain.json`。支持 `flat`、`stairs`、`boxes`、`corridor`、`poles`、`mixed`、`research`、`research_complex`。
基础地形为 8×8 米，research 和 research_complex 为 24×24 米，中心均保留至少 2.4×2.4 米出生区；难度范围 0～1。
这是独立预览用的单块网格。采集时使用相同几何生成函数，由 Isaac Lab 为各块选择难度并拼接；最终采集场景以采集目录中的 `terrain.ply` 为准。

## 加载策略并采集

`collect.py` 默认调用原 AS2W 的 `play_mode()`，使用原 play 的 5×5 地形、150 秒 episode、关闭随机推力和观测噪声，并关闭速度课程，避免首次 reset 把速度范围缩到 10%。

`collect.py` 会依次完成 **生成地形 → 配置雷达 → 加载策略 → 行走采集**，不需要先运行独立地形导出工具。

在项目根目录执行。下面使用本地已有的 GRU checkpoint；也可以替换为自己的权重路径。checkpoint 必须与当前环境观测和所选网络结构匹配，工具不会自动恢复训练时的环境配置。

```bash
cd /home/online/wx/qmini

.venv/bin/python tools/lidar_collection/collect.py \
  --checkpoint logs/rsl_rl/as2w_rough_gru/2026-11-03_20-14-26/model_7999.pt \
  --agent gru \
  --terrain existing \
  --lidar_pattern hemisphere \
  --lidar_rays 12000 \
  --lidar_offset 0.32 0 0.18 \
  --lidar_rpy 0 60 0 \
  --min_range 0.2 \
  --max_range 10 \
  --num_envs 1 \
  --steps 1000 \
  --scan_every 5 \
  --output tools/lidar_collection/data/runs/run_001 \
  --device cuda:0 \
  --viz none
```

MLP 策略使用 `--agent mlp`。策略加载逻辑见 [policy.py](core/policy.py)，只做推理，不进行训练。
工具不会自动选择 checkpoint，也不会覆盖已有输出目录；再次运行时换用 `run_002` 等新目录。
先以一个环境做短轨迹检查，再增加采集规模。依赖项目现有虚拟环境，不需要单独安装包。

| 参数 | 含义 |
|---|---|
| `--checkpoint` | 已训练策略的权重文件 |
| `--agent gru` | GRU 网络；普通 MLP 策略改为 `mlp` |
| `--cmd random` | 速度命令来源；改为 `gamepad` 使用遥控器 |
| `--terrain existing` | 默认复用原 play 地形；可选 `research_complex`、`research`、`mixed`、`stairs`、`boxes`、`corridor`、`poles`、`flat` |
| `--terrain_rows`、`--terrain_cols` | 新生成地形的块数；`existing` 使用原配置 |
| `--terrain_difficulty` | 新地形固定难度 `[0, 1]`；research / research_complex 默认 0.5，其他新地形默认随机；existing 使用原配置 |
| `--num_envs 1` | 同时运行的机器人环境数量 |
| `--steps 1000` | 运行的控制步数 |
| `--device cuda:0` | 使用的 GPU，可改为 `cuda:1` 等 |
| `--viz none` | 无界面运行 |
| `--output` | 新建的数据输出目录 |

默认 `--terrain existing` 复用原 play 地形；`--terrain_rows`、`--terrain_cols` 不覆盖原 play 配置。显式选择 research 或 research_complex 时默认 1×1 块，其他工具地形默认 2×2 块。默认采用原环境的速度命令采样，也可用遥控器指定速度；不包含覆盖规划器，因此不保证遍历所有区域。

## 遥控器控制采集

使用 `--cmd gamepad` 启用现有 Linux Xbox 兼容手柄接口。手柄需要连接到运行仿真的机器，并暴露 `/dev/input/js*` 设备。它控制策略的速度命令，策略继续输出关节和轮子动作。

```bash
cd /home/online/wx/qmini

.venv/bin/python tools/lidar_collection/collect.py \
  --checkpoint logs/rsl_rl/as2w_rough_gru/2026-11-03_20-14-26/model_7999.pt \
  --agent gru \
  --cmd gamepad \
  --gamepad_path /dev/input/js0 \
  --gamepad_deadzone 0.1 \
  --gamepad_scale 2 \
  --terrain existing \
  --lidar_pattern hemisphere \
  --lidar_offset 0.32 0 0.18 \
  --lidar_rpy 0 60 0 \
  --num_envs 1 --steps 10000 --scan_every 5 \
  --output tools/lidar_collection/data/runs/gamepad_001 \
  --device cuda:0 --viz none
```

| 操作 / 参数 | 含义 |
|---|---|
| 左摇杆前后 | 前进 / 后退速度 `vx` |
| 左摇杆左右 | 横移速度 `vy` |
| 右摇杆左右 | 转向角速度 `wz` |
| 摇杆回中 | 死区内速度命令归零 |
| `--gamepad_path` | 手柄设备路径，默认 `/dev/input/js0` |
| `--gamepad_deadzone` | 摇杆死区，默认 0.1，范围 `[0, 1)` |
| `--gamepad_scale` | 遥控器速度命令倍率，默认 2；设为 1 恢复原幅值 |
| `--cmd random` | 切回自动随机速度命令，也是默认模式 |

速度幅值以 `AS2WRoughEnvCfg.commands.base_velocity.ranges` 为基础乘以 `--gamepad_scale`。默认倍率 2 时，前后目标速度范围为 `[-2.0, 3.0]` m/s、横移为 `[-2.0, 2.0]` m/s、转向为 `[-4.0, 4.0]` rad/s。倍率作用于死区处理后的速度命令，不修改训练时的观测缩放；随机命令模式不受影响。这里是目标速度，实际速度取决于策略和地形。
多环境模式会给所有机器人发送同一手柄命令，手动采集建议使用 `--num_envs 1`。输入不依赖可视化窗口，`--viz none` 仍可读取手柄。仿真按运行性能推进，当前没有实时节拍限制。
手柄无法打开或运行中断开时会报错，不会自动切换为随机行走。若提示无法打开设备，请确认设备路径和当前用户的读取权限。

## 传感器配置

雷达在 [collect.py](collect.py) 的 `cfg.scene.collection_lidar` 中配置，挂载在与底盘高度扫描器相同的机身链接上。新增雷达用于数据采集，策略仍使用原有控制观测。

默认雷达安装在机身前上方 `(0.32, 0, 0.18)` 米，安装角为 `roll=0°、pitch=60°、yaw=0°`。这是根据模型头部位置设置的初值，可按实机调整。机身坐标使用 x 前、y 左、z 上；正 pitch 使传感器的 +x 射线向前下方倾斜。

| 模式 | 扫描几何 |
|---|---|
| `--lidar_pattern hemisphere` | 默认。传感器局部 +z 半球：水平 360°、仰角 0°～90°；均匀立体角分散采样，不是逐线扫描圈 |
| `--lidar_pattern mid360` | MID-360 视场近似：水平 360°、仰角 -7°～52°，同样使用分散采样 |
| `--lidar_pattern rings` | 旧的规则线束模式，默认 32 线、仰角 -30°～10° |

MID-360 实际视场不是完整半球，视场参数见 [Livox 官方规格](https://www.livoxtech.com/mid-360/specs)。两种分散模式都是**固定射线的几何近似**，不模拟真实 MID-360 的非重复时序扫描轨迹、逐点时间戳和运动畸变。默认半球经 pitch 60° 安装后朝前上方，前向射线可以覆盖地面；若要朝更低处扫描，可以进一步调整安装角度。

| 参数 | 含义 |
|---|---|
| `--lidar_rays 12000` | 半球/MID-360 视场近似模式的射线数；不保证每条射线都有回波 |
| `--lidar_offset 0.32 0 0.18` | 雷达相对机身链接的安装位置 xyz，单位米 |
| `--lidar_rpy 0 60 0` | 安装 roll/pitch/yaw，单位度，旋转顺序 Rz(yaw) Ry(pitch) Rx(roll) |
| `--vertical_fov LOW HIGH` | 可选，覆盖所选模式的默认仰角范围，单位度 |
| `--channels 32`、`--horizontal_res 1` | 仅 `rings` 模式使用：垂直线数和水平角间隔 |
| `--lidar_debug_vis` | 在已启用的可视化窗口中显示命中点；不改变采集内容 |
| `--min_range 0.2`、`--max_range 10` | 保存点云的有效距离范围，单位米 |
| `--scan_every 5` | 每段轨迹第一有效步开始采集，随后每 5 个控制步保存雷达帧 |

雷达配置封装在 [lidar_config.py](core/lidar_config.py)，分散射线和安装变换在 [lidar_geometry.py](core/lidar_geometry.py)。当前只对地形做射线检测，没有新增雷达外壳实体或质量，也没有机身自遮挡。

查看全部运行选项：

```bash
.venv/bin/python tools/lidar_collection/collect.py --help
```

雷达数据每 `--scan_every` 个控制步保存一次；位姿和角速度每个控制步保存一次，并非物理子步采样。
当前 AS2W 配置的控制周期为 `sim.dt × decimation = 0.005 × 4 = 0.02` 秒，因此上述命令记录 **50 Hz 的位姿和角速度、10 Hz 的雷达帧**，运行 20 秒仿真时间。重置当步跳过雷达帧，下一有效步立即开始新段采集；发生重置时实际帧数可能偏离 200。实际运行耗时取决于仿真和写盘速度，采样周期也会写入 `metadata.json`。

角速度直接读取仿真机身状态，无需额外配置 IMU 传感器；当前没有 IMU 噪声模型，也不计算加速度。

## 输出

```text
run_001/
  metadata.json           # 单位、坐标系、时间约定、采样参数
  env.yaml                # 采集环境配置
  agent.yaml              # 策略配置
  terrain.ply             # 仿真实际使用的完整地形网格
  states.jsonl            # 每个控制步的位姿、角速度及 episode/reset 标记
  env_000/
    episode_000000/
      episode.json        # 起止步数、帧数、结束原因
      states.jsonl        # 仅该段轨迹的状态
      frame_000000.npz
    episode_000001/       # reset 后自动创建的新数据包目录
      episode.json
      states.jsonl
      frame_000000.npz
```

每个 NPZ 包含：

| 字段 | 含义 |
|---|---|
| `time_s`, `step` | 全局仿真时间和控制步编号 |
| `env_id`, `episode_id` | 环境和轨迹编号 |
| `episode_time_s`, `episode_step` | 当前段的局部时间和步数；reset 时归零 |
| `lidar_points_w` | 有限量程内的有效雷达命中点，世界坐标，米 |
| `ray_indices` | 有效点对应的原始射线索引 |
| `base_pos_w`, `base_quat_xyzw` | 机身位置和朝向，四元数顺序 **xyzw** |
| `angular_velocity_b` | 机身坐标系角速度，rad/s |
| `sensor_pos_w`, `sensor_quat_xyzw` | 当前雷达坐标系到世界坐标系的位姿 |
| `gt_points_w` | 周围局部网格表面采样点，世界坐标，米 |

局部真值区域默认是以机身位置为中心、与世界轴对齐的 3.2 米立方体，密度为 4000 点/平方米，可通过 `--map_size` 和 `--gt_density` 调整。点数可变，不应直接当作固定长度数组批量堆叠。
雷达点保存完整量程内观测，训练前可按同一局部区域裁剪。
每个环境的 reset（终止或超时，由 `env.step` 返回 done）会关闭当前数据包，写入 `end_reason=reset`，并新建下一段目录。重置当步的状态归属新段、局部时间为 0；该步不保存雷达帧，下一有效步从 `frame_000000.npz` 开始。多个环境独立分段。正常退出或 Ctrl+C 会关闭当前文件并写入段信息；强制杀进程可能留下未完成摘要。这里的数据包是目录，不是 ROS bag。根目录 `states.jsonl` 继续作为全局索引。历史点云不能跨 episode 关联。

Schema v2 还修正了传感器位姿：当前 RayCaster 返回的是挂载链接位姿，保存前显式合成安装平移和旋转；距离过滤使用实际雷达原点。旧 schema v1 数据不会自动改写。

## 当前范围

启动时会读取 checkpoint 同目录下 `params/env.yaml` 与 `params/agent.yaml`，核对观测顺序、缩放、动作配置、网络配置和控制周期，并保存 `config_audit.json`。允许 play 模式关闭训练观测噪声，并在审计报告中列出该覆盖；其他接口不一致时报告差异并停止；缺少训练配置时提示未验证。该检查不验证机器人几何和观测函数源码是否变化。

- 结构化地形支持楼梯、箱体、走廊墙壁和立柱；当前几何尺寸为初始采集配置，未声称严格匹配论文数据分布。
- 雷达是对地形的瞬时射线扫描，包含地形遮挡和视场缺失；尚未模拟机器人自遮挡、扫描运动畸变、测距噪声。
- 角速度来自仿真机身状态，是理想陀螺仪信号；位姿也是仿真真值，未模拟里程计漂移。
- 真值来自三角网格全部表面，包含竖直面、遮挡面，也可能包含拼接网格的内部面和底面；不是经过实体布尔合并后的外表面。保留 `terrain.ply` 以便后续重新处理标签。
- 该工具需要可用的仿真运行环境和匹配的 checkpoint。命令行、几何采样可离线检查，端到端采集仍需在有 NVIDIA GPU 的机器验证。

## 离线验证

```bash
.venv/bin/python -m pytest tools/lidar_collection/tests -q
.venv/bin/ruff check tools/lidar_collection
```

测试覆盖局部真值裁剪、竖直面保留、地形复现与出生区、15 cm 台阶限制、密集布局平地占比与出生区、斜坡衔接、无效雷达点过滤及重置分段。策略与仿真端到端测试仍需可用的 GPU。
