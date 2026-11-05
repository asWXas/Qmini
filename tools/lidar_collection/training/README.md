# 地形重建网络训练包

本包离线读取 `data/runs` 中的 NPZ，训练“部分雷达观测 → 局部完整地形”的模型。底盘策略保持冻结，训练不启动 Isaac Sim，也不需要底盘 checkpoint。

**当前保留 v1 与 v3；默认使用 `sparse4d_v3`。** v2 已移除，旧的 `--model sparse4d` 和 v2 检查点会明确报错，不会静默转换架构。已有采集数据、训练结果和报告不删除。独立的 3D 基线继续保留。

| `--model` | 网络与输出 |
|---|---|
| `sparse4d_v1` | 两来源 4D 稀疏 U-Net，直接预测地图，兼容 v1 权重 |
| `sparse4d_v3` | 当前观测＋历史预测＋每帧画布，持续五级 GRU，网络残差更新（默认，无缓存补点） |
| `dense3d` | 独立 3D 稠密基线 |

v3 默认平板＋当前观测初始化，默认关闭额外相对位姿编码；位姿仍用于几何对齐。v1/v3 权重不能互换，切换架构需要新目录重新训练。完整 v3 结构和部署见 [v3 使用说明](docs/RECURRENT_V3.md)。

新 v3 训练默认启用每帧平面画布、关闭观测缓存，直接输出网络预测并反馈下一帧；`--canvas_mode volume` 可改为全体积候选，让网络判断保留/删除。GRU hidden 持续保存。旧权重加载/续训保留原配置，显式 `--no-observation_memory` 可确保关闭输出补点。画布改变结构需要重训，观测缓存只保留作可选对照。完整命令见 [纯网络画布训练与缓存对照](docs/CANVAS_MEMORY.md)。

v1 是依据论文方法实现的网络，v3 是本项目扩展，均不是作者公开源码；实现选择见 [论文对照与边界](docs/PAPER_NOTES.md)。

## 目录

```text
training/
├── train.py / evaluate.py / visualize.py / predictor.py  # 稳定入口与推理 API
├── models/       # v1.py、v3.py、dense3d.py、架构选择
├── datasets/     # 连续片段读取、地形划分、数据增强
├── geometry/     # 体素化、坐标工具、平板与残差运算
├── runtime/      # 递归状态、训练/评估步骤、多卡、学习率、Rich 输出
├── tests/        # 画布与观测记忆回归检查
├── deps/         # MinkowskiEngine 构建脚本、兼容补丁、许可证
└── docs/         # 论文对照、v3 说明、GPU 0 双进程训练命令
```

根目录 `tools/lidar_collection/train.py`、`evaluate.py`、`visualize.py` 及 `training.predictor` API 不变。内部 Python 模块已归类，不保留重复的旧文件转发层；外部代码若引用旧内部路径，应切换到上述目录。checkpoint 保存的是权重字典，现有 v1/v3 权重不受模块搬移影响。

所有保留代码均有调用；3D 基线仍有独立模型选项。安装补丁和许可证保留在 `deps/`，它们是重建稀疏依赖所需文件。清理的是失效字节码缓存和旧路径副本，不删除训练数据或模型结果。

两版同时在物理 GPU 0 训练的独立命令见 [GPU 0 并行训练](docs/GPU0_V1_V3.md)。

训练界面依赖 `rich`（当前环境已安装）；3D 基线使用 `torch`、`numpy`，4D 主模型额外依赖 MinkowskiEngine。

## 安装 4D 稀疏依赖

当前项目为 Python 3.12、PyTorch 2.11 / CUDA 13.0，GPU 为 RTX 5090。提供固定上游提交及兼容补丁的构建脚本：

```bash
cd /home/online/wx/qmini
.venv/bin/python tools/lidar_collection/training/deps/setup_minkowski.py --jobs 3
```

脚本只向当前 Python 环境添加编译依赖和 MinkowskiEngine，不降级 torch/numpy，不修改系统 CUDA。默认根据可见 GPU 选择编译架构，RTX 5090 为 12.0；也可显式指定 `--arch 12.0`。需要系统 C++ 编译器、BLAS 开发库、git 和 uv。该脚本针对当前 CUDA 13.0 环境，其他版本不能盲用同一构建命令。源码、构建产物和 wheel 存在 `~/.cache/qmini/`。

MinkowskiEngine 已能导入时不必每次重新构建。可运行：

```bash
OMP_NUM_THREADS=4 .venv/bin/python -c "from tools.lidar_collection.training.models.v1 import SparseSceneNet4D; print(SparseSceneNet4D(base_channels=4, resolution=16).__class__.__name__)"
```

这个导入检查只确认后端可加载；训练与评估流程请使用下方短测命令。

## 数据与坐标

v1/v3 与 3D 基线均使用机身为中心、世界轴对齐的局部地图；v3 的历史地图和持续隐藏网格按中心位移重定位，详见 [v3 坐标说明](docs/RECURRENT_V3.md#坐标与部署)。

- 数据根目录下每个 run 必须包含 `metadata.json`、`terrain.ply` 和 `env_*/episode_*/frame_*.npz`。
- 输入：`lidar_points_w`、`base_pos_w`、`base_quat_xyzw`；监督：`gt_points_w`。四元数为 xyzw。角速度已采集，但这版网络不使用 IMU、线加速度或角加速度。
- 地图以当前底盘位置为中心，坐标轴保持世界方向，范围默认 `[-1.6, 1.6)` 米。这与当前真值采样范围一致；不旋转真值立方体，以免引入没有监督的边界区域。
- 当前观测和历史地图都已在世界系，减去当前底盘位置后对齐；传感器旋转已包含在点云中，不能重复施加安装角。v3 新训练默认不额外编码相对位姿，位姿只用于几何对齐；v1 不额外编码位姿，dense3d 保留 4 个四元数通道。
- 默认 5 cm 体素、3.2 m 区域，即 `64³`。v1/v3 坐标为 `(batch,x,y,z,source)`，来源 0/1 分别为当前观测和历史地图；v3 启用画布时增加来源 2。每体素输入 3 个质心偏移，新训练不额外拼接位姿编码。dense3d 保留 12 通道稠密输入。
- 目标中的“未占据”表示没有采样到地形表面，**不是经过射线验证的自由空间，也不是实体内部/外部标签**。目前真值点云可能含旧地形网格底面或内部面；这版按已有标签学习，未自动清理它们。
- 训练默认取 12 帧连续片段：仅起点随机，内部按时间顺序递归，不跨 episode/reset；缺失编号会报错。整段可打乱，段内不能打乱。
- v1 默认保留旧的 `previous_gt` 训练初始化配方，评估从空地图开始；可显式 `--history_init empty` 统一训练/评估启动。v3 默认 `plane`，GT 只参与监督，首帧为平板＋当前雷达。
- v1/v3 都反馈离散预测地图并 detach。v3 另外保存五级隐藏特征，默认 `--bptt_steps 4`，截断梯度但保留特征值；新片段才重新初始化状态。
- 验证按完整 episode 连续推理，不按训练窗口长度重置；多卡按完整 episode 分配，导出采样不会跳过中间帧推理。

## v3 初始化与递归

历史初始化平板位于首帧机身位置沿世界 Z 轴向下 0.38 m，可用 `--plane_z` 调整；只初始化一次。新训练另有独立画布输入，每帧在当前中心重新生成平面或全体积候选。画布不直接写入输出或观测缓存。第一帧就运行网络并计入监督/指标，不用 GT、不等待预热。

历史地图和隐藏网格按中心位移重定位。各来源体素经 `3×3×3×3` 编码，五级 GRU 读取上次隐藏特征，再生成候选并逐层剪枝。输出占据先验＋残差、历史质心＋残差。全体积画布通过跳接保留最终全格候选，网络决定是否保留；默认直接反馈网络预测。只有显式启用 `--observation_memory` 的对照才会补点并反馈融合地图。

## 训练/验证划分

按 `terrain.ply` 文件的 SHA-256 分组，同一网格的全部 runs 始终进入同一集合。可显式选择验证 run；相同网格的其他 run 会一起保留用于验证。不指定时，用固定 seed 随机保留一个地形组。至少需要两种不同网格。

`split.json` 保存实际使用的完整帧列表，便于复查。分组只能防止完全相同网格泄漏；应另外采集不同 seed、布局与轨迹作最终测试集，不能仅凭相近场景的验证分数宣称泛化效果。

当前数据中，可以先保留 `research_complex_collect_001` 做验证，其余 runs 用于训练。这是跨场景验证，比将相邻帧随机打散更严格。

## 先检查流程

在项目根目录运行下面命令。它最多使用 4 帧训练和 3 帧验证，并将体素加粗到 20 cm，以便 CPU 快速检查数据、梯度和保存流程，**不代表正式训练效果**：

```bash
cd /home/online/wx/qmini

.venv/bin/python tools/lidar_collection/train.py \
  --data tools/lidar_collection/data/runs \
  --val_runs research_complex_collect_001 \
  --model sparse4d_v3 \
  --output tools/lidar_collection/data/training/gru_memory_v3_smoke_001 \
  --device cpu --epochs 1 \
  --seq_len 2 --batch_size 1 --base_channels 4 \
  --voxel_size 0.2 \
  --max_train_batches 2 --max_val_frames 3
```

## 正式训练命令

```bash
cd /home/online/wx/qmini

.venv/bin/python tools/lidar_collection/train.py \
  --data tools/lidar_collection/data/runs \
  --val_runs research_complex_collect_001 \
  --model sparse4d_v3 \
  --output tools/lidar_collection/data/training/gru_memory_v3_001 \
  --device cuda:0 \
  --epochs 30 --seq_len 12 --batch_size 1 \
  --map_size 3.2 --voxel_size 0.05 --base_channels 16 \
  --lr 0.01 --augmentation paper \
  --workers 0 --seed 42
```

输出目录必须不存在，每次新训练更换目录名。`--batch_size` 是每张 GPU 的序列数。sparse4d 将这些序列同一时间步的稀疏坐标合成一个真正的 batch，一次前向；时间步依次展开，地图反馈 detach，v3 隐藏特征按 BPTT 窗口反向传播。dense3d 仍逐序列累计梯度。双卡的全局序列数为 `2 × batch_size`。

sparse4d 使用四个解码尺度的普通 BCE 均值，加目标表面体素内的质心欧氏误差，默认权重比 1:1；默认平板模式下真值只用于监督，不进入历史初始化、不决定剪枝，也不替换预测反馈。Adam 初始学习率默认 0.01；新建 sparse4d 训练默认按验证损失自适应降低学习率，最低 0.0001。选择 `--lr_scheduler exponential` 可使用原先的指数衰减，此时 `--decay` 可覆盖衰减系数。梯度范数裁剪为 1。默认 `paper` 增强包含点噪声、倾斜、局部高度扰动、缺失区域、离群簇、位姿误差和一致的轨迹镜像，真值始终干净。`--point_noise_bound` 控制各轴均匀点噪声上限，默认 0.05 米。增强细节见 PAPER_NOTES。

dense3d 保持旧的单尺度加权 BCE、位置损失、0.001 学习率及每轮 0.98 衰减，默认 `basic` 增强使用 `--noise_std` 高斯点噪声和 `--point_dropout` 随机丢点。上述两个参数只对 basic 生效；`--augmentation none` 可关闭增强。

`--threshold` 默认 0.5，在 sparse4d 中也控制每层解码剪枝。sparse4d 默认 `--max_points 0`，不截断最终输出；dense3d 默认最多输出 20,000 点。显式设置正的数量限制会同时限制反馈和评估点云。显存不足时可降低 `--base_channels`；将 `--voxel_size` 改为 0.1 会降低空间分辨率。sparse4d 要求地图分辨率为 16 的倍数，dense3d 为 8 的倍数。

30 轮只是起始配置，不保证收敛。先检查输入和监督、少量轨迹拟合情况，再关注未见地形上的验证指标。正式训练不要保留 `--max_train_batches` 或 `--max_val_frames` 限制；有限验证的结果会标记 `partial_evaluation=true`。

## v3 单卡训练

完整命令见 [v3 单卡训练](docs/RECURRENT_V3.md#单卡训练)。先用每卡 batch 2、连续 12 帧、BPTT 4；持续隐藏状态和时间梯度需要额外显存，不能沿用其他结构的吞吐结论。

## v1 单卡训练参考

下面选择物理 GPU 1；启动前确认该卡有足够空闲显存。此处为 v1 的历史吞吐配方，5 cm、基础通道 16、连续 12 帧、batch 12：

```bash
cd /home/online/wx/qmini

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
.venv/bin/python tools/lidar_collection/train.py \
  --model sparse4d_v1 --device cuda:0 \
  --data tools/lidar_collection/data/runs \
  --val_runs research_complex_collect_001 \
  --output tools/lidar_collection/data/training/sparse4d_single_001 \
  --epochs 30 --seq_len 12 --batch_size 12 \
  --map_size 3.2 --voxel_size 0.05 --base_channels 16 \
  --window_sampling random --history_init previous_gt \
  --augmentation paper --point_noise_bound 0.02 \
  --voxel_dropout 0.02 --voxel_noise_std 0.05 \
  --lr 0.01 --lr_scheduler plateau --lr_metric val_loss \
  --lr_factor 0.5 --lr_patience 3 --min_lr 0.0001 \
  --workers 4 --prefetch_factor 2 --cpu_threads 4 --seed 42
```

`CUDA_VISIBLE_DEVICES=1` 只暴露物理 GPU 1，因此程序内对应 `--device cuda:0`。单卡不需要 torchrun；batch 12 是同时参与稀疏前向的 12 条序列。若从另一个工作区启动，`--data` 应指向 `/home/online/wx/qmini/tools/lidar_collection/data/runs`。

2026-11-04 的首轮短测，每组训练 3 个 batch、每段 12 帧，包含加载与训练时间，不含验证/保存时间：

| batch | CPU 线程 / workers | 吞吐 frames/s | 峰值 PyTorch 张量显存 | 结果 |
|---|---|---:|---:|---|
| 8 | 4 / 4 | 28.2 | 8.61 GiB | 完成 |
| 12 | 4 / 4 | **31.7** | 12.74 GiB | 完成，采用此配置 |
| 12 | 8 / 6 | 28.5 | 12.77 GiB | 增加 CPU 并行度未提升吞吐 |
| 14 | 4 / 4 | — | — | 稀疏卷积显存不足 |
| 16 | 4 / 4 | — | — | 稀疏卷积显存不足 |

张量统计不包含所有稀疏后端原生分配，不能用 `32 GiB - 张量显存` 来推算还能扩大多少 batch。以上是短测选型结果，正式训练吞吐随剪枝、点云密度与其他进程占用变化，不能保证持续 100% GPU 利用率。正式命令未添加 smoke 帧数限制，验证仍保留原来的 1638 帧，不注入噪声或初始化真值。

## 双卡训练、随机切片与噪声

当前数据保留 `research_complex_collect_001` 的 1638 帧（约 29.7%）做验证，其他三个 run 共 3876 帧（约 70.3%）训练；训练和验证网格文件哈希不重叠。随机切片仅发生在训练集合内部。

```bash
cd /home/online/wx/qmini

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 \
.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  tools/lidar_collection/train.py \
  --model sparse4d_v3 \
  --data tools/lidar_collection/data/runs \
  --val_runs research_complex_collect_001 \
  --output tools/lidar_collection/data/training/gru_memory_v3_dual_001 \
  --epochs 30 --seq_len 12 --batch_size 2 --bptt_steps 4 \
  --map_size 3.2 --voxel_size 0.05 --base_channels 16 \
  --window_sampling random --history_init plane --plane_z -0.38 \
  --augmentation paper --point_noise_bound 0.02 \
  --voxel_dropout 0.02 --voxel_noise_std 0.05 \
  --lr 0.01 --lr_scheduler plateau --lr_metric val_loss \
  --lr_factor 0.5 --lr_patience 3 --min_lr 0.0001 \
  --workers 4 --prefetch_factor 2 --cpu_threads 4 --seed 42
```

每卡 2 条序列、全局 batch 4；`--workers 4` 是每卡 4 个数据读取进程。`torchrun` 自动按 LOCAL_RANK 选卡，不要同时再启动两条普通训练命令。两卡使用相同初始权重，按全局有效帧数归一化梯度，在每次优化前用 NCCL 分桶求和。不同窗口长度、空的末尾 rank、剪枝产生的未使用参数都有处理。BatchNorm 使用每卡局部统计，运行缓冲区以 rank 0 为准，并非 SyncBatchNorm。

训练 sampler 不为凑齐双卡而重复或丢弃窗口索引；随机窗口本身可能重叠。验证按完整 episode 分给两卡，保持时间反馈，按 TP/FP/FN 与匹配点数加权汇总，不平均两个 F1。episode 长度不同时，验证阶段会出现一张卡先结束等待的情况。

Rich 只在 rank 0 输出：配置与划分表、训练/验证进度、loss、训练 frames/s、下一轮学习率、每卡峰值张量显存。验证进度条表示 rank 0 自己的分片，最终指标包含全部分片。`metrics.jsonl` 同时保存吞吐、耗时、卡数及显存数据。峰值张量显存不包括驱动、图形桌面或其他进程；日志重定向后 Rich 不持续刷新终端动画。

**随机片段的时间与初始化：**

- 默认平板模式从同一 episode 随机选择起点 `k ≥ 0`，在第 k 帧位置初始化平板，以该帧雷达作为当前观测，隐藏状态从零开始，随后训练 `k … k+seq_len-1`。
- 片段当前帧的 `sequence_step` 从 0 递增，`sequence_time_s` 从 0 开始；原 `time_s` 保留用于检查采集时间顺序。
- 平板仅初始化一次，GT 仅参与 loss，后续历史来自预测；单帧 episode 同样可训练。每条序列使用独立缓存。
- **第四维是来源标签**：v1 为 0/1；v3 启用画布时为 0/1/2，关闭时为 0/1。都不将片段时钟不断追加到第四维；12 帧是连续递归更新 12 步。
- 验证/部署的每段首帧也采用平板＋当前观测，不添加训练噪声，第一帧就计入指标；不用 GT 填充历史。平板初始化是用户指定变体，不是论文已核实配方。
- `--history_init empty --window_sampling sequential` 可恢复旧的空历史、非重叠窗口训练；随机切片的每轮样本可能重叠，不能把一次 epoch 当作所有原帧恰好遍历一次。

**适量噪声配置：** 上述命令使用点坐标各轴 ±2 cm 均匀扰动、随机丢弃 2% 当前观测体素、体素内质心偏移标准差 0.05 个体素（5 cm 网格下约 2.5 mm），偏移裁剪在原体素内。体素扰动仅支持 sparse4d，默认两个体素参数均为 0；只对新收到的观测生效，不改 GT 或历史预测。paper 的倾斜、局部缺失、高度扰动和离群簇仍生效；`--augmentation basic` 可使用仅高斯点噪声/丢点加可选体素扰动，`--augmentation none` 关闭全部训练噪声（镜像也关闭）。

GPU 利用率取决于稀疏占据率、CPU 坐标构建、读取速度与其他进程。增加 `batch_size` 会提升并行度并增加显存，也改变全局优化 batch；应看实际 frames/s 选择，不能保证始终 100%。已有训练运行时先停止它再启动新命令，其他项目占用 GPU 时吞吐也会变化。

## 自适应学习率

新建 sparse4d 训练默认使用 [ReduceLROnPlateau](https://docs.pytorch.org/docs/2.11/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)，dense3d 保留原来的指数衰减。可在训练命令里明确添加：

```bash
--lr_scheduler plateau --lr_metric val_loss \
--lr_factor 0.5 --lr_patience 3 --lr_threshold 0.001 --min_lr 0.0001
```

每轮验证后根据两卡汇总的验证 loss 更新，两卡得到相同学习率。默认只有比历史最佳 loss **降低超过 0.001** 才算有效改善；容忍 3 个连续未改善 epoch，第 4 个触发减半，之后重新计数。初始 LR 0.01 时为 `0.01 → 0.005 → 0.0025 → … → 0.0001`；不是按固定轮数降低，也不会自动升高。`--lr_metric val_f1` 可改为监控 F1 上升，阈值也是绝对变化量。

Rich 和 JSONL 同时记录本轮 LR、下一轮 LR，以及是否发生下降。正式训练应使用完整验证集，有限帧的 smoke 验证不能代表可靠的调度信号。`best.pt` 仍按验证 F1 选取，不随调度监控项变化。

断点续训默认恢复原调度类型、参数、最佳指标和等待计数，旧 checkpoint 缺少类型字段时按指数衰减处理。若要将已有指数训练切换为自适应，在原有续训命令上添加：

```bash
--lr_scheduler plateau --reset_scheduler \
--lr_metric val_loss --lr_factor 0.5 --lr_patience 3 --min_lr 0.0001
```

该开关仅重置调度器状态，保留模型、Adam 状态和 checkpoint 中的**当前学习率**；不会跳回初始 0.01。`min_lr` 不能高于当前恢复学习率。已经运行的训练不会自动更换调度器，需要从保存的 checkpoint 重启。输出仍应选择新目录。

保留论文所述指数衰减使用 `--lr_scheduler exponential`；`--decay` 只在指数模式下生效。自适应调度属于当前训练扩展。

## 3D 对照基线

3D 代码和已有 checkpoint 保留，显式选择即可：

```bash
.venv/bin/python tools/lidar_collection/train.py \
  --model dense3d \
  --data tools/lidar_collection/data/runs \
  --val_runs research_complex_collect_001 \
  --output tools/lidar_collection/data/training/dense3d_001 \
  --device cuda:0 --epochs 30 --seq_len 12 --batch_size 1 \
  --map_size 3.2 --voxel_size 0.05 --base_channels 8 \
  --lr 0.001 --decay 0.98 --augmentation basic \
  --workers 0 --seed 42
```

两者默认训练配方不同，不是仅替换卷积维度的严格消融。公平比较时固定数据划分、训练预算，并明确记录增强/优化器设置。

## 输出与恢复

```text
data/training/gru_memory_v3_001/
├── config.json     # 模型、网格、优化器及运行设置
├── split.json      # 地形分组与帧列表
├── metrics.jsonl   # 每轮训练损失、验证指标与原始观测基线
├── last.pt         # 最近一轮结束时的完整 checkpoint
└── best.pt         # 验证体素 F1 最优 checkpoint
```

checkpoint 包含模型、优化器、学习率调度器、轮数、随机状态及数据划分。使用原训练命令的相同模型、数据和训练参数，添加 `--resume 原目录/last.pt`，把 `--epochs` 改为更大的**总轮数**，并将 `--output` 改为新目录。不能用 20 cm smoke 权重直接恢复成 5 cm 正式模型，也不能互换 3D/4D 或 v1/v3 权重。未显式指定模型时，旧 v1 checkpoint 会自动选 `sparse4d_v1`，不会静默升级。v3 的相对位姿开关、BPTT 长度、坐标约定及平板高度在续训时也必须一致。恢复时未指定的架构、通道数、学习率、衰减、输出数量限制、窗口采样、历史初始化、平板高度及新噪声参数会从 checkpoint 恢复；其他设置应与原命令一致。改变 GPU 数需要显式加 `--allow_world_size_change`，它会改变全局 batch 和数据顺序；相同卡数会恢复每个 rank 的随机状态。旧 checkpoint 缺少采样/初始化字段时按旧的顺序窗口、空历史恢复，不能在续训时直接切换成另一种初始化配方。新目录的 `best.pt` 仅在超过已保存最佳 F1 时生成，原来的最佳文件仍在原目录。

## 独立评估

```bash
.venv/bin/python tools/lidar_collection/evaluate.py \
  --checkpoint tools/lidar_collection/data/training/gru_memory_v3_001/best.pt \
  --output tools/lidar_collection/data/training/eval_001 \
  --device cuda:0 --export_frames 10
```

默认使用 checkpoint 的验证帧列表，计算全部帧；`--export_frames 10` 只限制导出的可视化样本数量，不限制评估帧数。`--export_every N` 从第一帧起每隔 N 帧导出一次，默认 1；中间帧仍然连续推理并更新历史，不会跳过推理或重复初始化。`--max_frames` 才会截断评估。输出另保存 `config.json`，记录可视化使用的地图范围和体素分辨率。

评估新数据时额外传入 `--data 数据根目录 --runs run名称`。输出 `metrics.json` 和 `predictions/<run>/<env>/<episode>/frame_*.npz`，样本包含世界坐标系预测点、原观测、真值和位姿。文件里记录的原观测保持完整量程，指标只在局部地图内计算。

指标包括表面占据 precision / recall / F1，以及匹配体素内的质心误差（cm）。后者**不是整幅地图的 Chamfer 距离或高度 MAE**；遗漏的表面由 recall 反映。`observed_only` 报告不经网络、仅对雷达点体素化的基线。空预测 F1 为 0，无法计算的匹配误差为 null。评估数据与训练网格重复时会通过 `training_terrain_overlap` 明示，不能当成独立测试。

## 可视化验证结果

当前环境已有 `matplotlib` 和 `plotly`。先完整验证，并跨轨迹抽样导出，再生成可以直接用浏览器打开的离线 HTML 和 PNG：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
.venv/bin/python tools/lidar_collection/evaluate.py \
  --checkpoint tools/lidar_collection/data/training/gru_memory_v3_001/best.pt \
  --output tools/lidar_collection/data/training/eval_gru_memory_v3_001 \
  --device cuda:0 --export_frames 12 --export_every 140

.venv/bin/python tools/lidar_collection/visualize.py \
  --evaluation tools/lidar_collection/data/training/eval_gru_memory_v3_001
```

评估和可视化输出目录须为新目录。`visualization/report.html` 包含 Plotly 本体，不需要联网或启动服务器；滑块切换导出的帧，拖动旋转、滚轮缩放。四幅图分别为雷达观测体素、预测地图、真值体素和占据误差。前三图按相对机身高度着色；误差图绿色为匹配、红色为误检、黄色为漏检。所有图都以当前机身为原点、使用世界轴方向，裁剪到相同局部范围。

`overview.png` 展示首个、中间和末尾导出样本；`learning_curves.png` 展示训练/验证 loss 和验证 F1，并标记本次 checkpoint 轮数；`samples.json` 保存每个导出帧的指标。`--max_points 6000` 仅限制每路点云/错误类别的显示点数，指标始终使用完整有效体素。质心误差仍仅统计匹配体素。HTML 顶部的指标来自全部评估帧，单帧标题中的 F1 只对应当前样本。

可在 `visualize.py` 命令中追加 `--compare 旧版评估目录` 生成 `comparison.png`。对照必须使用相同帧、位姿、输入/真值、地图范围和分辨率；工具会检查这些条件。不同架构或训练轮数的权重对照不能视为受控消融实验。

旧评估报告保留在原输出目录，不会因移除模型代码而删除。已导出的点云仍可离线展示；重新推理仅支持 v1/v3 和 3D 基线。v3 的短程验证不代表正式训练收敛，也不能据此宣称性能优于 v1。

## 仅用雷达点和姿态推理

在项目根目录或已将项目根目录加入 Python 路径的程序中：

```python
from tools.lidar_collection.training.predictor import Reconstructor

reconstructor = Reconstructor("tools/lidar_collection/data/training/gru_memory_v3_001/best.pt", device="cuda:0")
# lidar_points_w: (N,3)，base_pos_w: (3,)，base_quat_xyzw: (4,)
prediction_points_w = reconstructor.update(lidar_points_w, base_pos_w, base_quat_xyzw)
# reset、切换机器人/轨迹或定位跳变时必须清空历史。
reconstructor.reset()
```

单个实例只维护一个机器人的历史。输入点必须已经通过雷达外参和机体位姿变换到世界坐标系。推理 API 不需要地形真值。这只是离线/程序接口，尚未接入实时仿真控制环。

## 验证

整理时使用现有测试副本运行了 73 项回归（源工作区原 tests 目录已不存在，没有恢复该目录）。测试覆盖 v1/v3 稀疏 CPU/CUDA 推理、连续片段与缺帧检查、地形划分、独立 batch 状态、持久隐藏特征及时间梯度、世界轴对齐、零残差与无邻域限制、初始化/reset、连续评估及可视化指标。v2 专属测试随实现移除；加载已移除架构有明确拒绝测试。

```bash
.venv/bin/python -m compileall -q tools/lidar_collection/training
.venv/bin/python -m ruff check tools/lidar_collection/training
```

短测验证流程和兼容性，正式模型仍需在留出地形上完整评估。复制工具到另一工作区时，若未复制采集数据，训练命令中的 `--data` 应指向原来的绝对数据路径。
