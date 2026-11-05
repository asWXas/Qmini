# 每帧画布与已观测地形记忆

v3 新训练默认使用 `--canvas --canvas_mode plane --no-observation_memory`：**输出和下一帧地图反馈均为网络自身预测，不使用缓存补点。** 也可选 `--canvas_mode volume`：每帧把整个局部立方体的体素激活为候选，让网络逐个判断保留或删除。v1 不变。旧 v3 checkpoint 保留原配置，不自动升级；续训默认继承 checkpoint 的缓存开关，纯网络续训请显式指定 `--no-observation_memory`。

## 数据流

```text
当前雷达体素 source=0 ─────────────────┐
对齐的上帧网络预测 source=1 ───────────┼→ 4D 编码 → 多尺度 GRU → 残差解码 → 当前网络地图
每帧重新生成的画布 source=2 ──────────┘                   ↑                       │
对齐的上帧隐藏特征 ─────────────────────────────────────┘                       └→ 下一帧反馈
```

下一帧反馈网络预测，并持续保存五级 GRU hidden。GRU 通过连续帧监督学习保留过去观测、利用新证据修正和补全；即使某个位置本帧没有雷达，也能依靠历史特征参与预测。没有新增独立的“上一帧原始点云”网络输入。

默认路径既不更新观测缓存，也不将当前雷达或画布直接复制到输出。网络可以新增、保留、删除或移动候选；首帧平板仍是已约定的输入先验，进入残差预测，并不是强制贴在输出上的地面。单独关闭缓存不会自动提高旧模型质量；新画布、平衡监督及持续特征需要实际训练后评估。

`--observation_memory` 只作为可选系统对照：启用时额外保存测量质心，与网络输出融合，再反馈融合地图。它的分数属于“网络＋几何缓存”系统，不能作为纯网络补全能力的分数。

| 画布模式 | 默认 64³ 下每帧体素数 | 作用 |
|---|---:|---|
| `plane` | 4,096 | 当前机身下方 `plane_z=-0.38 m` 的 XY 平面候选 |
| `volume` | 262,144 | 全 XYZ 范围候选，允许网络雕刻台阶、坡面、侧面等表面 |

画布的“占用”指激活稀疏坐标，不代表每个位置的真实占据概率为 1。画布不进入观测缓存，也不直接拼到最终输出。最终仍由历史占据先验加网络残差决定保留/删除，质心由历史质心加偏移残差决定。历史初始化仍是 `--history_init plane`；它与每帧的第三路画布是不同的东西。

每一级画布特征通过跳接到达解码器；全体积模式即使粗层全部剪空，最终层仍有完整的 `resolution³` 候选，不再因缺少细层坐标而无法补某个位置。第三路融合使用居中的三来源卷积，确实读取 source 0/1/2。**有候选不等于预测正确**，网络仍可能把未观测地形判空。

当前监督是地形**表面**体素，不是石块内部也填满的实心占据标签。全体积中大量位置没有表面标签，因此 volume 新训练默认 `--occupancy_loss balanced`：每个解码尺度分别计算占据/非占据平均 BCE，各占一半；某一类为空时使用普通 BCE。可显式选择 `bce` 作对照。类别平衡改变优化权重，不保证固定阈值下的概率校准或最佳召回率。

## 可选观测缓存对照（默认关闭）

- 只使用当前局部观测体素的质心。训练时使用噪声、丢点、体素扰动后的观测；GT 始终只用于监督和指标。画布及网络推测都不作为传感器证据。
- 用一个固定锚点保存表面质心，累积帧间变换；不依赖导航绝对位置。缓存栅格为网络体素尺寸的一半，同格观测加权平均，旧权重上限 64，防止反复移动地图时把微小位移逐次取整丢失。
- 默认 `--memory_size 12.8 --memory_max_voxels 200000`，范围跟随当前中心。离开当前 3.2 m 局部地图、仍在缓存范围内的已观测表面，返回时可以重新取回。超出缓存范围的部分被裁掉，超过容量时保留最近更新的格子。这不是无限大的全局地图。
- 最终融合时，同一体素内采用观测缓存质心，其他位置保留网络输出；网络不能删除当前缓存内的已观测体素。要求 `--max_points 0`，避免点数截断再次丢掉已观测表面。
- 缓存面向当前静态地形；它也会保留测量噪声，受到相对位姿漂移影响，尚无射线清空或动态物体删除机制。
- 新序列、episode 或显式 `reset()` 清空地图、GRU 状态及观测缓存。训练起点可以随机，随后严格连续；BPTT 截断只切断隐藏状态梯度，不清空记忆。

## 纯网络全体积训练命令

在可用的 GPU 上启动新进程；当前已运行的训练不会自动采用这次修改。下面用物理 GPU 0、batch 1、基础通道 8，降低全体积计算成本。保持连续 12 帧，每 2 帧截断梯度。输出目录必须不存在。

```bash
cd /home/online/wx/qmini
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
.venv/bin/python tools/lidar_collection/train.py \
  --model sparse4d_v3 --device cuda:0 \
  --data tools/lidar_collection/data/runs \
  --val_runs research_complex_collect_001 \
  --output tools/lidar_collection/data/training/v3_volume_neural_001 \
  --canvas --canvas_mode volume --occupancy_loss balanced \
  --no-observation_memory \
  --history_init plane --plane_z -0.38 --no-relative_pose \
  --epochs 60 --seq_len 12 --bptt_steps 2 --batch_size 1 \
  --map_size 3.2 --voxel_size 0.05 --base_channels 8 --max_points 0 \
  --window_sampling random \
  --augmentation basic --noise_std 0.01 --point_dropout 0.05 \
  --voxel_dropout 0.02 --voxel_noise_std 0.05 \
  --lr 0.001 --lr_scheduler plateau --lr_metric val_loss \
  --lr_factor 0.5 --lr_patience 3 --min_lr 0.0001 \
  --workers 2 --prefetch_factor 2 --cpu_threads 4 --seed 42
```

平面方案将 `--canvas_mode volume --occupancy_loss balanced` 改为 `--canvas_mode plane --occupancy_loss bce`，使用另一个输出目录。退回旧两来源方案用 `--no-canvas --no-observation_memory`。只有明确做缓存对照时，才改为 `--observation_memory --memory_size 12.8 --memory_max_voxels 200000`。

画布改变来源合并权重的形状，不能拿现有两来源 v3 权重直接 `--resume --canvas`；程序会明确报错。续训默认从 checkpoint 继承画布/记忆/损失配置，改变画布或损失配方需要新训练。加入观测记忆本身不改变网络参数形状，可在旧模型上独立开启。

## 纯网络评估与可选缓存对照

新模型按训练配置默认输出网络预测。也可显式指定，防止误用曾经启用缓存的 checkpoint：

```bash
.venv/bin/python tools/lidar_collection/evaluate.py \
  --checkpoint tools/lidar_collection/data/training/v3_volume_neural_001/best.pt \
  --output tools/lidar_collection/data/training/eval_v3_volume_neural_001 \
  --no-observation_memory --device cuda:0 \
  --export_frames 12 --export_every 140

.venv/bin/python tools/lidar_collection/visualize.py \
  --evaluation tools/lidar_collection/data/training/eval_v3_volume_neural_001
```

以下命令仅用于旧权重的缓存对照：

```bash
cd /home/online/wx/qmini
.venv/bin/python tools/lidar_collection/evaluate.py \
  --checkpoint tools/lidar_collection/data/training/v3_gpu0_001/best.pt \
  --output tools/lidar_collection/data/training/eval_v3_observation_memory_001 \
  --observation_memory --device cuda:0 \
  --export_frames 12 --export_every 140

.venv/bin/python tools/lidar_collection/visualize.py \
  --evaluation tools/lidar_collection/data/training/eval_v3_observation_memory_001
```

上例连续评估整个验证集，每 140 帧导出一帧，中间没有跳过推理。新画布模型同样用这两个入口，画布配置自动取自 checkpoint。

相对位姿部署接口用 `RelativeReconstructor(checkpoint, device="cuda:0", observation_memory=False)` 强制网络直接输出；不传该选项则继承 checkpoint 配置。只有对照时才设 `True`。`update` 的参数仍为当前机身系点云与 `T_current_from_previous`，返回当前中心、固定世界轴的局部地图。`reset()` 会清空地图、hidden 及可能存在的缓存。

## 指标与验证

默认 `prediction` 就是网络自身预测，`best.pt` 按这个输出的验证 F1 选取。Rich 和新生成的可视化报告明确标注最终输出类型。启用缓存对照时，`metrics.json` 同时保存：`prediction`（融合后地图）、`network_prediction`（本次递归的网络原始输出）、`retained_observations`（仅缓存）、`observed_only`（仅本帧）。训练 loss 始终监督网络原始输出；缓存模式的 best checkpoint 按融合地图验证 F1 选取。

导出的 NPZ 增加 `network_prediction_points_w` 与 `retained_observations_points_w`，与原 `prediction_points_w` 分开，避免把几何累积的收益误认为网络补全能力。因为融合地图会反馈下一帧，开启记忆后的网络原始输出也可能不同于未开启记忆的独立运行。

2026-11-04 验证：

- 最新默认纯网络路径的 14 项功能检查通过：包含网络判空时不复制雷达/平板、输出直接反馈、无缓存调用、持续 hidden 的跨帧梯度、GT 不进入推理及旧缓存对照兼容。纯网络 `64³` / 基础通道 8 / batch 1 / 连续 3 帧 / BPTT 2 的 CUDA 训练与 2 帧评估短测通过，峰值张量显存约 1.79 GiB；未完成正式训练，不能据此宣称补全效果改善。
- 原有 73 项回归检查通过；新增检查覆盖三来源融合、全体积候选不被粗剪枝永久删除、观测与 GT/画布隔离、跨帧梯度、测量扰动、相对位姿、出界返回、容量限制、状态重置及批内隔离。
- CUDA 5 cm / `64³` 全体积、基础通道 8、batch 1、连续 3 帧/BPTT 2 的训练和 2 帧评估短测通过；峰值 PyTorch 张量显存约 1.78 GiB，不含所有稀疏后端/驱动分配，不代表长期训练显存上限。尚未完成新画布模型的正式训练。
- 同一 epoch 31 v3 权重、同一留出地形 frame 144→155，仅开启观测记忆：F1 `0.452→0.555`，recall `30.4%→39.9%`；最后一帧 F1 `0.517→0.757`、recall `36.7%→62.1%`。这 12 帧中，融合结果均保留缓存投影到当前地图的全部体素。这是小段验证，不是完整验证集或新画布模型的结果。

对照报告：[连续 12 帧观测记忆验证](../../data/training/eval_memory_random12_seed42_001/index.html)。

功能回归检查：

```bash
.venv/bin/python -m pytest tools/lidar_collection/training/tests -q
```
