# v3：每帧画布、持续 GRU 记忆与全局候选残差

默认选择 `--model sparse4d_v3`。保留 `sparse4d_v1` 和独立 3D 基线；v2 及其旧 `sparse4d` 参数已移除。v3 必须使用新目录重新训练，不能直接续训 v1 或已移除的 v2 检查点。

新训练默认增加第三路画布，关闭输出端的独立观测缓存。上一帧地图本身携带历史几何，GRU hidden 携带学习到的特征；网络融合这两种历史与当前雷达，输出本帧地图并反馈。旧两来源 v3 权重仍兼容原配置。全体积训练命令和可选缓存对照见 [画布与观测记忆](CANVAS_MEMORY.md)。

## 实际结构

```text
当前雷达体素 ─────────────────────┐
上一帧网络地图 ──相对变换/体素化───┼→ 4D 编码器 → 来源卷积合并 → 各尺度 GRU → 生成式解码/剪枝
每帧平面/全体积画布 ───────────────┘
上一帧各尺度隐藏特征 ──相对变换──────→ 各尺度 GRU 的 hidden 输入                 ↓
                                         ↓                              占据/质心残差
                                保存新隐藏特征供下一帧                         ↓
                                                       对齐的历史地图 + 残差 → 网络地图 M_t
                                                                                 ↓
                                                                        输出并作为下一帧历史
```

- 当前观测和历史地图为来源 0/1，启用画布时另有来源 2；没有单独的上一帧原始点云网络输入。M_t 是网络内部融合与残差更新的结果，默认没有额外观测缓存分支，也不把雷达点直接并入输出。
- 编码使用 `3×3×3×3` 卷积跨来源混合；空间下采样 4 次，通道为 16/32/64/128/256（基础通道可调）。
- 瓶颈和四级跳接都用卷积合并来源，然后输入 GRUCell。GRU 的 hidden **是真正上一时刻保存的特征**，不是重新编码的上一张地图。
- 五个尺度分别保存稀疏坐标与隐藏特征；下一帧先变换特征格中心，再分配到该尺度网格，碰撞取均值，移出局部范围的部分丢弃。特征通道作为学习到的标量处理，不作为几何 XYZ 向量旋转；此实现不是严格旋转等变网络。重体素化的量化误差可能随时间累积。
- 保留生成式上采样、多尺度占据监督和预测概率剪枝。**没有观测邻域硬限制**。全体积画布通过跳接保留全部最终候选，但仍需要网络判断哪些位置有表面；平面画布只保证平面候选存在。两者都不保证未知区域预测正确。
- 最终占据 logit 为历史占据先验（默认阈值下 ±2）加预测残差；质心为旧质心加有界偏移残差。未占据候选从体素中心开始。跨体素移动通过新增/删除实现。
- 隐藏特征可保存过去证据并学习遗忘；生成式解码器可根据结构规律补全未观测区域。记忆、补全和正确性是三件事，必须通过完整训练及留出地形评估验证。

## 首帧与训练时序

按最新约定，默认 `--history_init plane --plane_z -0.38`：首帧历史地图为平板，另一路为当前雷达；相对变换为单位变换，五级隐藏状态从零开始。平板沿世界 Z 轴位于首帧机身下方 0.38 m；这是先验，不读取 GT，不随机身俯仰倾斜。只在初始化生成一次，之后随历史预测更新。

上述是历史地图的初始化。第三路画布则每帧围绕当前中心重新生成，不作为已确认占据或观测证据；当前中心升降时，平面画布仍位于当前中心的 `plane_z` 高度。所有输出候选包括初始平板和观测位置，都可由网络判断保留或删除。

后续反馈网络地图和隐藏特征。地图通过阈值解码，因此几何反馈 detach；**隐藏特征不逐帧 detach**。默认 `--bptt_steps 4`，连续 4 帧累计 loss 后反向传播，再 detach 隐藏特征以释放计算图，但保留数值继续递归；每个序列 batch 结束才做一次 optimizer step。`--bptt_steps 1` 可作为仅单步梯度的对照，仍保留跨帧特征值。增加此值会增加训练显存。只有显式 `--observation_memory` 的系统对照才会额外补点并反馈缓存融合地图。

**训练必须取连续片段。** `--window_sampling random` 只随机选择起点，例如取同一 episode 的第 100～111 帧；不会随机拼接 12 帧。DataLoader 打乱的是片段顺序，片段内部始终按时间推进，不跨环境或 reset，遇到缺失/跳过的帧编号会报错。这里的连续指连续雷达记录帧（如 `scan_every=5`），不是要求每个仿真步都有雷达。

每个随机窗口从平板地图和零隐藏状态开始，窗口内部不重置状态；不同序列/GPU/episode 的状态独立。验证和部署持续保留状态直到 reset，不按训练的 BPTT 长度清空。checkpoint 保存模型权重，不保存某一条机器人实时轨迹的隐藏状态；重新加载后从首帧初始化。

保留 `--history_init current`（两路均由当前观测初始化）和 `empty` 用于对照，默认仍为平板。v3 不支持 GT 初始化，避免训练部署初态不一致。

## 坐标与部署

**v3 与 v1 一样：地图中心跟随当前机身位置，XYZ 轴保持世界方向。** 采集 NPZ 继续使用世界坐标，不需要重采数据。观测和 GT 都减去当前机身位置，默认范围为 `[-1.6, 1.6)` 米；网络不输入绝对位置。

历史地图和五级隐藏网格只按中心位移重定位：

```text
当前局部点 = 世界点 - 当前机身位置
历史局部点移到当前地图 = 上帧局部点 + 上帧机身位置 - 当前机身位置
```

机器人原地转身时，历史地图和隐藏网格不旋转。部署输入的机身系雷达点需要先按当前朝向旋转到固定世界轴；位姿只用于几何对齐，新训练默认不把相对平移/旋转编码拼接到网络特征。

变换方向必须是 `T_current_from_previous`：

```text
x_current = R @ x_previous + translation
T_current_from_previous = inverse(T_world_current) @ T_world_previous
```

机器人向前走 0.2 m、无旋转时，此变换的 X 平移为 **-0.2 m**。若设备提供的是当前机身在上一帧中的位姿，需要先取逆。

部署需要当前机身系点云、帧间变换，以及初始化时的轴方向约定：

```python
import numpy as np
from tools.lidar_collection.training.predictor import RelativeReconstructor

reconstructor = RelativeReconstructor(
    ".../best.pt",
    device="cuda:0",
    initial_quaternion_xyzw=initial_body_orientation,  # 机身到世界，xyzw
)
map_local = reconstructor.update(lidar_points_body)  # 首帧默认单位变换

T_current_from_previous = np.eye(4, dtype=np.float32)
T_current_from_previous[0, 3] = -0.2  # 仅示意：实际必须来自每两帧间的运动估计
map_local = reconstructor.update(next_lidar_points_body, T_current_from_previous)
reconstructor.reset(initial_quaternion_xyzw=new_initial_orientation)  # 新轨迹/仿真 reset
```

雷达外参需要先把传感器点转换到机身坐标；接口不接收原始雷达系点。返回 `map_local` 是机身中心、固定世界轴方向的局部地图，不是机身旋转系点云。每个实例只服务一条轨迹。

不需要导航绝对位置。接口用帧间旋转更新朝向，将帧间平移转换到固定轴后移动历史地图。启用观测缓存时，在私有固定锚点中累积相对运动以保存已观测表面；这仍会受到里程计漂移影响。首帧姿态可来自 IMU；忽略初始四元数时以首帧机身方向定义固定轴，若开机时机身倾斜，这样的 Z 轴就不再与训练的重力方向一致。reset 后需重新提供初始朝向。

已有世界坐标/位姿时，仍可使用 `Reconstructor(checkpoint).update(lidar_points_w, base_pos_w, base_quat_xyzw)`，返回世界系地图。该接口只在适配层使用绝对位姿，与只输入相对变换的接口共享同一网络和状态更新逻辑。

### 记录真值范围与指标

现有 GT 采自世界轴方向的立方体，v3 保留该范围；机器人转身不会裁掉监督立方体角落。loss、指标和可视化均沿用相同局部范围，可以在相同留出帧、体素大小和阈值下对照 v1。网络仍没有观测邻域硬限制。

论文图 2 与 III 节描述将上一预测变换到当前测量坐标系，不能据此认定论文与这里的世界轴约定完全一致。参见 [论文原文](https://arxiv.org/pdf/2206.08077)。世界轴局部地图、持续 GRU 状态和 TBPTT 是本项目的实现选择和扩展。

## 单卡训练

在空闲 GPU 上启动。下面选择物理 GPU 0，batch 先用 2；持续特征和 TBPTT 的成本不同于旧版，不能直接沿用旧 batch 12 的性能结论。

```bash
cd /home/online/wx/qmini
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
.venv/bin/python tools/lidar_collection/train.py \
  --model sparse4d_v3 --device cuda:0 \
  --data tools/lidar_collection/data/runs \
  --val_runs research_complex_collect_001 \
  --output tools/lidar_collection/data/training/gru_canvas_neural_v3_001 \
  --epochs 30 --seq_len 12 --bptt_steps 4 --batch_size 2 \
  --map_size 3.2 --voxel_size 0.05 --base_channels 16 \
  --no-relative_pose --history_init plane --plane_z -0.38 \
  --canvas --canvas_mode plane --no-observation_memory \
  --window_sampling random \
  --augmentation paper --point_noise_bound 0.02 \
  --voxel_dropout 0.02 --voxel_noise_std 0.05 \
  --lr 0.01 --lr_scheduler plateau --lr_metric val_loss \
  --lr_factor 0.5 --lr_patience 3 --min_lr 0.0001 \
  --workers 4 --prefetch_factor 2 --cpu_threads 4 --seed 42
```

v3 的 `config.json` 明确记录 `architecture=minkowski_recurrent_residual_v3`、`map_frame=world`、`update_radius=null`、`relative_pose=false`、`bptt_steps`，以及 `canvas/canvas_mode/observation_memory/occupancy_loss`。不再提供观测邻域限制参数。续训时保持架构、初始化、画布与损失配方、相对位姿选项、坐标约定和 BPTT 配置一致；未显式设置的相关参数从 checkpoint 恢复。v1 checkpoint 不会自动升级，v2 checkpoint 会明确报错。旧检查点若保存 `relative_pose=true`，加载/续训仍保留其编码结构；无编码模型需在新目录重新训练。旧两来源权重不能直接开启第三路画布续训。

评估及可视化仍使用原入口：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/lidar_collection/evaluate.py \
  --checkpoint tools/lidar_collection/data/training/gru_canvas_neural_v3_001/best.pt \
  --output tools/lidar_collection/data/training/eval_gru_canvas_neural_v3_001 \
  --no-observation_memory --device cuda:0 --export_frames 12 --export_every 140

.venv/bin/python tools/lidar_collection/visualize.py \
  --evaluation tools/lidar_collection/data/training/eval_gru_canvas_neural_v3_001
```

## 验证范围

已验证世界轴保持、转身时隐藏网格不旋转、移动时地图/隐藏特征重定位、SE(3) 几何工具、碰撞平均与裁剪、后续帧 loss 回传到前帧隐藏特征、TBPTT 边界保值、空观测时远处新增/删除、序列重排时状态隔离、相对/世界接口一致、reset、GT 不进入预测、CPU/CUDA 及真实数据训练。5 cm、基础通道 16、batch 2、4 帧窗口、BPTT 3 的短测已通过；这不代表完成正式训练或重建性能优于 v1。
