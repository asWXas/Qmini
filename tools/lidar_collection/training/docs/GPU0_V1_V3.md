# 同一 GPU 0 上同时训练 v1 / v3

在两个终端分别执行下列命令。两个独立进程都使用物理 GPU 0，不使用 torchrun，也不共享模型、优化器、地图或隐藏状态。每个进程 batch 2、连续 12 帧；它们会共享显存和算力，无法保证比顺序训练更快。输出目录必须是新目录。

以下保留旧两来源对照配方，v3 显式关闭后来新增的画布与观测缓存。新方案见 [画布与观测记忆](CANVAS_MEMORY.md)。训练/验证划分、体素大小、通道数、噪声、学习率和种子一致。v1 使用空历史，不额外注入 GT；v3 使用固定平板和零 hidden。两者启动先验不同，因此并不是只比较 GRU 一项的严格消融。

## 终端 1：v1

```bash
cd /home/online/wx/qmini
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
.venv/bin/python tools/lidar_collection/train.py \
  --model sparse4d_v1 --device cuda:0 \
  --data tools/lidar_collection/data/runs \
  --val_runs research_complex_collect_001 \
  --output tools/lidar_collection/data/training/v1_gpu0_001 \
  --epochs 30 --seq_len 12 --batch_size 2 \
  --map_size 3.2 --voxel_size 0.05 --base_channels 16 \
  --window_sampling random \
  --history_init empty \
  --augmentation paper --point_noise_bound 0.02 \
  --voxel_dropout 0.02 --voxel_noise_std 0.05 \
  --lr 0.01 --lr_scheduler plateau --lr_metric val_loss \
  --lr_factor 0.5 --lr_patience 3 --min_lr 0.0001 \
  --workers 2 --prefetch_factor 2 --cpu_threads 2 --seed 42
```

## 终端 2：v3

```bash
cd /home/online/wx/qmini
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
.venv/bin/python tools/lidar_collection/train.py \
  --model sparse4d_v3 --device cuda:0 \
  --data tools/lidar_collection/data/runs \
  --val_runs research_complex_collect_001 \
  --output tools/lidar_collection/data/training/v3_gpu0_001 \
  --epochs 30 --seq_len 12 --batch_size 2 \
  --map_size 3.2 --voxel_size 0.05 --base_channels 16 \
  --window_sampling random \
  --history_init plane --plane_z -0.38 --bptt_steps 4 --no-relative_pose \
  --no-canvas --no-observation_memory \
  --augmentation paper --point_noise_bound 0.02 \
  --voxel_dropout 0.02 --voxel_noise_std 0.05 \
  --lr 0.01 --lr_scheduler plateau --lr_metric val_loss \
  --lr_factor 0.5 --lr_patience 3 --min_lr 0.0001 \
  --workers 2 --prefetch_factor 2 --cpu_threads 2 --seed 42
```

## 时间连续性

`--window_sampling random` 只随机选择起点 k；`--seq_len 12` 顺序读取 `k → k+1 → … → k+11`，到片段末尾结束，若 episode 较短则在 episode 末尾结束。这里的 end 是连续片段末尾，不是默认跑完剩余整条轨迹。DataLoader 只打乱整段，段内不跳帧、不重排、不跨 reset。每段重新初始化状态，段内持续反馈预测；v3 每 4 帧截断梯度，但隐藏特征值保留。

训练和评估共用连续性检查：同一环境/episode、递增时间、递增 step、相邻文件编号；缺帧或跨 reset 会报错。评估逐帧处理完整 episode，`--export_every` 仅控制保存哪些帧，中间帧仍参与递归。

整理时逐帧检查现有 5514 帧：gamepad_003 为 1378 帧/2 段，research_collect_001 为 1028 帧/1 段，research_collect_002 为 1470 帧/1 段，research_complex_collect_001 为 1638 帧/2 段。各段编号连续，时间递增，相邻记录 step 差均为 5，与采集 `scan_every=5` 一致。训练 3876 帧，验证 1638 帧。

## 本次验证边界

两个进程同时在 GPU 0 完成了 5 cm、基础通道 16、batch 2、连续 12 帧的短测（各一个训练 batch、3 帧验证）。峰值 PyTorch 张量显存分别约 2.26 GiB / 2.87 GiB，不包含全部驱动和稀疏后端分配，也不是长期显存上限。短测已退出，未启动这两条完整的 30 轮训练。随着剪枝和场景密度变化，显存与吞吐也会变化。
