# 动作数据

此目录用于存放真实训练动作。提取时原仓库没有提供 `.pkl` 动作文件，工具不附带完整训练数据集。

- 单动作示例需要 `humanoid/humanoid_spinkick.pkl`。
- 多动作示例需要 `../../config/datasets/dataset_humanoid_locomotion.yaml` 列出的文件，也可以把该列表改成自己的动作。
- 动作路径应与对应角色的骨架和自由度顺序一致。

## 文件放置与配置示例

假设你有旋转踢、跑步、走路三个动作文件，目录可以这样组织：

```text
smp_tools/
├── config/
│   ├── tinymdm_single_clip.yaml
│   ├── tinymdm_multi_clip.yaml
│   └── datasets/
│       └── dataset_humanoid_locomotion.yaml
└── data/
    └── motions/
        ├── README.md
        └── humanoid/
            ├── humanoid_spinkick.pkl
            ├── humanoid_run.pkl
            └── humanoid_walk.pkl
```

### 单动作：只训练旋转踢

在 `config/tinymdm_single_clip.yaml` 中，`motion_file` 直接指向一个 PKL：

```yaml
env_config: "humanoid_global_obs.yaml"
motion_file: "../data/motions/humanoid/humanoid_spinkick.pkl"
control_freq: 30
```

保留该配置的其余网络和训练参数，在 `smp_tools/` 目录执行：

```bash
python train_prior.py \
  --cfg_path config/tinymdm_single_clip.yaml \
  --out_dir output/smp_prior_spinkick
```

### 多动作：一起训练跑步和走路

如果只有跑步和走路两个文件，可以将 `config/datasets/dataset_humanoid_locomotion.yaml` 的内容替换为这个最小列表：

```yaml
motions:
  - file: "../../data/motions/humanoid/humanoid_run.pkl"
    weight: 3.0
  - file: "../../data/motions/humanoid/humanoid_walk.pkl"
    weight: 1.0
```

`weight` 是抽取动作的相对权重。此例中每次抽取动作时，跑步概率为 75%，走路概率为 25%，随后再从选中的动作中随机取时间窗口。保留原来的完整列表时，需要准备列表中的所有文件。

在 `config/tinymdm_multi_clip.yaml` 中，`motion_file` 指向上面的列表 YAML：

```yaml
env_config: "humanoid_local_obs.yaml"
motion_file: "datasets/dataset_humanoid_locomotion.yaml"
control_freq: 30
```

保留其余网络和训练参数，在 `smp_tools/` 目录执行：

```bash
python train_prior.py \
  --cfg_path config/tinymdm_multi_clip.yaml \
  --out_dir output/smp_prior_locomotion
```

**路径规则：YAML 内的相对路径，以该 YAML 文件所在目录为起点。** 训练配置位于 `config/`，所以访问动作目录用 `../data/motions/`；动作列表位于 `config/datasets/`，所以用 `../../data/motions/`。文件名可以替换成你自己的动作文件名。

## PKL 内容格式

每个动作 PKL 保存一个字典：

```python
{
    "loop_mode": 0,  # 0: CLAMP，1: WRAP
    "fps": 30,
    "frames": [...],
}
```

`frames` 的每一行包含 `[root_position(3), root_rotation_exp_map(3), joint_dofs]`。至少提供两帧，帧率需大于零。`anim/motion.py` 提供读写实现。

原 MimicKit README 提供的[资源下载链接](https://1sfu-my.sharepoint.com/:u:/g/personal/xbpeng_sfu_ca/EclKq9pwdOBAl-17SogfMW0Bved4sodZBQ_5eZCiz9O--w?e=bqXBaa)可作为获取数据的入口；下载后把其中的 `motions/` 内容放到本目录。该链接可能需要访问权限。
