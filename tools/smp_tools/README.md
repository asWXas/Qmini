# 独立 SMP 动作先验训练工具

整个 `smp_tools/` 文件夹可以直接复制到其他位置使用。数据加载、运动学、归一化、网络、训练、日志和可视化的本地代码依赖均包含在目录内；不需要原 MimicKit 仓库或物理仿真器。

```text
smp_tools/
├── train_prior.py          # 训练 / 采样入口
├── motion_prior_dataset.py # 动作加载、时间窗口采样与格式转换
├── motion_obs.py           # 观测编码
├── config_util.py         # 配置与相对路径处理
├── normalizer.py          # 观测归一化
├── tinymdm/                # TinyMDM、DiT、EMA、扩散采样
├── anim/                   # 动作库、运动学、MJCF / URDF / USD 骨架读取
├── util/                   # 张量运算、日志、进程工具及 GIF 可视化
├── config/                 # 单动作 / 多动作训练及观测配置
│   └── datasets/           # 动作文件列表和采样权重
├── data/
│   ├── assets/humanoid/   # 已包含 humanoid.xml 和 humanoid.usd
│   └── motions/           # 放置真实训练动作
├── tests/smoke_test.py     # 无需真实动作数据的训练与迁移测试
├── requirements.txt       # 全部第三方直接依赖
├── LICENSE
└── CITATION.cff
```

## 安装

复制文件夹后，在新位置执行：

```bash
cd smp_tools
pip install -r requirements.txt
```

第三方依赖为 PyTorch、diffusers、NumPy、PyYAML、Gymnasium、Matplotlib、Pillow 和 usd-core。`usd-core` 用于 USD 骨架读取；MJCF / URDF 通过 Python 自带 XML 库读取。无需安装 Isaac Gym、Isaac Lab、MuJoCo 或原项目的其他包。本工具已在 Python 3.12 环境验证。

## 准备动作数据

**已包含角色文件，但源仓库本身没有训练动作 `.pkl`，因此真实动作数据仍需自行提供。** 将动作放入本目录的 `data/motions/`，详见 [数据说明](data/motions/README.md)。

单动作配置默认读取 `data/motions/humanoid/humanoid_spinkick.pkl`。多动作配置读取 `config/datasets/dataset_humanoid_locomotion.yaml` 中列出的文件；可以将这个列表改为自己已有的动作及采样权重。

所有 YAML 中的相对路径均相对于**该 YAML 文件所在目录**，与启动命令的工作目录无关。例如：

```yaml
# config/tinymdm_multi_clip.yaml
env_config: "humanoid_local_obs.yaml"
motion_file: "datasets/dataset_humanoid_locomotion.yaml"
control_freq: 30
```

```yaml
# config/humanoid_local_obs.yaml
char_file: "../data/assets/humanoid/humanoid.xml"
```

```yaml
# config/datasets/dataset_humanoid_locomotion.yaml
motions:
  - file: "../../data/motions/humanoid/humanoid_run.pkl"
    weight: 3.0
  - file: "../../data/motions/humanoid/humanoid_walk.pkl"
    weight: 1.0
```

保留 `env_config` 字段名是为了沿用模型配置格式；它只描述角色及观测，不构建仿真环境。观测配置中的 `num_disc_obs_steps` 指定时间窗口长度，`global_obs`、`root_height_obs`、`disc_dof_vel_obs` 和 `key_bodies` 定义网络输入。维度由数据加载器自动计算。

## 训练网络

在 `smp_tools/` 内训练多动作先验：

```bash
python train_prior.py \
  --cfg_path config/tinymdm_multi_clip.yaml \
  --out_dir output/smp_prior
```

单动作训练：

```bash
python train_prior.py \
  --cfg_path config/tinymdm_single_clip.yaml \
  --out_dir output/smp_prior_spinkick
```

设备默认是 `cuda`，CPU 训练可追加 `--device cpu`。命令行中的相对路径以当前工作目录为基准；省略 `--cfg_path` 时自动使用本工具自带的多动作配置。也可以在文件夹的父目录运行 `python -m smp_tools.train_prior`。

训练过程包括动作采样、正向运动学、观测编码、归一化统计、加噪与噪声预测、AdamW 更新、EMA 更新及模型保存。训练参数位于 `config/tinymdm_*.yaml`，包括 `batch_size`、`lr`、`num_iterations`、`T`、`num_layers`、`num_attention_heads` 和 EMA 参数。

| 输出 | 内容 |
| --- | --- |
| `model.pt` | 网络、EMA 和归一化统计量的 state_dict |
| `diffusion_config.yaml` | 训练配置及计算后的输入维度 |
| `env_config.yaml` | 角色及观测配置副本 |
| `log.txt` | 训练损失 |
| `samples/` | 生成动作的 .pkl 和 .gif |

保存的配置使用相对路径。在 `smp_tools/` 内保存数据和输出后，可连同它们一起移动整个目录。若配置引用目录外的自定义数据，也需要一并迁移那些数据并更新路径。

## 采样验证

```bash
python train_prior.py --mode test \
  --cfg_path output/smp_prior/diffusion_config.yaml \
  --model_file output/smp_prior/model.pt \
  --out_dir output/smp_prior_samples
```

采样会读取角色和动作配置来建立数据维度并转换输出，因此仍需保留相应数据文件。

## 独立运行检查

```bash
python tests/smoke_test.py
```

测试用自带角色生成临时合成动作，验证数据加载、CPU 单步训练、EMA、权重保存、移动输出目录后的重载、动作及 GIF 生成，并禁止导入原项目模块。合成数据仅用于验证程序流程。

本工具的运动学和基础工具代码提取自 MimicKit，保留原项目的 [许可证](LICENSE) 和 [引用信息](CITATION.cff)。


现在文件夹已经独立放在 `/home/online/wx/online_ws/smp_tools`。我们先建立整体认识：**它负责把动作数据训练成一个能够生成动作片段的 SMP 先验网络。**

目录可以按职责这样看：

```text
smp_tools/
├── train_prior.py          ← 总入口：组织训练、保存模型、采样
│
├── config/                ← 决定用什么数据、怎样训练
│   ├── tinymdm_multi_clip.yaml
│   ├── tinymdm_single_clip.yaml
│   ├── humanoid_local_obs.yaml
│   ├── humanoid_global_obs.yaml
│   └── datasets/          ← 动作文件列表与采样权重
│
├── data/
│   ├── assets/humanoid/   ← 角色骨架
│   └── motions/           ← 训练动作
│
├── motion_prior_dataset.py ← 把动作变成训练样本
├── motion_obs.py           ← 把姿态、速度等编码成网络输入
├── normalizer.py           ← 对输入做归一化
│
├── tinymdm/                ← 扩散模型与神经网络
├── anim/                   ← 动作读取、插值、骨架与运动学
├── util/                   ← 数学、日志、可视化等辅助工具
│
├── config_util.py          ← 配置读取及路径解析
├── tests/smoke_test.py      ← 用合成数据验证整个流程
└── requirements.txt         ← 第三方依赖
```

**先重点看这五个文件，其他模块用到时再展开：**

| 文件 | 主要回答的问题 |
|---|---|
| [config/tinymdm_multi_clip.yaml](/home/online/wx/online_ws/smp_tools/config/tinymdm_multi_clip.yaml) | 用哪些数据？网络多大？训练多少次？ |
| [train_prior.py](/home/online/wx/online_ws/smp_tools/train_prior.py) | 整个训练流程如何组织？ |
| [motion_prior_dataset.py](/home/online/wx/online_ws/smp_tools/motion_prior_dataset.py) | 怎样从动作文件中取出一个 batch？ |
| [tinymdm/tinymdm_model.py](/home/online/wx/online_ws/smp_tools/tinymdm/tinymdm_model.py) | 怎样加噪、计算损失、生成动作？ |
| [tinymdm/arch.py](/home/online/wx/online_ws/smp_tools/tinymdm/arch.py) | 真正的 DiT 神经网络是什么结构？ |

一次训练迭代的数据流是：

```mermaid
flowchart LR
    A[动作 PKL 与角色骨架] --> B[随机抽取动作时间窗口]
    B --> C[运动学计算与观测编码]
    C --> D[归一化]
    D --> E[加噪]
    E --> F[DiT 预测噪声]
    F --> G[计算损失并更新网络]
```

当前多动作默认配置使用 **30 Hz 的采样频率、10 帧时间窗口、512 个样本的 batch**。因此，网络每次处理的是一批短动作片段。

有两个命名容易混淆：

- `env_config`：这里指**角色与观测配置**，不启动仿真环境。
- `tinymdm_model.py` 与 `arch.py`：前者负责扩散训练和采样逻辑，后者负责神经网络结构。

接下来阅读时，可以从 `train_prior.py` 的 `train()` 开始，先抓住这三行：

```python
dataset_env = MotionPriorData(config, device)  # 建立数据加载器
samples = dataset_env.fetch_obs_demo(batch_size)  # 取得动作样本
loss = model(samples)  # 计算扩散训练损失
```

沿着这三处往下看，就能串起这个目录最核心的代码。