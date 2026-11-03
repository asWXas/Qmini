import pickle
from pathlib import Path

# 仅加载来源可信的 PKL 文件
path = (
    Path(__file__).resolve().parent
    / "mothion_data/MimicKit_Data/motions/g1/g1_double_kong.pkl"
)

if not path.is_file():
    raise FileNotFoundError(f"文件不存在: {path}")

with path.open("rb") as f:
    data = pickle.load(f)

print("文件:", path.name)
print("数据类型:", type(data))

if not isinstance(data, dict):
    raise TypeError(f"预期为字典，实际为 {type(data).__name__}")

print("\n字段信息:")
for key, value in data.items():
    print(
        f"  {key}: 类型={type(value).__name__}, "
        f"shape={getattr(value, 'shape', None)}"
    )

required_keys = {"frames", "fps", "loop_mode"}
missing_keys = required_keys - data.keys()
if missing_keys:
    raise ValueError(f"缺少字段: {sorted(missing_keys)}")

frames = data["frames"]
fps = data["fps"]
loop_mode = data["loop_mode"]
num_frames = len(frames)

print("\n动作信息:")
print("循环模式:", loop_mode, {0: "不循环", 1: "循环"}.get(loop_mode, "未知"))
print("帧率:", fps)
print("总帧数:", num_frames)

if num_frames > 0:
    first_frame = frames[0]
    frame_dim = len(first_frame)

    print("每帧维度:", frame_dim)
    if fps > 0:
        print(f"首尾帧时间跨度: {(num_frames - 1) / fps:.3f} 秒")

    if frame_dim < 6:
        raise ValueError("每帧至少需要包含根位置 3 维和根旋转 3 维")

    print("关节自由度数量:", frame_dim - 6)
    print("\n第一帧:")
    print("根位置:", first_frame[:3])
    print("根旋转（指数映射）:", first_frame[3:6])
    print("关节自由度:", first_frame[6:])
else:
    print("动作数据为空")