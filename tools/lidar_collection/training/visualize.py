"""Offline, self-contained 3D validation report from exported evaluation NPZs."""

import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch

from .geometry.sparse_voxel import sparse_voxelize


def cloud(points, position, config, quaternion=None):
    """Use the same float32 cropping/voxelization as evaluation before rendering."""
    local = torch.as_tensor(points, dtype=torch.float32) - torch.as_tensor(position, dtype=torch.float32)
    if config.get("map_frame") == "body":
        from .geometry.temporal import rotation_matrix

        if quaternion is None:
            raise ValueError("Body-frame visualization requires base_quat_xyzw")
        quaternion = torch.as_tensor(quaternion, dtype=torch.float32)
        local = local @ rotation_matrix(quaternion)
    cells, offsets = sparse_voxelize(local, config["map_size"], config["resolution"])
    if config.get("map_frame") == "body":
        from .runtime.state import coverage_mask

        valid = coverage_mask(cells, 1, config, quaternion)
        cells, offsets = cells[valid], offsets[valid]
    xyz = (cells + offsets) * (config["map_size"] / config["resolution"]) - config["map_size"] / 2
    cells = cells.numpy().astype(np.int64)
    keys = (cells[:, 0] * config["resolution"] + cells[:, 1]) * config["resolution"] + cells[:, 2]
    return xyz.numpy(), keys


def comparison(prediction, target):
    points, keys = prediction
    truth, truth_keys = target
    matched = np.isin(keys, truth_keys)
    missing = ~np.isin(truth_keys, keys)
    tp, fp, fn = int(matched.sum()), int((~matched).sum()), int(missing.sum())
    common, p_index, t_index = np.intersect1d(keys, truth_keys, return_indices=True)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "matched_centroid_error_cm": (
            float(np.linalg.norm(points[p_index] - truth[t_index], axis=1).mean() * 100) if len(common) else None
        ),
    }, (points[matched], points[~matched], truth[missing])


def read_sample(path, root, config):
    with np.load(path, allow_pickle=False) as saved:
        position = saved["base_pos_w"]
        quaternion = saved["base_quat_xyzw"]
        observed = cloud(saved["lidar_points_w"], position, config, quaternion)
        predicted = cloud(saved["prediction_points_w"], position, config, quaternion)
        truth = cloud(saved["gt_points_w"], position, config, quaternion)
        scores, errors = comparison(predicted, truth)
        baseline, _ = comparison(observed, truth)
        record = dict(
            frame=str(path.relative_to(root / "predictions")),
            time_s=float(saved["time_s"]),
            step=int(saved["step"]),
            prediction=scores,
            observed_only=baseline,
            voxels=dict(observed=len(observed[0]), predicted=len(predicted[0]), truth=len(truth[0])),
        )
    return dict(record=record, observed=observed[0], predicted=predicted[0], truth=truth[0], errors=errors)


def thin(points, limit):
    # Deterministic uniform index sample, rendering only; all points contribute to metrics.
    return points if len(points) <= limit else points[np.linspace(0, len(points) - 1, limit, dtype=int)]


def label(sample):
    record = sample["record"]
    return f"{record['frame']} | t={record['time_s']:.1f}s | frame F1={record['prediction']['f1']:.3f}"


def interactive(samples, config, result, destination, limit):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    half = config["map_size"] / 2
    titles = ("LiDAR observation", "Predicted map", "Ground truth", "Voxel errors")
    figure = make_subplots(rows=1, cols=4, specs=[[{"type": "scene"}] * 4], subplot_titles=titles)

    def traces(sample):
        result = []
        for name in ("observed", "predicted", "truth"):
            xyz = thin(sample[name], limit)
            result.append(
                go.Scatter3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    mode="markers",
                    showlegend=False,
                    marker=dict(size=2, color=xyz[:, 2], colorscale="Viridis", cmin=-half, cmax=half),
                    name=name,
                    hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<br>z=%{z:.2f} m<extra>%{fullData.name}</extra>",
                )
            )
        for name, color, xyz in zip(
            ("Matched voxel", "Extra (FP)", "Missing (FN)"),
            ("#16a085", "#e74c3c", "#f0ad24"),
            sample["errors"],
            strict=True,
        ):
            xyz = thin(xyz, limit)
            result.append(
                go.Scatter3d(
                    x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name=name, marker=dict(size=2, color=color)
                )
            )
        for i, trace in enumerate(result):
            trace.scene = "scene" if i == 0 else f"scene{min(i + 1, 4)}"
        return result

    for i, trace in enumerate(traces(samples[0])):
        figure.add_trace(trace, row=1, col=min(i + 1, 4))
    figure.frames = [
        go.Frame(name=str(i), data=traces(sample), layout=dict(title_text=label(sample)))
        for i, sample in enumerate(samples)
    ]
    for i in range(1, 5):
        axis = dict(range=[-half, half], backgroundcolor="#f6f8fa", gridcolor="#d9dee5")
        figure.update_layout(
            **{
                f"scene{i if i > 1 else ''}": dict(
                    xaxis=dict(axis, title="X (m)"),
                    yaxis=dict(axis, title="Y (m)"),
                    zaxis=dict(axis, title="Z (m)"),
                    aspectmode="cube",
                    camera=dict(eye=dict(x=1.3, y=-1.7, z=1.1)),
                    uirevision="camera",
                )
            }
        )
    figure.update_layout(
        title_text=label(samples[0]),
        height=650,
        margin=dict(l=10, r=10, t=95, b=70),
        legend=dict(orientation="h", y=-0.05),
        template="plotly_white",
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Exported sample: "),
                steps=[
                    dict(
                        label=str(i + 1),
                        method="animate",
                        args=[
                            [str(i)],
                            dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0)),
                        ],
                    )
                    for i in range(len(samples))
                ],
            )
        ],
    )
    p, b = result["prediction"], result["observed_only"]
    summary = (
        f"{result['frames']} evaluated frames / {result['available_frames']} available; "
        f"checkpoint epoch {result.get('checkpoint_epoch', '?')}. "
        f"Prediction: precision {p['precision']:.3f}, recall {p['recall']:.3f}, F1 {p['f1']:.3f}. "
        f"LiDAR-only F1 {b['f1']:.3f}."
    )
    page = "<!doctype html><html><head><meta charset='utf-8'><title>Terrain reconstruction validation</title>"
    page += "<style>body{font:15px system-ui;margin:24px;color:#243047}p{line-height:1.6}h1{font-size:24px}</style>"
    page += "</head><body><h1>地形重建验证 · Terrain reconstruction</h1>"
    page += f"<p>{html.escape(result['architecture'])}<br>{html.escape(summary)}</p>"
    coordinate_note = (
        "当前机身坐标系；仅显示有真值覆盖的完整体素" if config.get("map_frame") == "body" else "机身中心、世界轴方向"
    )
    page += f"<p>滑块切换抽样帧，拖动旋转、滚轮缩放。坐标为{coordinate_note}；所有视图范围一致。"
    page += "前三图按相对机身高度着色；误差图：绿色匹配、红色多预测、黄色漏预测。"
    page += "指标基于全部有效体素；显示点数限制只影响渲染。"
    if result.get("evaluation_scope") == "random_contiguous_window":
        page += "本次随机选取连续片段，仅在片段起点重新初始化；段内每帧连续推理并保留历史，不使用起点之前的记忆。</p>"
    else:
        page += "抽样帧不是重新冷启动，保留此前全部帧的预测反馈。</p>"
    if config.get("observation_memory", False):
        raw, memory = result.get("network_prediction", {}), result.get("retained_observations", {})
        page += "<p><strong>本页 Prediction 为网络输出与历史雷达观测融合后的地图。</strong>"
        page += (
            f"本次递归中的网络原始输出 F1={raw.get('f1', float('nan')):.3f}；"
            f"仅观测记忆 F1={memory.get('f1', float('nan')):.3f}。"
        )
        page += "原始输出和观测记忆另存于 NPZ；记忆带来的提升不能等同于网络学会了未知区域补全。</p>"
    else:
        page += (
            "<p><strong>本页 Prediction 为网络自身的预测，未使用观测缓存补点。</strong>历史地图反馈也使用网络预测。</p>"
        )
    if config.get("canvas", False):
        mode = "全体积" if config.get("canvas_mode", "plane") == "volume" else "平面"
        page += f"<p>每帧使用{mode}画布提供候选；画布不是传感器证据，也不直接写入观测记忆。</p>"
    page += figure.to_html(full_html=False, include_plotlyjs=True)
    page += "<p>占据误差按相同体素位置判定；质心误差仅统计匹配体素，不代表整体地图距离误差。"
    page += "真值为已有网格表面采样，可能含内部面。不同 epoch/架构结果不能当作受控消融实验。</p></body></html>"
    (destination / "report.html").write_text(page)


def static_overview(samples, config, result, destination, limit, others=None, other_result=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = np.unique(np.linspace(0, len(samples) - 1, min(3, len(samples)), dtype=int))
    figure = plt.figure(figsize=(18, 5 * len(selected)))
    # 3D axis labels project outside the usual axes bounds; leave explicit row gaps.
    figure.subplots_adjust(left=0.035, right=0.96, bottom=0.045, top=0.91, wspace=0.16, hspace=0.25)
    half = config["map_size"] / 2
    for row, index in enumerate(selected):
        sample = samples[index]
        if others is None:
            panels = [
                ("LiDAR", sample["observed"]),
                ("Prediction", sample["predicted"]),
                ("Ground truth", sample["truth"]),
                ("Errors: green=TP / red=FP / yellow=FN", None),
            ]
        else:
            panels = [
                ("LiDAR", sample["observed"]),
                ("Reference prediction", others[index]["predicted"]),
                ("Current prediction", sample["predicted"]),
                ("Ground truth", sample["truth"]),
            ]
        for col, (name, points) in enumerate(panels):
            axis = figure.add_subplot(len(selected), 4, row * 4 + col + 1, projection="3d")
            if points is None:
                for xyz, color in zip(sample["errors"], ("#16a085", "#e74c3c", "#f0ad24"), strict=True):
                    xyz = thin(xyz, limit)
                    axis.scatter(*xyz.T, s=1.3, c=color, alpha=0.7, rasterized=True, depthshade=False)
            else:
                xyz = thin(points, limit)
                axis.scatter(
                    *xyz.T, s=1.3, c=xyz[:, 2], cmap="viridis", vmin=-half, vmax=half, rasterized=True, depthshade=False
                )
            axis.set(
                xlim=(-half, half),
                ylim=(-half, half),
                zlim=(-half, half),
                xlabel="X (m)",
                ylabel="Y (m)",
                zlabel="Z (m)",
            )
            axis.set_box_aspect((1, 1, 1))
            axis.view_init(elev=28, azim=-55)
            axis.tick_params(labelsize=7)
            score = sample["record"]["prediction"]["f1"]
            if col == 0:
                score = sample["record"]["observed_only"]["f1"]
            if others is not None and col == 1:
                score = others[index]["record"]["prediction"]["f1"]
            detail = "" if name == "Ground truth" else f" | frame F1={score:.3f}"
            axis.set_title(f"{name}\nt={sample['record']['time_s']:.1f}s{detail}", fontsize=10)
    title = f"Validation: {result['frames']} frames | prediction F1={result['prediction']['f1']:.3f}"
    title += f" | LiDAR-only F1={result['observed_only']['f1']:.3f}"
    if other_result:
        title += f" | reference F1={other_result['prediction']['f1']:.3f}"
    frame_label = (
        "Body-frame voxels within recorded GT coverage"
        if config.get("map_frame") == "body"
        else "World-aligned local surface voxels"
    )
    figure.suptitle(title + f"\n{frame_label}; first / middle / last exported samples", fontsize=13)
    figure.savefig(destination / ("comparison.png" if others else "overview.png"), dpi=140)
    plt.close(figure)


def learning_curves(result, destination):
    import matplotlib.pyplot as plt

    path = Path(result["checkpoint"]).parent / "metrics.jsonl"
    if not path.is_file():
        return
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    epochs = [r["epoch"] for r in rows]
    axes[0].plot(epochs, [r["train_loss"] for r in rows], label="Train loss")
    axes[0].plot(epochs, [r["validation"]["loss"] for r in rows], label="Validation loss")
    axes[1].plot(epochs, [r["validation"]["prediction"]["f1"] for r in rows], label="Validation F1")
    axes[1].axhline(result["observed_only"]["f1"], color="gray", linestyle="--", label="LiDAR-only F1")
    for axis in axes:
        if result.get("checkpoint_epoch") is not None:
            axis.axvline(result["checkpoint_epoch"], color="#e67e22", linestyle=":", label="Evaluated checkpoint")
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(destination / "learning_curves.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True, help="evaluate.py output directory")
    parser.add_argument("--output", type=Path, help="New directory; default EVALUATION/visualization")
    parser.add_argument("--compare", type=Path, help="Optional reference evaluation of the same exported frames")
    parser.add_argument(
        "--max_points", type=int, default=6000, help="Display limit per cloud/category; metrics use all"
    )
    args = parser.parse_args()
    if args.max_points < 1:
        parser.error("max_points must be positive")
    root = args.evaluation
    config = json.loads((root / "config.json").read_text())
    result = json.loads((root / "metrics.json").read_text())
    paths = sorted((root / "predictions").glob("*/*/*/*.npz"))
    if not paths:
        parser.error("No exported predictions; evaluate with --export_frames > 0 first")
    output = args.output or root / "visualization"
    if output.exists():
        parser.error("Output exists; select a new --output directory")
    torch.set_num_threads(2)
    samples = [read_sample(path, root, config) for path in paths]
    others, other_result = None, None
    if args.compare:
        other_config = json.loads((args.compare / "config.json").read_text())
        if other_config.get("map_frame", "world") != config.get("map_frame", "world"):
            parser.error("Comparison requires the same coordinate frame and supervised coverage")
        if any(other_config[key] != config[key] for key in ("resolution", "map_size")):
            parser.error("Comparison requires identical voxel/map sizes")
        other_result = json.loads((args.compare / "metrics.json").read_text())
        others = []
        for path in paths:
            other = args.compare / path.relative_to(root)
            with np.load(path) as a, np.load(other) as b:
                for key in ("base_pos_w", "base_quat_xyzw", "lidar_points_w", "gt_points_w", "time_s", "step"):
                    if not np.array_equal(a[key], b[key]):
                        parser.error(f"Comparison input differs: {path.name} / {key}")
            others.append(read_sample(other, args.compare, other_config))
    output.mkdir(parents=True)
    report = dict(
        evaluation=str(root.resolve()),
        metrics=result,
        display_limit=args.max_points,
        samples=[sample["record"] for sample in samples],
    )
    if others:
        report.update(comparison_metrics=other_result, comparison_samples=[s["record"] for s in others])
    (output / "samples.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    interactive(samples, config, result, output, args.max_points)
    static_overview(samples, config, result, output, args.max_points)
    if others:
        static_overview(samples, config, result, output, args.max_points, others, other_result)
    learning_curves(result, output)
    print(f"Saved {len(samples)} samples: {output / 'report.html'}")


if __name__ == "__main__":
    main()
