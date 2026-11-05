"""Rank-zero Rich reporting; JSONL remains the machine-readable source of metrics."""

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table


class TrainingDisplay:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.console = Console()
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed:.0f}/{task.total:.0f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[detail]}"),
            console=self.console,
            disable=not enabled,
            refresh_per_second=3,
        )
        self.task = None

    def __enter__(self):
        if self.enabled:
            self.progress.start()
        return self

    def __exit__(self, *args):
        if self.enabled:
            self.progress.stop()

    def setup(self, config, manifest):
        if not self.enabled:
            return
        table = Table(title="地形重建训练")
        table.add_column("配置")
        table.add_column("值")
        for key in (
            "model",
            "architecture",
            "relative_pose",
            "canvas",
            "canvas_mode",
            "occupancy_loss",
            "observation_memory",
            "memory_size",
            "world_size",
            "global_batch_size",
            "seq_len",
            "bptt_steps",
            "window_sampling",
            "history_init",
            "plane_z",
            "augmentation",
            "lr_scheduler",
            "lr_metric",
        ):
            table.add_row(key, str(config.get(key, "—")))
        table.add_row(
            "最终输出", "网络 + 观测缓存融合（对照）" if config.get("observation_memory", False) else "网络预测"
        )
        for name in ("train", "validation"):
            runs = manifest[name]
            count = sum(len(ep["frames"]) for run in runs for ep in run["episodes"])
            table.add_row(name, f"{count} 帧 · " + ", ".join(run["name"] for run in runs))
        self.console.print(table)

    def phase(self, name, total):
        if not self.enabled:
            return
        if self.task is not None:
            self.progress.remove_task(self.task)
        self.task = self.progress.add_task(name, total=total, detail="")

    def update(self, completed, detail=""):
        if self.enabled:
            self.progress.update(self.task, completed=completed, detail=detail)

    def epoch(self, record):
        if not self.enabled:
            return
        table = Table(title=f"Epoch {record['epoch']}")
        table.add_column("指标")
        table.add_column("值", justify="right")
        for label, value in (
            ("训练 loss", record["train_loss"]),
            ("验证 loss", record["validation"]["loss"]),
            ("验证 F1", record["validation"]["prediction"]["f1"]),
            ("训练吞吐 frames/s", record["train_frames_per_second"]),
            ("本轮 LR", record["learning_rate"]),
            ("下一轮 LR", record["next_lr"]),
        ):
            table.add_row(label, f"{value:.6g}")
        for key, label in (("network_prediction", "网络原始 F1"), ("retained_observations", "仅观测记忆 F1")):
            if key in record["validation"]:
                table.add_row(label, f"{record['validation'][key]['f1']:.6g}")
        if record["lr_reduced"]:
            table.add_row("学习率调整", "[yellow]已降低[/yellow]")
        for rank, peak in enumerate(record["peak_allocated_gib_per_rank"]):
            table.add_row(f"GPU/rank {rank} 峰值张量显存", f"{peak:.2f} GiB")
        table.add_row(
            "完整训练/验证", f"{not record['partial_training']} / {not record['validation']['partial_evaluation']}"
        )
        self.console.print(table)
