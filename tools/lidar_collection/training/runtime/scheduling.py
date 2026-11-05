"""Validation-driven LR schedules, with explicit legacy checkpoint compatibility."""

import math

import torch

SCHEDULE_DEFAULTS = dict(lr_metric="val_loss", lr_factor=0.5, lr_patience=3, lr_threshold=0.001, min_lr=0.0001)


def make_scheduler(optimizer, config):
    kind = config.get("lr_scheduler", "exponential")
    if kind == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["decay"])
    if kind != "plateau":
        raise ValueError(f"Unknown LR scheduler: {kind}")
    if config["min_lr"] > min(group["lr"] for group in optimizer.param_groups):
        raise ValueError("min_lr exceeds optimizer's current LR; choose a lower --min_lr")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min" if config["lr_metric"] == "val_loss" else "max",
        factor=config["lr_factor"],
        patience=config["lr_patience"],
        threshold=config["lr_threshold"],
        threshold_mode="abs",
        min_lr=config["min_lr"],
    )


def step_scheduler(scheduler, config, validation):
    if config.get("lr_scheduler", "exponential") == "plateau":
        metric = validation["loss"] if config["lr_metric"] == "val_loss" else validation["prediction"]["f1"]
        if not math.isfinite(metric):
            raise ValueError("Non-finite validation metric; refusing to update learning rate")
        scheduler.step(metric)
    else:
        scheduler.step()
