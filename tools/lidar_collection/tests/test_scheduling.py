"""Metric routing, plateau state restoration, floors and legacy scheduling."""

import copy

import pytest
import torch

from ..training.runtime.scheduling import SCHEDULE_DEFAULTS, make_scheduler, step_scheduler


def optimizer():
    return torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=0.01)


def test_validation_loss_plateau_restores_patience_and_respects_floor():
    config = dict(SCHEDULE_DEFAULTS, lr_scheduler="plateau", lr_patience=1, min_lr=0.0025)
    first = optimizer()
    scheduler = make_scheduler(first, config)
    for loss in (1.0, 1.1):
        step_scheduler(scheduler, config, {"loss": loss})
    second = optimizer()
    restored = make_scheduler(second, config)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    restored.load_state_dict(copy.deepcopy(scheduler.state_dict()))
    for loss, expected in [(1.2, 0.005), (0.8, 0.005), (0.9, 0.005), (1.0, 0.0025), (2.0, 0.0025), (3.0, 0.0025)]:
        for active in (scheduler, restored):
            step_scheduler(active, config, {"loss": loss})
            assert active.get_last_lr()[0] == pytest.approx(expected)
        assert scheduler.state_dict() == restored.state_dict()


def test_f1_uses_max_direction_and_rejects_nonfinite():
    config = dict(SCHEDULE_DEFAULTS, lr_scheduler="plateau", lr_metric="val_f1", lr_patience=0)
    active = make_scheduler(optimizer(), config)
    for score, expected in [(0.1, 0.01), (0.2, 0.01), (0.19, 0.005)]:
        step_scheduler(active, config, {"prediction": {"f1": score}})
        assert active.get_last_lr()[0] == pytest.approx(expected)
    before = copy.deepcopy(active.state_dict())
    with pytest.raises(ValueError, match="Non-finite"):
        step_scheduler(active, config, {"prediction": {"f1": float("nan")}})
    assert before == active.state_dict()


def test_legacy_config_keeps_exponential_schedule():
    config = {"decay": 0.98}
    opt = optimizer()
    active = make_scheduler(opt, config)
    opt.step()
    step_scheduler(active, config, {})
    assert active.get_last_lr()[0] == pytest.approx(0.0098)
