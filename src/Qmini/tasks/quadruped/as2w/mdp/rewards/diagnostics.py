"""Zero-return diagnostics for the configured gait/terrain mode selection."""

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from ..terrain_geometry import GridScanEdgeDetector, terrain_step_gate
from .gait import DiagonalGaitCfg, _gait_tracking_penalties, _pure_x_command_mask, diagonal_gait_targets
from .stability import base_height_l2


def _find_term(rewards, name):
    """Support both flat configurations and multi-critic reward dictionaries."""
    values = rewards if isinstance(rewards, dict) else vars(rewards)
    if name in values:
        return values[name]
    for group in values.values():
        if isinstance(group, dict) and name in group:
            return group[name]
    return None


class GaitScanTest(ManagerTermBase):
    """Log gait demand, individual wheel behavior and body height; return zero.

    Detailed pure-X metrics use blend >= 0.5, where gait is the dominant target.
    Conditional metrics are sample vectors: RSL-RL concatenates them over the
    rollout before averaging. Empty selections are not fake zero successes;
    an entire rollout without samples displays NaN alongside samples=0.
    These diagnostics describe behavior, not the cause of a learning failure.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        sensor = env.scene.sensors[cfg.params["height_sensor_cfg"].name]
        self._edges = GridScanEdgeDetector(sensor.cfg.pattern_cfg, env.device)
        self._last_print_step = None
        self._env = env
        self._latest_log = {}
        self._base_cfg = _find_term(env.cfg.rewards, "base_height_l2")
        self._base_term = base_height_l2(self._base_cfg, env) if self._base_cfg is not None else None

    def reset(self, env_ids=None):
        # The environment clears extras['log'] after computing terminal rewards.
        # Restore that step's snapshot without recomputing from reset robot data.
        self._env.extras["log"] = {**self._env.extras.get("log", {}), **self._latest_log}

    def __call__(
        self,
        env,
        height_sensor_cfg: SceneEntityCfg,
        print_interval: int = 100,
        leg_joint_cfg: SceneEntityCfg | None = None,
    ):
        # Zero disables terminal output while retaining the diagnostic logs.
        if isinstance(print_interval, bool) or not isinstance(print_interval, int) or print_interval < 0:
            raise ValueError("print_interval must be a nonnegative integer")
        hits = env.scene.sensors[height_sensor_cfg.name].data.ray_hits_w.torch
        edge, valid = self._edges(hits)
        # Managers resolve body/joint IDs only in their copied configurations.
        # The manager is available here, but not while this term is constructed.
        maneuver = self._managed_term(env, "explicit_diagonal_gait")
        longitudinal = self._managed_term(env, "pure_x_terrain_gait")
        gait_gate = torch.zeros_like(edge)
        scan_gate = torch.zeros_like(edge)
        maneuver_gate = torch.zeros_like(edge)
        longitudinal_gate = torch.zeros_like(edge)
        pure_x = torch.zeros_like(valid)
        if maneuver is not None and maneuver.weight != 0.0:
            params = maneuver.params
            command = env.command_manager.get_command(params.get("command_name", "base_velocity"))
            active = (command[:, 1:3].abs() > params.get("command_threshold", 0.1)).any(-1)
            maneuver_gate = (active & params.get("enabled", True)).to(edge.dtype)
            gait_gate = maneuver_gate
        if longitudinal is not None:
            params = longitudinal.params
            if params["height_sensor_cfg"].name != height_sensor_cfg.name:
                raise ValueError("GaitScanTest must use the same height scanner as pure_x_terrain_gait")
            scan_gate = (
                terrain_step_gate(edge, params.get("step_threshold", 0.03), params.get("step_full_height", 0.05))
                * valid
            )
            command = env.command_manager.get_command(params.get("command_name", "base_velocity"))
            threshold = params.get("command_threshold", 0.1)
            pure_x = _pure_x_command_mask(command, threshold, params.get("pure_x_tolerance"))
            enabled = params.get("enabled", True) and longitudinal.weight != 0.0
            longitudinal_gate = scan_gate * pure_x * enabled
            gait_gate = torch.maximum(gait_gate, longitudinal_gate)
        log = {
            "Test/gait_active_fraction": (gait_gate > 0.0).float().mean(),
            "Test/gait_blend_mean": gait_gate.mean(),
            "Test/scan_edge_fraction": (scan_gate > 0.0).float().mean(),
            "Test/scan_valid_fraction": valid.float().mean(),
            "Test/scan_edge_max_m": edge.max(),
        }
        self._log_yaw_tracking(env, log, maneuver or longitudinal)
        self._log_mode(env, log, "maneuver", maneuver, maneuver_gate, 0.8, leg_joint_cfg)
        self._log_mode(env, log, "pure_x_gait", longitudinal, longitudinal_gate, 0.4, leg_joint_cfg)
        if self._base_term is not None:
            params = self._base_cfg.params
            base_hits = env.scene.sensors[params["sensor_cfg"].name].data.ray_hits_w.torch
            heights = base_hits[:, self._base_term._scan_ids, 2]
            finite = torch.isfinite(heights)
            ground = heights.masked_fill(~finite, 0.0).sum(-1) / finite.sum(-1).clamp(min=1)
            asset_name = params["asset_cfg"].name if "asset_cfg" in params else "robot"
            height = env.scene[asset_name].data.root_pos_w.torch[:, 2] - ground
            valid_height = finite.any(-1) & torch.isfinite(height)
            log["Test/base_scan_valid_fraction"] = valid_height.float().mean()
            log["Test/base_height_m"] = height[valid_height]
            log["Test/rolling/base_height_m"] = height[valid_height & pure_x & valid & (scan_gate == 0)]
            log["Test/pure_x_gait/base_height_m"] = height[valid_height & (longitudinal_gate >= 0.5)]
            log["Test/maneuver/base_height_m"] = height[valid_height & (maneuver_gate >= 0.5)]
            # Match the reward's zero-ground fallback to expose invalid-scan spikes.
            error = (height - params["target_height"]).abs()
            log["Test/base_height_error_max_m"] = error.max()
        # Logger retains dictionary references; never mutate a previous step.
        # Drop our previous keys if a reward is disabled during training.
        previous = {key: value for key, value in env.extras.get("log", {}).items() if key not in self._latest_log}
        self._latest_log = log
        env.extras["log"] = {**previous, **log}
        step = env.common_step_counter
        if print_interval > 0 and (self._last_print_step is None or step - self._last_print_step >= print_interval):
            print(
                f"[GaitScanTest step={step}] "
                f"gait_active={log['Test/gait_active_fraction'].item():.1%} "
                f"gait_blend={log['Test/gait_blend_mean'].item():.3f} "
                f"scan_edge={log['Test/scan_edge_fraction'].item():.1%} "
                f"scan_valid={log['Test/scan_valid_fraction'].item():.1%} "
                f"scan_edge_max={log['Test/scan_edge_max_m'].item():.4f}m",
                flush=True,
            )
            self._last_print_step = step
        return torch.zeros_like(edge)

    @staticmethod
    def _managed_term(env, name):
        matches = [key for key in env.reward_manager.active_terms if key.rsplit("/", 1)[-1] == name]
        if len(matches) > 1:
            raise ValueError(f"Expected a single diagnostic reward term named {name}")
        return env.reward_manager.get_term_cfg(matches[0]) if matches else None

    @staticmethod
    def _log_yaw_tracking(env, log, term):
        params = term.params if term is not None else {}
        command = env.command_manager.get_command(params.get("command_name", "base_velocity"))
        asset_cfg = params.get("asset_cfg")
        asset_name = asset_cfg.name if asset_cfg is not None else "robot"
        actual = env.scene[asset_name].data.root_ang_vel_b.torch[:, 2]
        target = command[:, 2]
        threshold = params.get("command_threshold", 0.1)
        turning = target.abs() > threshold
        translating = (command[:, :2].abs() > threshold).any(-1)
        for mode, selected in (
            ("yaw_only", turning & ~translating),
            ("mixed_yaw", turning & translating),
            ("yaw_deadband", ~turning),
        ):
            prefix = f"Test/{mode}"
            log[f"{prefix}/samples"] = selected.float().sum()
            log[f"{prefix}/command_abs_rad_s"] = target[selected].abs()
            log[f"{prefix}/actual_abs_rad_s"] = actual[selected].abs()
            log[f"{prefix}/error_abs_rad_s"] = (target - actual).abs()[selected]
            if mode != "yaw_deadband":
                # Align left/right turns before averaging; a negative value is
                # rotation against the command, not cancellation of turn signs.
                log[f"{prefix}/actual_along_command_rad_s"] = (actual * target.sign())[selected]

    @staticmethod
    def _log_mode(env, log, mode, term, gate, default_period, leg_joint_cfg):
        if term is None or term.weight == 0.0 or not term.params.get("enabled", True):
            return
        params = term.params
        active = gate >= 0.5
        prefix = f"Test/{mode}"
        log[f"{prefix}/samples"] = active.float().sum()
        log[f"{prefix}/blend"] = gate[active]
        gait_cfg = DiagonalGaitCfg(
            **{
                key: params.get(key, default)
                for key, default in (
                    ("period", default_period),
                    ("frequency", None),
                    ("duty_factor", 0.5),
                    ("phase_offsets", (0.0, 0.5, 0.5, 0.0)),
                    ("command_threshold", 0.1),
                    ("command_name", "base_velocity"),
                )
            }
        )
        phase, _, stance = diagonal_gait_targets(env, gait_cfg)
        data = {}
        _gait_tracking_penalties(
            env,
            phase,
            stance,
            params["sensor_cfg"],
            params.get("contact_threshold", 1.0),
            params.get("asset_cfg"),
            params.get("wheel_sensor_names"),
            params.get("swing_height", 0.10 if mode == "maneuver" else 0.03),
            params.get("wheel_radius", 0.086),
            params.get("height_tracking_scale", 1.0),
            gait_cfg.duty_factor,
            gait_cfg.phase_offsets,
            params.get("height_std", 0.05),
            diagnostics=data,
        )
        matched = torch.where(stance, data["supported"], ~data["touching"])
        log[f"{prefix}/contact_pattern_match"] = matched.all(-1)[active].float()
        tracking_error = None
        if leg_joint_cfg is not None:
            asset = env.scene[leg_joint_cfg.name]
            ids = leg_joint_cfg.joint_ids
            target = asset.actuators.target_command.position.torch[:, ids]
            actual = asset.data.joint_pos.torch[:, ids]
            if target.shape[1] != 12:
                raise ValueError("Gait diagnostics require twelve joints ordered FL/FR/RL/RR, hip/thigh/calf")
            tracking_error = (target - actual).abs().reshape(-1, 4, 3).mean(-1)
        for i, leg in enumerate(("FL", "FR", "RL", "RR")):
            swing = active & ~stance[:, i]
            support = active & stance[:, i]
            log[f"{prefix}/{leg}/swing_samples"] = swing.float().sum()
            log[f"{prefix}/{leg}/swing_air_fraction"] = (~data["touching"][:, i])[swing].float()
            log[f"{prefix}/{leg}/stance_support_fraction"] = data["supported"][:, i][support].float()
            if "bottom_z" in data:
                valid = swing & data["valid_height"][:, i] & torch.isfinite(data["bottom_z"][:, i])
                actual_clearance = data["bottom_z"][:, i] - data["swing_ground"][:, i]
                target_clearance = data["target_z"][:, i] - data["swing_ground"][:, i]
                log[f"{prefix}/{leg}/swing_clearance_m"] = actual_clearance[valid]
                log[f"{prefix}/{leg}/swing_target_clearance_m"] = target_clearance[valid]
            if tracking_error is not None:
                log[f"{prefix}/{leg}/joint_target_error_rad"] = tracking_error[:, i][swing]
