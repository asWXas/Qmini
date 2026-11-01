"""One velocity command per curriculum task/episode; no mid-episode reclassification."""

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class LPVelocityCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids):
        controller = self._env.lp_curriculum
        self.vel_command_b[env_ids] = controller.commands[env_ids]
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = False

    def compute(self, dt):
        # Resample exclusively through reset(); the task ID stays valid for the
        # entire measured episode, even if its maximum length changes in config.
        self._update_metrics()


@configclass
class LPVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type = LPVelocityCommand
    heading_command = False
    rel_heading_envs = 0.0
    rel_standing_envs = 0.0
    resampling_time_range = (1.e9, 1.e9)
