# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm assets, joint-position actions and per-environment cube size variants."""

from isaaclab.assets import RigidObjectCfg
from isaaclab.cloner import InclusionSet
from isaaclab.cloner.cloner_strategies import random as random_clone_strategy
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import MultiAssetSpawnerCfg, UsdFileCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG

from . import mdp
from .lift_env_cfg import LiftEnvCfg

# Use the same joint selection for actor/critic observations and velocity penalties.
OPENARM_JOINTS = ["openarm_joint.*", "openarm_finger_joint.*"]

# Uniform size choices relative to the original 0.8 USD scale. Each environment
# retains its sampled collision/visual geometry throughout training.
OBJECT_BASE_SCALE = 0.8
OBJECT_SIZE_MULTIPLIERS = (0.8, 0.9, 1.0, 1.1, 1.2)


@configclass
class OpenArmCubeLiftEnvCfg(LiftEnvCfg):
    """Grasp a cube, transport it to its target pose, and regrasp after recoverable drops."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = replace(OPENARM_UNI_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.robot.spawn.func = mdp.spawn_openarm
        for properties in self.scene.robot.spawn.articulation_props:
            if isinstance(properties, PhysxArticulationCfg):
                properties.enabled_self_collisions = True

        for group in (self.observations.policy, self.observations.critic):
            for term_name in ("joint_pos", "joint_vel", "joint_pos_error"):
                getattr(group, term_name).params = {"asset_cfg": SceneEntityCfg("robot", joint_names=OPENARM_JOINTS)}
        self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=OPENARM_JOINTS)
        self.rewards.joint_acc.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=OPENARM_JOINTS)
        self.rewards.joint_limits.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=OPENARM_JOINTS)

        # Arm targets: default joint positions + 0.5 * policy actions (radians).
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_joint[1-7]"],
            scale=0.5,
            use_default_offset=True,
        )

        # One scalar controls both fingers: negative closes, non-negative opens.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_finger_joint.*"],
            open_command_expr={"openarm_finger_joint.*": 0.044},
            close_command_expr={"openarm_finger_joint.*": 0.0},
        )

        # Spawn separate size prototypes so PhysX parses distinct collision
        # geometry, instead of changing USD scale after physics replication.
        cube_spawn = UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            activate_contact_sensors=True,
            rigid_props=PhysxRigidBodyCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        )
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0, 0.055], rot=[0, 0, 0, 1]),
            spawn=MultiAssetSpawnerCfg(
                activate_contact_sensors=True,
                assets_cfg=[
                    replace(cube_spawn, scale=(OBJECT_BASE_SCALE * factor,) * 3) for factor in OBJECT_SIZE_MULTIPLIERS
                ],
                random_choice=False,  # Selection is handled by clone_cfg below.
            ),
        )
        self.scene.clone_cfg.clone_combinations = [InclusionSet(assets=["object"])]
        self.scene.clone_cfg.clone_strategy = random_clone_strategy
        # In the current PhysX/cloner version, inverted per-environment collision
        # groups suppress cube/table contacts in this multi-prototype scene.
        # Keep normal spatial broadphase collision detection; the 2.5 m spacing
        # separates workspaces without those groups.
        self.scene.filter_collisions = False

        # Separate single-body sensors support per-environment object filtering.
        for side in ("left", "right"):
            setattr(
                self.scene,
                f"{side}_finger_contact",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/openarm_{side}_finger",
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    update_period=0.0,
                    history_length=self.decimation,
                ),
            )
        # Include EVERY robot body: rewards subtract these normal forces from
        # the total object contact force to identify external tabletop support.
        self.scene.object_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            filter_prim_paths_expr=[
                f"{{ENV_REGEX_NS}}/Robot/{body}"
                for body in [
                    *(f"openarm_link{i}" for i in range(8)),
                    "openarm_hand",
                    "openarm_left_finger",
                    "openarm_right_finger",
                ]
            ],
            update_period=0.0,
            track_contact_points=True,
            max_contact_data_count_per_prim=16,
        )
