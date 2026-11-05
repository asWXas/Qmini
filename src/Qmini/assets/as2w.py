"""Unitree AS2W wheeled quadruped using the upstream URDF.

Limits come from the URDF. PD gains and the crouched initial pose are
simulation starting values, not validated hardware parameters.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.utils import clone
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

from . import QMINI_ASSETS_DIR

AS2W_ASSET_DIR = QMINI_ASSETS_DIR / "as2w_description"
# Explicit action ordering; simulator joint ordering may differ.
LEG_PREFIXES = ("FL", "FR", "RL", "RR")
LEG_JOINT_NAMES = [f"{leg}_{joint}" for leg in LEG_PREFIXES for joint in ("hip", "thigh", "calf")]
WHEEL_JOINT_NAMES = [f"{leg}_foot" for leg in LEG_PREFIXES]
JOINT_NAMES = [f"{leg}_{joint}" for leg in LEG_PREFIXES for joint in ("hip", "thigh", "calf", "foot")]
WHEEL_BODY_NAMES = WHEEL_JOINT_NAMES.copy()
RIGID_BODY_NAMES = ["base_link", *JOINT_NAMES, "JT128"]
FIXED_JOINT_NAMES = ["JT128"]


@clone
def spawn_as2w(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Filter base/thigh contacts before cloning the robot into other environments."""
    from isaaclab.sim.spawners.from_files import spawn_from_urdf

    root = spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)
    _filter_base_thigh_collisions(root)
    return root


def _filter_base_thigh_collisions(root):
    """Filter owned colliders, not nested rigid-body subtrees."""
    from pxr import Usd, UsdPhysics

    names = ("base_link", *(f"{leg}_thigh" for leg in LEG_PREFIXES))
    bodies = {}
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in names:
            name = prim.GetName()
            if name in bodies:
                raise ValueError(f"Ambiguous AS2W rigid body {name} under {root.GetPath()}")
            bodies[name] = prim
    missing = set(names) - bodies.keys()
    if missing:
        raise ValueError(f"Missing AS2W rigid bodies {sorted(missing)} under {root.GetPath()}")

    def owned_colliders(body):
        result = []
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            owner = prim
            while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if owner == body:
                result.append(prim)
        if not result:
            raise ValueError(f"No AS2W colliders owned by {body.GetPath()}")
        return result

    # Only the base colliders need editable API opinions. Expand their instance
    # roots locally, leaving leg meshes instanced and the source USD untouched.
    stage = root.GetStage()
    base_paths = [p.GetPath() for p in owned_colliders(bodies["base_link"])]
    for path in base_paths:
        prim = stage.GetPrimAtPath(path)
        ancestors = []
        while prim and prim != root:
            ancestors.append(prim.GetPath())
            prim = prim.GetParent()
        for ancestor in reversed(ancestors):
            prim = stage.GetPrimAtPath(ancestor)
            if prim.IsInstance():
                prim.SetInstanceable(False)

    thigh_paths = [
        collider.GetPath()
        for leg in LEG_PREFIXES
        for collider in owned_colliders(bodies[f"{leg}_thigh"])
    ]
    for path in base_paths:
        pairs = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path)).CreateFilteredPairsRel() # type: ignore
        for thigh_path in thigh_paths:
            pairs.AddTarget(thigh_path)


AS2W_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        func=spawn_as2w,
        asset_path=str(AS2W_ASSET_DIR / "as2w.urdf"),
        fix_base=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        collision_from_visuals=False,
        force_usd_conversion=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=True,  # 仅排除 base_link 与四条 thigh，见 spawn_as2w。
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            NewtonArticulationCfg(self_collision_enabled=False),
        ],
        rigid_props=[
            sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
            PhysxRigidBodyCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
        ],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={".*_hip": 0.0, ".*_thigh": 0.8, ".*_calf": -1.6, ".*_foot": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hips_and_thighs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip", ".*_thigh"],
            actuator_effort_limit=60.0,
            actuator_velocity_limit=25.0,
            joint_effort_limit=60.0,
            joint_velocity_limit=25.0,
            stiffness=60.0,
            damping=2.0,
        ),
        "calves": IdealPDActuatorCfg(
            joint_names_expr=[".*_calf"],
            actuator_effort_limit=90.0,
            actuator_velocity_limit=16.67,
            joint_effort_limit=90.0,
            joint_velocity_limit=16.67,
            stiffness=60.0,
            damping=2.0,
        ),
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=[".*_foot"],
            actuator_effort_limit=15.0,
            actuator_velocity_limit=98.47,
            joint_effort_limit=15.0,
            joint_velocity_limit=98.47,
            stiffness=0.0,
            damping=0.5,
        ),
    },
)
