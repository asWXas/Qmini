import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

from . import QMINI_ASSETS_DIR

UMV_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(QMINI_ASSETS_DIR / "UMV" / "umv.urdf"),
        fix_base=False,
        activate_contact_sensors=True,
        # Isaac Lab 3.0 中仍然有效。
        # 如果不希望每次启动都重新生成 USD，可以改成 False。
        force_usd_conversion=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            )
        ),
        
        # articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        #     enabled_self_collisions=True,
        #     solver_position_iteration_count=8,
        #     solver_velocity_iteration_count=4,
        # ),
        # rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #     disable_gravity=False,
        #     retain_accelerations=False,
        #     linear_damping=0.0,
        #     angular_damping=0.0,
        #     max_linear_velocity=1000.0,
        #     max_angular_velocity=1000.0,
        #     max_depenetration_velocity=1.0,
        # ),
        
        articulation_props=PhysxArticulationCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "q_w": 0.0,
            "q_frontwheel": 0.0,
            "q_s": 0.0,
            "q_n": 0.0,
            "q_h": 0.0,
        },
        joint_vel={
            ".*": 0.0,
        },
    ),
    actuators={
        # --------------------------------------------------
        # 1. 后驱动力轮
        # --------------------------------------------------
        "drive_wheel": ImplicitActuatorCfg(
            joint_names_expr=["q_w"],
            # Isaac Lab 3.0 推荐字段
            joint_effort_limit=50.0,
            joint_velocity_limit=100.0,
            # 速度型控制：
            # position stiffness = 0
            stiffness=0.0,
            damping=1.0,
        ),
        # --------------------------------------------------
        # 2. 前轮自由滚转
        # --------------------------------------------------
        "passive_wheel": ImplicitActuatorCfg(
            joint_names_expr=["q_frontwheel"],
            joint_effort_limit=0.0,
            joint_velocity_limit=100.0,
            stiffness=0.0,
            damping=0.01,
        ),
        # --------------------------------------------------
        # 3. 转向关节
        # --------------------------------------------------
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["q_s"],
            joint_effort_limit=6.25,
            joint_velocity_limit=160.0,
            stiffness=60.0,
            damping=3.0,
        ),
        # --------------------------------------------------
        # 4. 颈部 / 头部
        # --------------------------------------------------
        "upper_body": ImplicitActuatorCfg(
            joint_names_expr=[
                "q_n",
                "q_h",
            ],
            joint_effort_limit=200.0,
            joint_velocity_limit=30.0,
            stiffness=150.0,
            damping=10.0,
        ),
    },
)
