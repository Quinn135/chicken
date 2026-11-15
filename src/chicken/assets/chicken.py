"""Configuration for a my chicken!"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab_physx.sim.schemas import (
    PhysxArticulationRootPropertiesCfg,
    PhysxCollisionPropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
)

from . import CHICKEN_ASSETS_DIR

##
# Configuration
##

# Generated from robots/chicken_v6_5/urdf/chicken.urdf by scripts/convert_robot.py.
# (In 2.x this conversion re-ran on every import; now it is a one-off step.)
CHICKEN_USD_PATH = CHICKEN_ASSETS_DIR / "robots" / "usd" / "chicken" / "chicken.usda"

# The 3.x URDF importer nests each link under its parent link (2.x put every link directly under the
# robot prim), so sensors need each link's full path. Paths are relative to the robot prim.
# Regenerate this table if the URDF's link tree changes.
CHICKEN_BODY_PATHS = {
    "base": "Geometry/root/base",
    "rm_side_hip": "Geometry/root/base/rm_side_hip",
    "r_hip": "Geometry/root/base/rm_side_hip/r_hip",
    "rm_hip": "Geometry/root/base/rm_side_hip/r_hip/rm_hip",
    "r_upper_leg": "Geometry/root/base/rm_side_hip/r_hip/rm_hip/r_upper_leg",
    "rm_knee": "Geometry/root/base/rm_side_hip/r_hip/rm_hip/r_upper_leg/rm_knee",
    "r_lower_leg": "Geometry/root/base/rm_side_hip/r_hip/rm_hip/r_upper_leg/rm_knee/r_lower_leg",
    "rm_ankle": "Geometry/root/base/rm_side_hip/r_hip/rm_hip/r_upper_leg/rm_knee/r_lower_leg/rm_ankle",
    "r_foot": "Geometry/root/base/rm_side_hip/r_hip/rm_hip/r_upper_leg/rm_knee/r_lower_leg/rm_ankle/r_foot",
    "r_foot_pad": "Geometry/root/base/rm_side_hip/r_hip/rm_hip/r_upper_leg/rm_knee/r_lower_leg/rm_ankle/r_foot/r_foot_pad",  # noqa: E501
    "escmountcenter_7": "Geometry/root/base/rm_side_hip/r_hip/rm_hip/r_upper_leg/rm_knee/r_lower_leg/rm_ankle/escmountcenter_7",  # noqa: E501
    "escmountcenter_5": "Geometry/root/base/rm_side_hip/r_hip/rm_hip/r_upper_leg/rm_knee/escmountcenter_5",
    "escmountcenter_6": "Geometry/root/base/rm_side_hip/r_hip/rm_hip/escmountcenter_6",
    "escmountcenter_4": "Geometry/root/base/rm_side_hip/escmountcenter_4",
    "lm_side_hip": "Geometry/root/base/lm_side_hip",
    "l_hip": "Geometry/root/base/lm_side_hip/l_hip",
    "lm_hip": "Geometry/root/base/lm_side_hip/l_hip/lm_hip",
    "l_upper_leg": "Geometry/root/base/lm_side_hip/l_hip/lm_hip/l_upper_leg",
    "lm_knee": "Geometry/root/base/lm_side_hip/l_hip/lm_hip/l_upper_leg/lm_knee",
    "l_lower_leg": "Geometry/root/base/lm_side_hip/l_hip/lm_hip/l_upper_leg/lm_knee/l_lower_leg",
    "lm_ankle": "Geometry/root/base/lm_side_hip/l_hip/lm_hip/l_upper_leg/lm_knee/l_lower_leg/lm_ankle",
    "l_foot": "Geometry/root/base/lm_side_hip/l_hip/lm_hip/l_upper_leg/lm_knee/l_lower_leg/lm_ankle/l_foot",
    "l_foot_pad": "Geometry/root/base/lm_side_hip/l_hip/lm_hip/l_upper_leg/lm_knee/l_lower_leg/lm_ankle/l_foot/l_foot_pad",  # noqa: E501
    "escmountcenter_2": "Geometry/root/base/lm_side_hip/l_hip/lm_hip/l_upper_leg/lm_knee/l_lower_leg/lm_ankle/escmountcenter_2",  # noqa: E501
    "escmountcenter": "Geometry/root/base/lm_side_hip/l_hip/lm_hip/l_upper_leg/lm_knee/escmountcenter",
    "escmountcenter_3": "Geometry/root/base/lm_side_hip/l_hip/lm_hip/escmountcenter_3",
    "escmountcenter_1": "Geometry/root/base/lm_side_hip/escmountcenter_1",
    "bodybox": "Geometry/root/base/bodybox",
    "batfront": "Geometry/root/base/bodybox/batfront",
    "batback": "Geometry/root/base/bodybox/batback",
    "electronics": "Geometry/root/base/bodybox/electronics",
}

CHICKEN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(CHICKEN_USD_PATH),
        rigid_props=PhysxRigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=PhysxCollisionPropertiesCfg(
            collision_enabled=True,
        ),
        articulation_props=PhysxArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.0,
            stabilization_threshold=0.001,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.215, 0.223, 0.271), roughness=0.8, metallic=0.1),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 2.x used (0, 0, 0.02) for the URDF's empty "root" link. The 3.x importer drops that link and
        # makes "base" the articulation root, so add the root->base offset from fixed_node_to_root_joint
        # (0.0623, 0.0075, 0.1262) to spawn the robot in the same place.
        pos=(0.0623, 0.00748735, 0.02 + 0.126153),
        joint_pos={
            "l_side_hip": 0.0,
            "l_hip": 0.0,
            "l_knee": 0.0,
            "l_ankle": 0.0,
            "r_side_hip": 0.0,
            "r_hip": 0.0,
            "r_knee": 0.0,
            "r_ankle": 0.0,
        },
    ),
    actuators={
        "motor_actuator": DelayedPDActuatorCfg(
            joint_names_expr=["r_.*", "l_.*"],
            min_delay=1,
            max_delay=2,
            # 2.x names: effort_limit, effort_limit_sim, velocity_limit, velocity_limit_sim
            actuator_effort_limit=12.0,
            joint_effort_limit=15.0,
            actuator_velocity_limit=30.0,
            joint_velocity_limit=35.0,
            stiffness=20.0,
            damping=0.5,
            armature=0.01,
            friction=0.02,
            dynamic_friction=0.01,
            viscous_friction=0.02,
        ),
    },
)

"""it's beautiful!"""
