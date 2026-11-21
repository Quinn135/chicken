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

# Generated from robots/chicken_v4/urdf/chicken_v4.urdf by scripts/convert_robot.py.
# (In 2.x this conversion re-ran on every import; now it is a one-off step.)
CHICKEN_USD_PATH = CHICKEN_ASSETS_DIR / "robots" / "usd" / "chicken_v4_9" / "chicken_v4.usda"

# The 3.x URDF importer nests each link under its parent link (2.x put every link directly under the
# robot prim), so sensors need each link's full path. Paths are relative to the robot prim.
# Regenerate this table if the URDF's link tree changes.
CHICKEN_BODY_PATHS = {
    "hip": "Geometry/root/hip",  # body; articulation root
    "battery": "Geometry/root/hip/battery",
    "imu": "Geometry/root/hip/imu",
    "stm32": "Geometry/root/hip/stm32",
    "rm_side_hip": "Geometry/root/hip/rm_side_hip",
    "rhipframe": "Geometry/root/hip/rm_side_hip/rhipframe",
    "rm_hip_turn": "Geometry/root/hip/rm_side_hip/rhipframe/rm_hip_turn",
    "rhip": "Geometry/root/hip/rm_side_hip/rhipframe/rm_hip_turn/rhip",
    "rm_hip": "Geometry/root/hip/rm_side_hip/rhipframe/rm_hip_turn/rhip/rm_hip",
    "rlegtop": "Geometry/root/hip/rm_side_hip/rhipframe/rm_hip_turn/rhip/rm_hip/rlegtop",
    "rm_knee": "Geometry/root/hip/rm_side_hip/rhipframe/rm_hip_turn/rhip/rm_hip/rlegtop/rm_knee",
    "rlegbottom": "Geometry/root/hip/rm_side_hip/rhipframe/rm_hip_turn/rhip/rm_hip/rlegtop/rm_knee/rlegbottom",
    "rm_ankle": "Geometry/root/hip/rm_side_hip/rhipframe/rm_hip_turn/rhip/rm_hip/rlegtop/rm_knee/rlegbottom/rm_ankle",
    "rfoot": "Geometry/root/hip/rm_side_hip/rhipframe/rm_hip_turn/rhip/rm_hip/rlegtop/rm_knee/rlegbottom/rm_ankle"
    "/rfoot",
    "lm_side_hip": "Geometry/root/hip/lm_side_hip",
    "lhipframe": "Geometry/root/hip/lm_side_hip/lhipframe",
    "lm_hip_turn": "Geometry/root/hip/lm_side_hip/lhipframe/lm_hip_turn",
    "lhip": "Geometry/root/hip/lm_side_hip/lhipframe/lm_hip_turn/lhip",
    "lm_hip": "Geometry/root/hip/lm_side_hip/lhipframe/lm_hip_turn/lhip/lm_hip",
    "llegtop": "Geometry/root/hip/lm_side_hip/lhipframe/lm_hip_turn/lhip/lm_hip/llegtop",
    "lm_knee": "Geometry/root/hip/lm_side_hip/lhipframe/lm_hip_turn/lhip/lm_hip/llegtop/lm_knee",
    "llegbottom": "Geometry/root/hip/lm_side_hip/lhipframe/lm_hip_turn/lhip/lm_hip/llegtop/lm_knee/llegbottom",
    "lm_ankle": "Geometry/root/hip/lm_side_hip/lhipframe/lm_hip_turn/lhip/lm_hip/llegtop/lm_knee/llegbottom/lm_ankle",
    "lfoot": "Geometry/root/hip/lm_side_hip/lhipframe/lm_hip_turn/lhip/lm_hip/llegtop/lm_knee/llegbottom/lm_ankle"
    "/lfoot",
}

# Actuated joints, in the env's action order: the left leg, then the right leg, with the same joints in the
# same order in each half (the env splits this list in two). Turn hip = hip yaw. The ankle joints are named
# after the feet they move. (v6.5 had 8: l_side_hip, l_hip, l_knee, l_ankle, r_...)
CHICKEN_JOINTS = ["lturnhip", "lsidehip", "lhip", "lknee", "lfoot", "rturnhip", "rsidehip", "rhip", "rknee", "rfoot"]

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
        # The root body ("hip") sits at the URDF origin and the bottoms of the feet are 0.388 m below it with
        # straight legs. 0.376 puts the feet ~1 cm above the terrain (z = -0.02 at the spawn points).
        pos=(0.0, 0.0, 0.376),
        joint_pos=dict.fromkeys(CHICKEN_JOINTS, 0.0),
    ),
    actuators={
        "motor_actuator": DelayedPDActuatorCfg(
            joint_names_expr=CHICKEN_JOINTS,
            min_delay=1,
            max_delay=2,
            # 2.x names: effort_limit, effort_limit_sim, velocity_limit, velocity_limit_sim
            actuator_effort_limit=13.0,
            joint_effort_limit=15.0,
            actuator_velocity_limit=30.0,
            joint_velocity_limit=35.0,
            stiffness=20.0,
            damping=0.5,
            armature=0.15,
            friction=0.07,
            dynamic_friction=0.01,
            viscous_friction=0.02,
        ),
    },
)

"""it's beautiful!"""
