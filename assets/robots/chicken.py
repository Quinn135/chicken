"""Configuration for a my chicken!"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

CHICKEN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/isaaclab/source/chickenv30.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # joint_pos={
        #     "lr": 0.0,
        #     "l0": 8.969 * math.pi / 180.0,
        #     "l1": -18.869 * math.pi / 180.0,
        #     "l2": -9.901 * math.pi / 180.0,
        #     "rr": 0.0,
        #     "r0": -8.969 * math.pi / 180.0,
        #     "r1": 18.869 * math.pi / 180.0,
        #     "r2": 9.901 * math.pi / 180.0,
        # },
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
        "motor_actuator": ImplicitActuatorCfg(
            joint_names_expr=["r_.*", "l_.*"], effort_limit_sim=200.0, stiffness=12.5, damping=0.5
        ),
    },
)

"""it's beautiful!"""
