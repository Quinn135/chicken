"""Configuration for a my chicken!"""

import time

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg

# from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

##
# Configuration
##

urdf_path = "/workspace/isaaclab/source/chickens/v4 8/urdf/chicken.urdf"
usd_gen_dir = "/workspace/chicken/chicken/models/usd"
usd_gen_file_name = f"chicken_{round(time.time())}.usd"
print("converting urdf...")

# fix urdf file
with open(urdf_path) as f:
    content = f.read()

content = content.replace("package://chicken", "..")
with open(urdf_path, "w") as f:
    f.write(content)

# convert urdf to usd
converter = UrdfConverter(
    cfg=UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=usd_gen_dir,
        usd_file_name=f"{usd_gen_file_name}",
        fix_base=False,
        force_usd_conversion=True,
        merge_fixed_joints=False,
        link_density=1250,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=20.0, damping=0.5),  #
        ),
        collision_from_visuals=True,
        self_collision=False,
        collider_type="convex_hull",
        make_instanceable=True,
    ),
)

print("spawning...")

CHICKEN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{usd_gen_dir}/{usd_gen_file_name}",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
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
        pos=(0.0, 0.0, 0.0),
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
            max_delay=1,
            effort_limit=5.0,
            effort_limit_sim=6.0,
            velocity_limit=6.0,
            velocity_limit_sim=7.0,
            stiffness=20.0,
            damping=0.5,
            armature=0.02,
            friction=0.01,
            dynamic_friction=0.005,
            viscous_friction=0.05,
        ),
    },
)

"""it's beautiful!"""
