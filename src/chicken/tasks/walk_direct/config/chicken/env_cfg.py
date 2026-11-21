# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, PvaCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import CollisionBaseCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass, replace
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from chicken.assets import CHICKEN_ASSETS_DIR
from chicken.assets.chicken import CHICKEN_BODY_PATHS, CHICKEN_CFG, CHICKEN_JOINTS

# uv run isaaclab train --rl_library skrl --task Chicken-Walk-Direct --num_envs 4096
# uv run isaaclab play --rl_library skrl --task Chicken-Walk-Direct --num_envs 9 --checkpoint latest --viz newton_gl
# uv run tensorboard --logdir logs/skrl/chicken2

ROBOT = "{ENV_REGEX_NS}/Robot"

# Bodies whose contact with anything ends the episode (key -> link name). Same set as v6.5 under the v7
# link names; the feet are left out because lfoot/rfoot are the contact pads now. v7 has no ESC mounts.
GROUND_CONTACT_BODIES = {
    "l_knee": "lm_knee",
    "r_knee": "rm_knee",
    "l_leg": "llegtop",
    "r_leg": "rlegtop",
    "l_lowerleg": "llegbottom",
    "r_lowerleg": "rlegbottom",
    "lm_hip": "lm_hip",
    "rm_hip": "rm_hip",
    "lm_side_hip": "lm_side_hip",
    "rm_side_hip": "rm_side_hip",
    "lm_ankle": "lm_ankle",
    "rm_ankle": "rm_ankle",
    "l_hip": "lhip",
    "r_hip": "rhip",
    "l_hipframe": "lhipframe",
    "r_hipframe": "rhipframe",
    "lm_hip_turn": "lm_hip_turn",
    "rm_hip_turn": "rm_hip_turn",
    "body": "hip",
    "battery": "battery",
    "imu": "imu",
    "stm32": "stm32",
}


def body_path(name: str) -> str:
    return f"{ROBOT}/{CHICKEN_BODY_PATHS[name]}"


@configclass
class ChickenPhysicsCfg(PresetCfg):
    """Physics backends. The chicken's robot settings are PhysX-specific, so PhysX is the default.

    Select one with ``physics=<name>``; ``physx`` picks Isaac Sim PhysX or standalone OvPhysX,
    whichever is installed (``--extra isaacsim`` or ``--extra ovphysx``).
    """

    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default: PhysxAutoCfg = physx


@configclass
class ChickenSceneCfg(InteractiveSceneCfg):
    """Ground, robot, sensors and lights (2.x built these by hand in ``_setup_scene``)."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        # -1 = global: every env collides with it. The default (0) leaves it out of the per-env collision
        # groups that filter_collisions sets up, and the robots fall straight through.
        collision_group=-1,
        spawn=sim_utils.UsdFileCfg(
            # ground.usda is the 2.x env9.usd without its kinematic rigid body: a plain static collider.
            # As a kinematic body, 3.x PhysX didn't collide the robots with the triangle mesh (they fell
            # through), and a disabled body still crashed the newton_gl viewer. The terrain never moves.
            usd_path=str(CHICKEN_ASSETS_DIR / "terrain" / "ground.usda"),
            collision_props=CollisionBaseCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.14), roughness=0.1, metallic=0.1),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=1.0, dynamic_friction=0.9, restitution=0.0),
            scale=(2.5, 2.5, 0.4),
        ),
    )

    robot: ArticulationCfg = replace(CHICKEN_CFG, prim_path=ROBOT)

    # 2.x ``Imu`` was a full-state sensor; its 3.x equivalent is ``Pva``.
    imu = PvaCfg(prim_path=body_path("imu"))

    contact_l = ContactSensorCfg(prim_path=body_path("lfoot"), update_period=0.0, track_air_time=True)
    contact_r = ContactSensorCfg(prim_path=body_path("rfoot"), update_period=0.0, track_air_time=True)

    # Quaternions are (x, y, z, w) in 3.x; 2.x used (w, x, y, z).
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0), angle=34.3),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.78963, 0.0, 0.0, -0.61359)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0), exposure=0.4),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.30843, 0.63629, 0.63629, -0.30843)),
    )

    def __post_init__(self):
        # One contact sensor per body that must not touch the ground: added as ``contact_<key>``.
        for key, link in GROUND_CONTACT_BODIES.items():
            setattr(self, f"contact_{key}", ContactSensorCfg(prim_path=body_path(link), update_period=0.0))


@configclass
class ChickenEnvCfg(DirectRLEnvCfg):
    # env
    # num_envs = 1
    # num_envs = 2
    # num_envs = 9
    # num_envs = 64
    # num_envs = 1024
    num_envs = 4096

    spawn_size = 5

    decimation = 2
    episode_length_s = 8
    action_space = len(CHICKEN_JOINTS)  # one position target per motor

    history_length = 15
    history_interval = 1

    # joint velocities + sin/cos of joint angles + sin/cos of the 2 gait phases + gravity + ang vel + 3 commands
    original_observation_space = len(CHICKEN_JOINTS) * 3 + 4 + 3 + 3 + 3
    history_size = original_observation_space + action_space
    observation_space = original_observation_space + history_length // history_interval * history_size

    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation, physics=ChickenPhysicsCfg())

    # order:
    # "side_hip"
    # "hip"
    # "knee"
    # "ankle"

    # scene
    scene: ChickenSceneCfg = ChickenSceneCfg(
        num_envs=num_envs,
        env_spacing=spawn_size / math.sqrt(num_envs),  # recomputed from the actual num_envs in ChickenEnv
        replicate_physics=True,
        lazy_sensor_update=True,
        filter_collisions=True,
    )

    # def __post_init__(self):
    #     super().__post_init__()

    # # Double the default collision stack size from 64MB to 256MB
    # self.sim.physx.gpu_collision_stack_size = 67108864 * 2

    # # # PRO TIP: If you plan to scale to 4000+ envs, you should also increase
    # # # these other contact buffers now so they don't overflow next!
    # self.sim.physx.gpu_max_rigid_contact_count = 8388608 * 2  # default is 8388608
    # self.sim.physx.gpu_max_rigid_patch_count = 163840 * 2  # default is 163840
    # self.sim.physx.gpu_found_lost_pairs_capacity = 2097152 * 2  # default is 2097152
