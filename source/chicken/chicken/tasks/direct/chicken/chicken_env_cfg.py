# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import gymnasium as gym

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from assets.robots.chicken import CHICKEN_CFG

# python -m pip install -e source/chicken

# python scripts/skrl/train.py --task=Isaac-Chicken-Robot-v0 --headless
# python scripts/skrl/play.py --task=Isaac-Chicken-Robot-v0 --num_envs 9
# for vscode, python is at /isaac-sim/kit/python/bin/python3
# if you want to clone, clone into /workspace/chicken or /workspace/wherever_you_want
# and tensorboard command is: tensorboard --logdir logs/skrl/chicken3 --bind_all --logdir logs/skrl/chickenv3


@configclass
class ChickenEnvCfg(DirectRLEnvCfg):
    # env
    num_envs = 1
    # num_envs = 2
    # num_envs = 9
    # num_envs = 64
    # num_envs = 4096

    spawn_size = 5

    decimation = 2
    episode_length_s = 15
    action_space = 8  # command 8 motor positions

    history_length = 30
    history_interval = 1

    original_observation_space = 8 + 8 * 2 + 4 + 3 + 3
    history_size = original_observation_space + action_space
    observation_space = original_observation_space + history_length * history_size

    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    render_cfg = sim_utils.RenderCfg(rendering_mode="performance")

    # robot(s)
    robot_cfg: ArticulationCfg = CHICKEN_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # order:
    # "side_hip"
    # "hip"
    # "knee"
    # "ankle"

    root_str = "/Robot"

    # /root/base/center_mount/stuff_holder
    imu_cfg: ImuCfg = ImuCfg(prim_path=f"/World/envs/env_.*{root_str}/imu")

    contact_cfg_r: ContactSensorCfg = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*{root_str}/r_foot_pad",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground.*"],
        track_air_time=True,
    )
    contact_cfg_l: ContactSensorCfg = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*{root_str}/l_foot_pad",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground.*"],
        track_air_time=True,
    )

    contact_cfgs: dict[str, ContactSensorCfg] = {
        "l_knee": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/lm_knee",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "r_knee": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/rm_knee",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "l_foot": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/l_foot",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "r_foot": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/r_foot",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "l_leg": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/l_upper_leg",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "r_leg": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/r_upper_leg",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "l_lowerleg": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/l_lower_leg",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "r_lowerleg": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/r_lower_leg",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "lm_hip": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/lm_hip",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "rm_hip": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/rm_hip",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "lm_side_hip": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/lm_side_hip",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "rm_side_hip": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/rm_side_hip",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "lm_ankle": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/lm_ankle",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "rm_ankle": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/rm_ankle",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "l_hip": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/l_hip",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "r_hip": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/r_hip",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "base": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/base",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
        "mass": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/part_22",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground.*"],
        ),
    }
    # scene

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=spawn_size / math.sqrt(num_envs),
        replicate_physics=True,
        lazy_sensor_update=True,
        filter_collisions=True,
        clone_in_fabric=False,
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
