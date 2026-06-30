# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
    decimation = 1
    episode_length_s = 20
    action_space = 8  # command 8 motor positions

    history_length = 30
    history_interval = 1

    original_observation_space = 8 + 8 * 2 + 4 + 3 + 3
    observation_space = original_observation_space + history_length * original_observation_space

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
    pos_range = [50 * math.pi / 180.0, 80 * math.pi / 180.0, 140 * math.pi / 180.0, 80 * math.pi / 180.0]

    imu_cfg: ImuCfg = ImuCfg(prim_path="/World/envs/env_.*/Robot/base_link")

    contact_cfg_r: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_foot",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground"],
        track_air_time=True,
    )
    contact_cfg_l: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/l_foot",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground"],
        track_air_time=True,
    )

    contact_cfg_r_leg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rm_knee",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground"],
    )
    contact_cfg_l_leg: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/lm_knee",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground"],
    )

    contact_cfg_r_hip: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rm_hip",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground"],
    )
    contact_cfg_l_hip: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/lm_hip",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground"],
    )

    contact_cfg_r_thigh: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/r_thigh",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground"],
    )
    contact_cfg_l_thigh: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/l_thigh",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground"],
    )

    contact_cfg_r_ankle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/rm_ankle",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground"],
    )
    contact_cfg_l_ankle: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/lm_ankle",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground"],
    )

    contact_cfg_base: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground"],
    )
    # scene

    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=900, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2304, env_spacing=1.5, replicate_physics=True)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4900, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8100, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=10000, env_spacing=1.5, replicate_physics=True)
