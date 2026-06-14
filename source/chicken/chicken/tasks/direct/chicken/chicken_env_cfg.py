# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from assets.robots.chicken import CHICKEN_CFG

# python scripts/skrl/train.py --task=Isaac-Chicken-Robot-v0 --headless
# python scripts/skrl/play.py --task=Isaac-Chicken-Robot-v0 --num_envs 9


@configclass
class ChickenEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 30
    # - spaces definition
    action_space = 8  # command 8 motor positions

    # obs: motor vel, motor rot (sin, cos), lin_acc_b, ang_acc_b
    # imu...
    # history_length = 15
    observation_space = 8 + 8 * 2 + 3 * 2 + 2 + 2
    # observation_space = (8 + 8 * 2 + 3 * 4) + 8 * 2 * history_length  # remember actions

    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = CHICKEN_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    pos_range = [50 * math.pi / 180.0, 120 * math.pi / 180.0, 120 * math.pi / 180.0, 90 * math.pi / 180.0]

    imu_cfg: ImuCfg = ImuCfg(prim_path="/World/envs/env_.*/Robot/chicken/body")

    # scene
    # 9 envs if playing, otherwise 4096

    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=900, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.5, replicate_physics=True)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8100, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=10000, env_spacing=1.5, replicate_physics=True)
