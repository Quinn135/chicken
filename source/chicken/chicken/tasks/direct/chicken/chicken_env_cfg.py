# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

# python -m pip install -e source/chicken

# python scripts/skrl/train.py --task=Isaac-Chicken-Robot-v0 --headless
# python scripts/skrl/play.py --task=Isaac-Chicken-Robot-v0 --num_envs 9
# for vscode, python is at /isaac-sim/kit/python/bin/python3
# if you want to clone, clone into /workspace/chicken or /workspace/wherever_you_want
# and tensorboard command is: tensorboard --logdir logs/skrl/chicken3 --bind_all --logdir logs/skrl/chickenv3


@configclass
class ChickenEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10
    # - spaces definition
    action_space = 8  # command 8 motor positions

    # obs: motor vel, motor rot (sin, cos), lin_acc_b, ang_acc_b
    # imu...
    history_length = 5
    history_interval = 4
    # observation_space = 6 + 6 * 2 + 3 * 2 + 2 + 1 + 1 + 1
    observation_space = 8 + 8 * 2 + 3 * 2 + 2 + 1 + 1 + 1 + 8 * history_length + 3  # remember actions

    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = CHICKEN_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # order:
    # "side_hip"
    # "hip"
    # "knee"
    # "ankle"
    pos_range = [50 * math.pi / 180.0, 80 * math.pi / 180.0, 140 * math.pi / 180.0, 80 * math.pi / 180.0]

    imu_cfg: ImuCfg = ImuCfg(prim_path="/World/envs/env_.*/Robot/base_link")

    # scene

    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=900, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2304, env_spacing=1.5, replicate_physics=True)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4900, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8100, env_spacing=1.5, replicate_physics=True)
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=10000, env_spacing=1.5, replicate_physics=True)
