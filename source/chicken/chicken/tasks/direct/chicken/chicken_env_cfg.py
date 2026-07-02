# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import gymnasium as gym

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
    num_envs = 4096

    decimation = 1
    episode_length_s = 20
    action_space = 8  # command 8 motor positions

    history_length = 30
    history_interval = 1

    original_observation_space = 8 + 8 * 2 + 4 + 3 + 5
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
    # pos_range = [60 * math.pi / 180.0, 110 * math.pi / 180.0, 170 * math.pi / 180.0, 50 * math.pi / 180.0]

    root_str = "/Robot"

    # /root/base/center_mount/stuff_holder
    imu_cfg: ImuCfg = ImuCfg(prim_path=f"/World/envs/env_.*{root_str}/imu")

    contact_cfg_r: ContactSensorCfg = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*{root_str}/r_foot_pad",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground"],
        track_air_time=True,
    )
    contact_cfg_l: ContactSensorCfg = ContactSensorCfg(
        prim_path=f"/World/envs/env_.*{root_str}/l_foot_pad",
        update_period=0.0,
        filter_prim_paths_expr=["/World/ground"],
        track_air_time=True,
    )

    contact_cfgs: dict[str, ContactSensorCfg] = {
        "l_knee": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/lm_knee",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground"],
        ),
        "r_knee": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/rm_knee",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground"],
        ),
        "l_foot": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/l_foot",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground"],
        ),
        "r_foot": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/r_foot",
            update_period=0.0,
            filter_prim_paths_expr=["/World/ground"],
        ),
        "base": ContactSensorCfg(
            prim_path=f"/World/envs/env_.*{root_str}/base", update_period=0.0, filter_prim_paths_expr=["/World/ground"]
        ),
    }
    # scene

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=num_envs,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=True,
        filter_collisions=True,
        clone_in_fabric=False,
    )
