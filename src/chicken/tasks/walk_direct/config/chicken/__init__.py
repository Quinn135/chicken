# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# 2.x id: "Isaac-Chicken-Robot-v0" (3.x task ids drop the version suffix)
gym.register(
    id="Chicken-Walk-Direct",
    entry_point=f"{__name__}.env:ChickenEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ChickenEnvCfg",
        "default_agent": "skrl",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
