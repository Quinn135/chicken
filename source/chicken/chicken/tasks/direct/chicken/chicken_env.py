# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Imu
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.math import sample_uniform

from .chicken_env_cfg import ChickenEnvCfg


class ChickenEnv(DirectRLEnv):
    cfg: ChickenEnvCfg

    def __init__(self, cfg: ChickenEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # self._l0_dof_idx, _ = self.robot.find_joints("l0")
        # self._l1_dof_idx, _ = self.robot.find_joints("l1")
        # self._l2_dof_idx, _ = self.robot.find_joints("l2")
        # self._r0_dof_idx, _ = self.robot.find_joints("r0")
        # self._r1_dof_idx, _ = self.robot.find_joints("r1")
        # self._r2_dof_idx, _ = self.robot.find_joints("r2")
        self._l_joint_dof_idxs = [self.robot.find_joints("l0")[0], self.robot.find_joints("l1")[0], self.robot.find_joints("l2")[0]]
        self._r_joint_dof_idxs = [self.robot.find_joints("r0")[0], self.robot.find_joints("r1")[0], self.robot.find_joints("r2")[0]]

        self._body_idxs = self.robot.find_bodies("body")[0]

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        self.pos_range = self.cfg.pos_range
        
        # to fix IMU glitch:
        self.imu._dt = self.sim.get_physics_dt()
                
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        
        self.imu = Imu(self.cfg.imu_cfg)
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
            
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1200.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # apply joint actions
        for i, dof_idx in enumerate(self._l_joint_dof_idxs):
            # times the range of each joint
            self.robot.set_joint_position_target((self.actions[:, i] * self.pos_range[i]).unsqueeze(dim=1), joint_ids=dof_idx)
        for i, dof_idx in enumerate(self._r_joint_dof_idxs):
            self.robot.set_joint_position_target((self.actions[:, i + 3] * self.pos_range[i]).unsqueeze(dim=1), joint_ids=dof_idx)
        
        # self.robot.write_data_to_sim()

    def _sin_cos(self, angles: torch.Tensor) -> torch.Tensor:
        return torch.stack((
            torch.sin(angles),
            torch.cos(angles)
        ), dim=-1).flatten(start_dim=-2) # just flatten the last two dims

    def _get_observations(self) -> dict:
        # obs: motor vel, motor rot
        # imu
        self.lin_acc_b = self.imu.data.lin_acc_b # shape (N, 3)?
        self.ang_acc_b = self.imu.data.ang_acc_b # shape (N, 3)?
        self.lin_vel_b = self.imu.data.lin_vel_b # shape (N, 3)?
        
        # raise Exception("I am just testing!")
        
        obs = torch.cat((
            self.joint_vel[:, self._l_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1),
            self.joint_vel[:, self._r_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1),
            self._sin_cos(self.joint_pos[:, self._l_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1)),
            self._sin_cos(self.joint_pos[:, self._r_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1)),
            self.lin_acc_b.unsqueeze(dim=1),
            self.ang_acc_b.unsqueeze(dim=1)
        ), dim=-1)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # x speed (negative)
        self.x_vel = -self.lin_vel_b[:, 0]
        
        total_reward = torch.ones(self.num_envs, dtype=torch.float32, device=self.device) * 200
        total_reward += torch.sum(-torch.sum(torch.abs(self.joint_vel[:, self._l_joint_dof_idxs]), dim=-1) - torch.sum(torch.abs(self.joint_vel[:, self._r_joint_dof_idxs]), dim=-1), dim=-1)
        # print(total_reward.shape)
        
        body_pos_z = self.robot.data.root_pos_w[:, 2]
        total_reward += body_pos_z * 8
        
        body_quats = self.robot.data.body_quat_w[:, self._body_idxs, :].flatten(start_dim=-2)
        gravity_local = quat_apply_inverse(body_quats, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1))
        # -1 is up, 0 is side
        total_reward += -gravity_local[:, 2] * 1.5
        
        # total_reward += self.x_vel * 10
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # out_of_bounds = False
        # out_of_bounds = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        body_pos_z = self.robot.data.root_pos_w[:, 2]
        threshold = 0.14
        below_threshold = body_pos_z < threshold
        
        body_quats = self.robot.data.body_quat_w[:, self._body_idxs, :].flatten(start_dim=-2)
        gravity_local = quat_apply_inverse(body_quats, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1))
        titled_too_much = gravity_local[:, 2] > -0.94
        
        out_of_bounds = below_threshold | titled_too_much
        
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        
        self.imu.reset()

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        # joint_pos[:, self._pole_dof_idx] += sample_uniform(
        #     self.cfg.initial_pole_angle_range[0] * math.pi,
        #     self.cfg.initial_pole_angle_range[1] * math.pi,
        #     joint_pos[:, self._pole_dof_idx].shape,
        #     joint_pos.device,
        # )
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


# @torch.jit.script
# def compute_rewards(
#     rew_scale_alive: float,
#     rew_scale_terminated: float,
#     rew_scale_pole_pos: float,
#     rew_scale_cart_vel: float,
#     rew_scale_pole_vel: float,
#     pole_pos: torch.Tensor,
#     pole_vel: torch.Tensor,
#     cart_pos: torch.Tensor,
#     cart_vel: torch.Tensor,
#     reset_terminated: torch.Tensor,
# ):
#     rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
#     rew_termination = rew_scale_terminated * reset_terminated.float()
#     rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
#     rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
#     rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
#     total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
#     return total_reward