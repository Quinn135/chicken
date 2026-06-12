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

        self._l_joint_dof_idxs = [self.robot.find_joints("lr")[0], self.robot.find_joints("l0")[0], self.robot.find_joints("l1")[0], self.robot.find_joints("l2")[0]]
        self._r_joint_dof_idxs = [self.robot.find_joints("rr")[0], self.robot.find_joints("r0")[0], self.robot.find_joints("r1")[0], self.robot.find_joints("r2")[0]]

        self._body_idxs = self.robot.find_bodies("body")[0]

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        
        self.pos_range = self.cfg.pos_range
        
        # to fix IMU glitch:
        self.imu._dt = self.sim.get_physics_dt()
        
        # self.pos = torch.zeros((self.num_envs, 3), device=self.device)
        # self.last_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        # self.pos = torch.zeros((self.num_envs, 3), device=self.device)
        current_pos = self.robot.data.root_pos_w.clone()
        self.current_pos = current_pos
        self.last_pos = torch.zeros((self.num_envs, 3), device=self.device)
        
        # self.last_actions = torch.zeros((self.num_envs, self.cfg.history_length, 6 * 2), device=self.device)
        
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
        self.actions = torch.clamp(torch.nan_to_num(self.actions, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        
        # # shape (len(env_ids), len(body_ids), 3)
        # forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        # torques = torch.zeros_like(forces)
        
        # force_interval = 360
        # push_envs = self.episode_length_buf % force_interval == 0
        
        # max_force = 50.0
        # forces[push_envs] = sample_uniform(-max_force, max_force, forces[push_envs].shape, device=self.device)
        
        # self.robot.set_external_force_and_torque(forces, torques)
        # self.robot.write_data_to_sim()

    def _apply_action(self) -> None:
        # apply joint actions
        for i, dof_idx in enumerate(self._l_joint_dof_idxs):
            # times the range of each joint
            self.robot.set_joint_position_target((self.actions[:, i] * self.pos_range[i] / 2.0).unsqueeze(dim=1), joint_ids=dof_idx)
        for i, dof_idx in enumerate(self._r_joint_dof_idxs):
            self.robot.set_joint_position_target((self.actions[:, i + 4] * self.pos_range[i] / 2.0).unsqueeze(dim=1), joint_ids=dof_idx)
        
        # ones_to_update = self.episode_length_buf % 5 == 0
        
        # self.last_actions[ones_to_update] = torch.roll(self.last_actions[ones_to_update], shifts=1, dims=1)
        
        # # self.last_actions[ones_to_update, 0, :] = self.actions[ones_to_update]
        # self.last_actions[ones_to_update, 0, :] = self._sin_cos(torch.stack((
        #     self.joint_pos[ones_to_update, self._l_joint_dof_idxs].transpose(0, 1),
        #     self.joint_pos[ones_to_update, self._r_joint_dof_idxs].transpose(0, 1)),
        #         dim=1).flatten(start_dim=-2))

    def _sin_cos(self, angles: torch.Tensor) -> torch.Tensor:
        return torch.stack((
            torch.sin(angles),
            torch.cos(angles)
        ), dim=-1).flatten(start_dim=-2) # just flatten the last two dims

    def _get_info(self) -> dict:
        dt = self.cfg.sim.dt * self.cfg.decimation
        
        return {
            "x_vel": (self.current_pos[:, 0] - self.last_pos[:, 0]) / dt,
            "y_vel": (self.current_pos[:, 1] - self.last_pos[:, 1]) / dt,
        }

    def _get_observations(self) -> dict:
        # obs: motor vel, motor rot
        # imu
        self.lin_acc_b = torch.nan_to_num(self.imu.data.lin_acc_b, nan=0.0, posinf=0.0, neginf=0.0)
        self.ang_acc_b = torch.nan_to_num(self.imu.data.ang_acc_b, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.lin_vel_b = torch.nan_to_num(self.imu.data.lin_vel_b, nan=0.0, posinf=0.0, neginf=0.0)
        # self.lin_vel_w = torch.nan_to_num(self.imu.data.lin_vel_w, nan=0.0, posinf=0.0, neginf=0.0)
        self.ang_vel_b = torch.nan_to_num(self.imu.data.ang_vel_b, nan=0.0, posinf=0.0, neginf=0.0)
        
        # since IMU doesnt have world coords, use world velocity
        self.lin_vel_w = self.robot.data.root_lin_vel_w
        
        obs = torch.cat((
            self.joint_vel[:, self._l_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1),
            self.joint_vel[:, self._r_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1),
            self._sin_cos(self.joint_pos[:, self._l_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1)),
            self._sin_cos(self.joint_pos[:, self._r_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1)),
            self.lin_acc_b.unsqueeze(dim=1),
            self.ang_acc_b.unsqueeze(dim=1),
            # self.lin_vel_b.unsqueeze(dim=1),
            # self.lin_vel_w.unsqueeze(dim=1),
            # self.ang_vel_b.unsqueeze(dim=1),
            # self.last_actions.flatten(start_dim=-2).unsqueeze(dim=1)
        ), dim=-1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = torch.clamp(obs, -100.0, 100.0)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        current_pos = self.robot.data.root_pos_w.clone()
        self.current_pos = current_pos
        
        # total_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        total_reward = torch.ones(self.num_envs, dtype=torch.float32, device=self.device) * 10.0
        
        # total_reward += (below_threshold | titled_too_much).float() * -20.0
        out_of_bounds, _ = self._get_dones()
        total_reward += out_of_bounds.float() * -10.0

        # total_reward += torch.clamp(
        #     (torch.sum(-torch.sum(torch.abs(self.joint_vel[:, self._l_joint_dof_idxs]), dim=-1)
        #                            - torch.sum(torch.abs(self.joint_vel[:, self._r_joint_dof_idxs]), dim=-1), dim=-1)
        #                  ) * 1.0,
        #     min=-10.0,
        #     max=0.0
        # )

        # disincentivize large actions w/ self.actions
        total_reward -= torch.sum(torch.square(self.actions), dim=-1) * 0.1


        # reward it being close to target height
        # target_height = 0.0
        # body_pos_z = current_pos[:, 2] # Use current_pos
        # total_reward += torch.square(body_pos_z - target_height) * 4.0

        body_quats = self.robot.data.body_quat_w[:, self._body_idxs, :].flatten(start_dim=-2)
        gravity_local = quat_apply_inverse(body_quats, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32,
                                                                    device=self.device).repeat(self.num_envs, 1))
        total_reward += torch.square((-gravity_local[:, 2] + 1) / 2.0) * 1.0

        # Calculate delta distance securely
        # total_reward += -(current_pos[:, 0] - self.last_pos[:, 0]) * 3
        # total_reward -= torch.abs(current_pos[:, 1] - self.last_pos[:, 1]) * 10
        
        # # motivate walking!
        # target_vel = -0.25  # m/s
        # dt = self.cfg.sim.dt * self.cfg.decimation  # 3/120 = 0.025s
        # current_vel_x = (current_pos[:, 0] - self.last_pos[:, 0]) / dt
        # vel_reward = current_vel_x / target_vel
        # # cap reward to 1
        # total_reward += torch.clamp(vel_reward, max=1.0) * 2.0
        
        # # say sideways is bad!
        # current_vel_y = (current_pos[:, 1] - self.last_pos[:, 1]) / dt
        # total_reward -= torch.abs(current_vel_y / target_vel) * 0.5
        
        # total_reward -= torch.sum(torch.square(self.lin_vel_w), dim=-1) * 1
        # total_reward -= torch.sum(torch.square(self.ang_vel_b), dim=-1) * 1
        
        # total_reward -= torch.sum(torch.abs(self.joint_pos), dim=-1) * 0.3
        
        # Save memory
        self.last_pos = current_pos
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # out_of_bounds = False
        # out_of_bounds = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        body_pos_z = self.robot.data.root_pos_w[:, 2]
        threshold = -0.02
        below_threshold = body_pos_z < threshold
        
        body_quats = self.robot.data.body_quat_w[:, self._body_idxs, :].flatten(start_dim=-2)
        gravity_local = quat_apply_inverse(body_quats, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1))
        titled_too_much = gravity_local[:, 2] > -0.85
        
        # prevent too fast, flying away etc
        dt = self.cfg.sim.dt * self.cfg.decimation
        lin_vel_w = (self.current_pos[:, 0] - self.last_pos[:, 0]) / dt
        too_fast = torch.any(torch.abs(lin_vel_w) > 10.0, dim=-1)
        
        # too high
        body_pos_z = self.robot.data.root_pos_w[:, 2]
        too_high = body_pos_z > 3.0
        
        out_of_bounds = titled_too_much | too_fast | too_high | below_threshold
        # out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        
        self.imu.reset(env_ids)
        # self.last_actions[env_ids] = 0.0

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.last_pos[env_ids] = default_root_state[:, :3].clone()

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