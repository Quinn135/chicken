# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Imu
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply_inverse, sample_uniform

from .chicken_env_cfg import ChickenEnvCfg


class ChickenEnv(DirectRLEnv):
    cfg: ChickenEnvCfg

    def __init__(self, cfg: ChickenEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._l_joint_dof_idxs = [
            self.robot.find_joints("lr")[0],
            self.robot.find_joints("l0")[0],
            self.robot.find_joints("l1")[0],
            self.robot.find_joints("l2")[0],
        ]
        self._r_joint_dof_idxs = [
            self.robot.find_joints("rr")[0],
            self.robot.find_joints("r0")[0],
            self.robot.find_joints("r1")[0],
            self.robot.find_joints("r2")[0],
        ]

        self._body_idxs = self.robot.find_bodies("body")[0]

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.pos_range = self.cfg.pos_range

        # to fix IMU glitch:
        self.imu._dt = self.sim.get_physics_dt()

        # self.pos = torch.zeros((self.num_envs, 3), device=self.device)
        # self.last_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # self.pos = torch.zeros((self.num_envs, 3), device=self.device)
        # get pos of the "body" element (_body_idxs)
        self.current_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_pos = torch.zeros_like(self.current_pos)

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
        # self.actions = torch.clamp(torch.nan_to_num(self.actions, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

        # shape (len(env_ids), len(body_ids), 3)
        forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        torques = torch.zeros_like(forces)

        # force_interval = 60 * 3
        # push_envs = (self.episode_length_buf + 1) % force_interval == 0

        # 1/400 chance to push
        push_envs = torch.rand(self.num_envs, device=self.device) < 1.0 / 400.0

        if push_envs.any():
            max_force = 10.0
            forces[push_envs] = sample_uniform(-max_force, max_force, forces[push_envs].shape, device=self.device)

            self.robot.instantaneous_wrench_composer.set_forces_and_torques(
                forces=forces, torques=torques, body_ids=self._body_idxs
            )
        # self.robot.write_data_to_sim()

    def _apply_action(self) -> None:
        # apply joint actions
        for i, dof_idx in enumerate(self._l_joint_dof_idxs):
            # times the range of each joint
            self.robot.set_joint_position_target(
                (self.actions[:, i] * self.pos_range[i]).unsqueeze(dim=1), joint_ids=dof_idx
            )
        for i, dof_idx in enumerate(self._r_joint_dof_idxs):
            self.robot.set_joint_position_target(
                (self.actions[:, i + 4] * self.pos_range[i]).unsqueeze(dim=1), joint_ids=dof_idx
            )

        # ones_to_update = self.episode_length_buf % 5 == 0

        # self.last_actions[ones_to_update] = torch.roll(self.last_actions[ones_to_update], shifts=1, dims=1)

        # # self.last_actions[ones_to_update, 0, :] = self.actions[ones_to_update]
        # self.last_actions[ones_to_update, 0, :] = self._sin_cos(torch.stack((
        #     self.joint_pos[ones_to_update, self._l_joint_dof_idxs].transpose(0, 1),
        #     self.joint_pos[ones_to_update, self._r_joint_dof_idxs].transpose(0, 1)),
        #         dim=1).flatten(start_dim=-2))

    def _sin_cos(self, angles: torch.Tensor) -> torch.Tensor:
        return torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(
            start_dim=-2
        )  # just flatten the last two dims

    # def _get_info(self) -> dict:
    #     dt = self.cfg.sim.dt * self.cfg.decimation

    #     return {
    #         "log": {
    #             "chicken/mean_x_vel": torch.mean((self.current_pos[:, 0] - self.last_pos[:, 0]) / dt).item(),
    #             "chicken/mean_y_vel": torch.mean((self.current_pos[:, 1] - self.last_pos[:, 1]) / dt).item(),
    #             "chicken/mean_z_vel": torch.mean((self.current_pos[:, 2] - self.last_pos[:, 2]) / dt).item(),
    #         },
    #     }

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

        obs = torch.cat(
            (
                self.joint_vel[:, self._l_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1),
                self.joint_vel[:, self._r_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1),
                self._sin_cos(self.joint_pos[:, self._l_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1)),
                self._sin_cos(self.joint_pos[:, self._r_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1)),
                self.lin_acc_b.unsqueeze(dim=1),
                self.ang_acc_b.unsqueeze(dim=1),
                # self.lin_vel_b.unsqueeze(dim=1),
                self.lin_vel_w.unsqueeze(dim=1),
                self.ang_vel_b.unsqueeze(dim=1),
                # self.last_actions.flatten(start_dim=-2).unsqueeze(dim=1)
            ),
            dim=-1,
        )

        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = torch.clamp(obs, min=-100.0, max=100.0)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        self.current_pos = self.robot.data.body_pos_w[:, self._body_idxs, :].squeeze(1)

        # total_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        alive_award = torch.ones(self.num_envs, dtype=torch.float32, device=self.device) * 100.0

        # disincentivize large actions w/ self.joint_vel
        large_actions = -torch.sum(torch.pow(torch.abs(self.joint_vel), 0.5), dim=-1) * 0.1
        large_actions = torch.nan_to_num(large_actions, nan=0.0, posinf=0.0, neginf=0.0)
        large_actions = torch.clamp(large_actions, min=-20.0, max=20.0)

        # reward it being close to target height
        body_pos_z = self.current_pos[:, 2]  # Use current_pos
        target_height = 0.25
        close_to_target = -torch.abs(body_pos_z - target_height) * 25.0

        # reward uprightedness
        body_quats = self.robot.data.body_quat_w[:, self._body_idxs, :].flatten(start_dim=-2)
        gravity_local = quat_apply_inverse(
            body_quats, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        )
        uprightedness = torch.pow(-gravity_local[:, 2], 4) * 10.0 * torch.sign(-gravity_local[:, 2])

        # # motivate walking!
        dt = self.cfg.sim.dt * self.cfg.decimation
        target_vel = -0.1  # m/s
        target_dist = target_vel * self.episode_length_buf * dt
        current_dist_x = self.current_pos[:, 0] - self.robot.data.default_root_state[:, 0]
        x_reward = -torch.abs(current_dist_x - target_dist) * 5.0
        # current_vel_x = (self.current_pos[:, 0] - self.last_pos[:, 0]) / dt
        # vel_reward = -torch.abs(current_vel_x - target_vel) * 5.0

        # disincentivize movement
        # movement = -torch.clamp(torch.sum(torch.abs(self.lin_vel_w), dim=-1) * 20.0, min=0.0, max=10.0)

        # incentivize joints_pos being close to default
        joints_close_to_0 = -torch.sum(torch.abs(self.joint_pos), dim=-1) * 0.5

        # say sideways is bad!
        # current pos - default
        current_dist_y = self.current_pos[:, 1] - self.robot.data.default_root_state[:, 1] * 5.0
        # current_vel_y = (self.current_pos[:, 1] - self.last_pos[:, 1]) / dt
        # vel_y_penalty = torch.abs(current_vel_y) * 2.5

        # rotating bad
        ang_vel = -torch.sum(self.ang_vel_b, dim=-1) * 5

        # total_reward -= torch.sum(torch.square(self.lin_vel_w), dim=-1) * 1
        # total_reward -= torch.sum(torch.abs(self.joint_pos), dim=-1) * 0.3

        # print the components of the reward as a list
        raw_components = [
            alive_award,
            large_actions,
            close_to_target,
            uprightedness,
            x_reward,
            current_dist_y,
            joints_close_to_0,
            ang_vel,
        ]

        # clamp rewards to between -15 and 15
        safe_components = [torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0) for r in raw_components]
        safe_components = [torch.clamp(r, min=-100.0, max=100.0) for r in safe_components]

        out_of_bounds, _ = self._get_dones()
        reward_components = [torch.where(out_of_bounds, torch.zeros_like(r), r) for r in safe_components]

        # print([torch.mean(r).item() for r in reward_components])

        total_reward = torch.sum(torch.stack(reward_components), dim=0)

        self.last_pos = self.current_pos
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # out_of_bounds = False
        # out_of_bounds = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        body_pos_z = self.current_pos[:, 2]  # Use current_pos
        # print(torch.mean(body_pos_z).item())
        threshold = 0.14
        below_threshold = body_pos_z < threshold

        # print num below threshold
        # print(f"Number of envs below threshold: {torch.sum(below_threshold).item()}")

        body_quats = self.robot.data.body_quat_w[:, self._body_idxs, :].flatten(start_dim=-2)
        gravity_local = quat_apply_inverse(
            body_quats, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        )
        titled_too_much = gravity_local[:, 2] > -0.3

        # prevent too fast, flying away etc
        dt = self.cfg.sim.dt * self.cfg.decimation
        lin_vel_w = (self.current_pos - self.last_pos) / dt
        # pythagorean theorem
        too_fast = torch.norm(lin_vel_w, dim=-1) > 15.0

        # too high
        too_high = torch.abs(body_pos_z) > 0.8

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

        self.current_pos[env_ids] = default_root_state[:, :3]
        self.last_pos[env_ids] = self.current_pos[env_ids]

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
