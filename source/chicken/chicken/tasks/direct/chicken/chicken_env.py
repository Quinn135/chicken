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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, sample_uniform

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
        self._all_joint_dof_idxs = self._l_joint_dof_idxs + self._r_joint_dof_idxs

        self._body_idxs = self.robot.find_bodies("body")[0]
        self._foot_idxs = [self.robot.find_bodies("legRight2")[0], self.robot.find_bodies("legLeft2")[0]]

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.target_compass = torch.zeros((self.num_envs, 1), device=self.device)

        self.pos_range = self.cfg.pos_range
        self.all_pos_range = torch.tensor(self.pos_range + self.pos_range, device=self.device) / 2.0

        # to fix IMU glitch:
        self.imu._dt = self.sim.get_physics_dt()

        # self.pos = torch.zeros((self.num_envs, 3), device=self.device)
        # self.last_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # self.pos = torch.zeros((self.num_envs, 3), device=self.device)
        # get pos of the "body" element (_body_idxs)
        self.current_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_pos = torch.zeros_like(self.current_pos)

        # self.last_actions = torch.zeros((self.num_envs, self.cfg.history_length, 8 * 2), device=self.device)

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

        forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        torques = torch.zeros_like(forces)

        # 1/400 chance to push
        push_envs = torch.rand(self.num_envs, device=self.device) < 1.0 / 200.0

        if push_envs.any():
            max_force = 50.0
            forces[push_envs] = sample_uniform(-max_force, max_force, forces[push_envs].shape, device=self.device)

            self.robot.instantaneous_wrench_composer.set_forces_and_torques(
                forces=forces, torques=torques, body_ids=self._body_idxs
            )

    def _apply_action(self) -> None:
        # apply joint actions
        # for i, dof_idx in enumerate(self._l_joint_dof_idxs):
        #     # times the range of each joint
        #     self.robot.set_joint_position_target(
        #         (self.actions[:, i] * self.pos_range[i]).unsqueeze(dim=1), joint_ids=dof_idx
        #     )
        # for i, dof_idx in enumerate(self._r_joint_dof_idxs):
        #     self.robot.set_joint_position_target(
        #         (self.actions[:, i + 4] * self.pos_range[i]).unsqueeze(dim=1), joint_ids=dof_idx
        # )

        scaled_actions = self.actions[:, :8] * self.all_pos_range
        scaled_actions = scaled_actions.unsqueeze(dim=2)
        # ).unsqueeze(dim=1)
        self.robot.set_joint_position_target(scaled_actions, joint_ids=self._all_joint_dof_idxs)

        # ones_to_update = self.episode_length_buf % 3 == 0

        # # self.last_actions[ones_to_update, 0, :] = self.actions[ones_to_update]
        # self.last_actions[ones_to_update] = torch.roll(self.last_actions[ones_to_update], shifts=1, dims=1)
        # self.last_actions[ones_to_update, 0, :] = self._sin_cos(
        #     torch.stack(
        #         (
        #             self.joint_pos[ones_to_update, self._l_joint_dof_idxs].transpose(0, 1),
        #             self.joint_pos[ones_to_update, self._r_joint_dof_idxs].transpose(0, 1),
        #         ),
        #         dim=1,
        #     ).flatten(start_dim=-2)
        # )

    def _sin_cos(self, angles: torch.Tensor) -> torch.Tensor:
        return torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(
            start_dim=-2
        )  # just flatten the last two dims

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

        body_quats = self.robot.data.body_quat_w[:, self._body_idxs, :].flatten(start_dim=-2)
        roll, pitch, yaw = euler_xyz_from_quat(body_quats)
        self.compass = self._sin_cos(yaw.unsqueeze(dim=1))

        obs = torch.cat(
            (
                self.joint_vel[:, self._l_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1),
                self.joint_vel[:, self._r_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1),
                self._sin_cos(self.joint_pos[:, self._l_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1)),
                self._sin_cos(self.joint_pos[:, self._r_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1)),
                self.lin_acc_b.unsqueeze(dim=1),
                self.ang_acc_b.unsqueeze(dim=1),
                self.compass.unsqueeze(dim=1),
                # self.lin_vel_b.unsqueeze(dim=1),
                # self.lin_vel_w.unsqueeze(dim=1),
                # self.ang_vel_b.unsqueeze(dim=1),
                # self.last_actions.flatten(start_dim=-2).unsqueeze(dim=1),
            ),
            dim=-1,
        )

        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = torch.clamp(obs, min=-100.0, max=100.0)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # use e^(-x^2) for rewards
        # use -abs and -square for penalties

        self.current_pos = self.robot.data.body_pos_w[:, self._body_idxs, :].squeeze(1)

        # total_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        alive_award = torch.ones(self.num_envs, dtype=torch.float32, device=self.device) * 10.0

        # disincentivize large vels w/ self.joint_vel
        large_vels = -torch.sum(torch.pow(torch.abs(self.joint_vel), 0.5), dim=-1) * 0.04
        large_vels = torch.nan_to_num(large_vels, nan=0.0, posinf=0.0, neginf=0.0)
        large_vels = torch.clamp(large_vels, min=-25.0, max=0.0)

        large_actions = -torch.sum(torch.pow(torch.abs(self.actions), 2), dim=-1) * 10.0
        large_actions = torch.nan_to_num(large_actions, nan=0.0, posinf=0.0, neginf=0.0)

        # incentivize joints_pos being close to default
        joints_close_to_0 = -torch.sum(torch.abs(self.joint_pos), dim=-1) * 0.05

        # reward it being close to target height
        body_pos_z = self.current_pos[:, 2]  # Use current_pos
        target_height = 0.25
        close_to_target = torch.exp(-torch.square(body_pos_z - target_height) * 25) * 15

        # reward uprightedness
        body_quats = self.robot.data.body_quat_w[:, self._body_idxs, :].flatten(start_dim=-2)
        gravity_local = quat_apply_inverse(
            body_quats, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        )
        # uprightedness = torch.pow(-gravity_local[:, 2], 5) * 20.0 * torch.sign(-gravity_local[:, 2])
        uprightedness = torch.pow(-gravity_local[:, 2], 5) * 20.0

        # # motivate walking!
        dt = self.cfg.sim.dt * self.cfg.decimation

        current_vel_x = (self.current_pos[:, 0] - self.last_pos[:, 0]) / dt
        current_vel_y = (self.current_pos[:, 1] - self.last_pos[:, 1]) / dt

        target_vel = -0.3  # m/s
        target_vel_x = (target_vel * torch.cos(self.target_compass)).flatten(start_dim=-2)
        target_vel_y = (target_vel * torch.sin(self.target_compass)).flatten(start_dim=-2)

        vel_mean_square_error = torch.square(current_vel_x - target_vel_x) + torch.square(current_vel_y - target_vel_y)
        # Turns an error into a smooth reward.
        # The '5.0' is a tuning factor; higher means it has to be more precise to get points.
        vel_reward = torch.exp(-vel_mean_square_error * 45.0) * 100.0

        # vel_reward_x = torch.exp(-torch.square(current_vel_x - target_vel_x) * 10) * 200

        # # punish vertical velocity
        # vertical_vel = -torch.clamp(torch.abs(self.lin_vel_w[:, 2]) * 5.0, min=0.0, max=50.0)

        # disincentivize movement
        # movement = -torch.clamp(torch.sum(torch.abs(self.lin_vel_w), dim=-1) * 20.0, min=0.0, max=10.0)

        # incentivize feet being up
        # z of 0.22 is the height at the ground
        # so let's make the reward positive when the feet are above 0.22
        # foot_z = self.robot.data.body_pos_w[:, self._foot_idxs, 2]
        # feet_up = torch.clamp(foot_z - 0.22, min=0.0) * 100.0
        # feet_up = torch.flatten(feet_up, start_dim=-2)
        # feet_up = torch.sum(feet_up, dim=-1)
        # print(feet_up.shape)

        # rotating bad
        ang_vel = -torch.sum(torch.abs(self.ang_vel_b), dim=-1) * 10

        # print the components of the reward as a list
        raw_components = [
            alive_award,
            large_vels,
            large_actions,
            close_to_target,
            uprightedness,
            vel_reward,
            # vertical_vel,
            joints_close_to_0,
            ang_vel,
            # feet_up,
        ]

        # clamp rewards to between -15 and 15
        safe_components = [torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0) for r in raw_components]
        safe_components = [torch.clamp(r, min=-1000.0, max=1000.0) for r in safe_components]

        out_of_bounds, _ = self._get_dones()
        reward_components = [torch.where(out_of_bounds, torch.zeros_like(r), r) for r in safe_components]

        # print([torch.mean(r).item() for r in reward_components])

        total_reward = torch.sum(torch.stack(reward_components), dim=0)

        # total_reward = torch.clamp(total_reward, min=0.0)

        self.last_pos = self.current_pos.clone()
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # out_of_bounds = False
        # out_of_bounds = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        body_pos_z = self.current_pos[:, 2]  # Use current_pos
        # print(torch.mean(body_pos_z).item())
        threshold = 0.135
        below_threshold = body_pos_z < threshold

        # print num below threshold
        # print(f"Number of envs below threshold: {torch.sum(below_threshold).item()}")

        body_quats = self.robot.data.body_quat_w[:, self._body_idxs, :].flatten(start_dim=-2)
        gravity_local = quat_apply_inverse(
            body_quats, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        )
        titled_too_much = gravity_local[:, 2] > 0.0

        # prevent too fast, flying away etc
        dt = self.cfg.sim.dt * self.cfg.decimation
        lin_vel_w = (self.current_pos - self.last_pos) / dt
        # pythagorean theorem
        too_fast = torch.norm(lin_vel_w, dim=-1) > 20.0

        # too high
        too_high = torch.abs(body_pos_z) > 0.4

        out_of_bounds = titled_too_much | too_fast | too_high | below_threshold
        # out_of_bounds = titled_too_much | too_high | too_fast
        # out_of_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.imu.reset(env_ids)
        # self.last_actions[env_ids] = 0.0
        # self.last_actions = torch.zeros((self.num_envs, self.cfg.history_length, 8 * 2), device=self.device)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.current_pos[env_ids] = default_root_state[:, :3]
        self.last_pos[env_ids] = self.current_pos[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.target_compass[env_ids] = sample_uniform(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device)

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
