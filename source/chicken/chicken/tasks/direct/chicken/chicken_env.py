# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_from_euler_xyz, sample_uniform

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
        self._last_foot_idxs = [self.robot.find_joints("r2")[0], self.robot.find_joints("l2")[0]]

        self._body_idxs = self.robot.find_bodies("body")[0]
        self._left_foot_idx = self.robot.find_bodies("legLeft2")[0]
        self._right_foot_idx = self.robot.find_bodies("legRight2")[0]
        self._foot_idxs = [self.robot.find_bodies("legRight2")[0], self.robot.find_bodies("legLeft2")[0]]

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.target_vel = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self.target_horiz_vel = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self.target_yaw_rate = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)

        self.min_target_vel = 0.2
        self.max_target_vel = 1.0
        self.min_target_horiz_vel = 0
        self.max_target_horiz_vel = 0
        self.min_target_yaw_rate = -torch.pi / 2.0
        self.max_target_yaw_rate = torch.pi / 2.0

        # self.push_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        # self.push_force = torch.zeros((self.num_envs, 1, 3), device=self.device)

        self.pos_range = self.cfg.pos_range
        self.all_pos_range = torch.tensor(self.pos_range + self.pos_range, device=self.device) / 2.0

        # to fix IMU glitch:
        self.imu._dt = self.sim.get_physics_dt()

        self.current_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.last_pos = torch.zeros_like(self.current_pos)

        # FIX 3a: Use consistent (num_envs,) shape for yaw/last_yaw.
        # The original (num_envs, 1) init conflicted with the (num_envs,) tensors
        # returned by euler_xyz_from_quat, causing shape drift across steps.
        self.yaw = torch.zeros((self.num_envs,), device=self.device)
        self.last_yaw = torch.zeros_like(self.yaw)

        self.vel_mask = torch.zeros((self.num_envs), dtype=torch.bool, device=self.device)

        self.last_action = torch.zeros((self.num_envs, 8), device=self.device)
        self.start_rotation = torch.zeros((self.num_envs, 4), device=self.device)

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

        force_mag = 25

        force_idxs = torch.rand(self.num_envs, device=self.device) < 1.0 / 100.0
        forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        torques = torch.zeros((self.num_envs, 1, 3), device=self.device)
        if force_idxs.any():
            forces[force_idxs] = sample_uniform(-force_mag, force_mag, (force_idxs.sum(), 1, 3), device=self.device)

            self.robot.instantaneous_wrench_composer.set_forces_and_torques(
                forces=forces, torques=torques, body_ids=self._body_idxs
            )

        update_random_idxs = torch.rand(self.num_envs, device=self.device) < 1.0 / 120.0
        if update_random_idxs.any():
            update_env_ids = torch.where(update_random_idxs)[0]
            vel_mask = torch.rand(len(update_env_ids), device=self.device) < 0.5
            vel_env_ids = update_env_ids[vel_mask]
            yaw_env_ids = update_env_ids[~vel_mask]
            self.vel_mask[vel_env_ids] = True
            self.vel_mask[yaw_env_ids] = False

            self.target_vel[vel_env_ids] = sample_uniform(
                self.min_target_vel,
                self.max_target_vel,
                (vel_mask.sum(), 1),  # type: ignore
                device=self.device,
            )
            self.target_horiz_vel[vel_env_ids] = sample_uniform(
                self.min_target_horiz_vel,
                self.max_target_horiz_vel,
                (vel_mask.sum(), 1),  # type: ignore
                device=self.device,
            )
            self.target_yaw_rate[vel_env_ids] = 0.0

            self.target_vel[yaw_env_ids] = self.min_target_vel
            self.target_horiz_vel[yaw_env_ids] = 0.0
            self.target_yaw_rate[yaw_env_ids] = sample_uniform(
                self.min_target_yaw_rate,
                self.max_target_yaw_rate,
                ((~vel_mask).sum(), 1),  # type: ignore
                device=self.device,
            )

    def _apply_action(self) -> None:
        # apply joint actions
        scaled_actions = self.actions[:, :8] * self.all_pos_range
        scaled_actions = scaled_actions.unsqueeze(dim=2)
        self.robot.set_joint_position_target(scaled_actions, joint_ids=self._all_joint_dof_idxs)

    def _sin_cos(self, angles: torch.Tensor) -> torch.Tensor:
        return torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(start_dim=-2)

    def _get_observations(self) -> dict:
        self.lin_acc_b = torch.nan_to_num(self.imu.data.lin_acc_b, nan=0.0, posinf=0.0, neginf=0.0)
        self.ang_acc_b = torch.nan_to_num(self.imu.data.ang_acc_b, nan=0.0, posinf=0.0, neginf=0.0)

        self.lin_vel_b = torch.nan_to_num(self.imu.data.lin_vel_b, nan=0.0, posinf=0.0, neginf=0.0)
        self.ang_vel_b = torch.nan_to_num(self.imu.data.ang_vel_b, nan=0.0, posinf=0.0, neginf=0.0)

        self.lin_vel_w = self.robot.data.root_lin_vel_w

        body_quats = self.robot.data.body_quat_w[:, self._body_idxs, :].flatten(start_dim=-2)
        roll, pitch, yaw = euler_xyz_from_quat(body_quats)
        self.yaw = yaw.clone()

        obs = torch.cat(
            (
                self.joint_vel[:, self._l_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1),
                self.joint_vel[:, self._r_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1),
                self._sin_cos(self.joint_pos[:, self._l_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1)),
                self._sin_cos(self.joint_pos[:, self._r_joint_dof_idxs].flatten(start_dim=-2).unsqueeze(dim=1)),
                self.lin_acc_b.unsqueeze(dim=1),
                self.ang_acc_b.unsqueeze(dim=1),
                self._sin_cos(yaw.unsqueeze(dim=1)).unsqueeze(dim=1),
                self.target_vel.unsqueeze(dim=1),
                self.target_horiz_vel.unsqueeze(dim=1),
                self.target_yaw_rate.unsqueeze(dim=1),
            ),
            dim=-1,
        )

        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        obs = torch.clamp(obs, min=-100.0, max=100.0)

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        self.current_pos = self.robot.data.body_pos_w[:, self._body_idxs, :].squeeze(1)

        # ---- Shared computation (compute body_quats once) ----
        body_quats = self.robot.data.body_quat_w[:, self._body_idxs, :].flatten(start_dim=-2)

        # alive reward
        alive_award = torch.ones(self.num_envs, dtype=torch.float32, device=self.device) * 1.0

        # penalize large joint velocities
        large_vels = -torch.sum(torch.square(self.joint_vel), dim=-1) * 0.0005

        # uprightedness: reward staying upright
        gravity_local = quat_apply_inverse(
            body_quats,
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1),
        )
        uprightedness = torch.exp(40 * (torch.pow(-gravity_local[:, 2] - 1, 1.0)))

        # yaw from quaternion (needed for vel target projection to world frame)
        roll, pitch, yaw = euler_xyz_from_quat(body_quats)
        self.yaw = yaw.clone()

        # height reward
        body_pos_z = self.current_pos[:, 2]
        target_height = 0.1
        height = torch.exp(-torch.square(body_pos_z - target_height) * 50)

        # FIX 4: Reduce delta_reward weight so it doesn't dominate.
        # Original weight 0.5 was large enough to create a "stand still with constant
        # zero actions" local optimum. Reduced to 0.1 so it's a tie-breaker, not a goal.
        movement_delta = torch.sum(torch.square(self.actions - self.last_action), dim=-1)
        delta_reward = torch.exp(-movement_delta * 0.5) * 0.1

        # ---- Velocity / yaw-rate tracking ----
        lin_vel_w = self.robot.data.root_lin_vel_w  # (num_envs, 3), world frame
        yaw_rate = self.robot.data.root_ang_vel_w[:, 2]  # z-axis = yaw rate, rad/s

        target_vel = self.target_vel.flatten()
        target_horiz_vel = self.target_horiz_vel.flatten()
        target_yaw_rate = self.target_yaw_rate.flatten()

        # project body-frame velocity commands into world frame
        target_vel_x = target_vel * torch.cos(yaw) + target_horiz_vel * torch.cos(yaw + torch.pi / 2.0)
        target_vel_y = target_vel * torch.sin(yaw) + target_horiz_vel * torch.sin(yaw + torch.pi / 2.0)

        vel_mean_square_error = (
            torch.square(lin_vel_w[:, 0] - target_vel_x) + torch.square(lin_vel_w[:, 1] - target_vel_y)
        ) / 2.0

        vel_reward = torch.exp(-vel_mean_square_error * 20.0)
        yaw_reward = torch.exp(-torch.square(yaw_rate - target_yaw_rate) * 15.0)

        raw_components = [
            alive_award,  # [0]
            large_vels,  # [1]
            uprightedness * 10.0,  # [2]
            vel_reward * 10.0,  # [3]
            yaw_reward * 10.0,  # [4]
            height * 5.0,  # [5]
            delta_reward,  # [6]
        ]

        safe_components = [torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0) for r in raw_components]
        safe_components = [torch.clamp(r, min=-1000.0, max=1000.0) for r in safe_components]

        out_of_bounds, _ = self._get_dones()
        reward_components = [torch.where(out_of_bounds, torch.zeros_like(r), r) for r in safe_components]

        # FIX 5: Reward logging had wrong component indices.
        # reward_components[2] is uprightedness, NOT vel_reward.
        # reward_components[3] is vel_reward, NOT yaw_reward.
        self.extras["reward_components"] = {
            "alive_award": reward_components[0].mean().item(),
            "large_vels": reward_components[1].mean().item(),
            "uprightedness": reward_components[2].mean().item(),
            "vel_reward": reward_components[3].mean().item(),
            "yaw_reward": reward_components[4].mean().item(),
            "height": reward_components[5].mean().item(),
            "delta_reward": reward_components[6].mean().item(),
        }

        total_reward = torch.sum(torch.stack(reward_components), dim=0)

        # update state for next step
        self.last_pos = self.current_pos.clone()
        self.last_yaw = self.yaw.clone()
        self.last_action = self.actions.clone()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        body_pos_z = self.current_pos[:, 2]
        threshold = 0.0725
        below_threshold = body_pos_z < threshold
        # print(body_pos_z.mean().item())

        body_quats = self.robot.data.body_quat_w[:, self._body_idxs, :].flatten(start_dim=-2)
        gravity_local = quat_apply_inverse(
            body_quats,
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1),
        )
        tilted_too_much = gravity_local[:, 2] > -0.875

        # FIX 2 (continued): Use physics velocity instead of position differencing.
        # The original (current_pos - last_pos) / dt was stale on the very first step
        # after a reset and accumulated noise from position quantization.
        too_fast = torch.norm(self.robot.data.root_lin_vel_w, dim=-1) > 40.0

        too_high = torch.abs(body_pos_z) > 2.0

        xy_speed = torch.norm(self.robot.data.root_lin_vel_w[:, :2], dim=-1)

        # kill it if its not moving!!!!!!!!!!
        too_slow = xy_speed < 0.1
        grace_period_passed = self.episode_length_buf > 15
        lazy_death = too_slow & grace_period_passed

        out_of_bounds = tilted_too_much | too_fast | too_high | below_threshold | lazy_death
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self.imu.reset(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.current_pos[env_ids] = default_root_state[:, :3]
        self.last_pos[env_ids] = self.current_pos[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        random_rotation_yaw = sample_uniform(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device)  # type: ignore
        euler_rotation = torch.cat(
            [
                torch.zeros((len(env_ids), 2), device=self.device),  # type: ignore
                random_rotation_yaw,
            ],
            dim=-1,
        )
        random_rotation_quat = quat_from_euler_xyz(euler_rotation[:, 0], euler_rotation[:, 1], euler_rotation[:, 2])

        default_root_state[:, 3:7] = random_rotation_quat

        # FIX 3b: Reset last_yaw and last_action for the newly-reset environments.
        # Without this, the first step after a reset computed yaw_rate from stale yaw
        # values (potentially ±π away from the new orientation), producing a huge
        # spurious spike that poisoned the yaw_reward for that step. Similarly,
        # last_action carried over gait actions from the previous episode, making
        # delta_reward wrong on the first step.
        self.last_yaw[env_ids] = random_rotation_yaw.squeeze(-1)
        self.last_action[env_ids] = 0.0

        vel_mask = torch.rand(len(env_ids), device=self.device) < 0.5  # type: ignore
        vel_env_ids = env_ids[vel_mask]  # type: ignore
        yaw_env_ids = env_ids[~vel_mask]  # type: ignore
        self.vel_mask[vel_env_ids] = True
        self.vel_mask[yaw_env_ids] = False

        self.target_vel[vel_env_ids] = sample_uniform(
            self.min_target_vel,
            self.max_target_vel,
            (vel_mask.sum(), 1),  # type: ignore
            device=self.device,
        )
        self.target_horiz_vel[vel_env_ids] = sample_uniform(
            self.min_target_horiz_vel,
            self.max_target_horiz_vel,
            (vel_mask.sum(), 1),  # type: ignore
            device=self.device,
        )
        self.target_yaw_rate[vel_env_ids] = 0.0

        self.target_vel[yaw_env_ids] = self.min_target_vel
        self.target_horiz_vel[yaw_env_ids] = 0.0
        self.target_yaw_rate[yaw_env_ids] = sample_uniform(
            self.min_target_yaw_rate,
            self.max_target_yaw_rate,
            ((~vel_mask).sum(), 1),  # type: ignore
            device=self.device,
        )

        # self.push_step[env_ids] = 0.0
        # self.push_force[env_ids] = 0.0

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
