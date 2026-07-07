# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

# import math
# import random
from collections.abc import Sequence

import torch

# import omni.physx.scripts.physicsUtils as physicsUtils
# import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu
from isaaclab.sim.schemas import RigidBodyPropertiesCfg

# from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
# from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.shapes import CuboidCfg, spawn_cuboid

# from isaaclab.sim.spawners.shapes import MeshCfg, spawn_mesh
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_from_euler_xyz, sample_uniform

from .chicken_env_cfg import ChickenEnvCfg


class ChickenEnv(DirectRLEnv):
    cfg: ChickenEnvCfg

    def __init__(self, cfg: ChickenEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._l_joint_dof_idxs = [
            self.robot.find_joints("l_side_hip")[0],
            self.robot.find_joints("l_hip")[0],
            self.robot.find_joints("l_knee")[0],
            self.robot.find_joints("l_ankle")[0],
        ]
        self._r_joint_dof_idxs = [
            self.robot.find_joints("r_side_hip")[0],
            self.robot.find_joints("r_hip")[0],
            self.robot.find_joints("r_knee")[0],
            self.robot.find_joints("r_ankle")[0],
        ]
        self._all_joint_dof_idxs = self._l_joint_dof_idxs + self._r_joint_dof_idxs

        self._body_idxs = self.robot.find_bodies("imu")[0]
        self._foot_idxs = [self.robot.find_bodies("l_foot_pad")[0], self.robot.find_bodies("r_foot_pad")[0]]
        self._l_foot_idxs = self.robot.find_bodies("l_foot_pad")[0]
        self._r_foot_idxs = self.robot.find_bodies("r_foot_pad")[0]

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        self.lower_limits = self.robot.data.joint_pos_limits[:, self._all_joint_dof_idxs, 0]
        self.upper_limits = self.robot.data.joint_pos_limits[:, self._all_joint_dof_idxs, 1]

        self.target_vel = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self.target_horiz_vel = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
        self.target_yaw_rate = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)

        self.min_target_vel = -1.2
        self.max_target_vel = 1.6
        self.min_target_horiz_vel = -0.8
        self.max_target_horiz_vel = 0.8
        self.min_target_yaw_rate = -torch.pi
        self.max_target_yaw_rate = torch.pi

        self.min_freq = 2.5
        self.max_freq = 3.5

        self.target_height = 0.36

        self.current_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.yaw = torch.zeros((self.num_envs,), device=self.device)
        self.lin_vel_w = torch.zeros((self.num_envs, 3), device=self.device)

        self.last_action = torch.zeros((self.num_envs, 8), device=self.device)
        self.last_last_action = torch.zeros((self.num_envs, 8), device=self.device)

        self.history = torch.zeros(
            (self.num_envs, self.cfg.history_length * self.cfg.history_interval, self.cfg.history_size),
            device=self.device,
        )

        self.timing_ref = torch.zeros(
            (self.num_envs, 2), device=self.device
        )  # [tl, tr], expressed in obs as sin(2pitl) etc
        # goes between 0 and 1 at frequency desired to walk
        self.freq = torch.zeros((self.num_envs,), device=self.device)
        self.foot_offsets = torch.tensor([0.0, 0.5], device=self.device)

        # target_bodies = self._body_idxs
        # num_bodies = len(target_bodies)
        # self.forces = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)
        # self.force_time = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)

        self.touching_ground = torch.zeros(self.num_envs, device=self.device)

        self.sim_step_counter = 0

        self.is_teleop = False

    def set_teleop_commands(self, v_x, v_y, yaw_rate, freq_t):
        """Overrides the environment targets with manual inputs."""
        self.target_vel[:] = v_x
        self.target_horiz_vel[:] = v_y
        self.target_yaw_rate[:] = yaw_rate
        self.freq[:] = freq_t

    def _setup_scene(self):
        ground_cfg = CuboidCfg(
            size=(250.0, 250.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.14), roughness=0.1, metallic=0.1),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.95,
                dynamic_friction=0.85,
                restitution=0.0,
            ),
            rigid_props=RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                rigid_body_enabled=True,
            ),
            activate_contact_sensors=True,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        spawn_cuboid(prim_path="/World/ground", cfg=ground_cfg, translation=(0.0, 0.0, -0.5))

        # add lights
        light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0), angle=34.3)
        light_cfg.func("/World/DistantLight", light_cfg, orientation=(-0.61359, 0.78963, 0, 0))
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0), exposure=0.4)
        light_cfg.func("/World/DomeLight", light_cfg, orientation=(-0.30843, 0.30843, 0.63629, 0.63629))

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.imu = Imu(self.cfg.imu_cfg)
        self.scene.sensors["imu"] = self.imu

        self.contact_l = ContactSensor(self.cfg.contact_cfg_l)
        self.scene.sensors["contact_l"] = self.contact_l
        self.contact_r = ContactSensor(self.cfg.contact_cfg_r)
        self.scene.sensors["contact_r"] = self.contact_r

        self.contacts: list[ContactSensor] = []
        for key, value in self.cfg.contact_cfgs.items():
            self.contacts.append(ContactSensor(value))
            self.scene.sensors[key] = self.contacts[len(self.contacts) - 1]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

        if self.sim_step_counter == 25000:
            sim_utils.delete_prim("/World/ground")
            ground_usd_cfg = sim_utils.UsdFileCfg(
                usd_path="/workspace/isaaclab/source/models/env9.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.1, 0.14), roughness=0.1, metallic=0.1
                ),
                activate_contact_sensors=True,
            )
            sim_utils.spawn_from_usd("/World/ground", ground_usd_cfg)

            physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=0.95,
                dynamic_friction=0.85,
                restitution=0.0,
            )
            sim_utils.spawn_rigid_body_material("/World/gripmat", physics_material)
            sim_utils.bind_physics_material("/World/ground", "/World/gripmat")

        force_mag = 5.0 + 5.0 * (min(max(self.sim_step_counter - 20000, 0.0), 5000.0) / 5000.0)
        target_bodies = self._body_idxs
        num_bodies = len(target_bodies)

        force_idxs = torch.rand(self.num_envs, device=self.device) < 1.0 / 400.0
        torques = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)

        if force_idxs.any():
            forces = torch.zeros((self.num_envs, num_bodies, 3), device=self.device)
            forces[force_idxs] = sample_uniform(
                -force_mag, force_mag, (force_idxs.sum(), num_bodies, 3), device=self.device
            )
            #     self.force_time[force_idxs] = torch.round(
            #         sample_uniform(1, 3, (force_idxs.sum(), num_bodies, 3), device=self.device)
            #     )

            #     self.force_time = self.force_time - torch.ones_like(self.force_time)

            # self.forces = self.forces * (self.force_time >= 0)

            self.robot.instantaneous_wrench_composer.set_forces_and_torques(
                forces=forces, torques=torques, body_ids=target_bodies
            )

        if not getattr(self, "is_teleop", False):
            update_random_idxs = torch.rand(self.num_envs, device=self.device) < 1.0 / (120.0 * 10.0)
            if update_random_idxs.any():
                update_env_ids = torch.where(update_random_idxs)[0]

                self.target_vel[update_env_ids] = -sample_uniform(
                    self.min_target_vel,
                    self.max_target_vel,
                    (update_random_idxs.sum(), 1),  # type: ignore
                    device=self.device,
                )
                self.target_horiz_vel[update_env_ids] = sample_uniform(
                    self.min_target_horiz_vel,
                    self.max_target_horiz_vel,
                    (update_random_idxs.sum(), 1),  # type: ignore
                    device=self.device,
                )
                self.target_yaw_rate[update_env_ids] = sample_uniform(
                    self.min_target_yaw_rate,
                    self.max_target_yaw_rate,
                    (update_random_idxs.sum(), 1),  # type: ignore
                    device=self.device,
                )

        # update time
        self.timing_ref = self.timing_ref + self.freq.unsqueeze(1) * 1.0 / 120.0 * self.cfg.decimation
        self.timing_ref = torch.fmod(self.timing_ref, 1.0)

        self.lin_vel_w = self.robot.data.body_com_lin_vel_w[:, self._body_idxs, :].flatten(start_dim=-2)
        self.current_pos = self.robot.data.body_com_pos_w[:, self._body_idxs, :].squeeze(1)

    def _apply_action(self) -> None:
        # apply joint actions
        action_range = (self.upper_limits - self.lower_limits).flatten(start_dim=-2)
        action_midpoint = ((self.upper_limits + self.lower_limits) / 2.0).flatten(start_dim=-2)

        scaled_actions = self.actions[:, :8] * action_range / 2.0 + action_midpoint
        scaled_actions = scaled_actions.unsqueeze(dim=2)
        self.robot.set_joint_position_target(scaled_actions, joint_ids=self._all_joint_dof_idxs)

    def _sin_cos(self, angles: torch.Tensor) -> torch.Tensor:
        return torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(start_dim=-2)

    def _get_observations(self) -> dict:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        body_quats = self.robot.data.body_quat_w[:, self._body_idxs, :].flatten(start_dim=-2)
        self.roll, self.pitch, self.yaw = euler_xyz_from_quat(body_quats)

        real_history = self.history[
            :, list(range(0, self.cfg.history_length * self.cfg.history_interval, self.cfg.history_interval)), :
        ]

        obs_data = (
            self.joint_vel[:, self._l_joint_dof_idxs].flatten(start_dim=-2),
            self.joint_vel[:, self._r_joint_dof_idxs].flatten(start_dim=-2),
            self._sin_cos(self.joint_pos[:, self._l_joint_dof_idxs].flatten(start_dim=-2)),
            self._sin_cos(self.joint_pos[:, self._r_joint_dof_idxs].flatten(start_dim=-2)),
            self._sin_cos(2.0 * torch.pi * (self.timing_ref)),
            self.imu.data.projected_gravity_b,
            self.target_vel,
            self.target_horiz_vel,
            self.target_yaw_rate,
        )
        obs_data = [t.unsqueeze(-1) if t.dim() == 1 else t for t in obs_data]

        obs = torch.cat(
            [*obs_data, real_history.flatten(start_dim=-2)],
            dim=-1,
        )

        self.history = torch.roll(self.history, shifts=1, dims=1)
        self.history[:, 0, :] = torch.cat([*obs_data, self.actions], dim=-1).clone()

        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        self.sim_step_counter += 1

        roll, pitch, yaw = self.roll, self.pitch, self.yaw

        target_vel = self.target_vel.flatten()
        target_horiz_vel = self.target_horiz_vel.flatten()
        target_yaw_rate = self.target_yaw_rate.flatten()

        # project body-frame velocity commands into world frame
        target_vel_x = target_vel * torch.cos(yaw + torch.pi / 2.0) + target_horiz_vel * torch.cos(yaw)
        target_vel_y = target_vel * torch.sin(yaw + torch.pi / 2.0) + target_horiz_vel * torch.sin(yaw)

        vel_mse = (
            torch.square(self.lin_vel_w[:, 0] - target_vel_x) + torch.square(self.lin_vel_w[:, 1] - target_vel_y)
        ) / 2.0

        yaw_rate = self.robot.data.body_com_ang_vel_w[:, self._body_idxs, 2].flatten(
            start_dim=-2
        )  # z-axis = yaw rate, rad/s

        z_vel = self.lin_vel_w[:, 2]

        roll_pitch_vel = self.imu.data.ang_vel_b[:, :2]
        ang_vel = torch.norm(roll_pitch_vel, p=2, dim=1)

        target_pos = self.robot.data.joint_pos_target[:, self._all_joint_dof_idxs]
        joint_limit_violation = (target_pos > self.upper_limits) | (target_pos < self.lower_limits)

        joint_torques = self.robot.data.applied_torque[:, self._all_joint_dof_idxs]
        joint_vels = self.robot.data.joint_vel[:, self._all_joint_dof_idxs]
        joint_acc = self.robot.data.joint_acc[:, self._all_joint_dof_idxs]

        # 2d foot speed
        foot_vel = self.robot.data.body_com_lin_vel_w[:, self._foot_idxs, :2]  # for feet_slip
        foot_speed = torch.norm(foot_vel, dim=-1).flatten(start_dim=-2)

        # Squeeze the trailing body dimension so shapes become (num_envs,) before stacking
        force_l = torch.norm(self.contact_l.data.net_forces_w, dim=-1).squeeze(-1)
        force_r = torch.norm(self.contact_r.data.net_forces_w, dim=-1).squeeze(-1)

        should_up_l = (self.timing_ref[:, 0] > 0.5).float()
        should_up_r = (self.timing_ref[:, 1] > 0.5).float()
        should_down_l = 1.0 - should_up_l
        should_down_r = 1.0 - should_up_r

        contact_weight_l = torch.tanh(force_l / 5.0)  # smoothly ramps 0→1 as force increases
        contact_weight_r = torch.tanh(force_r / 5.0)

        feet_slip = should_down_l * foot_speed[:, 0] + should_down_r * foot_speed[:, 1]

        self.touching_ground = (
            torch
            .stack(
                [torch.norm(t.data.net_forces_w, dim=-1) > 0.0 for t in self.contacts],
                dim=1,
            )
            .flatten(start_dim=-2)
            .any(dim=-1)
        )

        rewards = [
            # task
            torch.exp(-8.0 * vel_mse) * 10.0,
            torch.exp(-0.15 * torch.square(yaw_rate - target_yaw_rate)) * 7.0,
            torch.exp(-8.0 * torch.square(z_vel)) * 1.0,
            torch.exp(-2.0 * torch.square(ang_vel)) * 0.5,
            torch.exp(-0.1 * (torch.square(pitch) + torch.square(roll))) * 12.5,  # 4
            torch.exp(-30.0 * torch.square(self.current_pos[:, 2] - self.target_height)) * 5.0,
            # gait height?
            # contact
            (1.0 - torch.abs(should_down_l - contact_weight_l)) * 6,
            (1.0 - torch.abs(should_down_r - contact_weight_r)) * 6,
            # reg
            feet_slip * -1.0,  # 8
            torch.sum(torch.square(joint_torques), dim=1).flatten() * -1e-3,
            torch.sum(torch.square(joint_acc), dim=1).flatten() * -2.5e-6,
            torch.sum(torch.square(joint_vels), dim=1).flatten() * -2e-5,  # 11
            torch.sum(torch.square(self.last_action - self.actions), dim=-1) * -0.15,
            torch.sum(torch.square(self.last_last_action - 2 * self.last_action + self.actions), dim=-1) * -0.045,
            torch.any(joint_limit_violation, dim=-2).flatten() * -8.0,  # 14
            self.touching_ground * -7.5,
            # survival
            torch.ones(self.num_envs, device=self.device) * 5.0,
        ]

        if self.sim_step_counter % 500 == 0:
            print("  ")
            for i, item in enumerate(rewards):
                print(i, item.mean().item())

        reward = torch.sum(torch.stack(rewards), dim=0)

        # update state for next step
        self.last_last_action = self.last_action.clone()
        self.last_action = self.actions.clone()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Per-environment randomized timeout between max_episode_length and max_episode_length + 120
        rand_offsets = sample_uniform(0.0, 120.0, (self.num_envs,), device=self.device)
        time_out = self.episode_length_buf >= (self.max_episode_length + rand_offsets) - 1

        body_pos_z = self.current_pos[:, 2]
        # threshold = 0.2
        # below_threshold = (body_pos_z < threshold) & (body_pos_z != 0.0)
        # print(self.target_height.mean().item(), body_pos_z.mean().item())
        # print(body_pos_z.mean().item())

        body_quats = self.robot.data.body_quat_w[:, self._body_idxs, :].flatten(start_dim=-2)
        gravity_local = quat_apply_inverse(
            body_quats,
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1),
        )
        tilted_too_much = gravity_local[:, 2] > -0.75

        too_fast = torch.norm(self.lin_vel_w, dim=-1) > 30.0

        too_high = torch.abs(body_pos_z) > 3.5

        # out_of_bounds = tilted_too_much | too_fast | too_high | self.touching_ground
        out_of_bounds = too_fast | too_high
        # out_of_bounds = out_of_bounds & (self.episode_length_buf > 4)
        # out_of_bounds = out_of_bounds * (1.0 - min(max(self.sim_step_counter - 20000, 0.0), 5000.0) / 5000.0)

        if not getattr(self, "is_teleop", False):
            return out_of_bounds, time_out
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        assert env_ids is not None
        env_ids_len = len(env_ids)
        super()._reset_idx(env_ids)

        self.imu.reset(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.current_pos[env_ids] = default_root_state[:, :3]
        self.history[env_ids] = 0.0

        self.freq[env_ids] = sample_uniform(self.min_freq, self.max_freq, (env_ids_len,), device=self.device)
        self.timing_ref[env_ids] = self.foot_offsets

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        random_rotation_roll = sample_uniform(
            -torch.pi * 30.0 / 180.0, torch.pi * 30.0 / 180.0, (env_ids_len, 1), device=self.device
        )  # type: ignore
        random_rotation_pitch = sample_uniform(
            -torch.pi * 30.0 / 180.0, torch.pi * 30.0 / 180.0, (env_ids_len, 1), device=self.device
        )  # type: ignore
        random_rotation_yaw = sample_uniform(-torch.pi, torch.pi, (env_ids_len, 1), device=self.device)  # type: ignore
        euler_rotation = torch.cat(
            [
                random_rotation_roll,
                random_rotation_pitch,
                random_rotation_yaw,
            ],
            dim=-1,
        )
        random_rotation_quat = quat_from_euler_xyz(euler_rotation[:, 0], euler_rotation[:, 1], euler_rotation[:, 2])

        default_root_state[:, 3:7] = random_rotation_quat

        self.last_action[env_ids] = 0.0
        self.last_last_action[env_ids] = 0.0

        if not getattr(self, "is_teleop", False):
            self.target_vel[env_ids] = sample_uniform(
                self.min_target_vel,
                self.max_target_vel,
                (len(env_ids), 1),  # type: ignore
                device=self.device,
            ) * (torch.rand((len(env_ids), 1), device=self.device) > 0.5).float().mul(2.0).sub(1.0)  # type: ignore
            self.target_horiz_vel[env_ids] = sample_uniform(
                self.min_target_horiz_vel,
                self.max_target_horiz_vel,
                (len(env_ids), 1),  # type: ignore
                device=self.device,
            ) * (torch.rand((len(env_ids), 1), device=self.device) > 0.5).float().mul(2.0).sub(1.0)  # type: ignore
            # self.target_yaw_rate[vel_env_ids] = 0.0

            # self.target_vel[yaw_env_ids] = 0.0
            # self.target_horiz_vel[yaw_env_ids] = 0.0
            self.target_yaw_rate[env_ids] = sample_uniform(
                self.min_target_yaw_rate,
                self.max_target_yaw_rate,
                (len(env_ids), 1),  # type: ignore
                device=self.device,
            ) * (torch.rand((len(env_ids), 1), device=self.device) > 0.5).float().mul(2.0).sub(1.0)  # type: ignore
        else:
            self.target_vel[env_ids] = 0
            self.target_horiz_vel[env_ids] = 0
            self.target_yaw_rate[env_ids] = 0

        # self.zero_target_vel[env_ids] = (torch.abs(self.target_vel[env_ids]) < 0.1).float()
        # self.target_vel[env_ids] = self.target_vel[env_ids] * self.zero_target_vel[env_ids]
        # self.target_horiz_vel[env_ids] = self.target_horiz_vel[env_ids] * self.zero_target_vel[env_ids]
        # self.target_yaw_rate[env_ids] = self.target_yaw_rate[env_ids] * self.zero_target_vel[env_ids]

        # self.push_step[env_ids] = 0.0
        # self.push_force[env_ids] = 0.0

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
