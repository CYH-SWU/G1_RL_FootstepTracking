# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_rotate

from .mdp.step_sequence import StepSequenceGenerator, WalkModes
from .mdp.config import G1EnvConfig

from .mdp.rewards import (
    foot_frc_clock_reward,
    foot_vel_clock_reward,
    body_orient_reward,
    height_reward,
    upper_body_stability_reward,
    step_reward,
    action_smoothness_reward,
    torque_smoothness_reward,
    posture_error_reward,
)

def quat_rotate_inv(quat, vec):
    q_conj = quat.clone()
    q_conj[:, 1:] = -q_conj[:, 1:]
    return quat_rotate(q_conj, vec)

def quat_to_euler_xyz(quat):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw

class G1FootstepTrackingEnv(DirectRLEnv):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._config = G1EnvConfig()
        self._step_gen = StepSequenceGenerator(
            self._config.step_length,
            self._config.step_width,
            self._config.total_duration,
            self._config.swing_duration,
            self._config.stance_duration,
        )
        self._left_foot_name = "left_ankle_roll_link"
        self._right_foot_name = "right_ankle_roll_link"
        self._pelvis_name = "pelvis"
        self.step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.mode = None
        self.sequence = None
        self.t1 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.t2 = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.target_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.target_reached_frames = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.difficulty = 0.0
        self.smooth_target = np.zeros((self.num_envs, 12))  # 保持 numpy 数组
        self.filtered_left_force = torch.zeros(self.num_envs, device=self.device)
        self.filtered_right_force = torch.zeros(self.num_envs, device=self.device)
        self.force_alpha = 0.15
        self.force_deadband = 5.0
        self._fixed_mode = None
        self.max_force = torch.zeros(self.num_envs, device=self.device)
        self.last_action = None
        self.last_torque = None
        self._prev_action = None
        self._prev_torque = None
        self._actions = None

        self._enable_domain_randomization = True
        self._training = True

        self._robot = None
        self._left_foot_id = None
        self._right_foot_id = None
        self._pelvis_id = None
        self._torso_id = None
        self._contact_sensor = None

        self._sequences = [None] * self.num_envs
        self._mode = [None] * self.num_envs

    def set_mode(self, mode):
        self._fixed_mode = mode

    def set_training(self, training: bool):
        self._training = training

    def _ensure_robot_init(self):
        if self._robot is None:
            self._robot = self.scene["robot"]
        if self._left_foot_id is None:
            self._left_foot_id = self._robot.body_names.index(self._left_foot_name)
            self._right_foot_id = self._robot.body_names.index(self._right_foot_name)
            self._pelvis_id = self._robot.body_names.index("pelvis")
            self._torso_id = self._robot.body_names.index("torso_link")
        if self._contact_sensor is None:
            try:
                self._contact_sensor = self.scene["contact_sensor"]
            except KeyError:
                self._contact_sensor = None

    def _get_pelvis_yaw(self):
        _, _, yaw = quat_to_euler_xyz(self._robot.data.root_quat_w)
        return yaw

    def _get_foot_forces(self):
        if self._contact_sensor is None:
            return torch.zeros(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device)
        force_data = self._contact_sensor.data.net_forces_w
        left_force = force_data[:, 0, 2]
        right_force = force_data[:, 1, 2]
        return left_force, right_force

    def _get_joint_torques(self):
        all_torques = self._robot.data.applied_torque
        joint_names_target = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]
        indices = [self._robot.joint_names.index(name) for name in joint_names_target]
        return all_torques[:, indices]

    def _randomize_dynamics(self):
        if not self._enable_domain_randomization:
            return
        mass_scale = torch.rand(self.num_envs, device=self.device) * 0.2 + 0.9
        self._robot.data.body_mass *= mass_scale.unsqueeze(-1)
        total_mass = self._robot.data.body_mass.sum(dim=-1)
        self.max_force = total_mass * 9.81 * 0.5
        self._robot.write_data_to_sim()

    def _reset(self):
        self._ensure_robot_init()
        self._randomize_dynamics()

        for i in range(self.num_envs):
            if self._fixed_mode is not None:
                mode = self._fixed_mode
            else:
                mode = np.random.choice(
                    [WalkModes.STANDING, WalkModes.CURVED, WalkModes.BACKWARD,
                     WalkModes.LATERAL, WalkModes.FORWARD],
                    p=self._config.mode_probs,
                )
            self._mode[i] = mode

            step_height = 0.0
            if mode == WalkModes.FORWARD:
                max_h = 0.05 * max(0.0, (self.difficulty - 0.273) / (1.0 - 0.273))
                step_height = np.random.choice([-max_h, max_h])

            num_steps = 20
            if mode == WalkModes.STANDING:
                num_steps = 1
            elif mode == WalkModes.CURVED:
                num_steps = 25

            phase = np.random.choice([0.0, 0.5])
            self.phase[i] = phase
            local_seq = self._step_gen.generate(mode, phase, num_steps, step_height)
            left_pos = self._robot.data.body_pos_w[i, self._left_foot_id].cpu().numpy()
            right_pos = self._robot.data.body_pos_w[i, self._right_foot_id].cpu().numpy()
            root_yaw = self._get_pelvis_yaw()[i].item()
            world_seq = self._step_gen.transform_to_world(local_seq, left_pos, right_pos, root_yaw)
            self._sequences[i] = world_seq

            if len(world_seq) > 0:
                first_step = world_seq[0]
                pelvis_pos = np.array([
                    first_step[0] - 0.05,
                    first_step[1],
                    first_step[2] + self._config.nominal_pelvis_height
                ])
                pelvis_quat = R.from_euler('z', first_step[3]).as_quat()
                self._robot.data.root_pos_w[i] = torch.tensor(pelvis_pos, device=self.device)
                self._robot.data.root_quat_w[i] = torch.tensor([pelvis_quat[3], pelvis_quat[0], pelvis_quat[1], pelvis_quat[2]], device=self.device)

            self.t1[i] = 0
            self.t2[i] = min(1, len(world_seq)-1) if len(world_seq) > 1 else 0
            self.target_reached[i] = False
            self.target_reached_frames[i] = 0

        self.step_counter[:] = 0
        self.smooth_target = np.zeros((self.num_envs, 12))
        self.filtered_left_force[:] = 0.0
        self.filtered_right_force[:] = 0.0
        self.max_force = self._robot.data.body_mass.sum(dim=-1) * 9.81 * 0.5

        self._robot.write_data_to_sim()
        return

    def _pre_physics_step(self, actions):
        self._ensure_robot_init()
        self._prev_action = self.last_action
        self._prev_torque = self.last_torque
        self._actions = actions
        self.last_action = actions.cpu().numpy() if torch.is_tensor(actions) else np.array(actions)
        self._apply_action()

    def _apply_action(self):
        if self._actions is None:
            return
        action_np = self._actions.cpu().numpy() if torch.is_tensor(self._actions) else np.array(self._actions)
        raw_target = self._config.nominal_angles + action_np * self._config.action_scale
        smooth = self._config.action_smoothing
        self.smooth_target = smooth * raw_target + (1 - smooth) * self.smooth_target
        target_qpos = self.smooth_target

        joint_names_target = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]
        for i, name in enumerate(joint_names_target):
            idx = self._robot.joint_names.index(name)
            self._robot.data.joint_pos_target[:, idx] = torch.tensor(target_qpos[:, i], dtype=torch.float32, device=self.device)
        self._robot.write_data_to_sim()

    def _post_physics_step(self):
        self._ensure_robot_init()
        self.step_counter += 1
        dt = self.cfg.sim.dt * self.cfg.decimation
        total_duration = self._config.total_duration
        self.phase = (self.step_counter.float() * dt % total_duration) / total_duration

        self.last_torque = self._get_joint_torques().cpu().numpy()

        for i in range(self.num_envs):
            mode = self._mode[i]
            seq = self._sequences[i]
            if mode != WalkModes.STANDING and seq is not None and len(seq) > 0 and self.t1[i] < len(seq):
                target_pos = seq[self.t1[i]][:3]
                left_pos = self._robot.data.body_pos_w[i, self._left_foot_id].cpu().numpy()
                right_pos = self._robot.data.body_pos_w[i, self._right_foot_id].cpu().numpy()
                l_dist = np.linalg.norm(left_pos - target_pos)
                r_dist = np.linalg.norm(right_pos - target_pos)
                if l_dist < self._config.target_radius or r_dist < self._config.target_radius:
                    self.target_reached[i] = True
                    self.target_reached_frames[i] += 1
                else:
                    self.target_reached[i] = False
                    self.target_reached_frames[i] = 0

                delay_frames = int(np.floor(self._config.swing_duration / self._config.control_dt))
                if self.target_reached[i] and self.target_reached_frames[i] >= delay_frames:
                    self.t1[i] = self.t2[i]
                    self.t2[i] = min(self.t2[i] + 1, len(seq) - 1)
                    self.target_reached[i] = False
                    self.target_reached_frames[i] = 0

    def _get_observations(self):
        self._ensure_robot_init()
        qpos = self._robot.data.joint_pos
        qvel = self._robot.data.joint_vel
        joint_names_target = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]
        joint_indices = [self._robot.joint_names.index(name) for name in joint_names_target]
        joint_angles = qpos[:, joint_indices]
        joint_vels = qvel[:, joint_indices]

        pelvis_z = self._robot.data.root_pos_w[:, 2]
        left_foot_z = self._robot.data.body_pos_w[:, self._left_foot_id, 2]
        right_foot_z = self._robot.data.body_pos_w[:, self._right_foot_id, 2]
        foot_z = torch.min(left_foot_z, right_foot_z) - self._config.foot_ankle_offset
        pelvis_height = pelvis_z - foot_z

        actor_obs_list = []
        for i in range(self.num_envs):
            seq = self._sequences[i]
            if seq is not None and len(seq) > 0:
                t1 = self.t1[i].item()
                t2 = self.t2[i].item()
                t1_w = seq[t1]
                t2_w = seq[t2]
                pelvis_pos = self._robot.data.root_pos_w[i].cpu().numpy()
                quat = self._robot.data.root_quat_w[i].cpu().numpy()
                r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
                R_wt = r.inv().as_matrix()
                t1_local = R_wt @ (t1_w[:3] - pelvis_pos)
                t2_local = R_wt @ (t2_w[:3] - pelvis_pos)
                pelvis_yaw = self._get_pelvis_yaw()[i].item()
                t1_yaw = t1_w[3] - pelvis_yaw
                t2_yaw = t2_w[3] - pelvis_yaw
                foot_dx, foot_dy, foot_dz = t1_local
                next_dx, next_dy, next_dz = t2_local
                foot_yaw = np.arctan2(np.sin(t1_yaw), np.cos(t1_yaw))
                next_yaw = np.arctan2(np.sin(t2_yaw), np.cos(t2_yaw))
            else:
                foot_dx = foot_dy = foot_dz = 0.0
                next_dx = next_dy = next_dz = 0.0
                foot_yaw = next_yaw = 0.0
            phase = self.phase[i].item()
            phase_signal = np.array([np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)])
            quat = self._robot.data.root_quat_w[i].cpu().numpy()
            r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            euler = r.as_euler('xyz')
            roll, pitch, yaw = euler
            pelvis_angvel = self._robot.data.root_ang_vel_w[i].cpu().numpy()
            actor_obs = np.concatenate([
                joint_angles[i].cpu().numpy(),
                joint_vels[i].cpu().numpy(),
                [pelvis_height[i].item()],
                [foot_dx, foot_dy, foot_dz],
                [next_dx, next_dy, next_dz],
                [foot_yaw, next_yaw],
                phase_signal,
                [roll, pitch, yaw],
                pelvis_angvel,
            ])
            actor_obs_list.append(actor_obs)
        actor_obs = np.stack(actor_obs_list, axis=0)

        if self._training and self._enable_domain_randomization:
            noise_std = 0.01
            actor_obs += np.random.normal(0, noise_std, actor_obs.shape)
            actor_obs = np.clip(actor_obs, -10.0, 10.0)

        actor_obs_tensor = torch.tensor(actor_obs, dtype=torch.float32, device=self.device)

        left_force, right_force = self._get_foot_forces()
        norm_left_frc = torch.clamp(left_force / (self.max_force + 1e-6), -1.0, 1.0)
        norm_right_frc = torch.clamp(right_force / (self.max_force + 1e-6), -1.0, 1.0)
        lin_vel = self._robot.data.root_lin_vel_w
        norm_lin_vel = torch.clamp(lin_vel / 2.0, -1.0, 1.0)
        torques = self._get_joint_torques()
        max_torques = torch.tensor([88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50], dtype=torch.float32, device=self.device)
        norm_torques = torch.clamp(torques / (max_torques + 1e-6), -1.0, 1.0)
        priv = torch.cat([
            norm_left_frc.unsqueeze(-1),
            norm_right_frc.unsqueeze(-1),
            norm_lin_vel,
            norm_torques,
        ], dim=-1)
        norm_actor = actor_obs_tensor / 10.0
        critic_obs = torch.cat([norm_actor, priv], dim=-1).to(torch.float32)

        return {
            "policy": actor_obs_tensor,
            "critic": critic_obs,
        }

    def _get_rewards(self):
        rewards = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.num_envs):
            rewards[i] = self._compute_reward_value(i)
        return rewards

    def _compute_reward_value(self, env_idx):
        left_pos = self._robot.data.body_pos_w[env_idx, self._left_foot_id].cpu().numpy()
        right_pos = self._robot.data.body_pos_w[env_idx, self._right_foot_id].cpu().numpy()
        left_vel = self._robot.data.body_lin_vel_w[env_idx, self._left_foot_id].cpu().numpy()
        right_vel = self._robot.data.body_lin_vel_w[env_idx, self._right_foot_id].cpu().numpy()
        left_force, right_force = self._get_foot_forces()
        left_force = left_force[env_idx].item()
        right_force = right_force[env_idx].item()

        pelvis_z = self._robot.data.root_pos_w[env_idx, 2].item()
        left_foot_z = self._robot.data.body_pos_w[env_idx, self._left_foot_id, 2].item()
        right_foot_z = self._robot.data.body_pos_w[env_idx, self._right_foot_id, 2].item()
        foot_z = min(left_foot_z, right_foot_z) - self._config.foot_ankle_offset

        pelvis_yaw = self._get_pelvis_yaw()[env_idx].item()
        pelvis_xy = self._robot.data.root_pos_w[env_idx, :2].cpu().numpy()
        head_xy = self._robot.data.body_pos_w[env_idx, self._torso_id, :2].cpu().numpy()

        seq = self._sequences[env_idx]
        t1 = self.t1[env_idx].item()
        if seq is not None and len(seq) > 0 and t1 < len(seq):
            target_yaw = seq[t1][3]
            target_pos = seq[t1][:3]
        else:
            target_yaw = 0.0
            target_pos = np.zeros(3)

        qpos = self._robot.data.joint_pos[env_idx].cpu().numpy()
        joint_names_target = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]
        current_joint_angles = np.array([qpos[self._robot.joint_names.index(name)] for name in joint_names_target])
        nominal_angles = np.array([
            -0.5235987756, 0.0, 0.0,
            0.872664626, -0.34906585, 0.0,
            -0.5235987756, 0.0, 0.0,
            0.872664626, -0.34906585, 0.0,
        ])

        swing_frac = self._config.swing_duration / self._config.total_duration
        is_stand = (self._mode[env_idx] == WalkModes.STANDING)

        max_force = self.max_force[env_idx].item()
        if max_force > 1e-6:
            r_frc = foot_frc_clock_reward(left_force, right_force, self.phase[env_idx].item(), max_force, swing_frac, is_stand)
        else:
            r_frc = 0.0
        r_vel = foot_vel_clock_reward(np.linalg.norm(left_vel), np.linalg.norm(right_vel), self.phase[env_idx].item(), self._config.max_foot_vel, swing_frac, is_stand)
        r_orient = body_orient_reward(pelvis_yaw, target_yaw)
        r_height = height_reward(pelvis_z, foot_z)
        r_stability = upper_body_stability_reward(head_xy, pelvis_xy)
        r_step = step_reward(left_pos, right_pos, target_pos, pelvis_xy, self.target_reached[env_idx].item())
        if self._prev_action is not None:
            action = self.last_action[env_idx] if self.last_action is not None else None
            prev = self._prev_action[env_idx] if self._prev_action is not None else None
            if action is not None and prev is not None:
                r_action = action_smoothness_reward(action, prev)
            else:
                r_action = 0.0
        else:
            r_action = 0.0
        if self._prev_torque is not None:
            torque = self.last_torque[env_idx] if self.last_torque is not None else None
            prev_torque = self._prev_torque[env_idx] if self._prev_torque is not None else None
            if torque is not None and prev_torque is not None:
                max_torques = np.array([88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50])
                r_torque = torque_smoothness_reward(torque, prev_torque, max_torques)
            else:
                r_torque = 0.0
        else:
            r_torque = 0.0
        r_posture = posture_error_reward(current_joint_angles, nominal_angles)

        weights = {
            "frc": 0.15,
            "vel": 0.175,
            "orient": 0.05,
            "height": 0.05,
            "step": 0.45,
            "stability": 0.05,
            "posture": 0.00,
            "action": 0.003,
            "torque": 0.002,
        }
        total = (weights["frc"] * r_frc + weights["vel"] * r_vel + weights["orient"] * r_orient +
                 weights["height"] * r_height + weights["step"] * r_step + weights["stability"] * r_stability +
                 weights["posture"] * r_posture + weights["action"] * r_action + weights["torque"] * r_torque)
        return total

    def _get_done(self):
        pelvis_z = self._robot.data.root_pos_w[:, 2]
        left_foot_z = self._robot.data.body_pos_w[:, self._left_foot_id, 2]
        right_foot_z = self._robot.data.body_pos_w[:, self._right_foot_id, 2]
        foot_z = torch.min(left_foot_z, right_foot_z) - self._config.foot_ankle_offset
        height = pelvis_z - foot_z
        return height < self._config.fall_height_threshold

    def _is_truncated(self):
        return self.step_counter >= self._config.max_episode_steps

    def _get_dones(self):
        return self._get_done(), self._is_truncated()

    def set_difficulty(self, progress):
        self.difficulty = np.clip(progress, 0.0, 1.0)
