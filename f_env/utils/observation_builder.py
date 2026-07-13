import numpy as np
from scipy.spatial.transform import Rotation as R

class ObservationBuilder:
    def __init__(self, config, joint_indices, joint_vel_indices, actuator_indices, max_torques):
        self.config = config
        self.joint_indices = joint_indices
        self.joint_vel_indices = joint_vel_indices
        self.actuator_indices = actuator_indices
        self.max_torques = max_torques

        # 构建归一化尺度
        self.critic_obs_scale = np.concatenate([
            [config.norm_params["joint_angles_max"]] * len(joint_indices),
            [config.norm_params["joint_vels_max"]] * len(joint_vel_indices),
            [config.norm_params["pelvis_height_max"]],
            config.norm_params["t1_pos_max"],
            config.norm_params["t2_pos_max"],
            [config.norm_params["t1_yaw_max"]],
            [config.norm_params["t2_yaw_max"]],
            [config.norm_params["phase_max"]] * 2,
            [config.norm_params["pelvis_orient_max"]] * 3,
            [config.norm_params["pelvis_angvel_max"]] * 3,
        ])

    def get_actor_obs(self, model, data, pelvis_id, left_foot_id, right_foot_id, sequence, t1, t2, phase):
        qpos = data.qpos
        qvel = data.qvel
        joint_angles = qpos[self.joint_indices]
        joint_vels = qvel[self.joint_vel_indices]

        pelvis_z = data.qpos[2]
        foot_z = min(data.xpos[left_foot_id][2], data.xpos[right_foot_id][2]) - self.config.foot_ankle_offset
        pelvis_height = pelvis_z - foot_z

        if len(sequence) > 0:
            t1_w = sequence[t1]
            t2_w = sequence[t2]
            pelvis_pos = data.xpos[pelvis_id]
            R_wt = self._get_R_world_to_pelvis(data, pelvis_id)
            t1_local = R_wt @ (t1_w[:3] - pelvis_pos)
            t2_local = R_wt @ (t2_w[:3] - pelvis_pos)
            t1_yaw = t1_w[3] - self._get_pelvis_yaw(data, pelvis_id)
            t2_yaw = t2_w[3] - self._get_pelvis_yaw(data, pelvis_id)
            foot_dx, foot_dy, foot_dz = t1_local[0], t1_local[1], t1_local[2]
            next_dx, next_dy, next_dz = t2_local[0], t2_local[1], t2_local[2]
            foot_yaw = np.arctan2(np.sin(t1_yaw), np.cos(t1_yaw))
            next_yaw = np.arctan2(np.sin(t2_yaw), np.cos(t2_yaw))
        else:
            foot_dx = foot_dy = foot_dz = 0.0
            next_dx = next_dy = next_dz = 0.0
            foot_yaw = next_yaw = 0.0

        phase_signal = np.array([np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)])

        quat = data.xquat[pelvis_id].copy()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = r.as_euler('xyz')
        roll, pitch, yaw = euler

        pelvis_angvel = data.qvel[3:6]

        obs = np.concatenate([
            joint_angles,
            joint_vels,
            [pelvis_height],
            [foot_dx, foot_dy, foot_dz],
            [next_dx, next_dy, next_dz],
            [foot_yaw, next_yaw],
            phase_signal,
            [roll, pitch, yaw],
            pelvis_angvel
        ])
        return obs.astype(np.float32)

    def get_critic_obs(self, model, data, pelvis_id, left_foot_id, right_foot_id, sequence, t1, t2, phase, actor_obs):
        norm_actor_obs = np.clip(actor_obs / self.critic_obs_scale, -1.0, 1.0)

        left_force = data.cfrc_ext[left_foot_id][2]
        right_force = data.cfrc_ext[right_foot_id][2]
        total_mass = sum(model.body_mass)
        max_force = total_mass * 9.81 * 0.5
        norm_left_frc = np.clip(left_force / max_force, -1.0, 1.0)
        norm_right_frc = np.clip(right_force / max_force, -1.0, 1.0)

        lin_vel = data.qvel[0:3]
        norm_lin_vel = np.clip(lin_vel / 2.0, -1.0, 1.0)

        torques = data.actuator_force[self.actuator_indices]
        norm_torques = np.clip(torques / (self.max_torques + 1e-6), -1.0, 1.0)

        priv = np.concatenate([
            [norm_left_frc, norm_right_frc],
            norm_lin_vel,
            norm_torques
        ])

        critic_obs = np.concatenate([norm_actor_obs, priv])
        return critic_obs.astype(np.float32)

    def _get_pelvis_yaw(self, data, pelvis_id):
        quat = data.xquat[pelvis_id].copy()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return r.as_euler('xyz')[2]

    def _get_R_world_to_pelvis(self, data, pelvis_id):
        quat = data.xquat[pelvis_id].copy()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return r.inv().as_matrix()