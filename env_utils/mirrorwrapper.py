import gymnasium as gym
import numpy as np

class MirrorWrapper(gym.Wrapper):
    """
    Data augmentation wrapper that randomly flips left/right at episode level
    using the robot's bilateral symmetry. Designed for asymmetric Actor-Critic
    environments with dictionary observations (actor_obs and critic_obs).
    Assumes actor_obs dim = 41, critic_obs = actor_obs + lin_vel(3) + torque(12).
    """
    def __init__(self, env, mirror_prob: float = 0.5):
        super().__init__(env)
        self.mirror_prob = mirror_prob
        self.mirror = False

        # Joint indices for 12 leg joints.
        self.left_indices = [0, 1, 2, 3, 4, 5]
        self.right_indices = [6, 7, 8, 9, 10, 11]
        self.sign_flip_indices = [1, 2, 5, 7, 8, 11]   # roll/yaw need sign flip.
        assert max(self.left_indices + self.right_indices) < 12, "Joint indices out of range."

        # Segment indices for actor_obs (41 dims).
        self.joint_dim = 12
        self.pos_start = 0
        self.vel_start = self.joint_dim
        self.height_idx = self.vel_start + self.joint_dim
        self.foot_dx_idx = self.height_idx + 1
        self.foot_dy_idx = self.foot_dx_idx + 1
        self.foot_dz_idx = self.foot_dy_idx + 1
        self.next_dx_idx = self.foot_dz_idx + 1
        self.next_dy_idx = self.next_dx_idx + 1
        self.next_dz_idx = self.next_dy_idx + 1
        self.foot_yaw_idx = self.next_dz_idx + 1
        self.next_yaw_idx = self.foot_yaw_idx + 1
        self.phase_sin_idx = self.next_yaw_idx + 1
        self.phase_cos_idx = self.phase_sin_idx + 1
        self.roll_idx = self.phase_cos_idx + 1
        self.pitch_idx = self.roll_idx + 1
        self.yaw_idx = self.pitch_idx + 1
        self.angvel_start = self.yaw_idx + 1          # 3 angular velocities.

        # Segment indices for critic_obs (total 56).
        # Structure: actor_obs (41) + lin_vel (3) + torque (12).
        self.critic_actor_len = 41
        self.critic_lin_vel_start = self.critic_actor_len
        self.critic_torque_start = self.critic_lin_vel_start + 3
        self.critic_torque_len = 12

        self.critic_obs_len = self.critic_actor_len + 3 + self.critic_torque_len

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.mirror = np.random.random() < self.mirror_prob
        if self.mirror:
            obs = self._mirror_obs(obs)
        return obs, info

    def step(self, action):
        if self.mirror:
            action = self._mirror_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.mirror:
            obs = self._mirror_obs(obs)
        return obs, reward, terminated, truncated, info

    # Action mirroring 
    def _mirror_action(self, action: np.ndarray) -> np.ndarray:
        mirrored = action.copy()
        mirrored[self.left_indices] = action[self.right_indices]
        mirrored[self.right_indices] = action[self.left_indices]
        mirrored[self.sign_flip_indices] *= -1.0
        return mirrored

    # Observation mirroring 
    def _mirror_obs(self, obs: dict) -> dict:
        new_obs = {k: v.copy() for k, v in obs.items()}
        new_obs["actor_obs"] = self._mirror_actor_obs(new_obs["actor_obs"])
        new_obs["critic_obs"] = self._mirror_critic_obs(new_obs["critic_obs"])
        return new_obs

    def _mirror_actor_obs(self, arr: np.ndarray) -> np.ndarray:
        return self._mirror_actor_array(arr)

    def _mirror_critic_obs(self, arr: np.ndarray) -> np.ndarray:
        """Mirror critic_obs (56 dims): actor_obs + lin_vel + torque."""
        arr = arr.copy()
        # Mirror the first 41 dims (normalized actor obs).
        arr[:self.critic_actor_len] = self._mirror_actor_array(arr[:self.critic_actor_len])
        # Lin vel y flips sign (index 42).
        arr[self.critic_lin_vel_start + 1] = -arr[self.critic_lin_vel_start + 1]
        # Mirror torque (12 dims).
        torque = arr[self.critic_torque_start:self.critic_torque_start + self.critic_torque_len]
        arr[self.critic_torque_start:self.critic_torque_start + self.critic_torque_len] = self._mirror_joint_array(torque)
        return arr

    def _mirror_actor_array(self, arr: np.ndarray) -> np.ndarray:
        """Mirror the 41-dim actor observation array."""
        arr = arr.copy()
        # Joint positions (0:12).
        joint_pos = arr[self.pos_start:self.pos_start + self.joint_dim]
        arr[self.pos_start:self.pos_start + self.joint_dim] = self._mirror_joint_array(joint_pos)
        # Joint velocities (12:24).
        joint_vel = arr[self.vel_start:self.vel_start + self.joint_dim]
        arr[self.vel_start:self.vel_start + self.joint_dim] = self._mirror_joint_array(joint_vel)
        # Pelvis height (index 24) unchanged.
        # Foot positions: dy flips (indices 26 and 29).
        arr[self.foot_dy_idx] = -arr[self.foot_dy_idx]
        arr[self.next_dy_idx] = -arr[self.next_dy_idx]
        # Foot yaw flips (indices 31, 32).
        arr[self.foot_yaw_idx] = -arr[self.foot_yaw_idx]
        arr[self.next_yaw_idx] = -arr[self.next_yaw_idx]
        # Phase (33, 34) flips sign.
        arr[self.phase_sin_idx] = -arr[self.phase_sin_idx]
        arr[self.phase_cos_idx] = -arr[self.phase_cos_idx]
        # Pelvis orientation: roll (35) and yaw (37) flip, pitch (36) unchanged.
        arr[self.roll_idx] = -arr[self.roll_idx]
        arr[self.yaw_idx] = -arr[self.yaw_idx]
        # Pelvis angular velocities: wx and wz flip, wy unchanged (38:41).
        angvel = arr[self.angvel_start:self.angvel_start + 3]
        arr[self.angvel_start] = -angvel[0]      # wx -> -wx
        arr[self.angvel_start + 2] = -angvel[2]  # wz -> -wz
        return arr

    def _mirror_joint_array(self, arr: np.ndarray) -> np.ndarray:
        """Mirror joint array (pos, vel, or torque): swap left/right and flip roll/yaw."""
        mirrored = arr.copy()
        mirrored[self.left_indices] = arr[self.right_indices]
        mirrored[self.right_indices] = arr[self.left_indices]
        mirrored[self.sign_flip_indices] *= -1.0
        return mirrored