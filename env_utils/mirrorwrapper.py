import gymnasium as gym
import numpy as np

class MirrorWrapper(gym.Wrapper):
    """
    镜像包装器：利用机器人左右对称性进行数据增强，在 episode 级别随机翻转左右。
    适用于非对称 Actor-Critic 环境（字典观测，包含 actor_obs 和 critic_obs）。
    假设：actor_obs 维度为 41（无腰部俯仰），critic_obs = actor_obs + 线速度(3) + 力矩(12)
    """
    def __init__(self, env, mirror_prob: float = 0.5):
        super().__init__(env)
        self.mirror_prob = mirror_prob
        self.mirror = False

        # ---------- 关节索引（12个关节） ----------
        self.left_indices = [0, 1, 2, 3, 4, 5]
        self.right_indices = [6, 7, 8, 9, 10, 11]
        self.sign_flip_indices = [1, 2, 5, 7, 8, 11]   # 左右对称取反
        assert max(self.left_indices + self.right_indices) < 12, "关节索引超出范围"

        # ---------- actor_obs 分段索引 (41维) ----------
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
        self.angvel_start = self.yaw_idx + 1          # 3个角速度

        # ---------- critic_obs 分段索引 (总长 56) ----------
        # 结构：actor_obs (41) + 线速度 (3) + 力矩 (12)
        self.critic_actor_len = 41
        self.critic_lin_vel_start = self.critic_actor_len
        self.critic_torque_start = self.critic_lin_vel_start + 3
        self.critic_torque_len = 12

        # 验证 critic_obs 总长度（可选）
        # 实际长度由环境决定，这里只用于索引计算
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

    # ---------- 动作镜像 ----------
    def _mirror_action(self, action: np.ndarray) -> np.ndarray:
        mirrored = action.copy()
        mirrored[self.left_indices] = action[self.right_indices]
        mirrored[self.right_indices] = action[self.left_indices]
        mirrored[self.sign_flip_indices] *= -1.0
        return mirrored

    # ---------- 观测镜像 ----------
    def _mirror_obs(self, obs: dict) -> dict:
        new_obs = {k: v.copy() for k, v in obs.items()}
        new_obs["actor_obs"] = self._mirror_actor_obs(new_obs["actor_obs"])
        new_obs["critic_obs"] = self._mirror_critic_obs(new_obs["critic_obs"])
        return new_obs

    def _mirror_actor_obs(self, arr: np.ndarray) -> np.ndarray:
        return self._mirror_actor_array(arr)

    def _mirror_critic_obs(self, arr: np.ndarray) -> np.ndarray:
        """镜像 critic_obs（56维）：actor_obs + 线速度 + 力矩"""
        arr = arr.copy()
        # 1. 镜像前 41 维（归一化 actor 观测）
        arr[:self.critic_actor_len] = self._mirror_actor_array(arr[:self.critic_actor_len])
        # 2. 线速度 y 取反（索引 42）
        arr[self.critic_lin_vel_start + 1] = -arr[self.critic_lin_vel_start + 1]
        # 3. 力矩镜像（12维）
        torque = arr[self.critic_torque_start:self.critic_torque_start + self.critic_torque_len]
        arr[self.critic_torque_start:self.critic_torque_start + self.critic_torque_len] = self._mirror_joint_array(torque)
        return arr

    def _mirror_actor_array(self, arr: np.ndarray) -> np.ndarray:
        """
        对 41 维的 Actor 观测数组执行镜像变换。
        """
        arr = arr.copy()
        # 关节角度 (0:12)
        joint_pos = arr[self.pos_start:self.pos_start + self.joint_dim]
        arr[self.pos_start:self.pos_start + self.joint_dim] = self._mirror_joint_array(joint_pos)
        # 关节速度 (12:24)
        joint_vel = arr[self.vel_start:self.vel_start + self.joint_dim]
        arr[self.vel_start:self.vel_start + self.joint_dim] = self._mirror_joint_array(joint_vel)
        # 骨盆高度 (24) 不变
        # 步点位置：dy 取反 (索引 26 和 29)
        arr[self.foot_dy_idx] = -arr[self.foot_dy_idx]
        arr[self.next_dy_idx] = -arr[self.next_dy_idx]
        # 步点偏航取反 (31, 32)
        arr[self.foot_yaw_idx] = -arr[self.foot_yaw_idx]
        arr[self.next_yaw_idx] = -arr[self.next_yaw_idx]
        # 相位 (33, 34) 取反
        arr[self.phase_sin_idx] = -arr[self.phase_sin_idx]
        arr[self.phase_cos_idx] = -arr[self.phase_cos_idx]
        # 骨盆姿态：roll (35) 和 yaw (37) 取反，pitch (36) 不变
        arr[self.roll_idx] = -arr[self.roll_idx]
        arr[self.yaw_idx] = -arr[self.yaw_idx]
        # 骨盆角速度：wx 和 wz 取反，wy 不变 (38:41)
        angvel = arr[self.angvel_start:self.angvel_start + 3]
        arr[self.angvel_start] = -angvel[0]      # wx -> -wx
        arr[self.angvel_start + 2] = -angvel[2]  # wz -> -wz
        return arr

    def _mirror_joint_array(self, arr: np.ndarray) -> np.ndarray:
        """镜像关节数组（位置、速度或力矩）：左右互换，髋滚转/偏航取反"""
        mirrored = arr.copy()
        mirrored[self.left_indices] = arr[self.right_indices]
        mirrored[self.right_indices] = arr[self.left_indices]
        mirrored[self.sign_flip_indices] *= -1.0
        return mirrored