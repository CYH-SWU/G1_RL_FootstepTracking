import gymnasium as gym
import numpy as np

class MirrorWrapper(gym.Wrapper):
    """
    镜像包装器（Mirror Wrapper）：
    利用机器人左右对称性进行数据增强，在 episode 级别随机翻转左右。

    该包装器不修改底层物理仿真，只对策略看到的观测和动作进行实时镜像变换，
    从而使策略学习到对称的步态，提高样本效率。

    使用方法：
        env = MirrorWrapper(env, mirror_prob=0.5)

    注意事项：
        1. 本实现假设观测和动作的维度及语义固定，具体索引需根据实际环境调整。
        2. 仅适用于“左右对称”的任务（如行走、站立），不适用于非对称任务。
        3. 站立模式下，虚拟步点的横向偏移和偏航为0，镜像后保持不变，因此安全。
    """

    def __init__(self, env, mirror_prob: float = 0.5):
        """
        初始化镜像包装器。

        参数:
            env: 基础环境（gym.Env）
            mirror_prob: 每个 episode 开启镜像的概率 (0~1)，默认 0.5
        """
        super().__init__(env)
        self.mirror_prob = mirror_prob
        self.mirror = False  # 当前 episode 是否启用镜像

        # ---------- 以下索引假设你的观测结构为 ----------
        # obs = [joint_pos(13), joint_vel(13), pelvis_euler(3), pelvis_ang_vel(3),
        #        target(4), stance(1), phase(2)]
        # 动作 = 关节角度增量 (13维)
        # 若你的环境不同，请修改下面的索引和映射规则 ----------
        self.joint_dim = 13
        self.pos_start = 0
        self.vel_start = self.joint_dim
        self.euler_start = self.vel_start + self.joint_dim
        self.angvel_start = self.euler_start + 3
        self.target_start = self.angvel_start + 3
        self.stance_idx = self.target_start + 4
        self.phase_idx = self.stance_idx + 1

        # 左右关节索引（假设顺序：左髋P,R,Y; 左膝; 左踝P,R; 右髋P,R,Y; 右膝; 右踝P,R; 腰部俯仰）
        # 若顺序不同，请调整以下列表
        self.left_indices = [0, 1, 2, 3, 4, 5]
        self.right_indices = [6, 7, 8, 9, 10, 11]
        # 需要取反的关节：髋关节的 Roll(1,7) 和 Yaw(2,8)
        self.sign_flip_indices = [1, 2, 7, 8]

        # 关节维度检查
        assert max(self.left_indices + self.right_indices) < self.joint_dim, \
            "关节索引超出关节维度，请检查索引定义"

    def reset(self, **kwargs):
        """
        重置环境并决定本 episode 是否镜像。

        返回:
            obs: 镜像后的观测（若本 episode 启用镜像）
            info: 原始 info
        """
        obs, info = self.env.reset(**kwargs)
        # 随机决定是否镜像
        self.mirror = np.random.random() < self.mirror_prob
        if self.mirror:
            obs = self._mirror_obs(obs)
        return obs, info

    def step(self, action):
        """
        执行一步仿真。

        流程：
            1. 若启用镜像，先将动作镜像，再传给环境。
            2. 环境执行物理步骤（物理世界未镜像）。
            3. 若启用镜像，将环境返回的观测镜像后再返回给策略。

        参数:
            action: 策略输出的原始动作（未镜像）

        返回:
            obs: 镜像后的观测（或原始观测）
            reward: 原始奖励（不受镜像影响）
            terminated, truncated, info: 原始终止信息
        """
        # 动作镜像（如果开启）
        if self.mirror:
            action = self._mirror_action(action)

        # 执行仿真（物理世界没有镜像）
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 观测镜像（如果开启）
        if self.mirror:
            obs = self._mirror_obs(obs)

        return obs, reward, terminated, truncated, info

    def _mirror_action(self, action: np.ndarray) -> np.ndarray:
        """
        将动作（策略输出）进行镜像变换。

        变换规则：
            - 左右腿关节位置互换
            - 髋关节的 Roll 和 Yaw 取反

        参数:
            action: 原始动作数组 (13维)

        返回:
            镜像后的动作数组
        """
        mirrored = action.copy()
        # 互换左右
        mirrored[self.left_indices] = action[self.right_indices]
        mirrored[self.right_indices] = action[self.left_indices]
        # 滚转和偏航取反
        mirrored[self.sign_flip_indices] *= -1.0
        return mirrored

    def _mirror_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        将观测进行镜像变换。

        变换规则：
            - 关节位置/速度：左右互换，髋关节滚转/偏航取反
            - 骨盆欧拉角：滚转和偏航取反，俯仰不变
            - 骨盆角速度：全部取反
            - 步点目标 (dx, dy, dz, dyaw)：dy 和 dyaw 取反，dx 和 dz 不变
            - 支撑腿标签：取反（-1 ↔ 1）
            - 步态相位：不变

        参数:
            obs: 原始观测数组

        返回:
            镜像后的观测数组
        """
        obs = obs.copy()

        # 1. 关节位置
        joint_pos = obs[self.pos_start:self.pos_start + self.joint_dim]
        obs[self.pos_start:self.pos_start + self.joint_dim] = self._mirror_joint_array(joint_pos)

        # 2. 关节速度
        joint_vel = obs[self.vel_start:self.vel_start + self.joint_dim]
        obs[self.vel_start:self.vel_start + self.joint_dim] = self._mirror_joint_array(joint_vel)

        # 3. 骨盆欧拉角 (roll, pitch, yaw)
        euler = obs[self.euler_start:self.euler_start + 3]
        obs[self.euler_start] = -euler[0]           # roll 取反
        # pitch 不变
        obs[self.euler_start + 2] = -euler[2]       # yaw 取反

        # 4. 骨盆角速度 (3维) 全部取反
        angvel = obs[self.angvel_start:self.angvel_start + 3]
        obs[self.angvel_start:self.angvel_start + 3] = -angvel

        # 5. 步点目标 (dx, dy, dz, dyaw)
        target = obs[self.target_start:self.target_start + 4]
        obs[self.target_start] = target[0]          # dx 不变
        obs[self.target_start + 1] = -target[1]     # dy 取反
        obs[self.target_start + 2] = target[2]      # dz 不变
        obs[self.target_start + 3] = -target[3]     # dyaw 取反

        # 6. 支撑腿标签 (-1/1)
        obs[self.stance_idx] = -obs[self.stance_idx]

        # 7. 步态相位 (sin, cos) 不变
        # 无需修改

        return obs

    def _mirror_joint_array(self, arr: np.ndarray) -> np.ndarray:
        """
        镜像关节数组（位置或速度）。

        规则：
            - 左右互换
            - 髋关节的滚转(索引1)和偏航(索引2)取反
        """
        mirrored = arr.copy()
        # 左右互换
        mirrored[self.left_indices] = arr[self.right_indices]
        mirrored[self.right_indices] = arr[self.left_indices]
        # 滚转和偏航取反
        mirrored[self.sign_flip_indices] *= -1.0
        return mirrored