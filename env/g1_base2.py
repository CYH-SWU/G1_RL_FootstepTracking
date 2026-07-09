# 1.1 0.75 0.35 action_scale = 0.30 soomth = 0.30 控制dt = 0.015 waist = 0.150 max_vel = 0.20基准
# 继承于g1_base1.py

import os
import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
from enum import Enum, auto
from pathlib import Path
import random

from planner_pipeline.reward_functions import (
    calc_foot_frc_clock_reward,
    calc_foot_vel_clock_reward,
    calc_body_orient_reward,
    calc_height_reward,
    calc_upper_body_stability,
    calc_torque_reward,
    calc_action_reward,
    calc_step_reward,
    calc_posture_error_reward,
)

class WalkModes(Enum):
    STANDING = auto()
    CURVED = auto()
    FORWARD = auto()
    BACKWARD = auto()
    INPLACE = auto()
    LATERAL = auto()

class G1TerrainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        robot_xml_path: str,
        max_episode_steps: int = 1500,
        control_dt: float = 0.015,
        physics_dt: float = 0.005,
        max_boxes: int = 30,  # 最大踏脚石数量
        **kwargs
    ):
        super().__init__()

        self.robot_xml_path = os.path.abspath(robot_xml_path)
        self.control_dt = control_dt
        self.physics_dt = physics_dt
        self.n_substeps = int(control_dt / physics_dt)
        self.max_episode_steps = max_episode_steps
        self.difficulty = 0

        # 模式概率（与 LHW 完全一致）
        self.mode_probs = [0.05, 0.15, 0.20, 0.30, 0.30] # STANDING, CURVED, BACKWARD, LATERAL, FORWARD
        # [0.05, 0.15, 0.20, 0.30, 0.30]
        self.mode_list = [WalkModes.STANDING, WalkModes.CURVED, WalkModes.BACKWARD,
                          WalkModes.LATERAL, WalkModes.FORWARD]

        # 动作空间：12 个关节（移除腰部）
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # 观测空间：actor_obs + critic_obs（非对称）
        actor_obs_dim = 12 + 12 + 1 + 8 + 2 + 3 + 3
        privileged_obs_dim = 2 + 3 + 12
        self.observation_space = spaces.Dict({
            "actor_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(actor_obs_dim,), dtype=np.float32),
            "critic_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(actor_obs_dim + privileged_obs_dim,), dtype=np.float32),
        })

        # 内部状态
        self.model = None
        self.data = None
        self.step_counter = 0
        self.phase = 0.0
        self.mode = None

        # 步态参数
        self.total_duration = 1.1
        self.swing_duration = 0.75
        self.stance_duration = 0.35
        self.step_length = 0.25
        self.step_width = 0.237
        self.max_foot_vel = 0.20

        # 步点序列相关
        self.sequence = []          # 世界坐标步点 (x,y,z,theta)
        self.t1 = 0
        self.t2 = 1
        self.target_radius = 0.20
        self.delay_frames = int(np.floor(self.swing_duration / self.control_dt))
        self.target_reached = False
        self.target_reached_frames = 0

        # 缓存ID
        self.pelvis_id = None
        self.left_foot_id = None
        self.right_foot_id = None
        self.torso_id = None
        self.head_id = None
        self.joint_indices = None
        self.actuator_indices = None
        self.max_torques = None

        # 摔倒阈值
        self.fall_height_threshold = 0.35

        # 标称姿态（12个关节）
        self.nominal_angles = np.array([
            -0.5235987756, 0.0, 0.0, 0.872664626, -0.34906585, 0.0,
            -0.5235987756, 0.0, 0.0, 0.872664626, -0.34906585, 0.0
        ])

        self.nominal_pelvis_height = 0.6937 + 0.0331
        self.foot_ankle_offset = 0.0331
        self.action_scale = 0.30
        self.smooth = 0.30
        self.last_action = None
        self.last_torque = None 
        self.smooth_target = np.zeros(12)

        # 最大踏脚石数量
        self.max_boxes = max_boxes

        self.norm_params = {
            "joint_angles_max": 1.5,
            "joint_vels_max": 10.0,
            "pelvis_height_max": 1.0,
            # T1 步点位置 (dx, dy, dz)
            "t1_pos_max": [0.30, 0.25, 0.9],      # 当前步点位置范围
            # T2 步点位置 (dx, dy, dz)
            "t2_pos_max": [0.5, 0.30, 0.9],      # 下一步点位置范围（可稍大）
            # T1 偏航
            "t1_yaw_max": 0.2,
            # T2 偏航
            "t2_yaw_max": 0.25,                 # 下一步点偏航可能稍大
            "phase_max": 1.0,
            "pelvis_orient_max": 0.3,
            "pelvis_angvel_max": 5.0,
        }


        # 构建缩放数组（顺序与 actor_obs 完全对应）
        self.critic_obs_scale = np.concatenate([
            [self.norm_params["joint_angles_max"]] * 12,
            [self.norm_params["joint_vels_max"]] * 12,
            [self.norm_params["pelvis_height_max"]],
            self.norm_params["t1_pos_max"],      # 3 个值
            self.norm_params["t2_pos_max"],      # 3 个值
            [self.norm_params["t1_yaw_max"]],
            [self.norm_params["t2_yaw_max"]],
            [self.norm_params["phase_max"]] * 2,
            [self.norm_params["pelvis_orient_max"]] * 3,
            [self.norm_params["pelvis_angvel_max"]] * 3,
        ])
        # 长度应为 12+12+1+3+3+1+1+2+3+3 = 41
        assert len(self.critic_obs_scale) == 41

        # ---------- 加载模型并预定义踏脚石 ----------
        self._load_model_with_boxes()
        # 缓存 ID
        self._cache_ids()

    def _load_model_with_boxes(self):
        """读取机器人XML，添加固定数量的踏脚石（box）、中心点和方向指示"""
        with open(self.robot_xml_path, 'r') as f:
            robot_xml = f.read()

        project_root = Path(__file__).parent.parent.absolute()
        mesh_abs_path = (project_root / "robot" / "assets").as_posix()
        robot_xml = robot_xml.replace('meshdir="assets"', f'meshdir="{mesh_abs_path}"')
        robot_xml = robot_xml.replace("meshdir='assets'", f"meshdir='{mesh_abs_path}'")

        markers_xml = ""
        for i in range(self.max_boxes):
            markers_xml += f'''
            <!-- 踏脚石 (box) -->
            <body name="step_{i}" pos="0 0 -10" quat="1 0 0 0">
                <geom type="box" size="0.125 1.0 0.05" rgba="0.8 0.8 0.8 1" group="1"/>
            </body>
            <!-- 中心点 (小球) -->
            <body name="step_dot_{i}" pos="0 0 -10" quat="1 0 0 0">
                <geom type="sphere" size="0.03" rgba="1.0 0.0 0.0 1" group="1"/>
            </body>
            <!-- 方向指示 (细长 box，指向 X 轴正方向) -->
            <body name="step_arrow_{i}" pos="0 0 -10" quat="1 0 0 0">
                <geom type="box" size="0.08 0.01 0.01" rgba="0.0 0.0 1.0 1" group="1"/>
            </body>
            '''
        full_xml = robot_xml.replace('</worldbody>', markers_xml + '</worldbody>')

        self.model = mujoco.MjModel.from_xml_string(full_xml)
        self.data = mujoco.MjData(self.model)

    def _cache_ids(self):
        model = self.model
        self.pelvis_id = model.body("pelvis").id
        self.left_foot_id = model.body("left_ankle_roll_link").id
        self.right_foot_id = model.body("right_ankle_roll_link").id
        self.torso_id = model.body("torso_link").id
        self.head_id = self.torso_id

        joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
        ]
        self.joint_indices = []
        self.joint_vel_indices = []
        for name in joint_names:
            joint = model.joint(name)
            self.joint_indices.append(joint.qposadr[0])
            self.joint_vel_indices.append(joint.dofadr[0])

        self.actuator_indices = []
        for name in joint_names:
            self.actuator_indices.append(model.actuator(name).id)

        # 最大力矩（12个关节）
        self.max_torques = np.array([
            88, 139, 88, 139, 50, 50,
            88, 139, 88, 139, 50, 50
        ])

    def _get_body_linvel(self, body_id):
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id, vel, 0
        )
        return np.linalg.norm(vel[:3])

    def _get_pelvis_yaw(self):
        quat = self.data.xquat[self.pelvis_id].copy()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return r.as_euler('xyz')[2]

    def _get_R_world_to_pelvis(self):
        quat = self.data.xquat[self.pelvis_id].copy()  # (w, x, y, z)
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        R_world_to_pelvis = r.inv().as_matrix()  # 逆矩阵即为世界→骨盆
        return R_world_to_pelvis

    def _generate_step_sequence(self, mode, num_steps=20, step_height=0.0):
        """生成步点序列（局部坐标系）"""
        if mode == WalkModes.CURVED:
            seq = []
            first_shift = np.random.uniform(0.100, 0.125)
            # 第一步偏移（与 FORWARD 一致）
            if np.isclose(self.phase, 0.0):
                seq.append([0.0, -first_shift, 0.0, 0.0])
                initial_y_sign = -1
                curve_dir = -1
            else:
                seq.append([0.0, first_shift, 0.0, 0.0])
                initial_y_sign = 1
                curve_dir = 1

            # 圆弧半径（2~4 米）
            R = np.random.uniform(2.5, 4.0)
            # 圆心在 y 轴上：使第一步落在圆弧上
            y0 = initial_y_sign * first_shift
            cy = y0 - curve_dir * R

            # 计算总角度，使每步弧长 ≈ step_length
            total_angle = (num_steps - 1) * (self.step_length - 0.025) / R
            # 增加随机扰动（可选）
            total_angle *= np.random.uniform(0.9, 1.1)

            dtheta = total_angle / (num_steps - 1)

            for i in range(1, num_steps):
                theta_i = i * dtheta
                radius_offset = ((-1) ** i) * ((self.step_width - 0.025) / 2)
                R_i = R + radius_offset
                x_local = R_i * np.sin(theta_i)
                y_local = curve_dir * R_i * np.cos(theta_i)
                x_world = x_local
                y_world = cy + y_local
                yaw = -theta_i * curve_dir  # 切线方向
                seq.append([x_world, y_world, 0.0, yaw])

            return np.array(seq)

        elif mode == WalkModes.LATERAL:
            seq = []
            y = 0
            c = np.random.choice([-1, 1])
            for i in range(1, num_steps + 1):
                if i % 2:
                    y += self.step_length * 0.8
                else:
                    y -= (2/3) * self.step_length * 0.8
                step = np.array([0, c * y, 0, 0])
                seq.append(step)
            return np.array(seq)

        elif mode == WalkModes.STANDING:
            return np.array([[0.0, 0.0, 0.0, 0.0]])

        elif mode == WalkModes.INPLACE:
            ss = np.random.uniform(-0.05, 0.05)
            seq = []
            for i in range(num_steps):
                x = ss * (i % 2)
                seq.append([x, 0.0, 0.0, 0.0])
            return np.array(seq)

        elif mode == WalkModes.BACKWARD:
            seq = []
            x = 0
            y = self.step_width / 2 * (1 if np.random.rand() > 0.5 else -1)
            for i in range(num_steps):
                x -= 0.1
                y = -y
                seq.append([x, y, 0.0, 0.0])
            return np.array(seq)

        else:  # FORWARD
            seq = []
            # 计算初始高度
            if step_height < 0:
                initial_z = -step_height * (num_steps - 1)
            else:
                initial_z = 0

            first_shift = np.random.uniform(0.100, 0.125)

            # 根据 phase 决定第一步侧移方向（与 LHW 一致）
            if np.isclose(self.phase, 0.0):
                # 相位为 0 时，第一步向左偏移
                seq.append([0.0, -first_shift, initial_z, 0.0])
                y = -self.step_width / 2
            else:  # phase 接近 0.5
                # 相位为 0.5 时，第一步向右偏移
                seq.append([0.0, first_shift, initial_z, 0.0])
                y = self.step_width / 2

            x = 0
            z = initial_z

            for i in range(1, num_steps):
                x += self.step_length
                y *= -1
                z += step_height   # 下楼梯时 step_height 为负，z 逐渐降低
                seq.append([x, y, z, 0.0])
            return np.array(seq)

    def _transform_sequence(self, sequence):
        """转换为世界坐标"""
        # 获取当前双脚中点和骨盆偏航
        left_pos = self.data.xpos[self.left_foot_id]
        right_pos = self.data.xpos[self.right_foot_id]
        mid_pt = (left_pos + right_pos) / 2
        root_yaw = self._get_pelvis_yaw()
        cos_y = np.cos(root_yaw)
        sin_y = np.sin(root_yaw)
        world_seq = []
        for x, y, z, theta in sequence:
            x_w = mid_pt[0] + x * cos_y - y * sin_y
            y_w = mid_pt[1] + x * sin_y + y * cos_y
            theta_w = root_yaw + theta
            world_seq.append(np.array([x_w, y_w, z, theta_w]))
        return np.array(world_seq)

    # ---------- 更新踏脚石位置 ----------
    def _update_terrain_boxes(self):
        num_steps = len(self.sequence)
        for i in range(self.max_boxes):
            box_body_name = f"step_{i}"
            box_body_id = self.model.body(box_body_name).id
            if i < num_steps:
                x, y, z, theta = self.sequence[i]
                # 踏脚石 (box)
                geom_addr = self.model.body_geomadr[box_body_id]
                box_h = self.model.geom_size[geom_addr][2]
                self.model.body_pos[box_body_id] = [x, y, z - box_h]
                quat_scipy = R.from_euler('z', theta).as_quat()
                quat_mujoco = [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]
                self.model.body_quat[box_body_id] = quat_mujoco

                # 中心点 (小球)
                dot_body_name = f"step_dot_{i}"
                dot_body_id = self.model.body(dot_body_name).id
                self.model.body_pos[dot_body_id] = [x, y, z]
                self.model.body_quat[dot_body_id] = quat_mujoco  # 小球朝向不重要，但保持与踏脚石一致

                # 方向箭头
                arrow_body_name = f"step_arrow_{i}"
                arrow_body_id = self.model.body(arrow_body_name).id
                self.model.body_pos[arrow_body_id] = [x, y, z]
                self.model.body_quat[arrow_body_id] = quat_mujoco
            else:
                # 隐藏所有标记
                for body_name in [box_body_name, f"step_dot_{i}", f"step_arrow_{i}"]:
                    body_id = self.model.body(body_name).id
                    self.model.body_pos[body_id] = [0, 0, -10]
                    self.model.body_quat[body_id] = [1, 0, 0, 0]

    # ---------- reset ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_reached = False
        self.target_reached_frames = 0

        # 选择模式
        self.mode = np.random.choice(self.mode_list, p=self.mode_probs)

        # 计算台阶高度（仅 FORWARD 模式）
        step_height = 0.0
        if self.mode == WalkModes.FORWARD:
            max_h = 0.1 * max(0.0, (self.difficulty - 0.273) / (1.0 - 0.273))
            step_height = np.random.choice([-max_h, max_h])

        num_steps = 20
        if self.mode == WalkModes.STANDING:
            num_steps = 1
        elif self.mode == WalkModes.CURVED:
            num_steps = 25

        # 生成局部步点序列
        local_seq = self._generate_step_sequence(self.mode, num_steps, step_height)

        # 转换到世界坐标（需要先有有效 data）
        # 由于模型已加载，但数据可能未重置，我们先重置关键帧
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            self.data.ctrl[:] = self.data.qpos[self.actuator_indices]
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.sequence = self._transform_sequence(local_seq)

        # 更新踏脚石位置
        self._update_terrain_boxes()

        # 设置骨盆位置为第一个步点上方
        if len(self.sequence) > 0:
            offset_back = -0.12  # 向后偏移 5cm
            first_step = self.sequence[0] 
            self.data.qpos[0] = first_step[0] + offset_back
            self.data.qpos[1] = first_step[1]
            self.data.qpos[2] = first_step[2] + self.nominal_pelvis_height
            yaw = first_step[3]
            quat = R.from_euler('z', yaw).as_quat()
            self.data.qpos[3] = quat[3]
            self.data.qpos[4] = quat[0]
            self.data.qpos[5] = quat[1]
            self.data.qpos[6] = quat[2]

        mujoco.mj_forward(self.model, self.data)

        # 初始化状态
        self.phase = random.choice([0.0, 0.5])
        self.step_counter = 0
        self.last_action = None
        self.last_torque = None
        self.t1 = 0
        self.t2 = min(1, len(self.sequence)-1) if len(self.sequence) > 1 else 0

        obs = {
            "actor_obs": self._get_actor_obs(),
            "critic_obs": self._get_critic_obs(),
        }
        info = {"mode": self.mode.name, "difficulty": self.difficulty}
        return obs, info

    # ---------- 观测 ----------
    def _get_actor_obs(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        joint_angles = qpos[self.joint_indices]
        joint_vels = qvel[self.joint_vel_indices]

        pelvis_z = self.data.qpos[2]
        foot_z = min(self.data.xpos[self.left_foot_id][2], self.data.xpos[self.right_foot_id][2]) - self.foot_ankle_offset
        pelvis_height = pelvis_z - foot_z

        if len(self.sequence) > 0:
            t1_idx = self.t1
            t2_idx = self.t2
            t1_w = self.sequence[t1_idx]
            t2_w = self.sequence[t2_idx]
            pelvis_pos = self.data.xpos[self.pelvis_id]
            R_wt = self._get_R_world_to_pelvis()
            t1_local = R_wt @ (t1_w[:3] - pelvis_pos)
            t2_local = R_wt @ (t2_w[:3] - pelvis_pos)
            t1_yaw = t1_w[3] - self._get_pelvis_yaw()
            t2_yaw = t2_w[3] - self._get_pelvis_yaw()
            foot_dx, foot_dy, foot_dz = t1_local[0], t1_local[1], t1_local[2]
            next_dx, next_dy, next_dz = t2_local[0], t2_local[1], t2_local[2]
            foot_yaw = np.arctan2(np.sin(t1_yaw), np.cos(t1_yaw))
            next_yaw = np.arctan2(np.sin(t2_yaw), np.cos(t2_yaw))
        else:
            foot_dx = foot_dy = foot_dz = 0.0
            next_dx = next_dy = next_dz = 0.0
            foot_yaw = next_yaw = 0.0

        phase_val = self.phase
        phase = np.array([np.sin(2*np.pi*phase_val), np.cos(2*np.pi*phase_val)])

        quat = self.data.xquat[self.pelvis_id].copy()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        euler = r.as_euler('xyz')
        roll, pitch, yaw = euler

        pelvis_angvel = self.data.qvel[3:6]

        obs = np.concatenate([
            joint_angles,
            joint_vels,
            [pelvis_height],
            [foot_dx, foot_dy, foot_dz],
            [next_dx, next_dy, next_dz],
            [foot_yaw, next_yaw],
            phase,
            [roll, pitch, yaw],
            pelvis_angvel
        ])
        return obs.astype(np.float32)

    def _get_critic_obs(self):
        actor_obs = self._get_actor_obs()
        # 使用预构建的缩放数组归一化基础观测
        norm_actor_obs = np.clip(actor_obs / self.critic_obs_scale, -1.0, 1.0)

        # 特权信息：足底力（法向）
        left_force = self.data.cfrc_ext[self.left_foot_id][2]
        right_force = self.data.cfrc_ext[self.right_foot_id][2]
        max_force = sum(self.model.body_mass) * 9.81 * 0.5
        norm_left_frc = np.clip(left_force / max_force, -1.0, 1.0)
        norm_right_frc = np.clip(right_force / max_force, -1.0, 1.0)

        # 特权信息：基座线速度（世界坐标系）
        lin_vel = self.data.qvel[0:3]
        norm_lin_vel = np.clip(lin_vel / 2.0, -1.0, 1.0)

        # 特权信息：关节力矩（12个关节）
        torques = self.data.actuator_force[self.actuator_indices]
        norm_torques = np.clip(torques / (self.max_torques + 1e-6), -1.0, 1.0)

        # 拼接所有特权信息
        priv = np.concatenate([
            [norm_left_frc, norm_right_frc],
            norm_lin_vel,
            norm_torques
        ])

        # 拼接归一化的 actor_obs 和特权信息
        critic_obs = np.concatenate([norm_actor_obs, priv])
        return critic_obs.astype(np.float32)

    # ---------- step ----------
    def step(self, action):
        assert self.model is not None

        self._apply_action(action)

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.step_counter += 1

        self.phase = (self.step_counter * self.control_dt % self.total_duration) / self.total_duration

        # 踩中判定
        if self.mode != WalkModes.STANDING and len(self.sequence) > 0 and self.t1 < len(self.sequence):
            target_pos = self.sequence[self.t1][:3]
            left_pos = self.data.xpos[self.left_foot_id]
            right_pos = self.data.xpos[self.right_foot_id]
            l_dist = np.linalg.norm(left_pos - target_pos)
            r_dist = np.linalg.norm(right_pos - target_pos)
            if l_dist < self.target_radius or r_dist < self.target_radius:
                self.target_reached = True
                self.target_reached_frames += 1
            else:
                self.target_reached = False
                self.target_reached_frames = 0

            if self.target_reached and self.target_reached_frames >= self.delay_frames:
                self.t1 = self.t2
                self.t2 = min(self.t2 + 1, len(self.sequence) - 1)
                self.target_reached = False
                self.target_reached_frames = 0

        reward = self._compute_reward(action)

        obs = {
            "actor_obs": self._get_actor_obs(),
            "critic_obs": self._get_critic_obs(),
        }

        terminated = self._check_termination()
        truncated = self.step_counter >= self.max_episode_steps

        info = {}
        return obs, reward, terminated, truncated, info

    def _apply_action(self, action):
        # 计算原始目标角度（标称姿态 + 动作缩放）
        raw_target = self.nominal_angles + action * self.action_scale

        # 指数移动平均平滑（平滑系数 0.5）
        smooth = self.smooth
        self.smooth_target = smooth * raw_target + (1 - smooth) * self.smooth_target
        target_qpos = self.smooth_target

        # 裁剪到关节限位
        for i, idx in enumerate(self.actuator_indices):
            low, high = self.model.actuator_ctrlrange[idx]
            target_qpos[i] = np.clip(target_qpos[i], low, high)

        # 发送到 MuJoCo 执行器
        self.data.ctrl[self.actuator_indices] = target_qpos

    def _compute_reward(self, action):
        left_force = self.data.cfrc_ext[self.left_foot_id][2]
        right_force = self.data.cfrc_ext[self.right_foot_id][2]
        left_vel = self._get_body_linvel(self.left_foot_id)
        right_vel = self._get_body_linvel(self.right_foot_id)

        pelvis_z = self.data.qpos[2]
        foot_z = min(self.data.xpos[self.left_foot_id][2], self.data.xpos[self.right_foot_id][2]) - self.foot_ankle_offset

        pelvis_yaw = self._get_pelvis_yaw()
        target_yaw = self.sequence[self.t1][3] if len(self.sequence) > 0 else 0.0

        pelvis_xy = self.data.xpos[self.pelvis_id][:2]
        head_xy = self.data.xpos[self.head_id][:2]

        total_mass = sum(self.model.body_mass)
        max_force = total_mass * 9.81 * 0.5

        swing_frac = self.swing_duration / self.total_duration

        is_stand = (self.mode == WalkModes.STANDING)

        if is_stand:
            r_frc = calc_foot_frc_clock_reward(
                swing_frac,
                left_force, right_force,
                self.phase, max_force,
                clock_left=1.0, clock_right=1.0
            )
            r_vel = calc_foot_vel_clock_reward(
                swing_frac,
                left_vel, right_vel,
                self.phase, self.max_foot_vel,
                clock_left=-1.0, clock_right=-1.0
            )
        else:
            r_frc = calc_foot_frc_clock_reward(swing_frac, left_force, right_force, self.phase, max_force)
            r_vel = calc_foot_vel_clock_reward(swing_frac, left_vel, right_vel, self.phase, self.max_foot_vel)

        r_orient = calc_body_orient_reward(pelvis_yaw, target_yaw)
        r_height = calc_height_reward(pelvis_z, foot_z, goal_height=self.nominal_pelvis_height, deadzone=0.023)

        if len(self.sequence) > 0 and self.t1 < len(self.sequence):
            target_pos = self.sequence[self.t1][:3]
            left_pos = self.data.xpos[self.left_foot_id]
            right_pos = self.data.xpos[self.right_foot_id]
            r_step = calc_step_reward(left_pos, right_pos, target_pos, pelvis_xy, self.target_reached)
        else:
            r_step = 0.0

        r_stability = calc_upper_body_stability(head_xy, pelvis_xy)

        r_action = calc_action_reward(action, self.last_action)
        self.last_action = action.copy()

        torques = self.data.actuator_force[self.actuator_indices]
        r_torque = calc_torque_reward(torques, self.last_torque)
        self.last_torque = torques.copy()

        current_joint_angles = self.data.qpos[self.joint_indices]
        r_posture = calc_posture_error_reward(current_joint_angles, self.nominal_angles)

        weights = {
            'frc': 0.15,
            'vel': 0.15,
            'orient': 0.05,
            'height': 0.05,
            'step': 0.45,
            'stability': 0.05,
            'posture': 0.00,
            'action': 0.00,
            'torque': 0.00
        }
        total = (weights['frc'] * r_frc +
                 weights['vel'] * r_vel +
                 weights['orient'] * r_orient +
                 weights['height'] * r_height +
                 weights['step'] * r_step +
                 weights['stability'] * r_stability +
                 weights['posture'] * r_posture +
                 weights['action'] * r_action +
                 weights['torque'] * r_torque)
        return total

    def _check_termination(self):
        pelvis_z = self.data.qpos[2]
        foot_z = min(self.data.xpos[self.left_foot_id][2], self.data.xpos[self.right_foot_id][2]) - self.foot_ankle_offset
        height = pelvis_z - foot_z
        if height < self.fall_height_threshold:
            return True
        return False

    def set_difficulty(self, progress: float):
        self.difficulty = np.clip(progress, 0.0, 1.0)

    def render(self):
        raise NotImplementedError("该环境不支持实时渲染，请使用独立的评估脚本。")

    def close(self):
        pass