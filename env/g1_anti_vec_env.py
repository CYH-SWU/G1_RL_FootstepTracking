'''
模仿LHW
使用双脚中点
删除obs的迈步标签
站立模式步点信息全为0
obs为两个时钟信号
使用与LHW一致的奖励函数系数
去除打印代码
取消<0.1强制站立
修改起伏地形的经验高度
使用LHW的踩中逻辑
使用LHW的步点跟踪奖励函数
加入标称姿态奖励函数
站立模式下足部速度奖励max_vel = 0.2
骨盆高度 = 世界坐标系下骨盆高度 - 双足的最低高度 + 0.0431
使用宇树g1标称姿态
使用宇树g1的控制方法
posture奖励为0.050
非对称actor critic
action_scale = 0.4
控制频率100hz
critic基础观测归一化
'''

import os
import time
import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional, Dict, Any

# 导入自定义模块
from planner_pipeline.terrain_generator import TerrainGenerator
from planner_pipeline.vision_processor import VisionProcessor
from planner_pipeline.footstep_planner import G1FootstepPlanner, Footstep
from planner_pipeline.reward_functions import (
    calc_foot_frc_clock_reward,
    calc_foot_vel_clock_reward,
    calc_body_orient_reward,
    calc_height_reward,
    calc_upper_body_stability,
    calc_action_penalty,
    calc_torque_penalty,
    calc_step_reward,
    calc_posture_error_reward,
    clock_frc
)



class G1TerrainEnv(gym.Env):
    """
    G1人形机器人复杂地形行走环境，集成视觉感知、步点规划与强化学习。
    符合Gymnasium规范，支持SB3的DummyVecEnv并行训练。
    """

    metadata = {"render_modes": []}  # 无渲染

    def __init__(
        self,
        robot_xml_path: str,
        mesh_dir: str,
        terrain_modes: list = None,
        probabilities: list = None,
        total_timesteps_for_max: int = 11000 * 1500,
        max_episode_steps: int = 2000,
        control_dt: float = 0.010,   
        physics_dt: float = 0.005,
        goal_radius: float = 7.5,
        **kwargs
    ):
        super().__init__()

        # 配置参数
        self.robot_xml_path = os.path.abspath(robot_xml_path)
        self.mesh_dir = os.path.abspath(mesh_dir)
        self.control_dt = control_dt
        self.physics_dt = physics_dt
        self.n_substeps = int(control_dt / physics_dt)
        self.goal_radius = goal_radius
        self.max_episode_steps = max_episode_steps
        self.total_timesteps_for_max = total_timesteps_for_max

        # 地形模式与概率
        if terrain_modes is None:
            terrain_modes = [
                "flat_stand", "rough_stand", "slope_stand",
                "flat_walk", "rough_walk", "step_walk"
            ]
        if probabilities is None:
            probabilities = [1, 0, 0, 0, 0, 0]
        # [0.05, 0.05, 0.05, 0.35, 0.30, 0.20]
        self.terrain_modes = terrain_modes
        self.probabilities = probabilities

        # 课程学习进度
        self.difficulty = 0.0

        # 动作空间: 13个关节
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )

        # 观测空间（移除支撑脚标识，维度39）
        actor_obs_dim = 13 + 13 + 1 + 3 + 1 + 2 + 3 + 3  # 去掉1维支撑脚
        privileged_obs_dim = 2 + 3 + 13 
        self.observation_space = spaces.Dict({
            "actor_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(actor_obs_dim,), dtype=np.float32),
            "critic_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(actor_obs_dim + privileged_obs_dim,), dtype=np.float32),
        })

        # 内部状态变量
        self.model = None
        self.data = None
        self.step_counter = 0
        self.phase = 0.0
        self.current_stance = -1  # 内部使用，不再出现在观测中
        self.target_footstep = None
        self.next_stance = None
        self.goal_pos = None
        self.terrain_mode = None
        self.terrain_gen = None
        self.vision_processor = None
        self.planner = None
        self.last_action = None

        # 缓存ID
        self.pelvis_id = None
        self.left_foot_id = None
        self.right_foot_id = None
        self.torso_id = None
        self.head_id = None
        self.joint_indices = None
        self.actuator_indices = None
        self.max_torques = None

        
        self.target_radius = 0.17          # 踩中半径 (m)，与 LHW 一致
        self.delay_frames = 10             # 稳定帧数延迟，可根据控制频率调整 (例如 10 帧 ≈ 0.2s)
    
        # 状态变量
        self.target_reached = False
        self.target_reached_frames = 0

        # 终点与摔倒阈值
        self.goal_distance_threshold = 0.5
        self.fall_height_threshold = 0.35

        nominal_list = [
            -0.1,  # left_hip_pitch
            0.0,   # left_hip_roll
            0.0,   # left_hip_yaw
            0.3,   # left_knee
            -0.2,  # left_ankle_pitch
            0.0,   # left_ankle_roll
            -0.1,  # right_hip_pitch
            0.0,   # right_hip_roll
            0.0,   # right_hip_yaw
            0.3,   # right_knee
            -0.2,  # right_ankle_pitch
            0.0,   # right_ankle_roll
            0.0    # waist_pitch
        ]
        self.nominal_angles = np.array(nominal_list)

        self.nominal_pelvis_height = 0.7823   # 标称骨盆高度，单位米
        self.foot_ankle_offset = 0.0331      # 脚踝到脚底的垂直偏移，单位米

        self.action_scale = 0.40

        self.obs_scale = np.array([
            # 1. 关节角度 (最大范围 ~ ±1.5 rad，但考虑到动作幅度可达 0.4，取 1.5 足够)
            1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,  # 13
            # 2. 关节速度 (最大范围 ~ ±10 rad/s)
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,  # 13
            # 3. 骨盆高度 (相对脚底，范围 ~ 0.7~0.9，以 1.0 为缩放)
            1.0,
            # 4. 步点相对位置 (dx, dy, dz，最大距离约 0.5m)
            0.5, 0.3, 0.8,
            # 5. 步点偏航 (最大转向角约 0.1~0.2 rad)
            0.2,
            # 6. 相位 sin/cos (已为 [-1,1]，缩放为 1)
            1.0, 1.0,
            # 7. 骨盆姿态 roll/pitch/yaw (最大约 ±0.3 rad)
            0.3, 0.3, 0.3,
            # 8. 骨盆角速度 (最大约 ±5 rad/s)
            5.0, 5.0, 5.0,
        ])
        assert len(self.obs_scale) == 39, "obs_scale 必须与 actor_obs 维度一致"
        

    def _get_body_linvel(self, body_id):
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id, vel, 0
        )
        return np.linalg.norm(vel[:3])


    # -------------------- 重置 --------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_reached = False
        self.target_reached_frames = 0

        # 根据难度选择地形模式
        if self.difficulty == 0:
            self.terrain_mode = "flat_stand"
        else:
            self.terrain_mode = np.random.choice(self.terrain_modes, p=self.probabilities)

        # 设置终点
        self._set_goal()

        # 生成地形
        if self.terrain_gen is None:
            self.terrain_gen = TerrainGenerator(
                robot_xml_path=self.robot_xml_path,
                mesh_dir=self.mesh_dir
            )

        mode_map = {
            "flat_stand": "flat", "flat_walk": "flat",
            "rough_stand": "rough", "rough_walk": "rough",
            "slope_stand": "slope", "slope_walk": "slope",
            "step_walk": "steps"
        }
        terrain_mode_str = mode_map.get(self.terrain_mode, "flat")
        self.model, self.data = self.terrain_gen.generate(
            mode=terrain_mode_str,
            difficulty=self.difficulty,
            goal_pos=(self.goal_pos[0], self.goal_pos[1])
        )

        # 缓存ID
        self._cache_ids()

        # 重置关键帧
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            self.data.ctrl[:] = self.data.qpos[self.actuator_indices]
        else:
            mujoco.mj_resetData(self.model, self.data)

        # 设置骨盆高度
        self._set_pelvis_height()
        mujoco.mj_forward(self.model, self.data)

        # 初始化步态相位等
        self.phase = 0.0
        self.step_counter = 0
        self.last_action = None

        # 初始化规划器
        if self.planner is None:
            self.planner = G1FootstepPlanner(
                step=0.3, step_width=0.237,
                max_step_len=0.40, min_step_len=0.15,
                max_turn_deg=6.0,
                max_step_height=0.20,
                max_slope_deg=20.0,
                clearance=0.03,
                w_step=1.0, w_angle=0.7, w_slope=0.5,
                step_discretization=0.05,
                turn_discretization=1.0
            )

        # 初始支撑腿（内部使用）
        self.current_stance = -1  # 左脚

        # 创建或更新视觉处理器
        if self.vision_processor is None:
            self.vision_processor = VisionProcessor(
                model=self.model, data=self.data,
                camera_name="chest_camera",
                pelvis_name="pelvis",
                width=320, height=240,
                fov_deg=60.0,
                depth_min=0.3, depth_max=2.0,
                crop_x_min=0.15, crop_x_max=0.8,
                crop_y_min=-0.5, crop_y_max=0.5,
                heightmap_resolution=0.025
            )
        else:
            self.vision_processor.update_model_data(self.model, self.data)

        # 更新高程图
        self._update_terrain_map()

        # 生成第一个目标步点
        if "walk" in self.terrain_mode:
            self._plan_next_footstep(force=True)
        else:
            self._setup_stand_mode()

        # 构建观测并返回
        obs = {
            "actor_obs": self._get_actor_obs(),
            "critic_obs": self._get_critic_obs(),
        }
        info = {"terrain_mode": self.terrain_mode, "difficulty": self.difficulty}
        return obs, info

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
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_pitch_joint"
        ]
        self.joint_indices = []
        self.joint_vel_indices = []
        for name in joint_names:
            try:
                joint = model.joint(name)
                self.joint_indices.append(joint.qposadr[0])
                dof_idx = joint.dofadr[0]
                self.joint_vel_indices.append(dof_idx)
            except Exception as e:
                raise ValueError(f"关节 {name} 未找到: {e}")

        actuator_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_pitch_joint"
        ]
        self.actuator_indices = []
        for name in actuator_names:
            try:
                idx = model.actuator(name).id
                self.actuator_indices.append(idx)
            except Exception as e:
                raise ValueError(f"执行器 {name} 未找到: {e}")

        # 最大力矩（硬编码）
        self.max_torques = np.array([
            88, 139, 88, 139, 50, 50,
            88, 139, 88, 139, 50, 50, 50
        ])

    def _set_pelvis_height(self):
        if "slope" in self.terrain_mode:
            z_terrain = 9.0 * self.difficulty / 4.0
        elif "rough" in self.terrain_mode:
            z_terrain = 0.18 * self.difficulty
        else:
            z_terrain = 0
        self.data.qpos[2] = z_terrain + self.nominal_pelvis_height + 0.01

    def _set_goal(self):
        if "slope" in self.terrain_mode or "step" in self.terrain_mode:
            angle = 0.0
        else:
            angle = np.random.uniform(-np.pi/3, np.pi/3)
        x_goal = self.goal_radius * np.cos(angle)
        y_goal = self.goal_radius * np.sin(angle)
        self.goal_pos = np.array([x_goal, y_goal, 0.0])

    def _setup_stand_mode(self):
        # 获取当前骨盆位置和偏航
        pelvis_pos = self.data.xpos[self.pelvis_id].copy()
        pelvis_quat = self.data.xquat[self.pelvis_id].copy()
        r = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
        yaw = r.as_euler('xyz')[2]

        self.target_footstep = {
            'x': pelvis_pos[0],
            'y': pelvis_pos[1],
            'z': pelvis_pos[2],
            'yaw': yaw,        # 关键：设为当前骨盆偏航，使 foot_yaw = 0
            'foot': 0
        }

    # -------------------- 步点规划与视觉 --------------------
    def _update_terrain_map(self):
        if self.vision_processor is None:
            return
        result = self.vision_processor.process(render_rgb=False, verbose=False)
        height_map = result['heightmap']
        slope_map = result['slopemap']
        x_edges = result['x_edges']
        y_edges = result['y_edges']
        res = 0.025
        self.planner.set_heightmap(height_map, slope_map, x_edges, y_edges, res)

    def _plan_next_footstep(self, force=False):
        if not force:
            pass

        # 获取支撑脚位置（用于规划器）
        stance_foot_id = self.left_foot_id if self.current_stance == -1 else self.right_foot_id
        foot_pos_world = self.data.xpos[stance_foot_id].copy()
        pelvis_pos = self.data.xpos[self.pelvis_id].copy()
        pelvis_quat = self.data.xquat[self.pelvis_id].copy()
        r = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
        yaw = r.as_euler('xyz')[2]
        R_yaw_to_world = R.from_euler('z', yaw).as_matrix()
        R_world_to_pelvis = R_yaw_to_world.T
        local_foot = R_world_to_pelvis @ (foot_pos_world - pelvis_pos)

        goal_local = R_world_to_pelvis @ (self.goal_pos - pelvis_pos)
        goal_local_xy = goal_local[:2]

        footstep, next_stance = self.planner.plan_next_footstep(
            current_foot_pos=(local_foot[0], local_foot[1], local_foot[2]),
            current_stance=self.current_stance,
            target_pos=(goal_local_xy[0], goal_local_xy[1])
        )


        local_pos = np.array([footstep.x, footstep.y, footstep.z])
        world_pos = pelvis_pos + R_yaw_to_world @ local_pos
        world_yaw = yaw + footstep.yaw
        world_yaw = np.arctan2(np.sin(world_yaw), np.cos(world_yaw))

        self.target_footstep = {
            'x': world_pos[0],
            'y': world_pos[1],
            'z': world_pos[2],
            'yaw': world_yaw,
            'foot': footstep.foot
        }
        self.next_stance = next_stance

    # -------------------- 环境步骤 --------------------
    def step(self, action):
        assert self.model is not None, "环境未重置"

        # 应用动作
        self._apply_action(action)

        # 推进物理
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.step_counter += 1

        # 更新相位
        self.phase = (self.step_counter * self.control_dt % 1.1) / 1.1

        # ---------- 站立模式：每步更新目标步点（使观测为 0） ----------
        if "stand" in self.terrain_mode:
            self._setup_stand_mode()

        # ---------- 踩中判定（LHW 风格：任一脚 + 延迟） ----------
        if "walk" in self.terrain_mode and self.target_footstep is not None:
            target_pos = np.array([self.target_footstep['x'],
                                   self.target_footstep['y'],
                                   self.target_footstep['z']])
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
                # 踩中，切换支撑腿
                self.current_stance = -self.current_stance
                self._update_terrain_map()
                self._plan_next_footstep(force=True)
                # 重置状态，避免重复触发
                self.target_reached = False
                self.target_reached_frames = 0

        # 计算奖励
        reward = self._compute_reward(action)

        # 构建观测
        obs = {
            "actor_obs": self._get_actor_obs(),
            "critic_obs": self._get_critic_obs(),
        }

        # 终止判断
        terminated = self._check_termination()
        truncated = self.step_counter >= self.max_episode_steps

        info = {}
        return obs, reward, terminated, truncated, info

    def _apply_action(self, action):
        # 目标角度 = 标称角度 + 动作 * 缩放因子
        target_qpos = self.nominal_angles + action * self.action_scale
        # 裁剪到关节限位
        for i, idx in enumerate(self.actuator_indices):
            low, high = self.model.actuator_ctrlrange[idx]
            target_qpos[i] = np.clip(target_qpos[i], low, high)
        self.data.ctrl[self.actuator_indices] = target_qpos

    def _compute_reward(self, action):
        # 获取脚力、速度等
        left_force = self.data.cfrc_ext[self.left_foot_id][2]
        right_force = self.data.cfrc_ext[self.right_foot_id][2]
        left_vel = self._get_body_linvel(self.left_foot_id)
        right_vel = self._get_body_linvel(self.right_foot_id)

        pelvis_z = self.data.qpos[2]
        foot_z = min(self.data.xpos[self.left_foot_id][2], self.data.xpos[self.right_foot_id][2]) - self.foot_ankle_offset

        pelvis_yaw = self._get_pelvis_yaw()
        target_yaw = self.target_footstep['yaw'] if self.target_footstep is not None else 0.0

        # ---------- LHW 风格步点跟踪奖励：不再需要 swing_pos ----------
        pelvis_xy = self.data.xpos[self.pelvis_id][:2]
        head_xy = self.data.xpos[self.head_id][:2]

        total_mass = sum(self.model.body_mass)
        max_force = total_mass * 9.81 * 0.5

        is_stand = "stand" in self.terrain_mode

        # 计算子奖励
        if is_stand:
            r_frc = calc_foot_frc_clock_reward(
                left_force, right_force,
                self.phase, max_force,
                clock_left=1.0, clock_right=1.0
            )
            r_vel = calc_foot_vel_clock_reward(
                left_vel, right_vel,
                self.phase, 0.2,
                clock_left=-1.0, clock_right=-1.0
            )
        else:
            r_frc = calc_foot_frc_clock_reward(left_force, right_force, self.phase, max_force)
            r_vel = calc_foot_vel_clock_reward(left_vel, right_vel, self.phase, 0.7)

        r_orient = calc_body_orient_reward(pelvis_yaw, target_yaw)
        nominal_pelvis_height = self.nominal_pelvis_height
        r_height = calc_height_reward(pelvis_z, foot_z, goal_height=nominal_pelvis_height, deadzone=0.023)

        # ---------- 使用新的 LHW 风格步点跟踪奖励 ----------
        if self.target_footstep is not None:
            target_pos = np.array([self.target_footstep['x'],
                                   self.target_footstep['y'],
                                   self.target_footstep['z']])
            left_pos = self.data.xpos[self.left_foot_id]
            right_pos = self.data.xpos[self.right_foot_id]
            r_step = calc_step_reward(left_pos, right_pos, target_pos, pelvis_xy, self.target_reached)
        else:
            r_step = 0.0

        r_stability = calc_upper_body_stability(head_xy, pelvis_xy)

        p_action, self.last_action = calc_action_penalty(action, self.last_action)
        torques = self.data.actuator_force[self.actuator_indices]
        p_torque = calc_torque_penalty(torques, self.max_torques)

        current_joint_angles = self.data.qpos[self.joint_indices]  # 13 维
        r_posture = calc_posture_error_reward(current_joint_angles, self.nominal_angles)

        weights = {
            'frc': 0.150, 
            'vel': 0.150, 
            'orient': 0.050,
            'height': 0.050,
            'step': 0.450, 
            'stability': 0.050, 
            'posture': 0.050, 
            'action': 0.000, 
            'torque': 0.000
        }
        total = (weights['frc'] * r_frc +
                 weights['vel'] * r_vel +
                 weights['orient'] * r_orient +
                 weights['height'] * r_height +
                 weights['step'] * r_step +
                 weights['stability'] * r_stability +
                 weights['posture'] * r_posture + 
                 weights['action'] * p_action +
                 weights['torque'] * p_torque)
        return total

    def _get_actor_obs(self):
        qpos = self.data.qpos
        joint_angles = qpos[self.joint_indices]

        qvel = self.data.qvel
        joint_vels = qvel[self.joint_vel_indices]

        pelvis_z = self.data.qpos[2]
        foot_z = min(self.data.xpos[self.left_foot_id][2], self.data.xpos[self.right_foot_id][2]) - self.foot_ankle_offset
        pelvis_height = pelvis_z - foot_z

        if self.target_footstep is not None:
            target_world = np.array([self.target_footstep['x'],
                                     self.target_footstep['y'],
                                     self.target_footstep['z']])
            pelvis_pos = self.data.xpos[self.pelvis_id].copy()
            pelvis_quat = self.data.xquat[self.pelvis_id].copy()
            r = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
            yaw = r.as_euler('xyz')[2]
            R_yaw_to_world = R.from_euler('z', yaw).as_matrix()
            R_world_to_pelvis = R_yaw_to_world.T
            local_target = R_world_to_pelvis @ (target_world - pelvis_pos)
            foot_dx, foot_dy, foot_dz = local_target[0], local_target[1], local_target[2]
            foot_yaw = self.target_footstep['yaw'] - yaw
            foot_yaw = np.arctan2(np.sin(foot_yaw), np.cos(foot_yaw))
        else:
            foot_dx, foot_dy, foot_dz = 0.0, 0.0, 0.0
            foot_yaw = 0.0

        # 步态相位（左右腿）
        phase_val = self.phase
        phase = np.array([np.sin(2 * np.pi * phase_val), np.cos(2 * np.pi * phase_val)])

        pelvis_quat = self.data.xquat[self.pelvis_id].copy()
        r = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
        euler = r.as_euler('xyz')
        roll, pitch, yaw = euler

        pelvis_angvel = self.data.qvel[3:6]

        obs = np.concatenate([
            joint_angles,
            joint_vels,
            [pelvis_height],
            [foot_dx, foot_dy, foot_dz],
            [foot_yaw],
            phase,
            [roll, pitch, yaw],
            pelvis_angvel
        ])
        return obs.astype(np.float32)
    
    def _get_critic_obs(self):
        raw_actor_obs = self._get_actor_obs()
        # 归一化基础观测
        norm_actor_obs = raw_actor_obs / self.obs_scale
        norm_actor_obs = np.clip(norm_actor_obs, -1.0, 1.0)
        priv_info = self._get_privileged_info()
        critic_obs = np.concatenate([norm_actor_obs, priv_info])
        return critic_obs.astype(np.float32)
    
    def _get_privileged_info(self):
        """返回特权信息向量（归一化后）"""
        # 1. 足底力（法向）
        left_frc = self.data.cfrc_ext[self.left_foot_id][2]
        right_frc = self.data.cfrc_ext[self.right_foot_id][2]
        max_force = sum(self.model.body_mass) * 9.81 * 0.5
        norm_left_frc = np.clip(left_frc / max_force, -1.0, 1.0)
        norm_right_frc = np.clip(right_frc / max_force, -1.0, 1.0)

        # 2. 基座世界线速度（自由关节线速度）
        lin_vel = self.data.qvel[0:3]
        max_lin_vel = 2.0  # 根据任务调整
        norm_lin_vel = np.clip(lin_vel / max_lin_vel, -1.0, 1.0)

        # 3. 关节力矩（归一化到最大力矩）
        torques = self.data.actuator_force[self.actuator_indices]
        norm_torques = np.clip(torques / (self.max_torques + 1e-6), -1.0, 1.0)

        # 可添加其他信息（如地面摩擦、坡度等）
        # friction = np.array([self.current_friction])  # 需在 reset 时记录

        return np.concatenate([
            np.array([norm_left_frc]),   
            np.array([norm_right_frc]),  
            norm_lin_vel,                
            norm_torques                 
        ])

    def _get_pelvis_yaw(self):
        quat = self.data.xquat[self.pelvis_id].copy()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return r.as_euler('xyz')[2]

    def _check_termination(self):
        pelvis_z = self.data.qpos[2]
        foot_z = min(self.data.xpos[self.left_foot_id][2], self.data.xpos[self.right_foot_id][2]) - self.foot_ankle_offset
        height = pelvis_z - foot_z
        if height < self.fall_height_threshold:
            return True

        pelvis_xy = self.data.xpos[self.pelvis_id][:2]
        if np.linalg.norm(pelvis_xy - self.goal_pos[:2]) < self.goal_distance_threshold:
            return True
        return False

    def set_difficulty(self, progress: float):
        self.difficulty = np.clip(progress, 0.0, 1.0)

    def render(self):
        raise NotImplementedError("该环境不支持实时渲染，请使用独立的评估脚本。")

    def close(self):
        pass