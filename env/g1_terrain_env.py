'''
初始版本
单个脚跟踪
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
    calc_step_reward,
    calc_upper_body_stability,
    calc_action_penalty,
    calc_torque_penalty,
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
        total_timesteps_for_max: int = 11000 * 1500,  # 预估每轮1500步
        max_episode_steps: int = 2000,
        control_dt: float = 0.02,
        physics_dt: float = 0.001,
        goal_radius: float = 7.5,
        **kwargs
    ):
        """
        初始化环境。

        :param robot_xml_path: 处理后的G1 XML文件路径
        :param mesh_dir: 网格文件目录
        :param terrain_modes: 地形模式列表，默认使用提供的六种
        :param probabilities: 各模式概率
        :param total_timesteps_for_max: 达到最大难度所需的总步数（课程学习）
        :param max_episode_steps: 单回合最大步数
        :param control_dt: 控制周期 (s)
        :param physics_dt: 物理积分步长 (s)
        :param goal_radius: 终点距离半径 (m)
        """
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
            probabilities = [0.05, 0.05, 0.05, 0.35, 0.30, 0.20]
        self.terrain_modes = terrain_modes
        self.probabilities = probabilities

        # 课程学习进度 (由外部回调设置)
        self.difficulty = 0.0

        # 动作空间: 13个关节的增量控制 (范围-1..1)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(13,), dtype=np.float32
        )

        # 观测空间
        obs_dim = 13 + 13 + 1 + 3 + 1 + 1 + 4 + 3 + 3  # 关节角度(13)+速度(13)+骨盆高(1)+落脚点相对位置(3)+偏航(1)+支撑脚(1)+相位(2)+欧拉角(3)+角速度(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 内部状态变量
        self.model = None
        self.data = None
        self.step_counter = 0
        self.phase = 0.0
        self.current_stance = -1  # -1: 左脚支撑, 1: 右脚支撑
        self.target_footstep = None  # 当前目标步点 (世界坐标系)
        self.next_stance = None
        self.goal_pos = None  # 世界坐标系 (x, y, 0)
        self.terrain_mode = None
        self.terrain_gen = None
        self.vision_processor = None
        self.planner = None
        self.last_action = None  # 用于动作平滑惩罚

        # 缓存MuJoCo ID
        self.pelvis_id = None
        self.left_foot_id = None
        self.right_foot_id = None
        self.torso_id = None
        self.head_id = None
        self.joint_indices = None  # 关节在qpos中的索引
        self.actuator_indices = None  # 执行器索引 (顺序与动作空间对应)

        # 最大力矩 (用于力矩惩罚)
        self.max_torques = None

        # 步点踩中阈值
        self.footstep_threshold = 0.08

        # 终点距离阈值 (判定到达)
        self.goal_distance_threshold = 0.5

        # 摔倒判定高度阈值 (骨盆离地高度)
        self.fall_height_threshold = 0.35

    def _get_body_linvel(self, body_id):
        """获取 body 在世界坐标系下的线速度模长。"""
        vel = np.zeros(6)  # 前3个是线速度，后3个是角速度
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            vel,
            0  # 0表示世界坐标系
        )
        return np.linalg.norm(vel[:3])

    # -------------------- 初始化与重置 --------------------
    def reset(self, seed=None, options=None):
        """重置环境，返回初始观测和info字典。"""
        super().reset(seed=seed)


        if self.difficulty < 0.1:  # 训练初期，只使用站立模式
            # 从所有站立模式中随机选一个（flat_stand, rough_stand, slope_stand）
            stand_modes = ["flat_stand", "rough_stand", "slope_stand"]
            self.terrain_mode = np.random.choice(stand_modes)
        else:
            # 正常随机选择
            self.terrain_mode = np.random.choice(self.terrain_modes, p=self.probabilities)


        # 2. 设置终点 (世界坐标系)
        self._set_goal()

        # 3. 生成地形
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
        # 难度传递给地形生成器
        self.model, self.data = self.terrain_gen.generate(
            mode=terrain_mode_str,
            difficulty=self.difficulty,
            goal_pos=(self.goal_pos[0], self.goal_pos[1])  # 终点标记柱位置
        )

        # 4. 缓存ID
        self._cache_ids()

        # 5. 重置关键帧 (stand)
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            # 同步电机命令
            self.data.ctrl[:] = self.data.qpos[self.actuator_indices]
        else:
            mujoco.mj_resetData(self.model, self.data)

        # 6. 设置骨盆高度 (根据地形)
        self._set_pelvis_height()

        # 更新派生量（确保 xquat、xpos 等有效）
        mujoco.mj_forward(self.model, self.data)

        # 7. 初始化步态相位
        self.phase = 0.0
        self.step_counter = 0
        self.last_action = None

        # 8. 初始化步点规划器 (如果需要)
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

        # 9. 初始支撑腿: 左脚 (使得第一步迈右脚)
        self.current_stance = -1  # 左脚支撑

        # 10. 创建视觉处理器（必须在规划之前）
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

        # 11. 更新高程图 (初次)
        self._update_terrain_map()

        # 12. 生成第一个目标步点 (如果是行走模式)
        if "walk" in self.terrain_mode:
            self._plan_next_footstep(force=True)
        else:
            # 站立模式：使用虚拟步点 (当前摆动脚的位置)
            self._setup_stand_mode()

        # 13. 构建初始观测
        obs = self._get_obs()
        info = {"terrain_mode": self.terrain_mode, "difficulty": self.difficulty}
        return obs, info

    def _cache_ids(self):
        """缓存常用body、joint、actuator的ID。"""
        model = self.model

        # Body IDs
        self.pelvis_id = model.body("pelvis").id
        self.left_foot_id = model.body("left_ankle_roll_link").id
        self.right_foot_id = model.body("right_ankle_roll_link").id
        self.torso_id = model.body("torso_link").id
        self.head_id = self.torso_id

        # 关节索引 (在qpos和qvel中的位置)
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
                self.joint_vel_indices.append(dof_idx)  # qvel 索引即为自由度索引
            except Exception as e:
                raise ValueError(f"关节 {name} 未找到: {e}")

        # 执行器索引 (在ctrl中的顺序)
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

        # 最大力矩（直接硬编码，单位 Nm）
        # 顺序与 actuator_indices 一致（即 13 个执行器）
        self.max_torques = np.array([
            88,   # left_hip_pitch
            139,  # left_hip_roll
            88,   # left_hip_yaw
            139,  # left_knee
            50,   # left_ankle_pitch
            50,   # left_ankle_roll
            88,   # right_hip_pitch
            139,  # right_hip_roll
            88,   # right_hip_yaw
            139,  # right_knee
            50,   # right_ankle_pitch
            50,   # right_ankle_roll
            50,   # waist_pitch
        ])

    def _set_pelvis_height(self):
        """根据地形模式设置骨盆高度。"""

        # 计算地形高度 (需要从地形生成器获取函数，这里简化处理)
        # 实际应用中应查询地形高度，我们根据模式经验设置
        if "slope" in self.terrain_mode:
            # 斜坡: 高度为 0.8 + 9.0*difficulty/4 (用户指定)
            z_terrain = 9.0 * self.difficulty / 4.0
        elif "rough" in self.terrain_mode:
            # 起伏面: 设为1.0 (经验)
            z_terrain = 0.15
        else:
            # 平地/台阶: 0.8
            z_terrain = 0

        self.data.qpos[2] = z_terrain + 0.80

    def _set_goal(self):
        """设置终点坐标 (世界坐标系)。"""
        if "slope" in self.terrain_mode or "step" in self.terrain_mode:
            # 台阶/斜坡: 正前方7.5m
            angle = 0.0
        else:
            # 平地/起伏: 随机 -60 ~ 60 度
            angle = np.random.uniform(-np.pi/3, np.pi/3)

        x_goal = self.goal_radius * np.cos(angle)
        y_goal = self.goal_radius * np.sin(angle)
        self.goal_pos = np.array([x_goal, y_goal, 0.0])

    def _setup_stand_mode(self):
        """站立模式：设置虚拟步点 (当前摆动脚位置)。"""
        # 确定摆动脚 (与支撑腿相反)
        swing_leg = 1 if self.current_stance == -1 else -1
        foot_id = self.left_foot_id if swing_leg == -1 else self.right_foot_id
        foot_pos_world = self.data.xpos[foot_id].copy()
        # 转换为骨盆坐标系 (仅偏航)
        pelvis_pos = self.data.xpos[self.pelvis_id].copy()
        pelvis_quat = self.data.xquat[self.pelvis_id].copy()
        r = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
        yaw = r.as_euler('xyz')[2]
        R_yaw_to_world = R.from_euler('z', yaw).as_matrix()
        R_world_to_pelvis = R_yaw_to_world.T
        local_pos = R_world_to_pelvis @ (foot_pos_world - pelvis_pos)
        # 构造虚拟步点 (世界坐标系下存储)
        self.target_footstep = {
            'x': foot_pos_world[0],
            'y': foot_pos_world[1],
            'z': foot_pos_world[2],
            'yaw': yaw,
            'foot': swing_leg
        }

    # -------------------- 步点规划与视觉 --------------------
    def _update_terrain_map(self):
        """调用视觉处理器更新高程图。"""
        if self.vision_processor is None:
            return
        # 处理视觉，但注意不要每步都做，只在这里触发
        result = self.vision_processor.process(render_rgb=False, verbose=False)
        height_map = result['heightmap']
        slope_map = result['slopemap']
        x_edges = result['x_edges']
        y_edges = result['y_edges']
        res = 0.025
        self.planner.set_heightmap(height_map, slope_map, x_edges, y_edges, res)

    def _plan_next_footstep(self, force=False):
        """
        规划下一个步点。如果force=True，即使未踩中也强制规划。
        """
        print(f"[规划] force={force}, 当前支撑腿: {'左' if self.current_stance==-1 else '右'}")
        if self.planner is None:
            print("错误：高程图为 None，请检查 _update_terrain_map() 是否被调用")
            return

        print(f"高程图尺寸: {self.planner.height_map.shape}")
        if self.planner.x_edges is not None:
            print(f"高程图有效范围: x=[{self.planner.x_edges[0]:.2f}, {self.planner.x_edges[-1]:.2f}], "
            f"y=[{self.planner.y_edges[0]:.2f}, {self.planner.y_edges[-1]:.2f}]")

        # 如果当前有目标步点，检查是否已踩中，未踩中且不强制则跳过
        if not force and self.target_footstep is not None:
            # 检查移动脚是否踩中目标步点 (世界坐标系)
            swing_leg = self.target_footstep['foot']
            foot_id = self.left_foot_id if swing_leg == -1 else self.right_foot_id
            foot_pos = self.data.xpos[foot_id]
            target_pos = np.array([self.target_footstep['x'],
                                   self.target_footstep['y'],
                                   self.target_footstep['z']])
            dx = foot_pos[0] - target_pos[0]
            dy = foot_pos[1] - target_pos[1]
            if np.hypot(dx, dy) > self.footstep_threshold:
                return  # 未踩中，保持原步点

        # 获取支撑脚位置 (当前支撑腿)
        stance_foot_id = self.left_foot_id if self.current_stance == -1 else self.right_foot_id
        foot_pos_world = self.data.xpos[stance_foot_id].copy()
        print(f"[规划] 支撑脚世界坐标: ({foot_pos_world[0]:.3f}, {foot_pos_world[1]:.3f}, {foot_pos_world[2]:.3f})")

        # 转换为骨盆坐标系
        pelvis_pos = self.data.xpos[self.pelvis_id].copy()
        pelvis_quat = self.data.xquat[self.pelvis_id].copy()
        r = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
        yaw = r.as_euler('xyz')[2]
        R_yaw_to_world = R.from_euler('z', yaw).as_matrix()
        R_world_to_pelvis = R_yaw_to_world.T
        local_foot = R_world_to_pelvis @ (foot_pos_world - pelvis_pos)
        print(f"[规划] 支撑脚骨盆坐标: ({local_foot[0]:.3f}, {local_foot[1]:.3f}, {local_foot[2]:.3f})")

        # 目标终点 (在骨盆坐标系中)
        goal_local = R_world_to_pelvis @ (self.goal_pos - pelvis_pos)
        goal_local_xy = goal_local[:2]

        # 调用规划器
        footstep, next_stance = self.planner.plan_next_footstep(
            current_foot_pos=(local_foot[0], local_foot[1], local_foot[2]),
            current_stance=self.current_stance,
            target_pos=(goal_local_xy[0], goal_local_xy[1])
        )

        if footstep is not None:
            print(f"[规划] 获得有效步点: ({footstep.x:.3f}, {footstep.y:.3f}, {footstep.z:.3f}) 偏航: {np.degrees(footstep.yaw):.1f}° 脚: {'左' if footstep.foot==-1 else '右'}")
        else:
            print("[规划] 规划器返回 None")

        # 将步点转换为世界坐标系并存储
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

        # 更新当前支撑腿 (实际在踩中时更新，但规划后我们预存)
        # 注意：实际踩中才会切换，我们在step中处理

    # -------------------- 环境步骤 --------------------
    def step(self, action):
        """
        执行一步控制。
        :param action: 13维动作 (归一化增量)
        :return: obs, reward, terminated, truncated, info
        """
        assert self.model is not None, "环境未重置"

        # 1. 应用动作 (增量控制)
        self._apply_action(action)

        # 2. 推进物理 (多子步)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_counter += 1

        # 3. 更新步态相位 (基于仿真时间)
        self.phase = (self.step_counter * self.control_dt % 1.1) / 1.1

        # 4. 检查是否需要规划新步点 (踩中判定)
        if self.target_footstep is not None:
            # 获取当前摆动脚位置 (与目标步点对应的脚)
            swing_foot = self.target_footstep['foot']
            foot_id = self.left_foot_id if swing_foot == -1 else self.right_foot_id
            foot_pos = self.data.xpos[foot_id]
            target_pos = np.array([self.target_footstep['x'],
                                   self.target_footstep['y'],
                                   self.target_footstep['z']])
            dx = foot_pos[0] - target_pos[0]
            dy = foot_pos[1] - target_pos[1]
            if np.hypot(dx, dy) < self.footstep_threshold:
                # 踩中，切换支撑腿
                self.current_stance = -self.current_stance
                # 根据模式更新步点
                if "walk" in self.terrain_mode:
                    # 行走模式：更新高程图并规划新步点
                    self._update_terrain_map()
                    self._plan_next_footstep(force=True)
                else:
                    # 站立模式：重新生成虚拟步点（基于新的支撑腿）
                    self._setup_stand_mode()

        # 5. 计算奖励
        reward = self._compute_reward(action)

        # 6. 构建观测
        obs = self._get_obs()

        # 7. 判断终止条件
        terminated = self._check_termination()
        truncated = self.step_counter >= self.max_episode_steps

        # 8. info (无额外信息)
        info = {}

        return obs, reward, terminated, truncated, info

    def _apply_action(self, action):
        """将动作应用到执行器 (增量控制)。"""
        # 当前关节角度
        qpos = self.data.qpos
        # 目标角度 = 当前角度 + action * max_delta (缩放因子)
        # 缩放因子根据关节范围设置 (这里简单使用0.2)
        max_delta = 0.2  # 可调整
        target_qpos = qpos[self.joint_indices] + action * max_delta
        # 裁剪到关节限位 (使用ctrlrange)
        for i, idx in enumerate(self.actuator_indices):
            low, high = self.model.actuator_ctrlrange[idx]
            target_qpos[i] = np.clip(target_qpos[i], low, high)
        # 设置ctrl
        self.data.ctrl[self.actuator_indices] = target_qpos

    def _compute_reward(self, action):
        """计算总奖励 (调用外部奖励函数)。"""
        # 获取所需状态
        left_force = self.data.cfrc_ext[self.left_foot_id][2]
        right_force = self.data.cfrc_ext[self.right_foot_id][2]
        left_vel = self._get_body_linvel(self.left_foot_id)
        right_vel = self._get_body_linvel(self.right_foot_id)

        pelvis_z = self.data.qpos[2]
        # 支撑脚高度 (当前支撑腿)
        stance_foot_id = self.left_foot_id if self.current_stance == -1 else self.right_foot_id
        foot_z = self.data.xpos[stance_foot_id][2]

        pelvis_yaw = self._get_pelvis_yaw()
        # 目标偏航 (从规划步点获取)
        target_yaw = self.target_footstep['yaw'] if self.target_footstep is not None else 0.0

        # 摆动脚位置 (与目标步点对应的脚)
        if self.target_footstep is not None:
            swing_foot = self.target_footstep['foot']
            swing_id = self.left_foot_id if swing_foot == -1 else self.right_foot_id
            swing_pos = self.data.xpos[swing_id]
            target_pos = np.array([self.target_footstep['x'],
                                   self.target_footstep['y'],
                                   self.target_footstep['z']])
        else:
            swing_pos = np.zeros(3)
            target_pos = np.zeros(3)

        pelvis_xy = self.data.xpos[self.pelvis_id][:2]
        goal_xy = self.goal_pos[:2]
        head_xy = self.data.xpos[self.head_id][:2]

        # 最大足底力
        total_mass = sum(self.model.body_mass)
        max_force = total_mass * 9.81 * 0.5

        # === 判断是否为站立模式 ===
        is_stand = "stand" in self.terrain_mode

        # 计算各子奖励
        if is_stand:
            # 站立模式：强制期望力为 1（踩实），期望速度为 -1（静止）
            r_frc = calc_foot_frc_clock_reward(
                left_force, right_force, 
                self.phase, max_force,
                clock_left=1.0, clock_right=1.0
            )
            r_vel = calc_foot_vel_clock_reward(
                left_vel, right_vel, 
                self.phase, 0.7,
                clock_left=-1.0, clock_right=-1.0
            )
        else:
            r_frc = calc_foot_frc_clock_reward(left_force, right_force, self.phase, max_force)
            r_vel = calc_foot_vel_clock_reward(left_vel, right_vel, self.phase, 0.7)

        r_orient = calc_body_orient_reward(pelvis_yaw, target_yaw)
        r_height = calc_height_reward(pelvis_z, foot_z, goal_height=0.75, deadzone=0.0235)
        r_step = calc_step_reward(swing_pos, target_pos, pelvis_xy, goal_xy)
        r_stability = calc_upper_body_stability(head_xy, pelvis_xy)

        # 惩罚项
        p_action, self.last_action = calc_action_penalty(action, self.last_action)
        torques = self.data.actuator_force[self.actuator_indices]
        p_torque = calc_torque_penalty(torques, self.max_torques)

        # 加权求和 (调整高度权重以强化站立稳定性)
        weights = {
            'frc': 0.145, 'vel': 0.145, 'orient': 0.150,
            'height': 0.150,          # 从 0.050 提高至 0.150
            'step': 0.450, 'stability': 0.050,
            'action': 0.005, 'torque': 0.005
        }
        total = (weights['frc'] * r_frc +
                 weights['vel'] * r_vel +
                 weights['orient'] * r_orient +
                 weights['height'] * r_height +
                 weights['step'] * r_step +
                 weights['stability'] * r_stability +
                 weights['action'] * p_action +
                 weights['torque'] * p_torque)

        return total

    def _get_obs(self):
        """构建观测向量。"""
        # 关节角度 (13)
        qpos = self.data.qpos
        joint_angles = qpos[self.joint_indices]

        # 关节速度 (13) - 使用 qvel 的正确索引
        qvel = self.data.qvel
        joint_vels = qvel[self.joint_vel_indices]

        # 骨盆高度 (离地高度)
        pelvis_z = self.data.qpos[2]
        stance_foot = self.left_foot_id if self.current_stance == -1 else self.right_foot_id
        foot_z = self.data.xpos[stance_foot][2]
        pelvis_height = pelvis_z - foot_z

        # 落脚点相对位置 (骨盆坐标系)
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

        # 支撑脚标识
        stance_flag = self.current_stance

        # 步态相位 (左右腿)
        phase_left = self.phase
        phase_right = (self.phase + 0.5) % 1.0
        sin_left = np.sin(2 * np.pi * phase_left)
        cos_left = np.cos(2 * np.pi * phase_left)
        sin_right = np.sin(2 * np.pi * phase_right)
        cos_right = np.cos(2 * np.pi * phase_right)
        phase = np.array([sin_left, cos_left, sin_right, cos_right])

        # 骨盆欧拉角
        pelvis_quat = self.data.xquat[self.pelvis_id].copy()
        r = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
        euler = r.as_euler('xyz')
        roll, pitch, yaw = euler

        # 骨盆角速度 (世界坐标系)
        pelvis_angvel = self.data.qvel[3:6]  # 自由关节的角速度分量

        # 拼接
        obs = np.concatenate([
            joint_angles,
            joint_vels,
            [pelvis_height],
            [foot_dx, foot_dy, foot_dz],
            [foot_yaw],
            [stance_flag],
            phase,
            [roll, pitch, yaw],
            pelvis_angvel
        ])
        return obs.astype(np.float32)

    def _get_pelvis_yaw(self):
        """获取骨盆偏航角。"""
        quat = self.data.xquat[self.pelvis_id].copy()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return r.as_euler('xyz')[2]

    def _check_termination(self):
        """检查是否终止 (摔倒或到达终点)。"""
        # 摔倒判定：骨盆离地高度 < 阈值
        pelvis_z = self.data.qpos[2]
        stance_foot = self.left_foot_id if self.current_stance == -1 else self.right_foot_id
        foot_z = self.data.xpos[stance_foot][2]
        height = pelvis_z - foot_z
        if height < self.fall_height_threshold:
            return True

        # 到达终点：骨盆xy距离目标 < 阈值
        pelvis_xy = self.data.xpos[self.pelvis_id][:2]
        if np.linalg.norm(pelvis_xy - self.goal_pos[:2]) < self.goal_distance_threshold:
            return True

        return False

    # -------------------- 课程学习接口 --------------------
    def set_difficulty(self, progress: float):
        """
        设置当前环境难度进度 (0~1)，由外部回调调用。
        """
        self.difficulty = np.clip(progress, 0.0, 1.0)

    # -------------------- 可选渲染 (无) --------------------
    def render(self):
        raise NotImplementedError("该环境不支持实时渲染，请使用独立的评估脚本。")

    def close(self):
        pass