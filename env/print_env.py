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
        total_timesteps_for_max: int = 11000 * 1500,
        max_episode_steps: int = 2000,
        control_dt: float = 0.02,
        physics_dt: float = 0.001,
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
        obs_dim = 13 + 13 + 1 + 3 + 1 + 1 + 4 + 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 内部状态变量
        self.model = None
        self.data = None
        self.step_counter = 0
        self.phase = 0.0
        self.current_stance = -1
        self.target_footstep = None
        self.next_stance = None
        self.goal_pos = None
        self.terrain_mode = None
        self.terrain_gen = None
        self.vision_processor = None
        self.planner = None
        self.last_action = None

        # 缓存MuJoCo ID
        self.pelvis_id = None
        self.left_foot_id = None
        self.right_foot_id = None
        self.torso_id = None
        self.head_id = None
        self.joint_indices = None
        self.actuator_indices = None

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
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            vel,
            0
        )
        return np.linalg.norm(vel[:3])

    # -------------------- 初始化与重置 --------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.difficulty < 0.1:
            stand_modes = ["flat_stand", "rough_stand", "slope_stand"]
            self.terrain_mode = np.random.choice(stand_modes)
        else:
            self.terrain_mode = np.random.choice(self.terrain_modes, p=self.probabilities)

        # DEBUG: 打印重置信息
        print(f"[RESET] terrain_mode: {self.terrain_mode}, difficulty: {self.difficulty:.3f}")

        self._set_goal()
        print(f"[RESET] goal_pos: {self.goal_pos}")

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

        self._cache_ids()

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            self.data.ctrl[:] = self.data.qpos[self.actuator_indices]
        else:
            mujoco.mj_resetData(self.model, self.data)

        self._set_pelvis_height()
        mujoco.mj_forward(self.model, self.data)

        self.phase = 0.0
        self.step_counter = 0
        self.last_action = None

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

        self.current_stance = -1

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

        self._update_terrain_map()

        if "walk" in self.terrain_mode:
            self._plan_next_footstep(force=True)
        else:
            self._setup_stand_mode()

        obs = self._get_obs()
        info = {"terrain_mode": self.terrain_mode, "difficulty": self.difficulty}
        print(f"[RESET] initial obs shape: {obs.shape}, first 10 values: {obs[:10]}")
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

        self.max_torques = np.array([
            88, 139, 88, 139, 50, 50,
            88, 139, 88, 139, 50, 50, 50
        ])

    def _set_pelvis_height(self):
        if "slope" in self.terrain_mode:
            z_terrain = 9.0 * self.difficulty / 4.0
        elif "rough" in self.terrain_mode:
            z_terrain = 0.15
        else:
            z_terrain = 0
        self.data.qpos[2] = z_terrain + 0.80
        print(f"[SET_PELVIS] z_terrain={z_terrain:.3f}, pelvis z={self.data.qpos[2]:.3f}")

    def _set_goal(self):
        if "slope" in self.terrain_mode or "step" in self.terrain_mode:
            angle = 0.0
        else:
            angle = np.random.uniform(-np.pi/3, np.pi/3)
        x_goal = self.goal_radius * np.cos(angle)
        y_goal = self.goal_radius * np.sin(angle)
        self.goal_pos = np.array([x_goal, y_goal, 0.0])

    def _setup_stand_mode(self):
        swing_leg = 1 if self.current_stance == -1 else -1
        foot_id = self.left_foot_id if swing_leg == -1 else self.right_foot_id
        foot_pos_world = self.data.xpos[foot_id].copy()
        pelvis_pos = self.data.xpos[self.pelvis_id].copy()
        pelvis_quat = self.data.xquat[self.pelvis_id].copy()
        r = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
        yaw = r.as_euler('xyz')[2]
        R_yaw_to_world = R.from_euler('z', yaw).as_matrix()
        R_world_to_pelvis = R_yaw_to_world.T
        local_pos = R_world_to_pelvis @ (foot_pos_world - pelvis_pos)
        self.target_footstep = {
            'x': foot_pos_world[0],
            'y': foot_pos_world[1],
            'z': foot_pos_world[2],
            'yaw': yaw,
            'foot': swing_leg
        }
        print(f"[STAND] target_footstep set to: {self.target_footstep}")

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
        print(f"[PLAN] force={force}, 当前支撑腿: {'左' if self.current_stance==-1 else '右'}")
        if self.planner is None:
            print("错误：规划器未初始化")
            return

        print(f"高程图尺寸: {self.planner.height_map.shape}")
        if self.planner.x_edges is not None:
            print(f"高程图范围: x=[{self.planner.x_edges[0]:.2f}, {self.planner.x_edges[-1]:.2f}], "
                  f"y=[{self.planner.y_edges[0]:.2f}, {self.planner.y_edges[-1]:.2f}]")

        if not force and self.target_footstep is not None:
            swing_leg = self.target_footstep['foot']
            foot_id = self.left_foot_id if swing_leg == -1 else self.right_foot_id
            foot_pos = self.data.xpos[foot_id]
            target_pos = np.array([self.target_footstep['x'],
                                   self.target_footstep['y'],
                                   self.target_footstep['z']])
            dx = foot_pos[0] - target_pos[0]
            dy = foot_pos[1] - target_pos[1]
            if np.hypot(dx, dy) > self.footstep_threshold:
                print(f"[PLAN] 未踩中，距离={np.hypot(dx, dy):.3f} > 阈值{self.footstep_threshold}")
                return

        stance_foot_id = self.left_foot_id if self.current_stance == -1 else self.right_foot_id
        foot_pos_world = self.data.xpos[stance_foot_id].copy()
        print(f"[PLAN] 支撑脚世界坐标: ({foot_pos_world[0]:.3f}, {foot_pos_world[1]:.3f}, {foot_pos_world[2]:.3f})")

        pelvis_pos = self.data.xpos[self.pelvis_id].copy()
        pelvis_quat = self.data.xquat[self.pelvis_id].copy()
        r = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
        yaw = r.as_euler('xyz')[2]
        R_yaw_to_world = R.from_euler('z', yaw).as_matrix()
        R_world_to_pelvis = R_yaw_to_world.T
        local_foot = R_world_to_pelvis @ (foot_pos_world - pelvis_pos)
        print(f"[PLAN] 支撑脚骨盆坐标: ({local_foot[0]:.3f}, {local_foot[1]:.3f}, {local_foot[2]:.3f})")

        goal_local = R_world_to_pelvis @ (self.goal_pos - pelvis_pos)
        goal_local_xy = goal_local[:2]

        footstep, next_stance = self.planner.plan_next_footstep(
            current_foot_pos=(local_foot[0], local_foot[1], local_foot[2]),
            current_stance=self.current_stance,
            target_pos=(goal_local_xy[0], goal_local_xy[1])
        )

        if footstep is not None:
            print(f"[PLAN] 获得有效步点: ({footstep.x:.3f}, {footstep.y:.3f}, {footstep.z:.3f}) 偏航: {np.degrees(footstep.yaw):.1f}° 脚: {'左' if footstep.foot==-1 else '右'}")
        else:
            print("[PLAN] 规划器返回 None")

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

        # DEBUG: 打印动作信息
        print(f"[STEP] step_counter={self.step_counter}, action mean={np.mean(action):.4f}, std={np.std(action):.4f}")

        self._apply_action(action)

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.step_counter += 1

        self.phase = (self.step_counter * self.control_dt % 1.1) / 1.1

        # 踩中判定
        if self.target_footstep is not None:
            swing_foot = self.target_footstep['foot']
            foot_id = self.left_foot_id if swing_foot == -1 else self.right_foot_id
            foot_pos = self.data.xpos[foot_id]
            target_pos = np.array([self.target_footstep['x'],
                                   self.target_footstep['y'],
                                   self.target_footstep['z']])
            dx = foot_pos[0] - target_pos[0]
            dy = foot_pos[1] - target_pos[1]
            dist = np.hypot(dx, dy)
            if dist < self.footstep_threshold:
                print(f"[STEP] 踩中！距离={dist:.3f}，切换支撑腿")
                self.current_stance = -self.current_stance
                if "walk" in self.terrain_mode:
                    self._update_terrain_map()
                    self._plan_next_footstep(force=True)
                else:
                    self._setup_stand_mode()

        reward = self._compute_reward(action)
        print(f"[STEP] reward={reward:.4f}")

        obs = self._get_obs()
        terminated = self._check_termination()
        truncated = self.step_counter >= self.max_episode_steps

        if terminated:
            print(f"[STEP] terminated=True (摔倒或到达终点)")
        if truncated:
            print(f"[STEP] truncated=True (超时)")

        info = {}
        return obs, reward, terminated, truncated, info

    def _apply_action(self, action):
        qpos = self.data.qpos
        max_delta = 0.2
        target_qpos = qpos[self.joint_indices] + action * max_delta
        for i, idx in enumerate(self.actuator_indices):
            low, high = self.model.actuator_ctrlrange[idx]
            target_qpos[i] = np.clip(target_qpos[i], low, high)
        self.data.ctrl[self.actuator_indices] = target_qpos
        # DEBUG: 打印前几个关节目标
        # print(f"[APPLY] target_qpos[:5]={target_qpos[:5]}")

    def _compute_reward(self, action):
        left_force = self.data.cfrc_ext[self.left_foot_id][2]
        right_force = self.data.cfrc_ext[self.right_foot_id][2]
        left_vel = self._get_body_linvel(self.left_foot_id)
        right_vel = self._get_body_linvel(self.right_foot_id)

        pelvis_z = self.data.qpos[2]
        stance_foot_id = self.left_foot_id if self.current_stance == -1 else self.right_foot_id
        foot_z = self.data.xpos[stance_foot_id][2]

        pelvis_yaw = self._get_pelvis_yaw()
        target_yaw = self.target_footstep['yaw'] if self.target_footstep is not None else 0.0

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

        total_mass = sum(self.model.body_mass)
        max_force = total_mass * 9.81 * 0.5

        is_stand = "stand" in self.terrain_mode

        if is_stand:
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

        p_action, self.last_action = calc_action_penalty(action, self.last_action)
        torques = self.data.actuator_force[self.actuator_indices]
        p_torque = calc_torque_penalty(torques, self.max_torques)

        weights = {
            'frc': 0.145, 'vel': 0.145, 'orient': 0.150,
            'height': 0.150,
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

        # DEBUG: 打印各子奖励
        print(f"[REWARD] r_frc={r_frc:.3f}, r_vel={r_vel:.3f}, r_orient={r_orient:.3f}, r_height={r_height:.3f}, r_step={r_step:.3f}, r_stability={r_stability:.3f}, p_action={p_action:.3f}, p_torque={p_torque:.3f}, total={total:.3f}")
        return total

    def _get_obs(self):
        qpos = self.data.qpos
        joint_angles = qpos[self.joint_indices]

        qvel = self.data.qvel
        joint_vels = qvel[self.joint_vel_indices]

        pelvis_z = self.data.qpos[2]
        stance_foot = self.left_foot_id if self.current_stance == -1 else self.right_foot_id
        foot_z = self.data.xpos[stance_foot][2]
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

        stance_flag = self.current_stance

        phase_left = self.phase
        phase_right = (self.phase + 0.5) % 1.0
        sin_left = np.sin(2 * np.pi * phase_left)
        cos_left = np.cos(2 * np.pi * phase_left)
        sin_right = np.sin(2 * np.pi * phase_right)
        cos_right = np.cos(2 * np.pi * phase_right)
        phase = np.array([sin_left, cos_left, sin_right, cos_right])

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
            [stance_flag],
            phase,
            [roll, pitch, yaw],
            pelvis_angvel
        ])
        return obs.astype(np.float32)

    def _get_pelvis_yaw(self):
        quat = self.data.xquat[self.pelvis_id].copy()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return r.as_euler('xyz')[2]

    def _check_termination(self):
        pelvis_z = self.data.qpos[2]
        stance_foot = self.left_foot_id if self.current_stance == -1 else self.right_foot_id
        foot_z = self.data.xpos[stance_foot][2]
        height = pelvis_z - foot_z
        if height < self.fall_height_threshold:
            print(f"[TERMINATE] 摔倒：骨盆高度 {height:.3f} < 阈值 {self.fall_height_threshold}")
            return True

        pelvis_xy = self.data.xpos[self.pelvis_id][:2]
        dist_to_goal = np.linalg.norm(pelvis_xy - self.goal_pos[:2])
        if dist_to_goal < self.goal_distance_threshold:
            print(f"[TERMINATE] 到达终点：距离 {dist_to_goal:.3f} < 阈值 {self.goal_distance_threshold}")
            return True

        return False

    # -------------------- 课程学习接口 --------------------
    def set_difficulty(self, progress: float):
        self.difficulty = np.clip(progress, 0.0, 1.0)
        print(f"[SET_DIFFICULTY] progress={progress:.3f}, difficulty={self.difficulty:.3f}")

    # -------------------- 可选渲染 (无) --------------------
    def render(self):
        raise NotImplementedError("该环境不支持实时渲染，请使用独立的评估脚本。")

    def close(self):
        pass