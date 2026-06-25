"""
奖励函数模块
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def clock_frc(phase, swing_frac=0.4545, relax=0.1):
    """
    计算足底力/速度期望时钟信号。
    返回 -1 (支撑相) 到 +1 (摆动相) 之间的值。
    
    :param phase: 步态相位 [0, 1)
    :param swing_frac: 摆动相占周期比例
    :param relax: 过渡区松弛度
    :return: 时钟信号 [-1, 1]
    """
    lower = swing_frac * (1 - relax)
    upper = swing_frac * (1 + relax)

    if phase < lower:
        return -1.0
    elif phase < upper:
        t = (phase - lower) / (upper - lower)
        return -1.0 + 2.0 * t
    else:
        return 1.0


def get_pelvis_yaw(data, pelvis_id):
    """从 MuJoCo data 中提取骨盆偏航角"""
    quat = data.xquat[pelvis_id].copy()  # (w,x,y,z)
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler = r.as_euler('xyz')
    return euler[2]


def calc_foot_frc_clock_reward(left_force, right_force, phase, max_force):
    """
    足底力相位匹配奖励。
    
    :param left_force: 左脚法向力 
    :param right_force: 右脚法向力 
    :param phase: 当前步态相位 [0, 1)
    :param max_force: 最大足底力归一化基准
    :return: 奖励值
    """
    norm_left = np.clip(left_force / max_force, -1.0, 1.0)
    norm_right = np.clip(right_force / max_force, -1.0, 1.0)

    phase_left = phase
    phase_right = (phase + 0.5) % 1.0

    clock_left = clock_frc(phase_left)
    clock_right = clock_frc(phase_right)

    score_left = np.tan(np.pi / 4 * clock_left * norm_left)
    score_right = np.tan(np.pi / 4 * clock_right * norm_right)

    return (score_left + score_right) / 2.0


def calc_foot_vel_clock_reward(left_vel, right_vel, phase, max_vel):
    """
    足部速度相位匹配奖励。
    
    :param left_vel: 左脚速度模长
    :param right_vel: 右脚速度模长
    :param phase: 当前步态相位 [0, 1)
    :param max_vel: 最大速度归一化基准 
    :return: 奖励值
    """
    norm_left = np.clip(left_vel / max_vel, -1.0, 1.0)
    norm_right = np.clip(right_vel / max_vel, -1.0, 1.0)

    phase_left = phase
    phase_right = (phase + 0.5) % 1.0

    clock_left = clock_frc(phase_left)
    clock_right = clock_frc(phase_right)

    score_left = np.tan(np.pi / 4 * clock_left * norm_left)
    score_right = np.tan(np.pi / 4 * clock_right * norm_right)

    return (score_left + score_right) / 2.0


def calc_body_orient_reward(pelvis_yaw, target_yaw):
    """
    躯干姿态奖励 (偏航对齐)。
    
    :param pelvis_yaw: 当前骨盆偏航角 
    :param target_yaw: 目标偏航角 
    :return: 奖励值
    """
    delta = pelvis_yaw - target_yaw
    delta = np.arctan2(np.sin(delta), np.cos(delta))  # 归一化到 [-pi, pi]
    return np.exp(-10.0 * delta**2)


def calc_height_reward(pelvis_z, foot_z, goal_height=0.75, deadzone=0.0235, k_height=100.0):
    """
    骨盆高度奖励。
    
    :param pelvis_z: 骨盆 Z 坐标 
    :param foot_z: 支撑脚 Z 坐标 
    :param goal_height: 期望骨盆离地高度 
    :param deadzone: 高度误差死区 
    :param k_height: 指数衰减系数
    :return: 奖励值
    """
    height_pelvis = pelvis_z - foot_z
    error = abs(height_pelvis - goal_height)
    error = max(0.0, error - deadzone)
    return np.exp(-k_height * error**2)


def calc_step_reward(swing_foot_pos, target_pos, pelvis_xy, goal_xy):
    """
    步点跟踪奖励。
    
    :param swing_foot_pos: 摆动脚世界坐标 (3,)
    :param target_pos: 目标步点世界坐标 (3,)
    :param pelvis_xy: 骨盆 XY 坐标 (2,)
    :param goal_xy: 终点 XY 坐标 (2,)
    :return: 奖励值
    """
    d = np.linalg.norm(swing_foot_pos - target_pos)
    hit_reward = 0.8 * np.exp(-d / 0.25)

    dist_root_to_goal = np.linalg.norm(pelvis_xy - goal_xy)
    progress_reward = 0.2 * np.exp(-dist_root_to_goal / 2.0)

    return hit_reward + progress_reward


def calc_upper_body_stability(head_xy, pelvis_xy):
    """
    上身稳定性奖励 (头部与骨盆 XY 距离)。
    
    :param head_xy: 头部 XY 坐标 (2,)
    :param pelvis_xy: 骨盆 XY 坐标 (2,)
    :return: 奖励值
    """
    dist = np.linalg.norm(head_xy - pelvis_xy)
    return np.exp(-10.0 * dist**2)


def calc_action_penalty(action, last_action):
    """
    动作平滑度惩罚。
    
    :param action: 当前动作 (np.ndarray)
    :param last_action: 上次动作 (np.ndarray)，若为 None 则返回 0
    :return: (惩罚值, 更新后的 last_action)
    """
    if last_action is None:
        return 0.0, action
    diff = action - last_action
    penalty = -np.mean(diff**2)
    return penalty, action


def calc_torque_penalty(torques, max_torques):
    """
    关节力矩惩罚。
    
    :param torques: 当前各关节力矩 (np.ndarray)
    :param max_torques: 各关节最大力矩 (np.ndarray)
    :return: 惩罚值
    """
    norm_torques = torques / (max_torques + 1e-6)
    return -np.mean(norm_torques**2)