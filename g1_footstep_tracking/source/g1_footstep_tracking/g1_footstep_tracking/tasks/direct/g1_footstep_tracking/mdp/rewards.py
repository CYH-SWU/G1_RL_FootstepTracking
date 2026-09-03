"""
奖励函数模块 - 迁移自原项目 RewardCalculator 和 reward_functions.py
使用 NumPy 实现，与 Isaac Lab 兼容
"""

import numpy as np

def clock_frc(phase, swing_frac, relax=0.1):
    """
    计算时钟信号，用于足底力/速度期望。
    返回 [-1, 1] 区间，-1 表示 stance，+1 表示 swing
    """
    lower = (1 - swing_frac) * (1 - relax)
    upper = (1 - swing_frac) * (1 + relax)
    if phase < lower:
        return -1.0
    elif phase < upper:
        t = (phase - lower) / (upper - lower)
        return -1.0 + 2.0 * t
    else:
        return 1.0

def foot_frc_clock_reward(left_force, right_force, phase, max_force, swing_frac, is_stand=False):
    """足底力与步态相位匹配奖励"""
    # 死区处理
    left_force = max(0.0, left_force)
    right_force = max(0.0, right_force)

    # 归一化到 [-1, 1]
    norm_left = min(left_force, max_force) / max_force * 2 - 1
    norm_right = min(right_force, max_force) / max_force * 2 - 1

    if is_stand:
        clock_left = -1.0
        clock_right = -1.0
    else:
        clock_left = -clock_frc(phase, swing_frac)
        clock_right = -clock_frc((phase + 0.5) % 1.0, swing_frac)

    score_left = np.tan(np.pi / 4 * clock_left * norm_left)
    score_right = np.tan(np.pi / 4 * clock_right * norm_right)
    return (score_left + score_right) / 2.0

def foot_vel_clock_reward(left_vel, right_vel, phase, max_vel, swing_frac, is_stand=False):
    """足速度与步态相位匹配奖励"""
    norm_left = min(left_vel, max_vel) / max_vel * 2 - 1
    norm_right = min(right_vel, max_vel) / max_vel * 2 - 1

    if is_stand:
        clock_left = -1.0
        clock_right = -1.0
    else:
        clock_left = clock_frc(phase, swing_frac)
        clock_right = clock_frc((phase + 0.5) % 1.0, swing_frac)

    score_left = np.tan(np.pi / 4 * clock_left * norm_left)
    score_right = np.tan(np.pi / 4 * clock_right * norm_right)
    return (score_left + score_right) / 2.0

def body_orient_reward(pelvis_yaw, target_yaw):
    """骨盆偏航对齐奖励"""
    delta = pelvis_yaw - target_yaw
    delta = np.arctan2(np.sin(delta), np.cos(delta))
    return np.exp(-10.0 * delta**2)

def height_reward(pelvis_z, foot_z, goal_height=0.7268, deadzone=0.0235, k_height=100.0):
    """骨盆高度奖励"""
    height = pelvis_z - foot_z
    error = abs(height - goal_height)
    error = max(0.0, error - deadzone)
    return np.exp(-k_height * error**2)

def upper_body_stability_reward(head_xy, pelvis_xy):
    """上体稳定性奖励（头-骨盆水平距离）"""
    dist = np.linalg.norm(head_xy - pelvis_xy)
    return np.exp(-10.0 * dist**2)

def step_reward(left_pos, right_pos, target_pos, pelvis_xy, target_reached):
    """足迹跟踪奖励（命中奖励 + 进度奖励）"""
    hit_reward = 0.0
    if target_reached:
        d_left = np.linalg.norm(left_pos[:3] - target_pos)
        d_right = np.linalg.norm(right_pos[:3] - target_pos)
        d = min(d_left, d_right)
        hit_reward = np.exp(-d / 0.25)

    target_xy = target_pos[:2]
    root_dist = np.linalg.norm(pelvis_xy - target_xy)
    progress_reward = np.exp(-root_dist / 2.0)
    return 0.8 * hit_reward + 0.2 * progress_reward

def action_smoothness_reward(action, prev_action):
    """动作平滑奖励"""
    if prev_action is None:
        return 0.0
    penalty = 5 * np.sum(np.abs(prev_action - action)) / len(action)
    return np.exp(-penalty)

def torque_smoothness_reward(torque, prev_torque, max_torques):
    """力矩平滑奖励"""
    if prev_torque is None:
        return 0.0
    penalty = 0.25 * np.sum(np.abs(prev_torque - torque) / (max_torques + 1e-6)) / len(torque)
    return np.exp(-penalty)

def posture_error_reward(current_joint_angles, nominal_angles):
    """姿态误差奖励"""
    error = np.linalg.norm(current_joint_angles - nominal_angles)
    return np.exp(-error)
