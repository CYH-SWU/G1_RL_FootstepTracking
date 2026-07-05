"""
奖励函数模块
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def get_pelvis_yaw(data, pelvis_id):
    """从 MuJoCo data 中提取骨盆偏航角"""
    quat = data.xquat[pelvis_id].copy()  # (w,x,y,z)
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler = r.as_euler('xyz')
    return euler[2]


def calc_foot_frc_clock_reward(left_force, right_force, phase, max_force,
                               clock_left=None, clock_right=None):
    """
    足底力相位匹配奖励。
    
    :param left_force: 左脚法向力 
    :param right_force: 右脚法向力 
    :param phase: 当前步态相位 [0, 1)
    :param max_force: 最大足底力归一化基准
    :param clock_left: 可选，左腿期望时钟信号（若为 None 则自动计算）
    :param clock_right: 可选，右腿期望时钟信号（若为 None 则自动计算）
    :return: 奖励值
    """
    left_force = max(0, left_force)
    right_force = max(0, right_force)

    norm_left = min(left_force, max_force) / max_force
    norm_right = min(right_force, max_force) / max_force
    norm_left = norm_left * 2 - 1
    norm_right = norm_right * 2 - 1

    if clock_left is None:
        raise ValueError("clock_left must be provided (function or numeric)")
    if clock_right is None:
        raise ValueError("clock_right must be provided (function or numeric)")

    if callable(clock_left):
        clock_left = clock_left(phase)
    if callable(clock_right):
        clock_right = clock_right(phase)

    score_left = np.tan(np.pi / 4 * clock_left * norm_left)
    score_right = np.tan(np.pi / 4 * clock_right * norm_right)

    return (score_left + score_right) / 2.0


def calc_foot_vel_clock_reward(left_vel, right_vel, phase, max_vel,
                               clock_left=None, clock_right=None):
    """
    足部速度相位匹配奖励。
    
    :param left_vel: 左脚速度模长
    :param right_vel: 右脚速度模长
    :param phase: 当前步态相位 [0, 1)
    :param max_vel: 最大速度归一化基准 
    :param clock_left: 可选，左腿期望时钟信号（若为 None 则自动计算）
    :param clock_right: 可选，右腿期望时钟信号（若为 None 则自动计算）
    :return: 奖励值
    """
    norm_left = min(left_vel, max_vel) / max_vel
    norm_right = min(right_vel, max_vel) / max_vel
    norm_left = norm_left * 2 - 1
    norm_right = norm_right * 2 - 1

    if clock_left is None:
        raise ValueError("clock_left must be provided (function or numeric)")
    if clock_right is None:
        raise ValueError("clock_right must be provided (function or numeric)")

    if callable(clock_left):
        clock_left = clock_left(phase)
    if callable(clock_right):
        clock_right = clock_right(phase)

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


def calc_height_reward(pelvis_z, foot_z, goal_height=0.7368, deadzone=0.0235, k_height=100.0):
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


def calc_upper_body_stability(head_xy, pelvis_xy):
    """
    上身稳定性奖励 (头部与骨盆 XY 距离)。
    
    :param head_xy: 头部 XY 坐标 (2,)
    :param pelvis_xy: 骨盆 XY 坐标 (2,)
    :return: 奖励值
    """
    dist = np.linalg.norm(head_xy - pelvis_xy)
    return np.exp(-10.0 * dist**2)


def calc_action_reward(action, prev_action):
    if prev_action is None:
        return 0.0  # 或者 1.0，习惯返回 0 表示无惩罚
    penalty = 5 * np.sum(np.abs(prev_action - action)) / len(action)
    return np.exp(-penalty)

def calc_torque_reward(torque, prev_torque):
    if prev_torque is None:
        return 0.0
    penalty = 0.25 * (np.sum(np.abs(prev_torque - torque)) / len(torque))
    return np.exp(-penalty)


def calc_step_reward(left_pos, right_pos, target_pos, pelvis_xy, target_reached):
    """
    步点跟踪奖励（LHW 风格）：取左右脚到目标的最小距离，计算命中奖励，
    并加上骨盆向目标移动的进度奖励。

    Args:
        left_pos: 左脚世界坐标 (3,)
        right_pos: 右脚世界坐标 (3,)
        target_pos: 目标步点世界坐标 (3,)
        pelvis_xy: 骨盆 XY 坐标 (2,)

    Returns:
        float: 步点跟踪奖励值
    """
    hit_reward = 0
    # 计算左右脚到目标的距离，取最小值
    if target_reached:
        d_left = np.linalg.norm(left_pos - target_pos)
        d_right = np.linalg.norm(right_pos - target_pos)
        d = min(d_left, d_right)

        # 命中奖励：指数衰减
        hit_reward = np.exp(-d / 0.25)

    # 进度奖励：骨盆到目标 XY 的距离
    target_xy = target_pos[:2]
    root_dist_to_target = np.linalg.norm(pelvis_xy - target_xy)
    progress_reward = np.exp(-root_dist_to_target / 2.0)

    return 0.8 * hit_reward + 0.2 * progress_reward

def calc_posture_error_reward(current_joint_angles, nominal_angles):
    """
    姿态误差奖励（LHW 风格）。
    计算当前关节角度与标称姿态的欧氏距离，返回指数衰减奖励。
    奖励值 = exp(-||current - nominal||)

    Args:
        current_joint_angles: 当前关节角度 (13,)
        nominal_angles: 标称关节角度 (13,)，顺序与 joint_indices 一致

    Returns:
        float: 奖励值，范围 (0, 1]
    """
    error = np.linalg.norm(current_joint_angles - nominal_angles)
    return np.exp(-error)



def create_phase_reward(swing_duration, stance_duration, strict_relaxer, stance_mode, FREQ=40):
    """Create phase-based reward functions for gait timing.

    Args:
        swing_duration: Duration of swing phase in seconds.
        stance_duration: Duration of stance phase in seconds.
        strict_relaxer: Fraction to relax phase boundaries.
        stance_mode: One of "grounded", "aerial", or "zero".
        FREQ: Control frequency in Hz.

    Returns:
        Tuple of (right_clock, left_clock) where each is [force_fn, velocity_fn].
    """
    from scipy.interpolate import PchipInterpolator

    right_swing = np.array([0.0, swing_duration]) * FREQ
    first_dblstance = np.array([swing_duration, swing_duration + stance_duration]) * FREQ
    left_swing = np.array([swing_duration + stance_duration, 2 * swing_duration + stance_duration]) * FREQ
    second_dblstance = np.array([2 * swing_duration + stance_duration, 2 * (swing_duration + stance_duration)]) * FREQ

    r_frc_phase_points = np.zeros((2, 8))
    r_vel_phase_points = np.zeros((2, 8))
    l_frc_phase_points = np.zeros((2, 8))
    l_vel_phase_points = np.zeros((2, 8))

    right_swing_relax_offset = (right_swing[1] - right_swing[0]) * strict_relaxer
    l_frc_phase_points[0, 0] = r_frc_phase_points[0, 0] = right_swing[0] + right_swing_relax_offset
    l_frc_phase_points[0, 1] = r_frc_phase_points[0, 1] = right_swing[1] - right_swing_relax_offset
    l_vel_phase_points[0, 0] = r_vel_phase_points[0, 0] = right_swing[0] + right_swing_relax_offset
    l_vel_phase_points[0, 1] = r_vel_phase_points[0, 1] = right_swing[1] - right_swing_relax_offset
    l_vel_phase_points[1, :2] = r_frc_phase_points[1, :2] = np.negative(np.ones(2))
    l_frc_phase_points[1, :2] = r_vel_phase_points[1, :2] = np.ones(2)

    dbl_stance_relax_offset = (first_dblstance[1] - first_dblstance[0]) * strict_relaxer
    l_frc_phase_points[0, 2] = r_frc_phase_points[0, 2] = first_dblstance[0] + dbl_stance_relax_offset
    l_frc_phase_points[0, 3] = r_frc_phase_points[0, 3] = first_dblstance[1] - dbl_stance_relax_offset
    l_vel_phase_points[0, 2] = r_vel_phase_points[0, 2] = first_dblstance[0] + dbl_stance_relax_offset
    l_vel_phase_points[0, 3] = r_vel_phase_points[0, 3] = first_dblstance[1] - dbl_stance_relax_offset
    if stance_mode == "aerial":
        l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.negative(np.ones(2))
        l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.ones(2)
    elif stance_mode == "zero":
        l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.zeros(2)
        l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.zeros(2)
    else:
        l_frc_phase_points[1, 2:4] = r_frc_phase_points[1, 2:4] = np.ones(2)
        l_vel_phase_points[1, 2:4] = r_vel_phase_points[1, 2:4] = np.negative(np.ones(2))

    left_swing_relax_offset = (left_swing[1] - left_swing[0]) * strict_relaxer
    l_frc_phase_points[0, 4] = r_frc_phase_points[0, 4] = left_swing[0] + left_swing_relax_offset
    l_frc_phase_points[0, 5] = r_frc_phase_points[0, 5] = left_swing[1] - left_swing_relax_offset
    l_vel_phase_points[0, 4] = r_vel_phase_points[0, 4] = left_swing[0] + left_swing_relax_offset
    l_vel_phase_points[0, 5] = r_vel_phase_points[0, 5] = left_swing[1] - left_swing_relax_offset
    l_vel_phase_points[1, 4:6] = r_frc_phase_points[1, 4:6] = np.ones(2)
    l_frc_phase_points[1, 4:6] = r_vel_phase_points[1, 4:6] = np.negative(np.ones(2))

    dbl_stance_relax_offset = (second_dblstance[1] - second_dblstance[0]) * strict_relaxer
    l_frc_phase_points[0, 6] = r_frc_phase_points[0, 6] = second_dblstance[0] + dbl_stance_relax_offset
    l_frc_phase_points[0, 7] = r_frc_phase_points[0, 7] = second_dblstance[1] - dbl_stance_relax_offset
    l_vel_phase_points[0, 6] = r_vel_phase_points[0, 6] = second_dblstance[0] + dbl_stance_relax_offset
    l_vel_phase_points[0, 7] = r_vel_phase_points[0, 7] = second_dblstance[1] - dbl_stance_relax_offset
    if stance_mode == "aerial":
        l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.negative(np.ones(2))
        l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.ones(2)
    elif stance_mode == "zero":
        l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.zeros(2)
        l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.zeros(2)
    else:
        l_frc_phase_points[1, 6:] = r_frc_phase_points[1, 6:] = np.ones(2)
        l_vel_phase_points[1, 6:] = r_vel_phase_points[1, 6:] = np.negative(np.ones(2))

    # Extend data to three cycles for continuity
    r_frc_prev_cycle = np.copy(r_frc_phase_points)
    r_vel_prev_cycle = np.copy(r_vel_phase_points)
    l_frc_prev_cycle = np.copy(l_frc_phase_points)
    l_vel_prev_cycle = np.copy(l_vel_phase_points)
    l_frc_prev_cycle[0] = r_frc_prev_cycle[0] = (
        r_frc_phase_points[0] - r_frc_phase_points[0, -1] - dbl_stance_relax_offset
    )
    l_vel_prev_cycle[0] = r_vel_prev_cycle[0] = (
        r_vel_phase_points[0] - r_vel_phase_points[0, -1] - dbl_stance_relax_offset
    )

    r_frc_second_cycle = np.copy(r_frc_phase_points)
    r_vel_second_cycle = np.copy(r_vel_phase_points)
    l_frc_second_cycle = np.copy(l_frc_phase_points)
    l_vel_second_cycle = np.copy(l_vel_phase_points)
    l_frc_second_cycle[0] = r_frc_second_cycle[0] = (
        r_frc_phase_points[0] + r_frc_phase_points[0, -1] + dbl_stance_relax_offset
    )
    l_vel_second_cycle[0] = r_vel_second_cycle[0] = (
        r_vel_phase_points[0] + r_vel_phase_points[0, -1] + dbl_stance_relax_offset
    )

    r_frc_phase_points_repeated = np.hstack((r_frc_prev_cycle, r_frc_phase_points, r_frc_second_cycle))
    r_vel_phase_points_repeated = np.hstack((r_vel_prev_cycle, r_vel_phase_points, r_vel_second_cycle))
    l_frc_phase_points_repeated = np.hstack((l_frc_prev_cycle, l_frc_phase_points, l_frc_second_cycle))
    l_vel_phase_points_repeated = np.hstack((l_vel_prev_cycle, l_vel_phase_points, l_vel_second_cycle))

    r_frc_phase_spline = PchipInterpolator(r_frc_phase_points_repeated[0], r_frc_phase_points_repeated[1])
    r_vel_phase_spline = PchipInterpolator(r_vel_phase_points_repeated[0], r_vel_phase_points_repeated[1])
    l_frc_phase_spline = PchipInterpolator(l_frc_phase_points_repeated[0], l_frc_phase_points_repeated[1])
    l_vel_phase_spline = PchipInterpolator(l_vel_phase_points_repeated[0], l_vel_phase_points_repeated[1])

    return [r_frc_phase_spline, r_vel_phase_spline], [l_frc_phase_spline, l_vel_phase_spline]





'''最初版本'''

def calc_foot_frc_clock_reward0(swing_frac, left_force, right_force, phase, max_force,
                               clock_left=None, clock_right=None):
    """
    足底力相位匹配奖励。
    
    :param left_force: 左脚法向力 
    :param right_force: 右脚法向力 
    :param phase: 当前步态相位 [0, 1)
    :param max_force: 最大足底力归一化基准
    :param clock_left: 可选，左腿期望时钟信号（若为 None 则自动计算）
    :param clock_right: 可选，右腿期望时钟信号（若为 None 则自动计算）
    :return: 奖励值
    """
    left_force = max(0, left_force)
    right_force = max(0, right_force)

    norm_left = min(left_force, max_force) / max_force
    norm_right = min(right_force, max_force) / max_force
    norm_left = norm_left * 2 - 1
    norm_right = norm_right * 2 - 1

    if clock_left is None:
        clock_left = -clock_frc(phase, swing_frac)
    if clock_right is None:
        clock_right = -clock_frc(phase+0.5, swing_frac)

    score_left = np.tan(np.pi / 4 * clock_left * norm_left)
    score_right = np.tan(np.pi / 4 * clock_right * norm_right)

    return (score_left + score_right) / 2.0


def calc_foot_vel_clock_reward0(swing_frac, left_vel, right_vel, phase, max_vel,
                               clock_left=None, clock_right=None):
    """
    足部速度相位匹配奖励。
    
    :param left_vel: 左脚速度模长
    :param right_vel: 右脚速度模长
    :param phase: 当前步态相位 [0, 1)
    :param max_vel: 最大速度归一化基准 
    :param clock_left: 可选，左腿期望时钟信号（若为 None 则自动计算）
    :param clock_right: 可选，右腿期望时钟信号（若为 None 则自动计算）
    :return: 奖励值
    """
    norm_left = min(left_vel, max_vel) / max_vel
    norm_right = min(right_vel, max_vel) / max_vel
    norm_left = norm_left * 2 - 1
    norm_right = norm_right * 2 - 1

    if clock_left is None:
        clock_left = clock_frc(phase, swing_frac)
    if clock_right is None:
        clock_right = clock_frc(phase+0.5, swing_frac)

    score_left = np.tan(np.pi / 4 * clock_left * norm_left)
    score_right = np.tan(np.pi / 4 * clock_right * norm_right)

    return (score_left + score_right) / 2.0


def clock_frc(phase, swing_frac=0.682, relax=0.1):
    """
    计算足底力/速度期望时钟信号。
    返回 -1 (支撑相) 到 +1 (摆动相) 之间的值。
    
    :param phase: 步态相位 [0, 1)
    :param swing_frac: 摆动相占周期比例
    :param relax: 过渡区松弛度
    :return: 时钟信号 [-1, 1]
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