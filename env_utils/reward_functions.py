"""
Reward function module for footstep tracking.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def clock_frc(phase, swing_frac, relax=0.1):
    """
    Compute clock signal for foot force/velocity expectation.
    Returns value between -1 (stance) and +1 (swing).

    :param phase: Gait phase in [0, 1)
    :param swing_frac: Swing phase fraction of the cycle
    :param relax: Transition zone relaxation factor
    :return: Clock signal in [-1, 1]
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


def get_pelvis_yaw(data, pelvis_id):
    """
    Extract the yaw angle of the pelvis from MuJoCo data.

    :param data: MuJoCo MjData object containing the simulation state.
    :param pelvis_id: Body ID of the pelvis in the model.
    :return: Pelvis yaw angle in radians.
    """
    quat = data.xquat[pelvis_id].copy()  # (w,x,y,z)
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler = r.as_euler('xyz')
    return euler[2]


def calc_foot_frc_clock_reward(swing_frac, left_force, right_force, phase, max_force,
                               clock_left=None, clock_right=None):
    """
    Reward for matching foot normal forces with gait phase expectations.

    :param left_force: Left foot normal force
    :param right_force: Right foot normal force
    :param phase: Current gait phase in [0, 1)
    :param max_force: Normalization reference for maximum force
    :param clock_left: Optional precomputed clock for left leg
    :param clock_right: Optional precomputed clock for right leg
    :return: Reward value
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
        clock_right = -clock_frc((phase + 0.5) % 1.0, swing_frac)

    score_left = np.tan(np.pi / 4 * clock_left * norm_left)
    score_right = np.tan(np.pi / 4 * clock_right * norm_right)

    return (score_left + score_right) / 2.0


def calc_foot_vel_clock_reward(swing_frac, left_vel, right_vel, phase, max_vel,
                               clock_left=None, clock_right=None):
    """
    Reward for matching foot velocity magnitude with gait phase expectations.

    :param left_vel: Left foot speed magnitude
    :param right_vel: Right foot speed magnitude
    :param phase: Current gait phase in [0, 1)
    :param max_vel: Normalization reference for maximum velocity
    :param clock_left: Optional precomputed clock for left leg
    :param clock_right: Optional precomputed clock for right leg
    :return: Reward value
    """
    norm_left = min(left_vel, max_vel) / max_vel
    norm_right = min(right_vel, max_vel) / max_vel
    norm_left = norm_left * 2 - 1
    norm_right = norm_right * 2 - 1

    if clock_left is None:
        clock_left = clock_frc(phase, swing_frac)
    if clock_right is None:
        clock_right = clock_frc((phase + 0.5) % 1.0, swing_frac)

    score_left = np.tan(np.pi / 4 * clock_left * norm_left)
    score_right = np.tan(np.pi / 4 * clock_right * norm_right)

    return (score_left + score_right) / 2.0


def calc_body_orient_reward(pelvis_yaw, target_yaw):
    """
    Reward for pelvis yaw alignment with target.

    :param pelvis_yaw: Current pelvis yaw angle
    :param target_yaw: Target yaw angle
    :return: Reward value
    """
    delta = pelvis_yaw - target_yaw
    delta = np.arctan2(np.sin(delta), np.cos(delta))  # Normalize to [-pi, pi]
    return np.exp(-10.0 * delta**2)


def calc_height_reward(pelvis_z, foot_z, goal_height=0.7368, deadzone=0.0235, k_height=100.0):
    """
    Reward for maintaining desired pelvis height above the ground.

    :param pelvis_z: Pelvis Z coordinate
    :param foot_z: Support foot Z coordinate
    :param goal_height: Desired pelvis height above foot
    :param deadzone: Tolerance band for height error
    :param k_height: Exponential decay coefficient
    :return: Reward value
    """
    height_pelvis = pelvis_z - foot_z
    error = abs(height_pelvis - goal_height)
    error = max(0.0, error - deadzone)
    return np.exp(-k_height * error**2)


def calc_upper_body_stability(head_xy, pelvis_xy):
    """
    Reward for upper body stability (head-pelvis horizontal distance).

    :param head_xy: Head XY position (2,)
    :param pelvis_xy: Pelvis XY position (2,)
    :return: Reward value
    """
    dist = np.linalg.norm(head_xy - pelvis_xy)
    return np.exp(-10.0 * dist**2)


def calc_action_reward(action, prev_action):
    """
    Penalize large action changes between steps.

    :param action: Current action
    :param prev_action: Previous action (None for first step)
    :return: Reward value (0 if no previous action)
    """
    if prev_action is None:
        return 0.0
    penalty = 5 * np.sum(np.abs(prev_action - action)) / len(action)
    return np.exp(-penalty)

def calc_torque_reward(torque, prev_torque):
    """
    Penalize large torque variations.

    :param torque: Current torque
    :param prev_torque: Previous torque (None for first step)
    :return: Reward value (0 if no previous torque)
    """
    if prev_torque is None:
        return 0.0
    penalty = 0.25 * (np.sum(np.abs(prev_torque - torque)) / len(torque))
    return np.exp(-penalty)


def calc_step_reward(left_pos, right_pos, target_pos, pelvis_xy, target_reached):
    """
    Footstep tracking reward: combines foot-to-target distance and pelvis progress.

    Args:
        left_pos: Left foot world position (3,)
        right_pos: Right foot world position (3,)
        target_pos: Target footstep world position (3,)
        pelvis_xy: Pelvis XY position (2,)
        target_reached: Boolean indicating whether the target has been reached.
                       If True, compute hit reward from the closest foot;
                       otherwise hit reward is zero.

    Returns:
        float: Reward value.
    """
    hit_reward = 0
    # Take minimum distance from either foot to target.
    if target_reached:
        d_left = np.linalg.norm(left_pos - target_pos)
        d_right = np.linalg.norm(right_pos - target_pos)
        d = min(d_left, d_right)
        hit_reward = np.exp(-d / 0.25)

    # Progress reward: encourage pelvis moving toward target.
    target_xy = target_pos[:2]
    root_dist_to_target = np.linalg.norm(pelvis_xy - target_xy)
    progress_reward = np.exp(-root_dist_to_target / 2.0)

    return 0.8 * hit_reward + 0.2 * progress_reward

def calc_posture_error_reward(current_joint_angles, nominal_angles):
    """
    Reward for staying close to nominal posture.

    Args:
        current_joint_angles: Current joint angles (13,)
        nominal_angles: Nominal joint angles (13,), order consistent with joint indices

    Returns:
        float: Reward in (0, 1]
    """
    error = np.linalg.norm(current_joint_angles - nominal_angles)
    return np.exp(-error)
