import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from env_utils.reward_functions import (
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
from .step_sequence import WalkModes

class RewardCalculator:
    """
    Computes dense rewards for the G1 footstep tracking environment.

    Aggregates multiple reward components with fixed weights to produce a scalar reward signal
    for each environment step. Reward terms include:

    - Foot contact force matching (frc): encourages foot forces to follow the gait cycle
    - Foot velocity matching (vel): encourages foot velocities to match swing/stance phase
    - Body orientation (orient): penalizes yaw deviation from the target heading
    - Pelvis height (height): maintains desired clearance above ground
    - Footstep tracking (step): rewards placing feet near target positions
    - Upper body stability (stability): penalizes head-pelvis horizontal displacement
    - Action smoothness (action): penalizes large action changes between steps
    - Torque smoothness (torque): penalizes large torque variations
    - Posture error (posture): penalizes deviation from nominal joint angles

    The final reward is a weighted sum (weights defined in compute_reward). Most weights are
    empirically tuned for stable learning. The standing mode uses fixed clock signals to enforce
    a static stance.

    Methods:
        compute_reward: Main entry point; returns the scalar reward for the current step.
        set_target_reached: Updates the internal target_reached flag used by the step reward.
        _get_body_linvel: Computes the linear velocity magnitude of a body.
        _get_pelvis_yaw: Extracts the yaw angle of the pelvis from the simulation data.
    """
    def __init__(self, config):
        self.config = config
        self.last_action = None
        self.last_torque = None
        self.target_reached = False  # Updated by the main environment.

    def set_target_reached(self, reached):
        self.target_reached = reached

    def compute_reward(self, model, data, pelvis_id, left_foot_id, right_foot_id, head_id,
                       joint_indices, actuator_indices, mode, phase, sequence, t1, action):
        left_force = data.cfrc_ext[left_foot_id][2]
        right_force = data.cfrc_ext[right_foot_id][2]
        left_vel = self._get_body_linvel(data, left_foot_id, model)
        right_vel = self._get_body_linvel(data, right_foot_id, model)

        pelvis_z = data.qpos[2]
        foot_z = min(data.xpos[left_foot_id][2], data.xpos[right_foot_id][2]) - self.config.foot_ankle_offset

        pelvis_yaw = self._get_pelvis_yaw(data, pelvis_id)
        target_yaw = sequence[t1][3] if len(sequence) > 0 else 0.0

        pelvis_xy = data.xpos[pelvis_id][:2]
        head_xy = data.xpos[head_id][:2]

        total_mass = sum(model.body_mass)
        max_force = total_mass * 9.81 * 0.5
        swing_frac = self.config.swing_duration / self.config.total_duration

        is_stand = (mode == WalkModes.STANDING)

        if is_stand:
            r_frc = calc_foot_frc_clock_reward(
                swing_frac,
                left_force, right_force,
                phase, max_force,
                clock_left=1.0, clock_right=1.0
            )
            r_vel = calc_foot_vel_clock_reward(
                swing_frac,
                left_vel, right_vel,
                phase, self.config.max_foot_vel,
                clock_left=-1.0, clock_right=-1.0
            )
        else:
            r_frc = calc_foot_frc_clock_reward(swing_frac, left_force, right_force, phase, max_force)
            r_vel = calc_foot_vel_clock_reward(swing_frac, left_vel, right_vel, phase, self.config.max_foot_vel)

        r_orient = calc_body_orient_reward(pelvis_yaw, target_yaw)
        r_height = calc_height_reward(pelvis_z, foot_z, goal_height=self.config.nominal_pelvis_height, deadzone=0.023)

        if len(sequence) > 0 and t1 < len(sequence):
            target_pos = sequence[t1][:3]
            left_pos = data.xpos[left_foot_id]
            right_pos = data.xpos[right_foot_id]
            r_step = calc_step_reward(left_pos, right_pos, target_pos, pelvis_xy, self.target_reached)
        else:
            r_step = 0.0

        r_stability = calc_upper_body_stability(head_xy, pelvis_xy)

        r_action = calc_action_reward(action, self.last_action)
        self.last_action = action.copy()

        torques = data.actuator_force[actuator_indices]
        r_torque = calc_torque_reward(torques, self.last_torque)
        self.last_torque = torques.copy()

        current_joint_angles = data.qpos[joint_indices]
        r_posture = calc_posture_error_reward(current_joint_angles, self.config.nominal_angles)

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

    def _get_body_linvel(self, data, body_id, model):
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            model, data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id, vel, 0
        )
        return np.linalg.norm(vel[:3])

    def _get_pelvis_yaw(self, data, pelvis_id):
        quat = data.xquat[pelvis_id].copy()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return r.as_euler('xyz')[2]