"""
Configuration dataclass for the G1 humanoid footstep tracking environment.

Holds all hyperparameters for physics, gait scheduling, control, normalization,
and curriculum/terrain settings. Used by G1Env and its helper components.

Key parameter groups:
- Physics: control_dt, physics_dt, max_episode_steps define simulation timing and episode length.
- Gait: total_duration, swing_duration, stance_duration, step_length, step_width, max_foot_vel,
  and target_radius control footstep generation and foot placement tolerance.
- Control: nominal_angles, nominal_pelvis_height, foot_ankle_offset, action_scale, action_smoothing
  define the default posture and action mapping.
- Curriculum: fall_height_threshold and max_boxes for terrain and termination.
- Mode probabilities: mode_probs determines the sampling distribution over walking modes.
- Normalization: norm_params provides scaling factors for observation normalization.

All fields have sensible defaults for training the G1 robot in footstep tracking tasks.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class G1EnvConfig:
    # Physics parameters.
    control_dt: float = 0.015
    physics_dt: float = 0.005
    max_episode_steps: int = 1500

    # Gait parameters.
    total_duration: float = 1.30
    swing_duration: float = 0.85
    stance_duration: float = 0.45
    step_length: float = 0.20
    step_width: float = 0.237
    max_foot_vel: float = 0.12
    target_radius: float = 0.16

    # Nominal posture (using default_factory).
    nominal_angles: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                -0.5235987756,
                0.0,
                0.0,
                0.872664626,
                -0.34906585,
                0.0,
                -0.5235987756,
                0.0,
                0.0,
                0.872664626,
                -0.34906585,
                0.0,
            ]
        )
    )
    nominal_pelvis_height: float = 0.7268
    foot_ankle_offset: float = 0.0331
    action_scale: float = 0.25
    action_smoothing: float = 0.20

    # Curriculum and terrain.
    fall_height_threshold: float = 0.35
    max_boxes: int = 30

    # Mode probabilities for terrain randomization.
    mode_probs: tuple = (0.05, 0.15, 0.20, 0.30, 0.30)

    # Normalization parameters (using default_factory).
    norm_params: dict = field(
        default_factory=lambda: {
            "joint_angles_max": 1.5,
            "joint_vels_max": 10.0,
            "pelvis_height_max": 1.0,
            "t1_pos_max": [0.30, 0.25, 0.9],
            "t2_pos_max": [0.5, 0.30, 0.9],
            "t1_yaw_max": 0.2,
            "t2_yaw_max": 0.25,
            "phase_max": 1.0,
            "pelvis_orient_max": 0.3,
            "pelvis_angvel_max": 5.0,
        }
    )
