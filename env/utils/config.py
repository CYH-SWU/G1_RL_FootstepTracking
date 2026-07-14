from dataclasses import dataclass, field
import numpy as np

@dataclass
class G1EnvConfig:
    # Physics parameters.
    control_dt: float = 0.015
    physics_dt: float = 0.005
    max_episode_steps: int = 1500

    # Gait parameters.
    total_duration: float = 1.1
    swing_duration: float = 0.75
    stance_duration: float = 0.35
    step_length: float = 0.20
    step_width: float = 0.237
    max_foot_vel: float = 0.30
    target_radius: float = 0.16

    # Nominal posture (using default_factory).
    nominal_angles: np.ndarray = field(default_factory=lambda: np.array([
        -0.5235987756, 0.0, 0.0, 0.872664626, -0.34906585, 0.0,
        -0.5235987756, 0.0, 0.0, 0.872664626, -0.34906585, 0.0
    ]))
    nominal_pelvis_height: float = 0.7268
    foot_ankle_offset: float = 0.0331
    action_scale: float = 0.30
    action_smoothing: float = 0.20

    # Curriculum and terrain.
    fall_height_threshold: float = 0.35
    max_boxes: int = 30

    # Mode probabilities for terrain randomization.
    mode_probs: tuple = (0.05, 0.15, 0.20, 0.30, 0.30)

    # Normalization parameters (using default_factory).
    norm_params: dict = field(default_factory=lambda: {
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
    })