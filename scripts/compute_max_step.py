#!/usr/bin/env python3
"""
Compute the maximum forward footstep amplitude of G1 robot legs near nominal posture.
Reads nominal angles and action_scale from env.config to stay consistent with the training environment.

Usage:
    uv run python scripts/compute_max_step.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from env.utils.config import G1EnvConfig

config = G1EnvConfig()
MODEL_PATH = PROJECT_ROOT / "robot" / "g1_processed.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# Joint names match the nominal_angles order in config.
hip_joint_name = "left_hip_pitch_joint"
knee_joint_name = "left_knee_joint"

hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, hip_joint_name)
knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, knee_joint_name)
hip_qpos_adr = model.joint(hip_id).qposadr[0]
knee_qpos_adr = model.joint(knee_id).qposadr[0]

ankle_body = model.body("left_ankle_roll_link").id
pelvis_body = model.body("pelvis").id

# Nominal angles for left hip pitch (index 0) and left knee (index 3).
nominal_hip = config.nominal_angles[0]
nominal_knee = config.nominal_angles[3]

action_scale = config.action_scale

# Search range: nominal ± action_scale.
hip_min = nominal_hip - action_scale
hip_max = nominal_hip + action_scale
knee_min = nominal_knee - action_scale
knee_max = nominal_knee + action_scale

print(f"Nominal hip angle: {nominal_hip:.4f} rad")
print(f"Nominal knee angle: {nominal_knee:.4f} rad")
print(f"Action scale: {action_scale}")
print(f"Search range: hip [{hip_min:.4f}, {hip_max:.4f}], knee [{knee_min:.4f}, {knee_max:.4f}]")

# Grid search at 0.01 rad resolution over the valid action ranges.
max_x = -np.inf
best_hip = best_knee = None

for hip in np.arange(hip_min, hip_max, 0.01):
    for knee in np.arange(knee_min, knee_max, 0.01):
        data.qpos[hip_qpos_adr] = hip
        data.qpos[knee_qpos_adr] = knee
        mujoco.mj_forward(model, data)

        ankle_pos = data.xpos[ankle_body].copy()
        pelvis_pos = data.xpos[pelvis_body].copy()
        rel_x = ankle_pos[0] - pelvis_pos[0]  # forward displacement

        if rel_x > max_x:
            max_x = rel_x
            best_hip = hip
            best_knee = knee

print(f"\nMaximum forward footstep amplitude: {max_x:.3f} m")
print(f"Corresponding hip angle: {best_hip:.3f} rad")
print(f"Corresponding knee angle: {best_knee:.3f} rad")