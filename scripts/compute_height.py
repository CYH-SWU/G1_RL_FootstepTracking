#!/usr/bin/env python3
"""
Compute the vertical height difference (Z-axis) between the G1 robot's pelvis and feet.
Loads robot/g1_processed.xml, applies the "stand" keyframe, and outputs the height difference.

Usage:
    uv run python scripts/compute_height.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "robot" / "g1_processed.xml"

from env.utils.config import G1EnvConfig

def main():
    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        print("Please run robot/gen_xml.py first to generate the model file.")
        return
    
    config = G1EnvConfig()

    print(f"Loading model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Switch to the "stand" keyframe if available.
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("Switched to 'stand' keyframe")
    else:
        print("Warning: 'stand' keyframe not found, using default reset.")
        mujoco.mj_resetData(model, data)

    # Forward pass to update all derived quantities (including xpos).
    mujoco.mj_forward(model, data)

    # Get body positions in world coordinates.
    pelvis_id = model.body("pelvis").id
    left_foot_id = model.body("left_ankle_roll_link").id
    right_foot_id = model.body("right_ankle_roll_link").id

    pelvis_z = data.xpos[pelvis_id][2]
    left_foot_z = data.xpos[left_foot_id][2]
    right_foot_z = data.xpos[right_foot_id][2]

    # Average foot height at the ankle.
    avg_foot_z = (left_foot_z + right_foot_z) / 2.0
    height_diff = pelvis_z - avg_foot_z
    pelvis_height = pelvis_z - avg_foot_z + config.foot_ankle_offset

    print("\n--- Computation Results ---")
    print(f"Pelvis Z coordinate:         {pelvis_z:.4f} m")
    print(f"Left ankle Z coordinate:     {left_foot_z:.4f} m")
    print(f"Right ankle Z coordinate:    {right_foot_z:.4f} m")
    print(f"Average foot Z coordinate:   {avg_foot_z:.4f} m")
    print(f"Pelvis - average foot Z:     {height_diff:.4f} m")
    print(f"Actual pelvis height:        {pelvis_height:.4f} m")


if __name__ == "__main__":
    main()