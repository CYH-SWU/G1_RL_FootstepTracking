#!/usr/bin/env python3
"""
Test G1 robot model loading and keyframe switching.
Loads robot/g1_processed.xml, switches to the "stand" keyframe, and launches the MuJoCo viewer.
Gravity is forced to zero to observe joint motions without gravitational effects.
"""

import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MODEL_PATH = PROJECT_ROOT / "robot" / "g1_processed.xml"

# Optional config import for future extensions (not used here)
# from env.config import G1EnvConfig


def main():
    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        print("Please run robot/gen_xml.py first to generate the model file.")
        return

    print(f"Loading model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # Zero out gravity to disable its effect.
    model.opt.gravity = [0.0, 0.0, 0.0]
    print("Gravity set to zero (no gravity)")

    # Switch to the "stand" keyframe if available.
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("Switched to 'stand' keyframe")
    else:
        print("Warning: 'stand' keyframe not found, using default reset.")
        mujoco.mj_resetData(model, data)

    # Forward pass to update derived quantities (including xpos).
    mujoco.mj_forward(model, data)

    print("Launching MuJoCo viewer. Press Esc or close the window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            # Maintain real-time simulation speed.
            elapsed = time.time() - step_start
            time_to_sleep = model.opt.timestep - elapsed
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)


if __name__ == "__main__":
    main()