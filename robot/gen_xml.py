"""
Process the raw Unitree G1 MuJoCo model for reinforcement learning.
Filters leg joints (hip, knee, ankle) as active actuators, locks other joints,
assigns PD gains, updates the stand keyframe with nominal posture, and adds
contact exclusions plus visual ground/light for debugging.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

DEFAULT_INPUT = Path(__file__).parent / "unitree_g1.xml"
DEFAULT_OUTPUT = Path(__file__).parent / "g1_processed.xml"

# Joints to retain for leg actuation.
KEEP_JOINT_KEYWORDS = ["hip", "knee", "ankle"]

# Stand keyframe target angles (rad).
STAND_ANGLES = {
    "left_hip_pitch_joint": -0.5235987756,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.872664626,
    "left_ankle_pitch_joint": -0.34906585,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.5235987756,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.872664626,
    "right_ankle_pitch_joint": -0.34906585,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.150,
    "left_shoulder_pitch_joint": 0.2000,
    "left_shoulder_roll_joint": 0.2000,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.5235987756,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2000,
    "right_shoulder_roll_joint": -0.2000,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.5235987756,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# Position gain (KP) per joint.
KP_MAP = {
    "left_hip_pitch_joint": 115,
    "left_hip_roll_joint": 115,
    "left_hip_yaw_joint": 115,
    "left_knee_joint": 172,
    "left_ankle_pitch_joint": 46,
    "left_ankle_roll_joint": 46,
    "right_hip_pitch_joint": 115,
    "right_hip_roll_joint": 115,
    "right_hip_yaw_joint": 115,
    "right_knee_joint": 172,
    "right_ankle_pitch_joint": 46,
    "right_ankle_roll_joint": 46,
}

def get_dampratio(joint_name: str) -> float:
    if "hip" in joint_name.lower():
        return 0.65
    elif "knee" in joint_name.lower():
        return 0.55
    elif "ankle" in joint_name.lower():
        return 0.40
    else:
        return 0.55

def process_g1_model(input_path=None, output_path=None):
    in_path = Path(input_path) if input_path else DEFAULT_INPUT
    out_path = Path(output_path) if output_path else DEFAULT_OUTPUT

    if not in_path.exists():
        print(f"Error: Input model not found: {in_path}")
        return None

    tree = ET.parse(in_path)
    root = tree.getroot()

    # Collect all hinge joint names (excluding floating base).
    joint_order = []
    for joint in root.findall(".//joint"):
        jname = joint.get("name")
        if jname and jname != "floating_base_joint":
            joint_order.append(jname)
    print(f"Extracted {len(joint_order)} hinge joints.")

    # Process actuators: keep only leg-related ones.
    actuator_node = root.find(".//actuator")
    kept_joint_names = []  # In actuator order.
    if actuator_node is not None:
        kept = 0
        for actuator in list(actuator_node.findall("position")):
            joint_name = actuator.get("joint")
            if joint_name is None:
                continue

            keep = any(kw in joint_name.lower() for kw in KEEP_JOINT_KEYWORDS)

            if keep:
                kept += 1
                kept_joint_names.append(joint_name)
                if "inheritrange" in actuator.attrib:
                    del actuator.attrib["inheritrange"]
                actuator.set("inheritrange", "0")
                kp = KP_MAP.get(joint_name, 120)
                actuator.set("kp", str(kp))
                dampratio = get_dampratio(joint_name)
                actuator.set("dampratio", str(dampratio))
                if "kd" in actuator.attrib:
                    del actuator.attrib["kd"]

                joint = root.find(f".//joint[@name='{joint_name}']")
                if joint is not None:
                    joint_range = joint.get("range")
                    if joint_range:
                        actuator.set("ctrlrange", joint_range)
                    else:
                        print(f"Warning: joint '{joint_name}' has no range, ctrlrange not set.")

                    joint_force_range = joint.get("actuatorfrcrange")
                    if joint_force_range:
                        actuator.set("forcerange", joint_force_range)
                    else:
                        print(f"Warning: joint '{joint_name}' has no actuatorfrcrange, forcerange not set.")
                else:
                    print(f"Warning: no joint found for actuator '{joint_name}', ctrlrange not set.")
            else:
                actuator_node.remove(actuator)
                joint = root.find(f".//joint[@name='{joint_name}']")
                if joint is not None:
                    joint.set("type", "hinge")
                    joint.set("range", "0 0")
                    joint.set("damping", "10000")
                    joint.set("armature", "0")
                    if "actuatorfrcrange" in joint.attrib:
                        del joint.attrib["actuatorfrcrange"]
                    if "frictionloss" in joint.attrib:
                        del joint.attrib["frictionloss"]
        print(f"Kept {kept} actuators (leg joints + waist_pitch).")

    # Add contact exclusions to prevent self-collision artifacts
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    # Upper body vs. torso and leg cross-exclusions.
    ET.SubElement(contact, "exclude", body1="torso_link", body2="left_shoulder_pitch_link")
    ET.SubElement(contact, "exclude", body1="torso_link", body2="right_shoulder_pitch_link")
    ET.SubElement(contact, "exclude", body1="left_shoulder_pitch_link", body2="left_elbow_link")
    ET.SubElement(contact, "exclude", body1="right_shoulder_pitch_link", body2="right_elbow_link")
    ET.SubElement(contact, "exclude", body1="left_shoulder_pitch_link", body2="pelvis")
    ET.SubElement(contact, "exclude", body1="right_shoulder_pitch_link", body2="pelvis")
    ET.SubElement(contact, "exclude", body1="left_wrist_yaw_link", body2="left_hip_pitch_link")
    ET.SubElement(contact, "exclude", body1="right_wrist_yaw_link", body2="right_hip_pitch_link")
    ET.SubElement(contact, "exclude", body1="left_elbow_link", body2="left_hip_pitch_link")
    ET.SubElement(contact, "exclude", body1="right_elbow_link", body2="right_hip_pitch_link")
    ET.SubElement(contact, "exclude", body1="left_hip_pitch_link", body2="right_hip_pitch_link")
    ET.SubElement(contact, "exclude", body1="left_knee_link", body2="right_knee_link")
    ET.SubElement(contact, "exclude", body1="pelvis", body2="waist_yaw_link")
    ET.SubElement(contact, "exclude", body1="pelvis", body2="waist_roll_link")
    ET.SubElement(contact, "exclude", body1="pelvis", body2="torso_link")

    # Remove legacy "ctrl" attributes from keyframes
    keyframe = root.find(".//keyframe")
    if keyframe is not None:
        if "ctrl" in keyframe.attrib:
            del keyframe.attrib["ctrl"]
        for key in keyframe.findall("key"):
            if "ctrl" in key.attrib:
                del key.attrib["ctrl"]

    # Update the "stand" keyframe with target angles 
    stand_key = keyframe.find("key[@name='stand']") if keyframe is not None else None
    if stand_key is not None:
        qpos_str = stand_key.get("qpos")
        if qpos_str:
            qpos_values = np.array([float(x) for x in qpos_str.split()], dtype=np.float64)
            name_to_idx = {}
            for idx, name in enumerate(joint_order):
                name_to_idx[name] = 7 + idx
            for name, angle in STAND_ANGLES.items():
                if name in name_to_idx:
                    idx = name_to_idx[name]
                    if idx < len(qpos_values):
                        qpos_values[idx] = angle
                    else:
                        print(f"Warning: joint {name} index {idx} out of qpos length {len(qpos_values)}")
                else:
                    print(f"Warning: joint {name} not found in joint_order, cannot set initial angle.")
            print(f"Updated qpos (first 10): {qpos_values[:10]}")
            new_qpos_str = ' '.join([f"{v:.6f}" for v in qpos_values])
            stand_key.set("qpos", new_qpos_str)

            # Build control values for retained actuators.
            ctrl_values = []
            for name in kept_joint_names:
                if name in STAND_ANGLES:
                    ctrl_values.append(STAND_ANGLES[name])
                else:
                    print(f"Warning: joint {name} missing in STAND_ANGLES, ctrl set to 0")
                    ctrl_values.append(0.0)
            ctrl_str = ' '.join([f"{v:.6f}" for v in ctrl_values])
            stand_key.set("ctrl", ctrl_str)
            print(f"Set ctrl: {len(ctrl_values)} values, first 5: {ctrl_values[:5]}")
            print("Updated stand keyframe qpos and ctrl to official stand posture.")
        else:
            print("Warning: stand keyframe qpos is empty.")
    else:
        print("Warning: stand keyframe not found; initial posture not set.")

    # Add ground, texture, and light
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    # Ground texture (checkerboard).
    tex = ET.SubElement(asset, "texture")
    tex.set("name", "ground_tex")
    tex.set("type", "2d")
    tex.set("builtin", "checker")
    tex.set("rgb1", "0.2 0.3 0.4")
    tex.set("rgb2", "0.6 0.7 0.8")
    tex.set("width", "300")
    tex.set("height", "300")
    tex.set("mark", "edge")
    tex.set("random", "0.01")
    mat = ET.SubElement(asset, "material")
    mat.set("name", "groundplane")
    mat.set("texture", "ground_tex")
    mat.set("texrepeat", "4 4")
    mat.set("texuniform", "true")
    mat.set("reflectance", "0.2")

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    # Add a directional light if none exists.
    light_exists = False
    for light in worldbody.findall("light"):
        light_exists = True
        break
    if not light_exists:
        light = ET.SubElement(worldbody, "light")
        light.set("pos", "0 0 3")
        light.set("dir", "0 0 -1")
        light.set("directional", "true")

    # Ground plane with semi‑transparent material.
    ground_geom = ET.SubElement(worldbody, "geom")
    ground_geom.set("type", "plane")
    ground_geom.set("size", "20 20 0.1")
    ground_geom.set("pos", "0 0 0")
    ground_geom.set("material", "groundplane")
    ground_geom.set("rgba", "0.5 0.7 0.8 0.5")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Processed model saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    process_g1_model()