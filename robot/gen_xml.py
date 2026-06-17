import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

DEFAULT_INPUT = Path(__file__).parent / "unitree_g1.xml"
DEFAULT_OUTPUT = Path(__file__).parent / "g1_processed.xml"

# 保留的关键词：髋、膝、踝、以及腰部俯仰（waist_pitch）
KEEP_JOINT_KEYWORDS = ["hip", "knee", "ankle", "waist_pitch"]

# ---- 官方站立姿态角度（所有关节，无论是否可控） ----
# 这些角度来自原始 XML 的 stand keyframe，已验证正确
STAND_ANGLES = {
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 1.2800,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 1.2800,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2000,
    "left_shoulder_roll_joint": 0.2000,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 1.2800,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2000,
    "right_shoulder_roll_joint": -0.2000,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 1.2800,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

def process_g1_model(input_path=None, output_path=None):
    in_path = Path(input_path) if input_path else DEFAULT_INPUT
    out_path = Path(output_path) if output_path else DEFAULT_OUTPUT

    if not in_path.exists():
        print(f"Error: Input model not found: {in_path}")
        return None

    tree = ET.parse(in_path)
    root = tree.getroot()

    # ---- 1. 收集所有铰链关节名称（排除 freejoint），按出现顺序 ----
    joint_order = []
    for joint in root.findall(".//joint"):
        jname = joint.get("name")
        if jname and jname != "floating_base_joint":
            joint_order.append(jname)
    print(f"提取到 {len(joint_order)} 个铰链关节")

    # ---- 2. 处理执行器（保留/移除） ----
    actuator_node = root.find(".//actuator")
    if actuator_node is not None:
        kept = 0
        for actuator in list(actuator_node.findall("position")):
            joint_name = actuator.get("joint")
            if joint_name is None:
                continue

            keep = any(kw in joint_name.lower() for kw in KEEP_JOINT_KEYWORDS)

            if keep:
                kept += 1
                if "inheritrange" in actuator.attrib:
                    del actuator.attrib["inheritrange"]
                actuator.set("inheritrange", "0")
                actuator.set("kp", "250")
                actuator.set("dampratio", "1")

                joint = root.find(f".//joint[@name='{joint_name}']")
                if joint is not None:
                    joint_range = joint.get("range")
                    if joint_range:
                        actuator.set("ctrlrange", joint_range)
                    else:
                        print(f"Warning: joint '{joint_name}' has no range, ctrlrange not set.")
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
        print(f"Kept {kept} actuators (legs + waist_pitch).")

    # ---- 3. 添加接触排除 ----
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    ET.SubElement(contact, "exclude", body1="torso_link", body2="left_shoulder_pitch_link")
    ET.SubElement(contact, "exclude", body1="torso_link", body2="right_shoulder_pitch_link")
    ET.SubElement(contact, "exclude", body1="left_shoulder_pitch_link", body2="left_elbow_link")
    ET.SubElement(contact, "exclude", body1="right_shoulder_pitch_link", body2="right_elbow_link")
    ET.SubElement(contact, "exclude", body1="left_shoulder_pitch_link", body2="pelvis")
    ET.SubElement(contact, "exclude", body1="right_shoulder_pitch_link", body2="pelvis")


    # ---- 4. 清理 keyframe 中的 ctrl 属性 ----
    keyframe = root.find(".//keyframe")
    if keyframe is not None:
        if "ctrl" in keyframe.attrib:
            del keyframe.attrib["ctrl"]
        for key in keyframe.findall("key"):
            if "ctrl" in key.attrib:
                del key.attrib["ctrl"]

    # ---- 5. 更新 stand keyframe 的 qpos ----
    stand_key = keyframe.find("key[@name='stand']") if keyframe is not None else None
    if stand_key is not None:
        qpos_str = stand_key.get("qpos")
        if qpos_str:
            qpos_values = np.array([float(x) for x in qpos_str.split()], dtype=np.float64)

            # 构建名称到索引的映射（索引 = 7 + 在 joint_order 中的位置）
            name_to_idx = {}
            for idx, name in enumerate(joint_order):
                name_to_idx[name] = 7 + idx

            # 更新所有关节的角度
            for name, angle in STAND_ANGLES.items():
                if name in name_to_idx:
                    idx = name_to_idx[name]
                    if idx < len(qpos_values):
                        qpos_values[idx] = angle
                    else:
                        print(f"Warning: 关节 {name} 的索引 {idx} 超出 qpos 长度 {len(qpos_values)}")
                else:
                    print(f"Warning: 关节 {name} 未在 joint_order 中找到，无法设置初始角度。")

            # 打印前几个值以供验证
            print(f"更新后的 qpos (前10个): {qpos_values[:10]}")

            new_qpos_str = ' '.join([f"{v:.6f}" for v in qpos_values])
            stand_key.set("qpos", new_qpos_str)
            print("已更新 stand keyframe 的 qpos 为官方站立姿态。")
        else:
            print("Warning: stand keyframe 的 qpos 为空。")
    else:
        print("Warning: 未找到 stand keyframe，无法设置初始姿态。")

    # ---- 6. 输出处理后的模型 ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Processed model saved to: {out_path}")
    return out_path