import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

DEFAULT_INPUT = Path(__file__).parent / "unitree_g1.xml"
DEFAULT_OUTPUT = Path(__file__).parent / "g1_processed.xml"

# 保留的关键词：髋、膝、踝、以及腰部俯仰
KEEP_JOINT_KEYWORDS = ["hip", "knee", "ankle", "waist_pitch"]

# keyframe数据
STAND_ANGLES = {
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,          
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
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

KP_MAP = {
    "left_hip_pitch_joint": 116,
    "left_hip_roll_joint": 116,
    "left_hip_yaw_joint": 116,
    "left_knee_joint": 145,
    "left_ankle_pitch_joint": 46,
    "left_ankle_roll_joint": 46,
    "right_hip_pitch_joint": 116,
    "right_hip_roll_joint": 116,
    "right_hip_yaw_joint": 116,
    "right_knee_joint": 145,
    "right_ankle_pitch_joint": 46,
    "right_ankle_roll_joint": 46,
    "waist_pitch_joint": 100,  
}

def process_g1_model(input_path=None, output_path=None):
    in_path = Path(input_path) if input_path else DEFAULT_INPUT
    out_path = Path(output_path) if output_path else DEFAULT_OUTPUT

    if not in_path.exists():
        print(f"Error: Input model not found: {in_path}")
        return None

    tree = ET.parse(in_path)
    root = tree.getroot()

    # 收集所有铰链关节名称
    joint_order = []
    for joint in root.findall(".//joint"):
        jname = joint.get("name")
        if jname and jname != "floating_base_joint":
            joint_order.append(jname)
    print(f"提取到 {len(joint_order)} 个铰链关节")

    # 处理执行器
    actuator_node = root.find(".//actuator")
    kept_joint_names = []  # 按执行器顺序存储保留的关节名称
    if actuator_node is not None:
        kept = 0
        for actuator in list(actuator_node.findall("position")):
            joint_name = actuator.get("joint")
            if joint_name is None:
                continue

            keep = any(kw in joint_name.lower() for kw in KEEP_JOINT_KEYWORDS)

            if keep:
                kept += 1
                kept_joint_names.append(joint_name)   # 记录顺序
                if "inheritrange" in actuator.attrib:
                    del actuator.attrib["inheritrange"]
                actuator.set("inheritrange", "0")
                kp = KP_MAP.get(joint_name, 120)  
                actuator.set("kp", str(kp))
                actuator.set("dampratio", "1.0")
                if "kd" in actuator.attrib:
                    del actuator.attrib["kd"]

                joint = root.find(f".//joint[@name='{joint_name}']")
                if joint is not None:
                    joint_range = joint.get("range")
                    if joint_range:
                        actuator.set("ctrlrange", joint_range)
                    else:
                        print(f"Warning: joint '{joint_name}' has no range, ctrlrange not set.")

                    # 设置力矩限幅
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
        print(f"Kept {kept} actuators (legs + waist_pitch).")

    # 添加接触排除 
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    # 上肢与躯干（避免固定手臂穿透）
    ET.SubElement(contact, "exclude", body1="torso_link", body2="left_shoulder_pitch_link")
    ET.SubElement(contact, "exclude", body1="torso_link", body2="right_shoulder_pitch_link")
    ET.SubElement(contact, "exclude", body1="left_shoulder_pitch_link", body2="left_elbow_link")
    ET.SubElement(contact, "exclude", body1="right_shoulder_pitch_link", body2="right_elbow_link")
    ET.SubElement(contact, "exclude", body1="left_shoulder_pitch_link", body2="pelvis")
    ET.SubElement(contact, "exclude", body1="right_shoulder_pitch_link", body2="pelvis")

    # 手部与大腿（末端手腕与髋）
    ET.SubElement(contact, "exclude", body1="left_wrist_yaw_link", body2="left_hip_pitch_link")
    ET.SubElement(contact, "exclude", body1="right_wrist_yaw_link", body2="right_hip_pitch_link")

    # 手臂与腿（肘部与大腿）
    ET.SubElement(contact, "exclude", body1="left_elbow_link", body2="left_hip_pitch_link")
    ET.SubElement(contact, "exclude", body1="right_elbow_link", body2="right_hip_pitch_link")

    # 腿与腿（左右大腿、膝盖之间）
    ET.SubElement(contact, "exclude", body1="left_hip_pitch_link", body2="right_hip_pitch_link")
    ET.SubElement(contact, "exclude", body1="left_knee_link", body2="right_knee_link")

    # 骨盆与腰部（避免关节连接处额外接触）
    ET.SubElement(contact, "exclude", body1="pelvis", body2="waist_yaw_link")
    ET.SubElement(contact, "exclude", body1="pelvis", body2="waist_roll_link")
    ET.SubElement(contact, "exclude", body1="pelvis", body2="torso_link")  

    # 添加胸部相机
    torso_body = root.find(".//body[@name='torso_link']")
    if torso_body is not None:
        cam = ET.SubElement(torso_body, "camera")
        cam.set("name", "chest_camera")
        cam.set("pos", "0.1 0 0")           
        cam.set("euler", "0 -0.5236 -1.5708")          
        cam.set("fovy", "60")
        print("已添加胸部相机，位于躯干前方 40mm，俯仰角 -30°。")
    else:
        print("警告：未找到 torso_link，无法添加相机。")

    # 清理 keyframe 中的 ctrl 属性
    keyframe = root.find(".//keyframe")
    if keyframe is not None:
        if "ctrl" in keyframe.attrib:
            del keyframe.attrib["ctrl"]
        for key in keyframe.findall("key"):
            if "ctrl" in key.attrib:
                del key.attrib["ctrl"]

    # 更新 stand keyframe 的 qpos 并添加正确的 ctrl 
    stand_key = keyframe.find("key[@name='stand']") if keyframe is not None else None
    if stand_key is not None:
        qpos_str = stand_key.get("qpos")
        if qpos_str:
            qpos_values = np.array([float(x) for x in qpos_str.split()], dtype=np.float64)

            # 构建名称到索引的映射
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

            
            print(f"更新后的 qpos (前10个): {qpos_values[:10]}")

            new_qpos_str = ' '.join([f"{v:.6f}" for v in qpos_values])
            stand_key.set("qpos", new_qpos_str)

            
            # 构建 ctrl 值列表，顺序与 kept_joint_names 相同
            ctrl_values = []
            for name in kept_joint_names:
                if name in STAND_ANGLES:
                    ctrl_values.append(STAND_ANGLES[name])
                else:
                    print(f"Warning: 关节 {name} 没有在 STAND_ANGLES 中，ctrl 设为 0")
                    ctrl_values.append(0.0)

            ctrl_str = ' '.join([f"{v:.6f}" for v in ctrl_values])
            stand_key.set("ctrl", ctrl_str)
            print(f"已设置 ctrl: {len(ctrl_values)} 个值，前5个: {ctrl_values[:5]}")

            print("已更新 stand keyframe 的 qpos 和 ctrl 为官方站立姿态。")
        else:
            print("Warning: stand keyframe 的 qpos 为空。")
    else:
        print("Warning: 未找到 stand keyframe，无法设置初始姿态。")

    # 输出处理后的模型
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Processed model saved to: {out_path}")
    return out_path 