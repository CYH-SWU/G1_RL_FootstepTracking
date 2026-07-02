import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

DEFAULT_INPUT = Path(__file__).parent / "unitree_g1.xml"
DEFAULT_OUTPUT = Path(__file__).parent / "g1_processed.xml"

# 保留的关键词：髋、膝、踝、以及腰部俯仰
KEEP_JOINT_KEYWORDS = ["hip", "knee", "ankle"]

# keyframe数据
STAND_ANGLES = {
    "left_hip_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "left_ankle_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.1,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
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

KP_MAP = {
    "left_hip_pitch_joint": 85,
    "left_hip_roll_joint": 85,
    "left_hip_yaw_joint": 85,
    "left_knee_joint": 127,
    "left_ankle_pitch_joint": 34,
    "left_ankle_roll_joint": 34,
    "right_hip_pitch_joint": 85,
    "right_hip_roll_joint": 85,
    "right_hip_yaw_joint": 85,
    "right_knee_joint": 127,
    "right_ankle_pitch_joint": 34,
    "right_ankle_roll_joint": 34,
    "waist_pitch_joint": 85,   
}

def get_dampratio(joint_name: str) -> float:
    """根据关节名称返回推荐的 dampratio 值（action_scale=0.4 时）"""
    if "hip" in joint_name.lower():
        return 1.2
    elif "knee" in joint_name.lower():
        return 1.3
    elif "ankle" in joint_name.lower():
        return 1.5
    elif "waist" in joint_name.lower():
        return 1.2
    else:
        return 1.0

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
        print(f"Kept {kept} actuators (legs + waist_pitch).")

    # ---- 添加接触排除 ----
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    # 上肢与躯干
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


    # ---- 清理 keyframe 中的 ctrl 属性 ----
    keyframe = root.find(".//keyframe")
    if keyframe is not None:
        if "ctrl" in keyframe.attrib:
            del keyframe.attrib["ctrl"]
        for key in keyframe.findall("key"):
            if "ctrl" in key.attrib:
                del key.attrib["ctrl"]

    # ---- 更新 stand keyframe ----
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
                        print(f"Warning: 关节 {name} 的索引 {idx} 超出 qpos 长度 {len(qpos_values)}")
                else:
                    print(f"Warning: 关节 {name} 未在 joint_order 中找到，无法设置初始角度。")
            print(f"更新后的 qpos (前10个): {qpos_values[:10]}")
            new_qpos_str = ' '.join([f"{v:.6f}" for v in qpos_values])
            stand_key.set("qpos", new_qpos_str)

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

    # ==================== 添加地面、纹理和光源 ====================
    # 1. 创建 asset（如果不存在）
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    # 添加纹理和材质
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

    # 2. 添加光源和地面到 worldbody
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    # 添加光源（如果已存在则跳过，这里直接添加，确保位置正确）
    # 注意：原 XML 可能已有光源，但为了统一，我们添加一个平行光
    # 检查是否已有 light，若没有则添加
    light_exists = False
    for light in worldbody.findall("light"):
        light_exists = True
        break
    if not light_exists:
        light = ET.SubElement(worldbody, "light")
        light.set("pos", "0 0 3")
        light.set("dir", "0 0 -1")
        light.set("directional", "true")

    # 添加地面平面
    ground_geom = ET.SubElement(worldbody, "geom")
    ground_geom.set("type", "plane")
    ground_geom.set("size", "20 20 0.1")
    ground_geom.set("pos", "0 0 0")
    ground_geom.set("material", "groundplane")
    # 半透明使地面更美观（可选）
    ground_geom.set("rgba", "0.5 0.7 0.8 0.5")

    # 输出处理后的模型
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Processed model saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    process_g1_model()