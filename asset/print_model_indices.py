#!/usr/bin/env python3
"""
打印 G1 处理模型中所有关节和执行器的索引映射。
在 asset 目录下运行。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot.gen_xml import process_g1_model

import mujoco

def main():
    robot_dir = PROJECT_ROOT / "robot"
    processed_xml = robot_dir / "g1_processed.xml"
    if not processed_xml.exists():
        print("生成 g1_processed.xml...")
        process_g1_model()
    else:
        print(f"使用已有模型: {processed_xml}")

    # 加载模型
    model = mujoco.MjModel.from_xml_path(str(processed_xml))

    print("\n===== 关节 (Joint) 信息 =====")
    print(f"{'名称':<30} {'qpos 索引':<12} {'qvel 索引':<12} {'类型':<8}")
    print("-" * 70)
    for i in range(model.njnt):
        name = model.joint(i).name
        jnt_type = model.jnt_type[i]
        qpos_adr = model.jnt_qposadr[i]
        qvel_adr = model.jnt_dofadr[i]  # 对于铰链关节，通常 qvel 索引与 qpos 索引相同（除 freejoint）
        # 对于 freejoint，qpos 占 7 个，qvel 占 6 个
        print(f"{name:<30} {qpos_adr:<12} {qvel_adr:<12} {jnt_type}")

    print("\n===== 执行器 (Actuator) 信息 =====")
    print(f"{'名称':<30} {'ctrl 索引':<12} {'关联关节':<20}")
    print("-" * 70)
    for i in range(model.nu):
        name = model.actuator(i).name
        joint_id = model.actuator(i).trnid[0]  # 第一个 trnid 是关节索引
        joint_name = model.joint(joint_id).name
        print(f"{name:<30} {i:<12} {joint_name:<20}")

if __name__ == "__main__":
    main()