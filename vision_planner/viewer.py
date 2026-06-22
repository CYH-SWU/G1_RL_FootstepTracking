#!/usr/bin/env python3
"""
加载整合后的场景（地形+机器人），切换到 stand 关键帧，开启 MuJoCo 可视化查看器。
用于检验地形和机器人的正确性。
在 vision_planner 目录下运行。
"""

import sys
import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot.gen_xml import process_g1_model

import mujoco
import mujoco.viewer

def main():
    # 确保机器人模型存在
    robot_dir = PROJECT_ROOT / "robot"
    processed_xml = robot_dir / "g1_processed.xml"
    if not processed_xml.exists():
        print("生成 g1_processed.xml...")
        process_g1_model()
    else:
        print(f"使用已有模型: {processed_xml}")

    # 确保场景 XML 存在（由之前的脚本生成）
    asset_dir = PROJECT_ROOT / "asset"
    scene_xml_path = asset_dir / "scene_with_robot.xml"
    if not scene_xml_path.exists():
        print(f"错误: 场景 XML 文件不存在: {scene_xml_path}")
        print("请先运行 vision_planner/test.py 生成场景 XML。")
        sys.exit(1)

    # 加载模型
    model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
    data = mujoco.MjData(model)

    # 重置到 stand 关键帧
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("已重置到 'stand' 关键帧。")
        # 同步 ctrl 与 qpos（避免弹跳）
        #actuator_qpos_indices = [7,8,9,10,11,12,13,14,15,16,17,18,21]  # 从打印表获取
        #data.ctrl[:] = data.qpos[actuator_qpos_indices]
        data.qpos[2] = 1.8
        print("已同步 ctrl。")
    else:
        print("警告: 未找到 'stand' 关键帧，将使用默认重置。")
        mujoco.mj_resetData(model, data)

    # 推进一次 forward 更新运动学
    mujoco.mj_forward(model, data)

    left_foot = data.site_xpos[model.site("left_foot").id]
    right_foot = data.site_xpos[model.site("right_foot").id]
    foot_distance = np.linalg.norm(left_foot - right_foot)
    print(foot_distance)
    # 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n按 'Esc' 或关闭窗口退出。")
        print("您可以使用鼠标旋转/缩放视角。")
        print("按 'C' 键可切换相机。")
        
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()