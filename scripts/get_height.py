#!/usr/bin/env python3
"""
计算 G1 机器人骨盆与脚掌的垂直高度差（Z 向）。
加载 robot/g1_processed.xml，应用 stand 关键帧，输出高度差。
用法：python compute_height.py
"""

import numpy as np
import mujoco
from pathlib import Path

# 项目根目录（与 test_model.py 保持一致）
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MODEL_PATH = PROJECT_ROOT / "robot" / "g1_processed.xml"

def main():
    if not MODEL_PATH.exists():
        print(f"错误：模型文件不存在: {MODEL_PATH}")
        print("请先运行 robot/gen_xml.py 生成模型文件。")
        return

    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # 切换到 stand 关键帧
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("已切换到 'stand' 关键帧")
    else:
        print("警告：未找到 'stand' 关键帧，使用默认重置。")
        mujoco.mj_resetData(model, data)

    # 前向计算，更新所有派生量（包括 xpos）
    mujoco.mj_forward(model, data)

    # 获取 body 位置（世界坐标系）
    pelvis_id = model.body("pelvis").id
    left_foot_id = model.body("left_ankle_roll_link").id
    right_foot_id = model.body("right_ankle_roll_link").id

    pelvis_z = data.xpos[pelvis_id][2]
    left_foot_z = data.xpos[left_foot_id][2]
    right_foot_z = data.xpos[right_foot_id][2]

    # 脚掌高度取平均（踝关节位置）
    avg_foot_z = (left_foot_z + right_foot_z) / 2.0

    # 高度差 = 骨盆 Z - 脚掌平均 Z
    height_diff = pelvis_z - avg_foot_z

    # 输出结果
    print("\n--- 计算结果 ---")
    print(f"骨盆 Z 坐标:         {pelvis_z:.4f} m")
    print(f"左脚踝 Z 坐标:       {left_foot_z:.4f} m")
    print(f"右脚踝 Z 坐标:       {right_foot_z:.4f} m")
    print(f"脚掌平均 Z 坐标:     {avg_foot_z:.4f} m")
    print(f"骨盆 - 脚掌平均 Z:   {height_diff:.4f} m")
    print("\n提示：该值为标称站立高度，即环境中的 nominal_pelvis_height。")

if __name__ == "__main__":
    main()