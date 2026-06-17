#!/usr/bin/env python3
"""
动态查看机器人姿态（无重力），推进仿真步，默认自由视角。
在 asset 目录下运行。
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot.gen_xml import process_g1_model

import mujoco
import mujoco.viewer

def main():
    robot_dir = PROJECT_ROOT / "robot"
    processed_xml = robot_dir / "g1_processed.xml"
    if not processed_xml.exists():
        print("生成 g1_processed.xml...")
        process_g1_model()
    else:
        print(f"使用已有模型: {processed_xml}")

    # 检查 STL 目录
    assets_dir = robot_dir / "assets"
    if not assets_dir.exists():
        print(f"错误: 未找到 STL 目录: {assets_dir}")
        sys.exit(1)

    # ---- 生成临时场景 XML ----
    asset_dir = PROJECT_ROOT / "asset"
    scene_xml_path = asset_dir / "temp_scene.xml"
    robot_rel_path = os.path.relpath(processed_xml, start=asset_dir)

    xml_content = f'''<mujoco model="temp_scene">
  <include file="{robot_rel_path}"/>
  <compiler meshdir="../robot/assets"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom type="plane" size="5 5 0.1" pos="0 0 0" rgba="0.5 0.5 0.5 1"/>
  </worldbody>
</mujoco>'''

    with open(scene_xml_path, 'w') as f:
        f.write(xml_content)
    print(f"\n临时场景 XML 已保存至: {scene_xml_path}")

    # ---- 加载场景 ----
    scene_model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
    scene_data = mujoco.MjData(scene_model)

    # ---- 重置到 stand 关键帧 ----
    key_id = mujoco.mj_name2id(scene_model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(scene_model, scene_data, key_id)
        print("已重置到 'stand' 关键帧。")
    else:
        print("警告: 未找到 'stand' 关键帧，将使用默认重置。")
        mujoco.mj_resetData(scene_model, scene_data)

    # 取消重力（便于观察初始姿态）
    scene_model.opt.gravity[:] = [0, 0, 0]

    # 更新运动学（仅计算前向动力学，不推进时间）
    mujoco.mj_forward(scene_model, scene_data)

    # ---- 启动动态查看器 ----
    with mujoco.viewer.launch_passive(scene_model, scene_data) as viewer:
        print("\n按 'Esc' 或关闭窗口退出。")
        print("仿真正在运行（无重力），机器人保持初始姿态。")
        print("您可以使用鼠标旋转/缩放视角。")

        while viewer.is_running():
            mujoco.mj_step(scene_model, scene_data)
            viewer.sync()

if __name__ == "__main__":
    main()