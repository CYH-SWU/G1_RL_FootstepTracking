"""
该脚本视觉感知测试。
切换地形请手动修改地形标签字符串。
为使机器人骨盆高度正常,请手动设置机器人的骨盆高度。
"""

import os
import sys
import time
import numpy as np
import mujoco

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from planner_pipeline.terrain_generator import TerrainGenerator
from planner_pipeline.vision_processor import VisionProcessor


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    robot_xml = os.path.join(project_root, "robot", "g1_processed.xml")
    mesh_dir = os.path.join(project_root, "robot", "assets")

    if not os.path.exists(robot_xml):
        print(f"错误: 找不到机器人 XML 文件：{robot_xml}")
        return
    if not os.path.exists(mesh_dir):
        print(f"警告: STL 目录不存在：{mesh_dir}，将尝试从 XML 所在目录自动推断")

    print("初始化地形生成器...")
    terrain_gen = TerrainGenerator(robot_xml_path=robot_xml, mesh_dir=mesh_dir)

    MODE = "steps"          # 可选: flat, rough, slope, steps
    DIFFICULTY = 1.0

    print(f"生成地形: {MODE}, 难度: {DIFFICULTY}")
    model, data = terrain_gen.generate(
        mode=MODE,
        difficulty=DIFFICULTY,
        goal_pos=(7.5, 0.0)
    )

    # 关键帧重置 + 手动骨盆高度 
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        data.qpos[2] = 0.80   
        #data.qpos[3:7] = [0.8660254, 0, 0, -0.5] # 重置机器人旋转四元组
        print("已重置到 'stand' 关键帧，并手动设置骨盆高度。")
    else:
        print("警告: 未找到 'stand' 关键帧，使用默认重置。")
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)  

    # 视觉处理 
    print("\n初始化视觉处理器...")
    processor = VisionProcessor(
        model=model,
        data=data,
        camera_name="chest_camera",
        pelvis_name="pelvis",
        width=320,
        height=240,
        fov_deg=60.0,
        depth_min=0.5,
        depth_max=2.0,
        crop_x_min=0.15,
        crop_x_max=0.8,
        crop_y_min=-0.5,
        crop_y_max=0.5,
        heightmap_resolution=0.025,
    )

    print("执行视觉处理（含 RGB 渲染）...")
    result = processor.process(render_rgb=True, verbose=True)

    # 可视化所有结果
    print("\n=== 可视化结果 ===")
    processor.visualize_pointcloud(subsample=10000)
    processor.visualize_heightmap()
    processor.visualize_slopemap(vmax=30.0)
    processor.visualize_rgb()  

    print("\n所有可视化窗口已显示。关闭图形窗口后程序结束。")


if __name__ == "__main__":
    main()