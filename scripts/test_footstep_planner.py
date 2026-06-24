"""
该脚本用于视觉感知 + 步点规划器测试。
切换地形请手动修改地形标签字符串。
请手动设置机器人的骨盆高度。
支撑腿坐标请手动设置（骨盆坐标系下）。
手动设置终点位置（世界坐标系）。
"""

import os
import sys
import numpy as np
import mujoco

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from planner_pipeline.terrain_generator import TerrainGenerator
from planner_pipeline.vision_processor import VisionProcessor
from planner_pipeline.footstep_planner import G1FootstepPlanner
from scipy.spatial.transform import Rotation as R


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
        data.qpos[2] = 0.8                       # 手动设置骨盆位置
        data.qpos[0] = 0
        data.qpos[1] = 0
        # 设置机器人旋转四元组
        # data.qpos[3:7] = [0.8660254, 0, 0, -0.5] 
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

    # 步点规划器测试
    print("\n========== 步点规划器测试 ==========")

    # 获取高程图数据
    heightmap = result['heightmap']
    slopemap = result['slopemap']
    x_edges = result['x_edges']
    y_edges = result['y_edges']
    res = 0.025

    # 实例化规划器
    planner = G1FootstepPlanner(
        step=0.3,
        step_width=0.237,
        max_step_len=0.40,
        min_step_len=0.15,
        max_turn_deg=8.0,
        max_step_height=0.20,
        max_slope_deg=20.0,
        clearance=0.03,
        w_step=1.0,
        w_angle=0.7,
        w_slope=0.5,
        step_discretization=0.05,
        turn_discretization=1.0
    )
    planner.set_heightmap(heightmap, slopemap, x_edges, y_edges, res)

    # 获取骨盆位姿
    pelvis_pos = data.xpos[model.body("pelvis").id].copy() 
    pelvis_quat = data.xquat[model.body("pelvis").id].copy()
    r_quat = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
    euler = r_quat.as_euler('xyz')
    yaw = euler[2]
    R_yaw_to_world = R.from_euler('z', yaw).as_matrix()            
    R_world_to_pelvis = R_yaw_to_world.T                           

    # 手动设置支撑腿坐标（骨盆坐标系）
    current_stance = 1                     # -1: 左脚, 1: 右脚
    foot_pelvis_pos = np.array([0.05, 0.12, -0.79])   

    print(f"使用支撑腿: {'左' if current_stance == -1 else '右'}脚")
    print(f"支撑腿骨盆坐标: ({foot_pelvis_pos[0]:.4f}, {foot_pelvis_pos[1]:.4f}, {foot_pelvis_pos[2]:.4f})")

    # 目标终点
    goal_world = np.array([7.5, 0, 0.0])                     # 扩展为 3D
    goal_pelvis_3d = R_world_to_pelvis @ (goal_world - pelvis_pos)   
    goal_pelvis_xy = goal_pelvis_3d[:2]                       # 只取 XY
    print(f"终点在骨盆坐标系下的位置 (XY): ({goal_pelvis_xy[0]:.4f}, {goal_pelvis_xy[1]:.4f})")

    # 调用规划器
    footstep, next_stance = planner.plan_next_footstep(
        current_foot_pos=(foot_pelvis_pos[0], foot_pelvis_pos[1], foot_pelvis_pos[2]),
        current_stance=current_stance,
        target_pos=(goal_pelvis_xy[0], goal_pelvis_xy[1])
    )

    # 打印骨盆坐标系下的步点
    print("\n--- 规划步点（骨盆坐标系） ---")
    print(f"  位置: ({footstep.x:.4f}, {footstep.y:.4f}, {footstep.z:.4f}) m")
    print(f"  偏航角: {np.degrees(footstep.yaw):.2f}°")
    print(f"  使用脚: {'左' if footstep.foot == -1 else '右'}脚")

    # 将步点转换到世界坐标系
    local_pos = np.array([footstep.x, footstep.y, footstep.z])
    world_pos = pelvis_pos + R_yaw_to_world @ local_pos
    world_yaw = yaw + footstep.yaw
    world_yaw = np.arctan2(np.sin(world_yaw), np.cos(world_yaw))   # 归一化到 [-pi, pi]

    print("\n--- 规划步点（世界坐标系） ---")
    print(f"  位置: ({world_pos[0]:.4f}, {world_pos[1]:.4f}, {world_pos[2]:.4f}) m")
    print(f"  偏航角: {np.degrees(world_yaw):.2f}°")
    print("=====================================\n")

    # 可视化
    print("\n=== 可视化结果 ===")
    processor.visualize_pointcloud(subsample=10000)
    processor.visualize_heightmap()
    processor.visualize_slopemap(vmax=30.0)
    processor.visualize_rgb()

    print("\n所有可视化窗口已显示。关闭图形窗口后程序结束。")


if __name__ == "__main__":
    main()