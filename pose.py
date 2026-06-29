#!/usr/bin/env python3
"""
调整 G1 机器人标称姿态的辅助脚本，并检查重心投影是否在支撑面内。

功能：
1. 加载处理后的 G1 模型，创建地面，关闭重力。
2. 切换到 stand 关键帧。
3. 可手动设置骨盆高度（默认由关键帧决定）。
4. 计算并打印骨盆与脚底的高度差。
5. 计算全身质心（COM）位置。
6. 获取脚掌支撑面的多边形（四个接触点）。
7. 判断质心投影是否落在支撑面内。
8. 启动 MuJoCo 查看器，方便可视化微调。
"""

import os
import sys
import time
import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()
MODEL_PATH = PROJECT_ROOT / "robot" / "g1_processed.xml"


def get_support_polygon(model, data):
    """
    获取双脚脚掌接触点（四个球体）的世界坐标，形成支撑多边形。
    返回左右脚各自的接触点列表（每个列表包含四个点）和合并后的多边形点集。
    """
    # 找到所有 class="foot" 的 geom
    foot_geoms = []
    for i in range(model.ngeom):
        # 检查 geom 的类名（在 XML 中为 class="foot"）
        # 在 MuJoCo 中，class 信息存储在 model.geom_class 中？
        # 更可靠的方式：检查 geom 的 name 或通过 body 推断。
        # 简单方法：根据 body 名称来找。
        body_id = model.geom_bodyid[i]
        body_name = model.body(body_id).name
        if "ankle_roll_link" in body_name:
            foot_geoms.append(i)

    # 但我们的脚掌接触点有四个球体，它们属于同一个 body（ankle_roll_link）
    # 实际上，在 XML 中，它们直接位于 ankle_roll_link 下，没有独立的 body。
    # 获取这些 geom 的位置需要从 data.geom_xpos 中读取，但我们需要知道它们的 ID。
    # 更好的方法：通过 geom 的 pos 属性在 body 局部坐标，但为了简化，我们可以遍历所有 geom，
    # 检查其 size 是否为 (0.005,)，即球体半径，且其 parent body 为 ankle_roll_link。
    # 但更直接：在 XML 中它们被标记为 class="foot"，但 MuJoCo 的 Python API 不直接暴露 class。
    # 替代方案：硬编码 geom 名称（如果知道）。
    # 在您的 XML 中，这些 geom 没有名称。所以我们需要通过位置关系推断：
    # 它们相对于 ankle_roll_link 的位置在 XML 中为 pos="-0.05 0.025 -0.03" 等。
    # 但更简单：直接在 reset 后从 data.geom_xpos 中取出那些 z 值最低的 geom。
    # 但为了准确，我建议您给这些 geom 添加名称，或者在脚本中通过坐标近似。

    # 此处，我们采用一种近似方法：获取左右脚踝位置，然后根据固定偏移计算接触点。
    # 但为了精确，更好的办法是修改 XML 给脚掌 geom 添加名称。
    # 为了演示，我假设您已经添加了名称，或者在您当前的 XML 中，脚掌 geom 有特定的 ID。
    # 由于无法确定，我改为直接从 XML 中读取这些 geom 的 pos 并转换到世界坐标。
    # 使用 model.geom_pos 和 model.geom_quat 计算。
    # 但 muoco 中 geom 的 pos 是相对于 body 的局部位置，需要结合 body 的变换。
    # 另一种方式：从 data.geom_xpos 直接获取，只要我们知道哪些 geom 是脚掌。
    # 我们可以通过 geom 的 size 和类型来识别（四个半径为 0.005 的球体）。
    # 下面实现一个更通用的方法：

    foot_contact_points = []
    foot_geom_ids = []
    for i in range(model.ngeom):
        # 检查几何体是否为球体，且大小约为 0.005
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE:
            size = model.geom_size[i]
            if np.isclose(size[0], 0.005, atol=1e-4):
                foot_geom_ids.append(i)

    if len(foot_geom_ids) != 8:
        print(f"警告：找到 {len(foot_geom_ids)} 个可能的脚掌接触点，期望 8 个。")
        # 如果不匹配，可能因为 XML 结构变化，可尝试其他识别方式。
        # 此处继续使用识别出的点。

    # 按 body 分组（左右脚）
    foot_geoms_by_body = {}
    for gid in foot_geom_ids:
        body_id = model.geom_bodyid[gid]
        if body_id not in foot_geoms_by_body:
            foot_geoms_by_body[body_id] = []
        foot_geoms_by_body[body_id].append(gid)

    # 取两个 body（左右脚）
    if len(foot_geoms_by_body) >= 2:
        bodies = list(foot_geoms_by_body.keys())
        left_points = []
        right_points = []
        for i, body_id in enumerate(bodies):
            points = []
            for gid in foot_geoms_by_body[body_id]:
                world_pos = data.geom_xpos[gid].copy()
                points.append(world_pos[:2])  # 取 XY 平面投影
            if i == 0:
                left_points = points
            else:
                right_points = points
        # 合并左右脚的所有点
        all_points = left_points + right_points
        return all_points, left_points, right_points
    else:
        print("无法识别脚掌接触点，请检查 XML 中脚掌 geom 的设置。")
        return [], [], []


def is_point_in_polygon(point, polygon_points):
    """判断点是否在多边形内（包括边界）"""
    if len(polygon_points) < 3:
        return False
    poly = Polygon(polygon_points)
    pt = Point(point)
    return poly.contains(pt) or poly.touches(pt)


def main():
    if not MODEL_PATH.exists():
        print(f"错误：模型文件不存在: {MODEL_PATH}")
        print("请确认模型路径，或修改脚本中的 MODEL_PATH。")
        return

    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # 取消重力
    model.opt.gravity = np.zeros(3)
    print("重力已关闭 (重力加速度设为 0)")

    # 切换到 stand 关键帧
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("已切换到 'stand' 关键帧")
    else:
        print("警告：未找到 'stand' 关键帧，使用默认重置。")
        mujoco.mj_resetData(model, data)

    # 可选：设置骨盆高度（例如 0.70），注释掉则使用关键帧原生高度
    # data.qpos[2] = 0.70
    # print(f"骨盆高度已设置为 {data.qpos[2]:.4f} m")

    # 前向计算，更新派生量
    mujoco.mj_forward(model, data)

    # ----- 1. 计算骨盆与脚底高度差 -----
    pelvis_id = model.body("pelvis").id
    try:
        left_foot_id = model.body("left_ankle_roll_link").id
        right_foot_id = model.body("right_ankle_roll_link").id
    except:
        left_foot_id = model.body("left_foot").id
        right_foot_id = model.body("right_foot").id

    pelvis_z = data.xpos[pelvis_id][2]
    left_foot_z = data.xpos[left_foot_id][2]
    right_foot_z = data.xpos[right_foot_id][2]
    foot_z_min = min(left_foot_z, right_foot_z)
    height_diff = pelvis_z - foot_z_min

    print(f"\n--- 当前姿态高度信息 ---")
    print(f"骨盆高度 (z): {pelvis_z:.4f} m")
    print(f"左脚高度 (z): {left_foot_z:.4f} m")
    print(f"右脚高度 (z): {right_foot_z:.4f} m")
    print(f"骨盆与较低脚的高度差: {height_diff:.4f} m")

    # ----- 2. 计算质心（COM）位置 -----
    mujoco.mj_comPos(model, data)
    # data.subtree_com[0] 为根 body（pelvis）的子树质心，即全身质心
    com_world = data.subtree_com[0].copy()
    print(f"\n--- 质心位置 ---")
    print(f"质心 (x, y, z): ({com_world[0]:.4f}, {com_world[1]:.4f}, {com_world[2]:.4f}) m")

    # ----- 3. 获取脚掌支撑多边形 -----
    all_points, left_points, right_points = get_support_polygon(model, data)
    if len(all_points) < 3:
        print("警告：未能获取足够的脚掌支撑点，无法判断重心投影。")
    else:
        # 由于脚掌接触点可能有左右脚各4个，合并成一个凸包
        # 使用凸包来近似支撑面（可能包含内部点，但凸包已涵盖）
        # 注意：实际支撑面是左右脚接触点的凸包，但不一定所有点都在凸包边界上。
        # 更准确的是分别判断质心是否在左脚或右脚多边形内，但通常左右脚同时着地时，支撑面是两者的并集。
        # 简单合并所有点，然后计算凸包。
        if len(all_points) >= 3:
            hull = ConvexHull(all_points)
            hull_points = [all_points[i] for i in hull.vertices]
            # 判断质心投影是否在凸包内
            com_xy = com_world[:2]
            inside = is_point_in_polygon(com_xy, hull_points)
            print(f"\n--- 重心投影检查 ---")
            print(f"质心投影 (x, y): ({com_xy[0]:.4f}, {com_xy[1]:.4f})")
            print(f"支撑面凸包顶点: {hull_points}")
            print(f"质心投影在支撑面内: {'是' if inside else '否'}")
            if not inside:
                print("警告：质心投影落在支撑面外，该姿态在重力作用下可能不稳定。")
        else:
            print("支撑点数不足，无法形成凸包。")

    print("-------------------------\n")

    # 启动查看器
    print("启动 MuJoCo 查看器。按 Esc 或关闭窗口退出。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.time() - step_start
            time_to_sleep = model.opt.timestep - elapsed
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)


if __name__ == "__main__":
    main()