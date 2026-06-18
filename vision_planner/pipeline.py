#!/usr/bin/env python3
"""
集成 G1 机器人与台阶地形，使用胸部相机渲染深度图，
生成点云并转换到骨盆坐标系（Y 向前），进行步点规划。
"""

import sys
import os
import time
import numpy as np
import mujoco
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.signal import convolve2d
from scipy.ndimage import distance_transform_edt, sobel

from planner import G1FootstepPlanner

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot.gen_xml import process_g1_model

# ==================== 1. 确保机器人模型存在 ====================
robot_dir = PROJECT_ROOT / "robot"
processed_xml = robot_dir / "g1_processed.xml"
if not processed_xml.exists():
    print("生成 g1_processed.xml...")
    process_g1_model()
else:
    print(f"使用已有模型: {processed_xml}")

# ==================== 2. 构建场景 XML ====================
# ==================== 2. 构建场景 XML（含棋盘纹理 + 台阶 + 机器人） ====================
asset_dir = PROJECT_ROOT / "asset"
scene_xml_path = asset_dir / "scene_with_robot.xml"
robot_rel_path = os.path.relpath(processed_xml, start=asset_dir)

# 台阶定义（使用材质以应用纹理）
steps = '''
    <!-- 台阶1: 上表面 Z=0.10，中心 X=0.75 -->
    <geom type="box" size="0.25 1.0 0.05" pos="0.75 0 0.05" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶2: 上表面 Z=0.20，中心 X=1.25 -->
    <geom type="box" size="0.25 1.0 0.05" pos="1.25 0 0.15" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶3: 上表面 Z=0.30，中心 X=1.75 -->
    <geom type="box" size="0.25 1.0 0.05" pos="1.75 0 0.25" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶4: 上表面 Z=0.40，中心 X=2.25 -->
    <geom type="box" size="0.25 1.0 0.05" pos="2.25 0 0.35" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶5: 上表面 Z=0.50，中心 X=2.75 -->
    <geom type="box" size="0.25 1.0 0.05" pos="2.75 0 0.45" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶6: 上表面 Z=0.60? 但原定义是0.55，我们保持0.55 -->
    <geom type="box" size="0.25 1.0 0.05" pos="3.25 0 0.55" material="step_mat" rgba="0.8 0.6 0.4 1"/>
'''

xml_content = f'''<mujoco model="g1_with_steps">
  <!-- 1. 引入机器人模型 -->
  <include file="{robot_rel_path}"/>
  
  <!-- 2. 覆盖资源路径，使STL从 robot/assets 加载 -->
  <compiler meshdir="../robot/assets"/>
  
  <!-- 3. 定义场景的纹理和材质 -->
  <asset>
    <texture name="ground_tex" type="2d" builtin="checker" 
             rgb1="0.2 0.3 0.4" rgb2="0.6 0.7 0.8" 
             width="300" height="300" mark="edge" random="0.01"/>
    <material name="groundplane" texture="ground_tex" texrepeat="4 4" 
              texuniform="true" reflectance="0.2"/>
    <material name="step_mat" rgba="0.8 0.6 0.4 1" reflectance="0.3"/>
  </asset>
  
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <!-- 地面（带棋盘纹理） -->
    <geom type="plane" size="5 5 0.1" pos="0 0 0" material="groundplane"/>
    <!-- 台阶 -->
    {steps}
  </worldbody>
</mujoco>'''

with open(scene_xml_path, 'w') as f:
    f.write(xml_content)
print(f"场景 XML 已保存至: {scene_xml_path}")

# ==================== 3. 加载场景并准备渲染 ====================
model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
data = mujoco.MjData(model)

# 重置到 stand 关键帧
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    actuator_qpos_indices = [7,8,9,10,11,12,13,14,15,16,17,18,21]
    data.ctrl[:] = data.qpos[actuator_qpos_indices]
    print("已重置到 'stand' 关键帧并同步 ctrl。")
else:
    print("警告: 未找到 'stand' 关键帧，使用默认重置。")
    mujoco.mj_resetData(model, data)

mujoco.mj_forward(model, data)

# 获取胸部相机
camera_name = "chest_camera"
try:
    camera_id = model.camera(camera_name).id
    print(f"找到胸部相机: {camera_name}")
except Exception as e:
    print(f"警告: 未找到相机 '{camera_name}'，使用默认视角。")
    camera_id = -1

width, height = 320, 240
renderer_rgb = mujoco.Renderer(model, width=width, height=height)
renderer_depth = mujoco.Renderer(model, width=width, height=height)
renderer_depth.enable_depth_rendering()

mujoco.mj_step(model, data)

start = time.perf_counter()

if camera_id != -1:
    renderer_rgb.update_scene(data, camera=camera_id)
    renderer_depth.update_scene(data, camera=camera_id)
else:
    renderer_rgb.update_scene(data)
    renderer_depth.update_scene(data)

rgb = renderer_rgb.render()
depth = renderer_depth.render()
print(f"Depth range: {depth.min():.3f} ~ {depth.max():.3f} m")
# ==================== 4. 相机坐标系点云 ====================
fov_deg = 60.0
focal_px = 0.5 * height / np.tan(0.5 * np.radians(fov_deg))
fx = fy = focal_px
cx, cy = width/2.0, height/2.0

rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
u, v = cols.flatten(), rows.flatten()
z = depth.flatten()

valid = np.isfinite(z) & (z > 0)
u, v, z = u[valid], v[valid], z[valid]

MIN_DEPTH, MAX_DEPTH = 0.5, 2.0
mask = (z >= MIN_DEPTH) & (z <= MAX_DEPTH)
u, v, z = u[mask], v[mask], z[mask]

Xc = (u - cx) * z / fx
Yc = (cy - v) * z / fy
Zc = -z
points_cam = np.stack((Xc, Yc, Zc), axis=-1)

print("\n相机坐标系点云范围:")
print(f"  X: [{points_cam[:,0].min():.3f}, {points_cam[:,0].max():.3f}]")
print(f"  Y: [{points_cam[:,1].min():.3f}, {points_cam[:,1].max():.3f}]")
print(f"  Z: [{points_cam[:,2].min():.3f}, {points_cam[:,2].max():.3f}]")
# ==================== 5. 相机 → 世界 ====================
if camera_id != -1:
    cam_pos = data.cam_xpos[camera_id].copy()
    cam_rot = data.cam_xmat[camera_id].copy().reshape(3, 3)
else:
    cam_pos = np.array([0, 0, 1])
    cam_rot = np.eye(3)

print("\n相机外参 (世界坐标系):")
print(f"位置: {cam_pos}")
print(f"旋转矩阵 (相机→世界):\n{cam_rot}")

points_world = (cam_rot @ points_cam.T).T + cam_pos

print("\n世界坐标系点云范围:")
print(f"  X: [{points_world[:,0].min():.3f}, {points_world[:,0].max():.3f}]")
print(f"  Y: [{points_world[:,1].min():.3f}, {points_world[:,1].max():.3f}]")
print(f"  Z: [{points_world[:,2].min():.3f}, {points_world[:,2].max():.3f}]")

# ==================== 6. 世界 → 骨盆坐标系（仅保留偏航角） ====================
pelvis_body_id = model.body("pelvis").id
pelvis_pos = data.xpos[pelvis_body_id].copy()
pelvis_quat = data.xquat[pelvis_body_id].copy()  # (w, x, y, z)

from scipy.spatial.transform import Rotation as R

# 将四元数转换为欧拉角 (xyz 顺序：roll, pitch, yaw)
r = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
euler = r.as_euler('xyz')
yaw = euler[2]  # 仅取绕 Z 轴的旋转角（偏航）

# 构建仅绕 Z 轴的旋转矩阵（骨盆→世界）
R_yaw_to_world = R.from_euler('z', yaw).as_matrix()

# 世界→骨盆（转置，即反向旋转）
R_world_to_pelvis = R_yaw_to_world.T

# 轴交换矩阵（根据你的定义：最终 X 向右，Y 向前，Z 向上）
# 如果你最终希望 Y 向前（即你之前的修正版本）
swap = np.eye(3)

# 变换点云
points_pelvis_tmp = (R_world_to_pelvis @ (points_world - pelvis_pos).T).T
points_pelvis = (swap @ points_pelvis_tmp.T).T

print("\n骨盆坐标系点云范围 (校正后):")
print(f"  X: [{points_pelvis[:,0].min():.3f}, {points_pelvis[:,0].max():.3f}]")
print(f"  Y: [{points_pelvis[:,1].min():.3f}, {points_pelvis[:,1].max():.3f}]")
print(f"  Z: [{points_pelvis[:,2].min():.3f}, {points_pelvis[:,2].max():.3f}]")

# ==================== 7. 裁剪 ====================

pts = points_pelvis
mask = (pts[:,0] > 0.15) & (pts[:,0] < 0.8) & (np.abs(pts[:,1]) < 0.5)
pts_cropped = pts[mask]

print("\n骨盆坐标系点云范围 (裁剪后):")
print(f"  X: [{pts_cropped[:,0].min():.3f}, {pts_cropped[:,0].max():.3f}]")
print(f"  Y: [{pts_cropped[:,1].min():.3f}, {pts_cropped[:,1].max():.3f}]")
print(f"  Z: [{pts_cropped[:,2].min():.3f}, {pts_cropped[:,2].max():.3f}]")

end = time.perf_counter()
gap1 = (end - start) * 1000
print(f"点云生成耗时: {gap1:.3f} 毫秒")

# ==================== 8. 可视化点云 ====================
if len(pts_cropped) > 10000:
    idx = np.random.choice(len(pts_cropped), 10000, replace=False)
    points_small = pts_cropped[idx]
else:
    points_small = pts_cropped

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_small[:,0], points_small[:,1], points_small[:,2],
           c=points_small[:,2], cmap='jet', s=0.5)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Point Cloud in Pelvis Frame (Y forward)")
ax.set_box_aspect([np.ptp(points_small[:,0]),
                   np.ptp(points_small[:,1]),
                   np.ptp(points_small[:,2])])
plt.show()

cv2.imshow("RGB Image", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
print("按任意键关闭 RGB 窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# ==================== 9. 高程图 ====================
print("\n生成高程图...")
start = time.perf_counter()
if len(pts_cropped) == 0:
    print("警告：裁剪后点云为空，无法生成高程图")
    height_map = np.zeros((1, 1))
    slope_map = np.zeros((1, 1))
else:
    x_min, x_max = 0.15, 0.8      # X 为前向（机器人前方）
    y_min, y_max = -0.5, 0.5      # Y 为侧向（左右）
    res = 0.05
    nx = int((x_max - x_min) / res) + 1
    ny = int((y_max - y_min) / res) + 1

    height_map = np.full((nx, ny), -np.inf)
    for px, py, pz in pts_cropped:
        ix = int((px - x_min) / res)
        iy = int((py - y_min) / res)
        if 0 <= ix < nx and 0 <= iy < ny:
            if pz > height_map[ix, iy]:
                height_map[ix, iy] = pz

    missing = ~np.isfinite(height_map) | (height_map == -np.inf)
    if np.any(missing):
        indices = distance_transform_edt(missing, return_distances=False, return_indices=True)
        height_map[missing] = height_map[tuple(indices)][missing]

end = time.perf_counter()
gap2 = (end - start) * 1000
print(f"高程图生成耗时: {gap2:.3f} 毫秒")
print(f"高程图尺寸: {nx} x {ny}")

# ==================== 10. 高程图可视化 ====================
fig2, ax2 = plt.subplots(figsize=(10, 6))
im = ax2.imshow(height_map.T, origin='lower',
                extent=[x_min, x_max, y_min, y_max],
                cmap='terrain', aspect='auto')
plt.colorbar(im, label='Height (m)')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('Elevation Map (Pelvis Frame)')
plt.show()

# ==================== 11. 坡度图 ====================
res = 0.05
start = time.perf_counter()
grad_x = sobel(height_map, axis=0) / res
grad_y = sobel(height_map, axis=1) / res
slope_map = np.arctan(np.sqrt(grad_x**2 + grad_y**2)) * 180.0 / np.pi
slope_map = np.nan_to_num(slope_map)
end = time.perf_counter()
gap3 = (end - start)*1000
print(f"坡度图计算耗时: {gap3:.3f} 毫秒")

fig3, ax3 = plt.subplots(figsize=(10, 6))
im2 = ax3.imshow(slope_map.T, origin='lower',
                 extent=[x_min, x_max, y_min, y_max],
                 cmap='hot', aspect='auto', vmin=0, vmax=30)
plt.colorbar(im2, label='Slope (deg)')
ax3.set_xlabel('X (m)')
ax3.set_ylabel('Y (m)')
ax3.set_title('Slope Map (Pelvis Frame)')
plt.show()

# ==================== 12. 规划器 ====================
print("\n开始步点规划...")

if 'x_edges' not in locals():
    x_edges = np.linspace(x_min, x_max, nx)
    y_edges = np.linspace(y_min, y_max, ny)

planner = G1FootstepPlanner()
planner.set_heightmap(height_map, slope_map, x_edges, y_edges, res)

current_foot = (0.125, -0.115, -0.8)
current_stance = -1
target = (1, 1)

start = time.perf_counter()
footstep, next_stance = planner.plan_next_footstep(current_foot, current_stance, target)
end = time.perf_counter()
gap4 = (end - start)*1000
print(f"步点生成耗时: {gap4:.3f} 毫秒")
print(f"落脚点: ({footstep.x:.3f}, {footstep.y:.3f}, {footstep.z:.3f}), 朝向 {np.degrees(footstep.yaw):.1f}°, 下一步用 {'左' if next_stance==-1 else '右'}脚")
print(f"总流程耗时: {(gap1 + gap2 + gap3 + gap4):.3f} 毫秒")