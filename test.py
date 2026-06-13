#!/usr/bin/env python3
"""
最终修正版：台阶地面 + 固定相机（高1.0m，俯角60°）
- 使用减法公式 points_world = (R @ P_c.T).T - cam_pos
- 骨盆坐标系平移 (0,0,0.8) 后 Y 取反
- 输出地面水平且 Z=0，Y 正向为左的点云
"""
import time
import numpy as np
import mujoco
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

script_dir = Path(__file__).parent

# ==================== 1. 构建 XML ====================
xml = '''<mujoco model="plane_with_camera">
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
    
    <!-- 扩大后的基础平面地面（5m×5m，位于 Z=0） -->
    <geom type="plane" size="5 5 0.1" pos="0 0 0" material="groundplane" rgba="0.6 0.8 1.0 1"/>
    
    <!-- 台阶1: 上表面 Z=0.10，中心 Y=0.75 -->
    <geom type="box" size="1.0 0.25 0.05" pos="0 0.75 0.05" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶2: 上表面 Z=0.20，中心 Y=1.25 -->
    <geom type="box" size="1.0 0.25 0.05" pos="0 1.25 0.15" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶3: 上表面 Z=0.30，中心 Y=1.75 -->
    <geom type="box" size="1.0 0.25 0.05" pos="0 1.75 0.25" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶4: 上表面 Z=0.40，中心 Y=2.25 -->
    <geom type="box" size="1.0 0.25 0.05" pos="0 2.25 0.35" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶5: 上表面 Z=0.50，中心 Y=2.75 -->
    <geom type="box" size="1.0 0.25 0.05" pos="0 2.75 0.45" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <geom type="box" size="1.0 0.25 0.05" pos="0 3.25 0.55" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 固定相机：位置(0,0,1.0)，绕X轴旋转60°，视野60度 -->
    <camera name="fixed_cam" pos="0 0 1.0" euler="30 30 0" fovy="60"/>
  </worldbody>
</mujoco>'''

# ==================== 2. 加载并渲染 ====================
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

camera_name = "fixed_cam"
camera_id = model.camera(camera_name).id
width, height = 640, 480

start = time.perf_counter()

renderer_rgb = mujoco.Renderer(model, width=width, height=height)
renderer_depth = mujoco.Renderer(model, width=width, height=height)
renderer_depth.enable_depth_rendering()

mujoco.mj_forward(model, data)
renderer_rgb.update_scene(data, camera=camera_id)
renderer_depth.update_scene(data, camera=camera_id)
rgb = renderer_rgb.render()
depth = renderer_depth.render()          # 垂直距离（米）

print(f"Depth range: {depth.min():.3f} ~ {depth.max():.3f} m")

# ==================== 3. 相机坐标系点云（直接用 z = depth）====================
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

# ==================== 4. 世界坐标系（使用减法修正）====================
cam_pos = data.cam_xpos[camera_id].copy()
cam_rot = data.cam_xmat[camera_id].copy().reshape(3, 3)

print("\n相机外参 (世界坐标系):")
print(f"位置: {cam_pos}")
print(f"旋转矩阵 (相机→世界):\n{cam_rot}")


points_world = (cam_rot @ points_cam.T).T + cam_pos


print("\n世界坐标系点云范围:")
print(f"  X: [{points_world[:,0].min():.3f}, {points_world[:,0].max():.3f}]")
print(f"  Y: [{points_world[:,1].min():.3f}, {points_world[:,1].max():.3f}]")
print(f"  Z: [{points_world[:,2].min():.3f}, {points_world[:,2].max():.3f}]")


# ==================== 5. 骨盆坐标系 ====================
pelvis_pos = np.array([0.0, 0.0, 0.8])      # 骨盆在世界中的位置
points_pelvis = points_world - pelvis_pos  
points_final = points_pelvis

print("\n骨盆坐标系点云范围 (校正后):")
print(f"  X: [{points_final[:,0].min():.3f}, {points_final[:,0].max():.3f}]")
print(f"  Y: [{points_final[:,1].min():.3f}, {points_final[:,1].max():.3f}]")
print(f"  Z: [{points_final[:,2].min():.3f}, {points_final[:,2].max():.3f}]")

# ==================== 6.范围裁剪 ====================
pts = points_final
mask = (pts[:,1] > 0.15) & (pts[:,1] < 0.8) & (np.abs(pts[:,0]) < 0.5)      
pts_cropped = pts[mask]

print("\n骨盆坐标系点云范围 (裁剪后):")
print(f"  X: [{pts_cropped[:,0].min():.3f}, {pts_cropped[:,0].max():.3f}]")
print(f"  Y: [{pts_cropped[:,1].min():.3f}, {pts_cropped[:,1].max():.3f}]")
print(f"  Z: [{pts_cropped[:,2].min():.3f}, {pts_cropped[:,2].max():.3f}]")

end = time.perf_counter()
elapsed_ms = (end - start) * 1000
print(f"执行耗时: {elapsed_ms:.3f} 毫秒")

# ==================== 7. 可视化 ====================
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
ax.set_title("Point Cloud in Pelvis Frame (Final)")
ax.set_box_aspect([np.ptp(points_small[:,0]),
                   np.ptp(points_small[:,1]),
                   np.ptp(points_small[:,2])])
plt.show()

cv2.imshow("RGB Image", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
print("按任意键关闭窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# ==================== 8. 生成高程图 ====================
print("\n生成高程图...")
start = time.perf_counter()
# 检查点云是否为空
if len(pts_cropped) == 0:
    print("警告：裁剪后点云为空，无法生成高程图")
    # 创建一个默认的高程图（全零）作为占位
    height_map = np.zeros((1, 1))
    slope_map = np.zeros((1, 1))
else:
    # 定义高程图范围（骨盆坐标系，X侧向，Y前向）
    x_min, x_max = -0.5, 0.5      # X 范围（左右）
    y_min, y_max = 0.15, 0.8      # Y 范围（前向）
    res = 0.05                    # 网格分辨率（米）
    nx = int((x_max - x_min) / res) + 1
    ny = int((y_max - y_min) / res) + 1
    
    # 初始化高程图（每个网格存储最高点，初始为 -inf）
    height_map = np.full((nx, ny), -np.inf)
    
    # 填充高程图
    for px, py, pz in pts_cropped:
        ix = int((px - x_min) / res)
        iy = int((py - y_min) / res)
        if 0 <= ix < nx and 0 <= iy < ny:
            if pz > height_map[ix, iy]:
                height_map[ix, iy] = pz
    
    # 处理未覆盖的网格（使用最近邻插值填充）
    from scipy.ndimage import distance_transform_edt
    missing = ~np.isfinite(height_map) | (height_map == -np.inf)
    if np.any(missing):
        indices = distance_transform_edt(missing, return_distances=False, return_indices=True)
        height_map[missing] = height_map[tuple(indices)][missing]

    

end_elev = time.perf_counter()
print(f"高程图生成耗时: {(end_elev - start)*1000:.3f} 毫秒")
print(f"高程图尺寸: {nx} x {ny}")

# ==================== 9. 可视化高程图（2D热力图）====================
fig2, ax2 = plt.subplots(figsize=(10, 6))
im = ax2.imshow(height_map.T, origin='lower', 
                extent=[x_min, x_max, y_min, y_max],
                cmap='terrain', aspect='auto')
plt.colorbar(im, label='Height (m)')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_title('Elevation Map (Pelvis Frame)')
plt.show()