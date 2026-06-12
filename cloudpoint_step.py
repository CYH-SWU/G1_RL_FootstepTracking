#!/usr/bin/env python3
"""
最终修正版：台阶地面 + 固定相机（高1.0m，俯角60°）
- 使用减法公式 points_world = (R @ P_c.T).T - cam_pos
- 骨盆坐标系平移 (0,0,0.8) 后 Y 取反
- 输出地面水平且 Z=0，Y 正向为左的点云
"""

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
    <!-- 基础平面地面 -->
    <geom type="plane" size="3 3 0.1" pos="0 0 0" material="groundplane" rgba="0.6 0.8 1.0 1"/>
    
    <!-- 台阶1: Y方向 -0.5 ~ 1.0，高度 0.05 (上表面 Z=0.05) -->
    <geom type="box" size="1.0 0.25 0.05" pos="0 0.75 0.05" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶2: Y方向 -1.0 ~ 1.5，高度 0.15 (上表面 Z=0.15) -->
    <geom type="box" size="1.0 0.25 0.15" pos="0 1.25 0.15" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶3: Y方向 -1.5 ~ 2.0，高度 0.25 (上表面 Z=0.25) -->
    <geom type="box" size="1.0 0.25 0.25" pos="0 1.75 0.25" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    
    <!-- 固定相机 -->
    <camera name="fixed_cam" pos="0 0 1.0" euler="60 0 0" fovy="60"/>
  </worldbody>
</mujoco>'''

# ==================== 2. 加载并渲染 ====================
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

camera_name = "fixed_cam"
camera_id = model.camera(camera_name).id
width, height = 640, 480

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

MIN_DEPTH, MAX_DEPTH = 0.3, 3.0
mask = (z >= MIN_DEPTH) & (z <= MAX_DEPTH)
u, v, z = u[mask], v[mask], z[mask]

Xc = (u - cx) * z / fx
Yc = (v - cy) * z / fy
Zc = z
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

points_world = -(cam_rot @ points_cam.T).T + cam_pos
points_world[:,0] = -points_world[:,0]


print("\n世界坐标系点云范围:")
print(f"  X: [{points_world[:,0].min():.3f}, {points_world[:,0].max():.3f}]")
print(f"  Y: [{points_world[:,1].min():.3f}, {points_world[:,1].max():.3f}]")
print(f"  Z: [{points_world[:,2].min():.3f}, {points_world[:,2].max():.3f}]")

# ==================== 5. 水平性验证 ====================
pca = PCA(n_components=3)
pca.fit(points_world[::100])
normal = pca.components_[2]
print(f"世界坐标系下点云平面法线: {normal}")

center_mask = (np.abs(points_world[:,0])<0.5) & (np.abs(points_world[:,1])<0.5)
if np.any(center_mask):
    center_z = np.mean(points_world[center_mask,2])
    print(f"地面中心区域平均 Z 值: {center_z:.3f} m (应接近 0)")

# ==================== 6. 骨盆坐标系 ====================
pelvis_pos = np.array([0.0, 0.0, 0.8])      # 骨盆在世界中的位置
points_pelvis = points_world - pelvis_pos  
points_final = points_pelvis

print("\n骨盆坐标系点云范围 (校正后):")
print(f"  X: [{points_final[:,0].min():.3f}, {points_final[:,0].max():.3f}]")
print(f"  Y: [{points_final[:,1].min():.3f}, {points_final[:,1].max():.3f}]")
print(f"  Z: [{points_final[:,2].min():.3f}, {points_final[:,2].max():.3f}]")

# ==================== 7. 保存点云（可选）====================
try:
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_final)
    o3d.io.write_point_cloud(str(script_dir / "pointcloud_final.ply"), pcd)
    print(f"点云已保存至: {script_dir / 'pointcloud_final.ply'}")
except ImportError:
    np.savetxt(script_dir / "pointcloud_final.xyz", points_final, delimiter=' ')
    print(f"点云已保存至: {script_dir / 'pointcloud_final.xyz'}")

# ==================== 8. 可视化 ====================
if len(points_final) > 10000:
    idx = np.random.choice(len(points_final), 10000, replace=False)
    points_small = points_final[idx]
else:
    points_small = points_final

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