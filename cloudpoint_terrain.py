#!/usr/bin/env python3
"""
测试：随机起伏地形 + 固定相机视角（高度1.0m，俯角60度）。
拍摄彩色图和深度图，生成点云并转换到骨盆坐标系（地面水平）。
"""

import numpy as np
import mujoco
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import convolve2d
from sklearn.decomposition import PCA

# 当前脚本所在目录
script_dir = Path(__file__).parent

# ==================== 1. 生成随机起伏地形（高度场）====================
print("生成随机起伏地形...")
nrow, ncol = 80, 80               # 网格分辨率
x_len, y_len = 5.0, 5.0           # 地形物理尺寸（米）
z_min, z_max = 0.0, 0.2           # 高度范围（米）

# 随机高度场 + 平滑
hf = np.random.randn(nrow, ncol).astype(np.float32)
kernel = np.ones((4,4), dtype=np.float32) / 16
hf = convolve2d(hf, kernel, mode='same')
rel_h = (hf - hf.min()) / (hf.max() - hf.min())
elev_int = (rel_h * 65535).astype(np.uint32)
elev_str = ' '.join(elev_int.flatten('C').astype(str))

sx = x_len / 2
sy = y_len / 2
sz_half = (z_max - z_min) / 2
z_mean = (z_min + z_max) / 2

# ==================== 2. 构建完整 XML（地形 + 相机 + 光源）====================
xml = f'''<mujoco model="rough_terrain_with_camera">
  <asset>
    <texture name="ground_tex" type="2d" builtin="checker" 
             rgb1="0.2 0.3 0.4" rgb2="0.6 0.7 0.8" 
             width="300" height="300" mark="edge" random="0.01"/>
    <material name="groundplane" texture="ground_tex" texrepeat="2 2" 
              texuniform="true" reflectance="0.2"/>
    <hfield name="ground" size="{sx} {sy} {sz_half} {z_mean}" 
            nrow="{nrow}" ncol="{ncol}" 
            elevation="{elev_str}"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <geom type="hfield" hfield="ground" material="groundplane" rgba="0.6 0.8 1.0 1"/>
    <!-- 固定相机：位置(0,0,1.0)，绕X轴旋转60°（向下看） -->
    <camera name="fixed_cam" pos="0 0 1" euler="60 0 0" fovy="60"/>
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