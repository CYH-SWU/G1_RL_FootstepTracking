#!/usr/bin/env python3
"""
测试固定相机视角（高度0.5m，俯角60度）的地面场景。
拍摄彩色图和深度图，然后直接生成点云并可视化（使用 matplotlib）。
"""

import numpy as np
import mujoco
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 当前脚本所在目录
script_dir = Path(__file__).parent

# 创建 XML 模型
xml = '''<mujoco model="test_camera_view">
  <asset>
    <texture name="ground_tex" type="2d" builtin="checker" 
             rgb1="0.2 0.3 0.4" rgb2="0.6 0.7 0.8" 
             width="300" height="300" mark="edge" random="0.01"/>
    <material name="groundplane" texture="ground_tex" texrepeat="4 4" 
              texuniform="true" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/> 
    <geom type="box" size="5 5 0.05" pos="0 0 -0.05" material="groundplane" rgba="0.6 0.8 1.0 1"/>
    <camera name="fixed_cam" pos="0 0 0.5" euler="60 0 0" fovy="60"/>
  </worldbody>
</mujoco>'''

# 加载模型
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# 相机参数
camera_name = "fixed_cam"
camera_id = model.camera(camera_name).id
width, height = 640, 480

# 创建渲染器
renderer_rgb = mujoco.Renderer(model, width=width, height=height)
renderer_depth = mujoco.Renderer(model, width=width, height=height)
renderer_depth.enable_depth_rendering()

# 前向仿真一次
mujoco.mj_forward(model, data)

# 渲染
renderer_rgb.update_scene(data, camera=camera_id)
renderer_depth.update_scene(data, camera=camera_id)
rgb = renderer_rgb.render()      # (height, width, 3), RGB
depth = renderer_depth.render()  # (height, width), float, 单位: 米 相机坐标系下的Z轴深度

print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")
print(f"Depth range: {depth.min():.3f} ~ {depth.max():.3f} m")

# 保存 RGB 图像
rgb_path = script_dir / "rgb_image.png"
cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
print(f"RGB 图像已保存至: {rgb_path}")

# 保存原始深度图
depth_raw_path = script_dir / "depth.npy"
np.save(depth_raw_path, depth)
print(f"原始深度数据已保存至: {depth_raw_path}")

# 深度图可视化（剪切）
DEPTH_MIN_DISP = 0.3
DEPTH_MAX_DISP = 3.0
depth_clipped = np.clip(depth, DEPTH_MIN_DISP, DEPTH_MAX_DISP)
depth_norm = (depth_clipped - DEPTH_MIN_DISP) / (DEPTH_MAX_DISP - DEPTH_MIN_DISP) * 255
depth_norm = depth_norm.astype(np.uint8)
depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
depth_color_path = script_dir / "depth_color.png"
cv2.imwrite(str(depth_color_path), depth_color)
print(f"深度伪彩色图已保存至: {depth_color_path}")

# ========== 从深度图生成点云 ==========
print("\n正在生成点云...")

# 相机内参
fov_deg = 60.0
focal_px = 0.5 * height / np.tan(0.5 * np.radians(fov_deg))
fx = fy = focal_px
cx = width / 2.0
cy = height / 2.0
print(f"相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

# 像素网格
rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
u = cols.flatten()
v = rows.flatten()
z = depth.flatten()

# 过滤有效深度
valid = np.isfinite(z) & (z > 0)
u = u[valid]
v = v[valid]
z = z[valid]

# 深度范围剪切
MIN_DEPTH_PC = 0.3
MAX_DEPTH_PC = 3.0
mask = (z >= MIN_DEPTH_PC) & (z <= MAX_DEPTH_PC)
u = u[mask]
v = v[mask]
z = z[mask]

print(f"有效点数量: {len(z)}")

# 计算点云（相机坐标系）
Xc = (u - cx) * z / fx
Yc = (v - cy) * z / fy
Zc = z
points_cam = np.stack((Xc, Yc, Zc), axis=-1)


print("点云范围 (相机坐标系):")
print(f"  X: [{points_cam[:,0].min():.3f}, {points_cam[:,0].max():.3f}]")
print(f"  Y: [{points_cam[:,1].min():.3f}, {points_cam[:,1].max():.3f}]")
print(f"  Z: [{points_cam[:,2].min():.3f}, {points_cam[:,2].max():.3f}]")

# 保存点云为 PLY 文件（使用 open3d 如果可用，否则保存为 XYZ）
try:
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_cam)
    o3d.io.write_point_cloud(str(script_dir / "pointcloud.ply"), pcd)
    print(f"点云已保存至: {script_dir / 'pointcloud.ply'}")
except ImportError:
    np.savetxt(script_dir / "pointcloud.xyz", points_cam, delimiter=' ')
    print(f"点云已保存至: {script_dir / 'pointcloud.xyz'} (XYZ 文本格式)")

# ========== 可视化点云（matplotlib）==========
print("使用 matplotlib 可视化点云（可能需要几秒钟）...")
# 随机下采样到 10000 点，避免卡死
if len(points_cam) > 10000:
    idx = np.random.choice(len(points_cam), 10000, replace=False)
    points_small = points_cam[idx]
else:
    points_small = points_cam

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_small[:,0], points_small[:,1], points_small[:,2],
           c=points_small[:,2], cmap='jet', s=0.5)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Point Cloud from Depth (Camera Frame)")
plt.show()

# 显示 RGB 和深度图窗口
cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
cv2.imshow("Depth (clipped)", depth_color)
print("按任意键关闭所有窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()