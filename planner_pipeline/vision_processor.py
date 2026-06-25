import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import cv2
from typing import Optional, Tuple, Dict, Any

class VisionProcessor:
    """
    视觉处理器：从胸部相机获取深度图，生成点云，转换到骨盆坐标系，
    裁剪、生成高程图和坡度图。
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        camera_name: str = "chest_camera",
        pelvis_name: str = "pelvis",
        width: int = 320,
        height: int = 240,
        fov_deg: float = 60.0,
        depth_min: float = 0.3,
        depth_max: float = 2.0,
        crop_x_min: float = 0.15,
        crop_x_max: float = 0.8,
        crop_y_min: float = -0.5,
        crop_y_max: float = 0.5,
        heightmap_resolution: float = 0.025,
    ):
        """
        初始化视觉处理器。
        
        :param model: MuJoCo 模型
        :param data: MuJoCo 数据
        :param camera_name: 胸部相机名称
        :param pelvis_name: 骨盆 body 名称
        :param width: 渲染宽度
        :param height: 渲染高度
        :param fov_deg: 相机视场角
        :param depth_min: 有效深度最小值
        :param depth_max: 有效深度最大值
        :param crop_x_min: 骨盆坐标系下 X 方向裁剪最小值
        :param crop_x_max: 骨盆坐标系下 X 方向裁剪最大值
        :param crop_y_min: 骨盆坐标系下 Y 方向裁剪最小值
        :param crop_y_max: 骨盆坐标系下 Y 方向裁剪最大值
        :param heightmap_resolution: 高程图网格分辨率
        """
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.pelvis_name = pelvis_name
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.crop_x_min = crop_x_min
        self.crop_x_max = crop_x_max
        self.crop_y_min = crop_y_min
        self.crop_y_max = crop_y_max
        self.heightmap_res = heightmap_resolution
        
        # 查找相机 ID
        try:
            self.camera_id = model.camera(camera_name).id
        except Exception:
            self.camera_id = -1
            print(f"警告: 未找到相机 '{camera_name}'，将使用默认视角。")
        
        # 查找骨盆 body ID
        try:
            self.pelvis_id = model.body(pelvis_name).id
        except Exception:
            raise ValueError(f"未找到 body: {pelvis_name}")
        
        # 创建深度渲染器
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        self.renderer.enable_depth_rendering()
        
        # 缓存最近一次处理的结果
        self.last_depth = None
        self.last_rgb = None
        self.last_points_cam = None
        self.last_points_world = None
        self.last_points_pelvis = None
        self.last_points_cropped = None
        self.last_heightmap = None
        self.last_slopemap = None
        self.last_x_edges = None
        self.last_y_edges = None

    def update_model_data(self, model, data):
        '''更新持有的model与data'''
        self.model = model
        self.data = data
        self.renderer = mujoco.Renderer(model, width=self.width, height=self.height)
        self.renderer.enable_depth_rendering()
        
    def process(self, render_rgb: bool = False, verbose: bool = False) -> Dict[str, Any]:
        """
        完整视觉处理流程
        
        :param render_rgb: 是否同时渲染 RGB 图像
        :param verbose: 是否打印详细耗时信息
        :return: 包含以下键的字典：
            - 'depth': 深度图 (H, W)
            - 'rgb': RGB 图像 (H, W, 3) 或 None
            - 'points_cam': 相机坐标系点云 (N, 3)
            - 'points_world': 世界坐标系点云 (N, 3)
            - 'points_pelvis': 骨盆坐标系点云 (N, 3)
            - 'points_cropped': 裁剪后骨盆坐标系点云 (M, 3)
            - 'heightmap': 高程图 (nx, ny)
            - 'slopemap': 坡度图 (nx, ny)
            - 'x_edges': X 方向边缘数组 (nx,)
            - 'y_edges': Y 方向边缘数组 (ny,)
            - 'timing': 各步骤耗时字典
        """
        timing = {}
        start_total = time.perf_counter()
        
        # 渲染深度图
        start = time.perf_counter()
        self.renderer.update_scene(self.data, camera=self.camera_id if self.camera_id != -1 else None)
        depth = self.renderer.render()
        timing['depth_render'] = (time.perf_counter() - start) * 1000
        
        rgb = None
        if render_rgb:
            start = time.perf_counter()
            self.renderer.disable_depth_rendering()
            self.renderer.update_scene(self.data, camera=self.camera_id if self.camera_id != -1 else None)
            rgb = self.renderer.render()
            self.renderer.enable_depth_rendering()  # 恢复
            timing['rgb_render'] = (time.perf_counter() - start) * 1000  
        
        self.last_depth = depth
        self.last_rgb = rgb
        
        # 计算相机内参
        start = time.perf_counter()
        fov_rad = np.radians(self.fov_deg)
        focal_px = 0.5 * self.height / np.tan(0.5 * fov_rad)
        fx = fy = focal_px
        cx, cy = self.width / 2.0, self.height / 2.0
        
        rows, cols = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
        u, v = cols.flatten(), rows.flatten()
        z_flat = depth.flatten()
        
        valid = np.isfinite(z_flat) & (z_flat > 0)
        u, v, z = u[valid], v[valid], z_flat[valid]
        
        # 深度范围滤波
        mask = (z >= self.depth_min) & (z <= self.depth_max)
        u, v, z = u[mask], v[mask], z[mask]
        
        # 轴转换
        Xc = (u - cx) * z / fx
        Yc = (cy - v) * z / fy
        Zc = -z
        points_cam = np.stack((Xc, Yc, Zc), axis=-1)
        timing['pointcloud_cam'] = (time.perf_counter() - start) * 1000
        self.last_points_cam = points_cam
        
        # 相机 → 世界 
        start = time.perf_counter()
        if self.camera_id != -1:
            cam_pos = self.data.cam_xpos[self.camera_id].copy()
            cam_rot = self.data.cam_xmat[self.camera_id].copy().reshape(3, 3)
        else:
            cam_pos = np.array([0, 0, 1])
            cam_rot = np.eye(3)
        
        points_world = (cam_rot @ points_cam.T).T + cam_pos
        timing['world_transform'] = (time.perf_counter() - start) * 1000
        self.last_points_world = points_world
        
        # 世界 → 骨盆
        start = time.perf_counter()
        pelvis_pos = self.data.xpos[self.pelvis_id].copy()
        pelvis_quat = self.data.xquat[self.pelvis_id].copy()  # (w, x, y, z)
        
        r_quat = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]])
        euler = r_quat.as_euler('xyz')
        yaw = euler[2]
        
        # 构建绕 Z 轴的旋转矩阵
        R_yaw_to_world = R.from_euler('z', yaw).as_matrix()
        R_world_to_pelvis = R_yaw_to_world.T  
        
        swap = np.eye(3)  # 无轴交换
        points_pelvis_tmp = (R_world_to_pelvis @ (points_world - pelvis_pos).T).T
        points_pelvis = (swap @ points_pelvis_tmp.T).T
        timing['pelvis_transform'] = (time.perf_counter() - start) * 1000
        self.last_points_pelvis = points_pelvis
        
        # 裁剪 
        start = time.perf_counter()
        pts = points_pelvis
        mask = (pts[:,0] > self.crop_x_min) & (pts[:,0] < self.crop_x_max) & \
               (np.abs(pts[:,1]) > self.crop_y_min) & (np.abs(pts[:,1]) < self.crop_y_max)
        points_cropped = pts[mask]
        timing['crop'] = (time.perf_counter() - start) * 1000
        self.last_points_cropped = points_cropped
        
        # 生成高程图 
        start = time.perf_counter()
        if len(points_cropped) == 0:
            heightmap = np.zeros((1, 1))
            slopemap = np.zeros((1, 1))
            x_edges = np.array([self.crop_x_min, self.crop_x_max])
            y_edges = np.array([self.crop_y_min, self.crop_y_max])
        else:
            x_min, x_max = self.crop_x_min, self.crop_x_max
            y_min, y_max = self.crop_y_min, self.crop_y_max
            res = self.heightmap_res
            nx = int((x_max - x_min) / res) + 1
            ny = int((y_max - y_min) / res) + 1
            
            heightmap = np.full((nx, ny), -np.inf)
            for px, py, pz in points_cropped:
                ix = int((px - x_min) / res)
                iy = int((py - y_min) / res)
                if 0 <= ix < nx and 0 <= iy < ny:
                    if pz > heightmap[ix, iy]:
                        heightmap[ix, iy] = pz
            
            missing = ~np.isfinite(heightmap) | (heightmap == -np.inf)
            if np.any(missing):
                indices = distance_transform_edt(missing, return_distances=False, return_indices=True)
                heightmap[missing] = heightmap[tuple(indices)][missing]
            
            # 计算坡度图(手动向量化差分)
            nx, ny = heightmap.shape
            grad_x = np.zeros((nx, ny))
            grad_y = np.zeros((nx, ny))
            
            grad_x[1:-1, :] = (heightmap[2:, :] - heightmap[:-2, :]) / (2.0 * res)
            grad_x[0, :] = (heightmap[1, :] - heightmap[0, :]) / res
            grad_x[-1, :] = (heightmap[-1, :] - heightmap[-2, :]) / res
            
            grad_y[:, 1:-1] = (heightmap[:, 2:] - heightmap[:, :-2]) / (2.0 * res)
            grad_y[:, 0] = (heightmap[:, 1] - heightmap[:, 0]) / res
            grad_y[:, -1] = (heightmap[:, -1] - heightmap[:, -2]) / res
            
            slopemap = np.arctan(np.sqrt(grad_x**2 + grad_y**2)) * 180.0 / np.pi
            slopemap = np.nan_to_num(slopemap, nan=0.0)
            
            x_edges = np.linspace(x_min, x_max, nx)
            y_edges = np.linspace(y_min, y_max, ny)
        
        timing['heightmap'] = (time.perf_counter() - start) * 1000
        self.last_heightmap = heightmap
        self.last_slopemap = slopemap
        self.last_x_edges = x_edges
        self.last_y_edges = y_edges
        
        timing['total'] = (time.perf_counter() - start_total) * 1000
        
        if verbose:
            self._print_timing(timing)
            self._print_stats(points_cam, points_world, points_pelvis, points_cropped,
                              heightmap, slopemap)
        
        return {
            'depth': depth,
            'rgb': rgb,
            'points_cam': points_cam,
            'points_world': points_world,
            'points_pelvis': points_pelvis,
            'points_cropped': points_cropped,
            'heightmap': heightmap,
            'slopemap': slopemap,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'timing': timing,
        }
    
    def _print_timing(self, timing: Dict[str, float]):
        """打印耗时信息"""
        print("\n=== 视觉流程耗时 ===")
        for key, val in timing.items():
            print(f"  {key}: {val:.3f} ms")
    
    def _print_stats(self, points_cam, points_world, points_pelvis, points_cropped,
                     heightmap, slopemap):
        """打印点云和高程图统计信息"""
        if len(points_cam) > 0:
            print("\n=== 相机坐标系点云 ===")
            print(f"  X: [{points_cam[:,0].min():.3f}, {points_cam[:,0].max():.3f}]")
            print(f"  Y: [{points_cam[:,1].min():.3f}, {points_cam[:,1].max():.3f}]")
            print(f"  Z: [{points_cam[:,2].min():.3f}, {points_cam[:,2].max():.3f}]")
        if len(points_world) > 0:
            print("\n=== 世界坐标系点云 ===")
            print(f"  X: [{points_world[:,0].min():.3f}, {points_world[:,0].max():.3f}]")
            print(f"  Y: [{points_world[:,1].min():.3f}, {points_world[:,1].max():.3f}]")
            print(f"  Z: [{points_world[:,2].min():.3f}, {points_world[:,2].max():.3f}]")
        if len(points_pelvis) > 0:
            print("\n=== 骨盆坐标系点云 ===")
            print(f"  X: [{points_pelvis[:,0].min():.3f}, {points_pelvis[:,0].max():.3f}]")
            print(f"  Y: [{points_pelvis[:,1].min():.3f}, {points_pelvis[:,1].max():.3f}]")
            print(f"  Z: [{points_pelvis[:,2].min():.3f}, {points_pelvis[:,2].max():.3f}]")
        if len(points_cropped) > 0:
            print("\n=== 裁剪后点云 ===")
            print(f"  X: [{points_cropped[:,0].min():.3f}, {points_cropped[:,0].max():.3f}]")
            print(f"  Y: [{points_cropped[:,1].min():.3f}, {points_cropped[:,1].max():.3f}]")
            print(f"  Z: [{points_cropped[:,2].min():.3f}, {points_cropped[:,2].max():.3f}]")
        if heightmap.size > 1:
            print(f"\n=== 高程图尺寸: {heightmap.shape[0]} x {heightmap.shape[1]} ===")
            print(f"  min: {heightmap.min():.3f}, max: {heightmap.max():.3f}, mean: {heightmap.mean():.3f}")
        if slopemap.size > 1:
            print(f"=== 坡度图统计 ===")
            print(f"  min: {slopemap.min():.2f}°, max: {slopemap.max():.2f}°, mean: {slopemap.mean():.2f}°, std: {slopemap.std():.2f}°")
    
    # 可视化接口
    def visualize_pointcloud(self, subsample: int = 10000):
        """
        可视化当前裁剪后的点云(3D散点图)。
        :param subsample: 采样点数限制
        """
        pts = self.last_points_cropped
        if pts is None or len(pts) == 0:
            print("点云为空，无法可视化。请先运行 process()")
            return
        if len(pts) > subsample:
            idx = np.random.choice(len(pts), subsample, replace=False)
            pts = pts[idx]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=pts[:,2], cmap='jet', s=0.5)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Point Cloud in Pelvis Frame")
        ax.set_box_aspect([np.ptp(pts[:,0]), np.ptp(pts[:,1]), np.ptp(pts[:,2])])
        plt.show()
    
    def visualize_heightmap(self):
        """可视化当前高程图"""
        hm = self.last_heightmap
        if hm is None or hm.size <= 1:
            print("高程图无效，请先运行 process()")
            return
        x_edges = self.last_x_edges
        y_edges = self.last_y_edges
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(hm.T, origin='lower',
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                       cmap='terrain', aspect='auto')
        plt.colorbar(im, label='Height (m)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Elevation Map')
        plt.show()
    
    def visualize_slopemap(self, vmax: float = 30.0):
        """
        可视化坡度图。
        :param vmax: 颜色映射最大值
        """
        sm = self.last_slopemap
        if sm is None or sm.size <= 1:
            print("坡度图无效，请先运行 process()")
            return
        x_edges = self.last_x_edges
        y_edges = self.last_y_edges
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(sm.T, origin='lower',
                       extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                       cmap='hot', aspect='auto', vmin=0, vmax=vmax)
        plt.colorbar(im, label='Slope (deg)')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Slope Map')
        plt.show()
    
    def visualize_rgb(self):
        """显示最近一次渲染的 RGB 图像"""
        rgb = self.last_rgb
        if rgb is None:
            print("未找到 RGB 图像，请在 process() 时设置 render_rgb=True")
            return
        cv2.imshow("RGB Image", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def get_timing(self) -> Dict[str, float]:
        """返回最近一次 process() 的耗时字典"""
        return self.last_timing if hasattr(self, 'last_timing') else {}