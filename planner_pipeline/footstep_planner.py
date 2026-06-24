import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class Footstep:
    x: float
    y: float
    z: float
    yaw: float        # 落脚点朝向（弧度，相对于骨盆X轴）
    foot: int         # -1 左，1 右

class G1FootstepPlanner:
    """
    步点规划器
    - 支撑腿标识：-1=左，1=右
    - 候选步生成：先沿Y偏移固定步宽，再沿X延伸步长，整体绕支撑腿旋转
    - 成本函数：步长偏差 + 角度偏差 + 坡度惩罚
    - 若无可行候选，则退回原地迈步或随机选一个
    """

    def __init__(self,
                 step: float = 0.3,                # 标称步长
                 step_width: float = 0.237,        # 固定步宽
                 max_step_len: float = 0.40,       # 最大前向步长
                 min_step_len: float = 0.15,       # 最小前向步长
                 max_turn_deg: float = 8.0,       # 最大转向角
                 max_step_height: float = 0.20,    # 最大抬腿高度差
                 max_slope_deg: float = 20.0,      # 最大地形坡度
                 clearance: float = 0.03,          # 脚底安全间隙
                 w_step: float = 1.0,              # 步长偏差惩罚权重
                 w_angle: float = 0.7,             # 角度偏差惩罚权重
                 w_slope: float = 0.5,             # 坡度惩罚权重
                 step_discretization: float = 0.05,   # 步长离散化间隔
                 turn_discretization: float = 1.0     # 转向角离散化间隔
                 ):
        self.step = step
        self.step_width = step_width
        self.max_step_len = max_step_len
        self.min_step_len = min_step_len
        self.max_turn_rad = np.radians(max_turn_deg)
        self.max_step_height = max_step_height
        self.max_slope_deg = max_slope_deg
        self.clearance = clearance
        self.w_step = w_step
        self.w_angle = w_angle
        self.w_slope = w_slope
        self.step_discretization = step_discretization
        self.turn_discretization = np.radians(turn_discretization)

        # 生成离散化步长和偏航角
        self.step_lengths = self._discretize_step_lengths()
        self.yaw_steps = self._discretize_yaw_steps()

        # 预生成候选 (dx, theta) 组合
        self.candidates = self._build_candidates()

        # 高程图数据
        self.height_map = None
        self.slope_map = None
        self.x_edges = None
        self.y_edges = None
        self.res = None

    
    def _discretize_step_lengths(self) -> List[float]:
        steps = np.arange(self.min_step_len, self.max_step_len + self.step_discretization, self.step_discretization)
        return [round(s, 3) for s in steps]

    def _discretize_yaw_steps(self) -> List[float]:
        num = max(3, int(2 * self.max_turn_rad / self.turn_discretization) + 1)
        yaws = np.linspace(-self.max_turn_rad, self.max_turn_rad, num)
        return [round(y, 4) for y in yaws]

    def _build_candidates(self) -> List[Tuple[float, float]]:
        candidates = []
        for dx in self.step_lengths:
            for theta in self.yaw_steps:
                candidates.append((dx, theta))
        
        return candidates

    def set_heightmap(self, height_map: np.ndarray, slope_map: np.ndarray,
                      x_edges: np.ndarray, y_edges: np.ndarray, resolution: float):
        self.height_map = height_map
        self.slope_map = slope_map
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.res = resolution

    def _get_terrain(self, x: float, y: float) -> Tuple[Optional[float], Optional[float]]:
        if self.height_map is None:
            return None, None
        ix = (x - self.x_edges[0]) / self.res
        iy = (y - self.y_edges[0]) / self.res
        nx, ny = self.height_map.shape
        if ix < 0 or ix >= nx - 1 or iy < 0 or iy >= ny - 1:
            return None, None
        ix0, ix1 = int(ix), int(ix) + 1
        iy0, iy1 = int(iy), int(iy) + 1
        wx = ix - ix0
        wy = iy - iy0

        h00, h10 = self.height_map[ix0, iy0], self.height_map[ix1, iy0]
        h01, h11 = self.height_map[ix0, iy1], self.height_map[ix1, iy1]
        if any(np.isnan([h00, h10, h01, h11])):
            return None, None
        h0 = h00 * (1 - wx) + h10 * wx
        h1 = h01 * (1 - wx) + h11 * wx
        height = h0 * (1 - wy) + h1 * wy

        s00, s10 = self.slope_map[ix0, iy0], self.slope_map[ix1, iy0]
        s01, s11 = self.slope_map[ix0, iy1], self.slope_map[ix1, iy1]
        if any(np.isnan([s00, s10, s01, s11])):
            return height, None
        s0 = s00 * (1 - wx) + s10 * wx
        s1 = s01 * (1 - wx) + s11 * wx
        slope = s0 * (1 - wy) + s1 * wy
        return height, slope

    def plan_next_footstep(self,
                           current_foot_pos: Tuple[float, float, float],
                           current_stance: int,           # -1 左, 1 右
                           target_pos: Tuple[float, float]
                           ) -> Tuple[Optional[Footstep], int]:
        if self.height_map is None:
            print("错误：未设置高程图")
            return None, current_stance

        cx, cy, cz = current_foot_pos
        next_stance = -current_stance

        # 目标方向角
        target_dx = target_pos[0]
        target_dy = target_pos[1]
        if (target_dx**2 + target_dy**2) > 1e-6:
            target_dir = np.arctan2(target_dy, target_dx)
        else:
            target_dir = 0.0

        # 侧向偏移量
        dy_offset = -current_stance * self.step_width

        best_cost = float('inf')
        best_step = None

        for dx, theta in self.candidates:
            nx = cx + dx * np.cos(theta) - dy_offset * np.sin(theta)
            ny = cy + dx * np.sin(theta) + dy_offset * np.cos(theta)

            # 边界检查
            if nx < self.x_edges[0] or nx > self.x_edges[-1] or ny < self.y_edges[0] or ny > self.y_edges[-1]:
                continue

            # 地形信息
            terrain_h, slope = self._get_terrain(nx, ny)
            if terrain_h is None or slope is None:
                continue
            if slope > self.max_slope_deg:
                continue

            # 高度差检查
            dz = terrain_h - cz
            if abs(dz) > self.max_step_height:
                continue

            # 落脚点Z值
            nz = terrain_h + self.clearance

            # 成本函数 
            # 步长成本
            max_step_dev = max(abs(self.step - self.min_step_len), abs(self.max_step_len - self.step))
            step_cost = ((dx - self.step) / max_step_dev) ** 2

            # 角度成本
            angle_error = target_dir - theta
            angle_error = np.clip(angle_error, -self.max_turn_rad, self.max_turn_rad)
            angle_cost = (angle_error / self.max_turn_rad) ** 2

            # 坡度成本
            slope_cost = (slope / self.max_slope_deg) ** 2 if self.max_slope_deg > 0 else 0.0

            # 总成本
            cost = self.w_step * step_cost + self.w_angle * angle_cost + self.w_slope * slope_cost

            if cost < best_cost:
                best_cost = cost
                best_step = Footstep(x=nx, y=ny, z=nz, yaw=theta, foot=next_stance)

        if best_step is None:
            # 尝试原地迈步
            fallback_x = cx + self.step * 0.5  # 小步
            fallback_y = cy + dy_offset
            # 查地形
            terrain_h, _ = self._get_terrain(fallback_x, fallback_y)
            if terrain_h is None:
                terrain_h = cz
            nz = terrain_h + self.clearance
            best_step = Footstep(x=fallback_x, y=fallback_y, z=nz, yaw=0.0, foot=next_stance)
            print("警告：无可行候选步，采用原地迈步回退。")

        return best_step, next_stance