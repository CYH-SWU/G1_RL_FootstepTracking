import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class Footstep:
    """落脚点信息（骨盆坐标系，Y向前，X向右，Z向上）"""
    x: float          # 侧向位置 (m)
    y: float          # 前向位置 (m)
    z: float          # 地形高度 + 安全间隙 (m)
    yaw: float        # 落脚点朝向 (rad)
    foot: str         # 'left' 或 'right'

class G1FootstepPlanner:
    """
    宇树G1步点规划器（支持平滑转向）
    步长 = 前进方向位移（相对机器人自身），步宽 = 0.23m（侧向偏移绝对值）
    转向角变化范围限制在 max_turn_rad 以内，默认 ±0.2 rad
    """

    def __init__(self, step_len: float = 0.25, step_variation: float = 0.05,
                 step_width: float = 0.23, max_turn_rad: float = 0.2):
        """
        参数:
            step_len: 标称步长（前进方向位移，米）
            step_variation: 步长变化范围（米）
            step_width: 固定步宽（侧向偏移绝对值，米）
            max_turn_rad: 最大转向角变化（弧度），默认 0.2 ≈ 11.5°
        """
        self.step_len = step_len
        self.step_variation = step_variation
        self.step_width = step_width
        self.max_turn_rad = max_turn_rad          # 限制转向幅度

        # 运动学约束
        self.max_step_height = 0.12               # 最大抬腿高度差 (米)
        self.max_slope_deg = 15.0                 # 最大地形坡度 (度)
        self.clearance = 0.03                     # 脚底离地安全间隙 (米)

        # 成本函数权重
        self.w_dist = 1.0                         # 到终点的距离权重
        self.w_turn = 0.3                         # 转向惩罚权重（鼓励小转向）

        # 预生成候选步 (dx_local, dy_local, dyaw) 相对于机器人自身坐标系
        # dx_local: 侧向偏移（绝对值 = step_width）
        # dy_local: 前向位移（步长）
        # dyaw: 转向角变化（弧度）
        self.candidates = self._build_candidates()

        # 高程图数据（由外部设置）
        self.height_map = None
        self.slope_map = None
        self.x_edges = None
        self.y_edges = None
        self.res = None

    # ------------------------------------------------------------------
    # 候选步生成
    # ------------------------------------------------------------------
    def _build_candidates(self) -> List[Tuple[float, float, float]]:
        """生成候选步 (dx_local, dy_local, dyaw)"""
        candidates = []
        # 步长列表（前进位移）
        step_lengths = [self.step_len - self.step_variation,
                        self.step_len,
                        self.step_len + self.step_variation]
        # 转向角列表（限制在 [-max_turn_rad, max_turn_rad] 内，取 3 个离散值）
        yaw_steps = np.linspace(-self.max_turn_rad, self.max_turn_rad, 3)

        for dy in step_lengths:
            for dyaw in yaw_steps:
                # 侧向偏移绝对值固定为 step_width，符号后续由支撑脚决定
                candidates.append((self.step_width, dy, dyaw))

        # 原地转向候选（无前进，无侧移）
        for dyaw in yaw_steps:
            if abs(dyaw) > 1e-3:
                candidates.append((0.0, 0.0, dyaw))

        # 去重
        unique = []
        for c in candidates:
            if not any(np.allclose(c, u, atol=1e-5) for u in unique):
                unique.append(c)
        return unique

    # ------------------------------------------------------------------
    # 高程图接口
    # ------------------------------------------------------------------
    def set_heightmap(self, height_map: np.ndarray, slope_map: np.ndarray,
                      x_edges: np.ndarray, y_edges: np.ndarray, resolution: float):
        self.height_map = height_map
        self.slope_map = slope_map
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.res = resolution

    def _get_terrain(self, x: float, y: float) -> Tuple[Optional[float], Optional[float]]:
        """双线性插值获取地形高度和坡度"""
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
        s0 = s00 * (1 - wx) + s10 * wx
        s1 = s01 * (1 - wx) + s11 * wx
        slope = s0 * (1 - wy) + s1 * wy
        return height, slope

    # ------------------------------------------------------------------
    # 步点规划主接口
    # ------------------------------------------------------------------
    def plan_next_footstep(self,
                           current_foot_pos: Tuple[float, float, float],   # (x, y, z) 当前支撑脚位置（骨盆坐标系）
                           current_stance: str,                           # 'left' 或 'right'
                           robot_yaw: float,                              # 机器人当前偏航角（弧度）
                           target_pos: Tuple[float, float]                # (x, y) 目标终点（骨盆坐标系）
                           ) -> Tuple[Optional[Footstep], str]:
        """
        返回: (落脚点对象, 下一步支撑脚)
        """
        if self.height_map is None:
            print("错误：未设置高程图")
            return None, current_stance

        best_cost = float('inf')
        best_step = None
        next_stance = 'right' if current_stance == 'left' else 'left'

        cx, cy, cz = current_foot_pos

        # 根据当前脚确定侧向偏移的局部符号：左脚 -> 向右（正局部X），右脚 -> 向左（负局部X）
        sign_x_local = 1 if current_stance == 'left' else -1

        for dx_local_abs, dy_local, dyaw in self.candidates:
            # 局部坐标系下的位移（相对于机器人当前朝向）
            dx_local = sign_x_local * dx_local_abs
            # 旋转到世界坐标系（骨盆坐标系）
            cos_yaw = np.cos(robot_yaw)
            sin_yaw = np.sin(robot_yaw)
            dx_world = dx_local * cos_yaw - dy_local * sin_yaw
            dy_world = dx_local * sin_yaw + dy_local * cos_yaw

            nx = cx + dx_world
            ny = cy + dy_world

            # 边界检查
            if nx < self.x_edges[0] or nx > self.x_edges[-1] or ny < self.y_edges[0] or ny > self.y_edges[-1]:
                continue

            terrain_h, slope = self._get_terrain(nx, ny)
            if terrain_h is None or slope is None:
                continue
            if slope > self.max_slope_deg:
                continue

            dz = terrain_h - cz
            if abs(dz) > self.max_step_height:
                continue

            nz = terrain_h + self.clearance
            # 落脚点朝向 = 机器人当前偏航 + 转向角变化
            foot_yaw = robot_yaw + dyaw

            # 成本函数：到终点的距离 + 转向惩罚
            dist = np.hypot(nx - target_pos[0], ny - target_pos[1])
            turn_penalty = self.w_turn * abs(dyaw)
            cost = self.w_dist * dist + turn_penalty

            if cost < best_cost:
                best_cost = cost
                best_step = Footstep(x=nx, y=ny, z=nz, yaw=foot_yaw, foot=next_stance)

        # 回退：原地落脚（保持当前朝向）
        if best_step is None:
            terrain_h, _ = self._get_terrain(cx, cy)
            if terrain_h is None:
                terrain_h = cz
            fallback_z = terrain_h + self.clearance
            best_step = Footstep(x=cx, y=cy, z=fallback_z, yaw=robot_yaw, foot=next_stance)

        return best_step, next_stance