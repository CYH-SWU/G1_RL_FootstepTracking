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
    foot: int         # -1 表示左脚，1 表示右脚

class G1FootstepPlanner:
    """
    宇树G1步点规划器（骨盆坐标系Y轴为机器人朝向）
    - 支撑腿标识：-1=左，1=右
    - 所有坐标均在骨盆坐标系下
    - 机器人始终朝向骨盆坐标系Y轴正向
    - 步长离散化，步宽固定
    - 转向角范围可配置，离散化间隔可调
    - 成本函数：到目标的距离 + 步长偏差惩罚 + 朝向目标奖励
    - 若无可行候选步，则随机选择一个候选步（强制迈步）
    """

    def __init__(self,
                 step: float = 0.25,               # 标称步长（前向位移，米）
                 step_width: float = 0.23,         # 固定步宽（侧向偏移绝对值，米）
                 max_step_len: float = 0.35,       # 最大前向步长（米）
                 min_step_len: float = 0.10,       # 最小前向步长（米）
                 max_turn_deg: float = 20.0,       # 最大转向角（度）
                 max_step_height: float = 0.15,    # 最大抬腿高度差（米）
                 max_slope_deg: float = 15.0,      # 最大地形坡度（度）
                 clearance: float = 0.03,          # 脚底安全间隙（米）
                 w_dist: float = 1.0,              # 到目标距离权重
                 w_step: float = 0.5,              # 步长偏差惩罚权重
                 w_angle: float = 0.2,             # 朝向目标奖励权重
                 step_discretization: float = 0.05,   # 步长离散化间隔（米）
                 turn_discretization: float = 5.0     # 转向角离散化间隔（度）
                 ):
        """
        参数:
            step: 标称步长（前向位移，米）
            step_width: 固定步宽（米）
            max_step_len: 最大前向步长（米）
            min_step_len: 最小前向步长（米）
            max_turn_deg: 最大转向角（度）
            max_step_height: 最大抬腿高度差（米）
            max_slope_deg: 最大地形坡度（度）
            clearance: 脚底安全间隙（米）
            w_dist, w_step, w_angle: 成本函数权重
            step_discretization: 步长离散化间隔（米）
            turn_discretization: 转向角离散化间隔（度）
        """
        self.step = step
        self.step_width = step_width
        self.max_step_len = max_step_len
        self.min_step_len = min_step_len
        self.max_turn_rad = np.radians(max_turn_deg)
        self.max_step_height = max_step_height
        self.max_slope_deg = max_slope_deg
        self.clearance = clearance
        self.w_dist = w_dist
        self.w_step = w_step
        self.w_angle = w_angle
        self.step_discretization = step_discretization
        self.turn_discretization = np.radians(turn_discretization)

        # 生成离散化步长和转向角
        self.step_lengths = self._discretize_step_lengths()
        self.yaw_steps = self._discretize_yaw_steps()

        # 构建候选步 (dx_abs, dy, dyaw)，其中 dx_abs 为侧向偏移绝对值
        self.candidates = self._build_candidates()

        # 高程图数据（外部设置）
        self.height_map = None
        self.slope_map = None
        self.x_edges = None
        self.y_edges = None
        self.res = None

    # ------------------------------------------------------------------
    # 离散化参数
    # ------------------------------------------------------------------
    def _discretize_step_lengths(self) -> List[float]:
        """生成离散步长列表（前向位移）"""
        steps = np.arange(self.min_step_len, self.max_step_len + self.step_discretization, self.step_discretization)
        return [round(s, 3) for s in steps]

    def _discretize_yaw_steps(self) -> List[float]:
        """生成离散转向角列表（弧度）"""
        num = max(3, int(2 * self.max_turn_rad / self.turn_discretization) + 1)
        yaws = np.linspace(-self.max_turn_rad, self.max_turn_rad, num)
        return [round(y, 4) for y in yaws]

    # ------------------------------------------------------------------
    # 候选步生成
    # ------------------------------------------------------------------
    def _build_candidates(self) -> List[Tuple[float, float, float]]:
        """生成候选步 (dx_abs, dy, dyaw)"""
        candidates = []
        lateral_abs = [0.0, self.step_width]   # 允许无侧移或固定步宽

        for dy in self.step_lengths:
            for dyaw in self.yaw_steps:
                for dx_abs in lateral_abs:
                    candidates.append((dx_abs, dy, dyaw))

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
        """设置高程图和坡度图（骨盆坐标系）"""
        self.height_map = height_map
        self.slope_map = slope_map
        self.x_edges = x_edges
        self.y_edges = y_edges
        self.res = resolution

    def _get_terrain(self, x: float, y: float) -> Tuple[Optional[float], Optional[float]]:
        """双线性插值获取地形高度和坡度，返回 (高度, 坡度) 或 (None, None)"""
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

        # 高度插值
        h00, h10 = self.height_map[ix0, iy0], self.height_map[ix1, iy0]
        h01, h11 = self.height_map[ix0, iy1], self.height_map[ix1, iy1]
        if any(np.isnan([h00, h10, h01, h11])):
            return None, None
        h0 = h00 * (1 - wx) + h10 * wx
        h1 = h01 * (1 - wx) + h11 * wx
        height = h0 * (1 - wy) + h1 * wy

        # 坡度插值
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
                           current_stance: int,                           # -1: 左脚, 1: 右脚
                           target_pos: Tuple[float, float]                # (x, y) 目标终点（骨盆坐标系）
                           ) -> Tuple[Optional[Footstep], int]:
        """
        返回: (落脚点对象, 下一步支撑脚标识)
        假设机器人始终朝向骨盆坐标系 Y 轴正向（即 robot_yaw = 0）。
        """
        if self.height_map is None:
            print("错误：未设置高程图")
            return None, current_stance

        best_cost = float('inf')
        best_step = None
        next_stance = -current_stance   # 左右交替

        cx, cy, cz = current_foot_pos

        # 计算从当前脚指向目标的方向角（相对于 Y 轴正向）
        target_dx = target_pos[0] - cx
        target_dy = target_pos[1] - cy
        if (target_dx**2 + target_dy**2) > 1e-6:
            target_dir = np.arctan2(target_dx, target_dy)   # 相对于 Y 轴的角度
        else:
            target_dir = 0.0

        for dx_abs, dy, dyaw in self.candidates:
            # 侧向偏移符号由支撑脚决定：左脚（-1）-> 向右（正X），右脚（1）-> 向左（负X）
            sign_x = -current_stance
            dx = sign_x * dx_abs
            # 候选落脚点绝对坐标（骨盆坐标系）
            nx = cx + dx
            ny = cy + dy

            # 边界检查
            if nx < self.x_edges[0] or nx > self.x_edges[-1] or ny < self.y_edges[0] or ny > self.y_edges[-1]:
                continue

            # 地形查询
            terrain_h, slope = self._get_terrain(nx, ny)
            if terrain_h is None or slope is None:
                continue
            if slope > self.max_slope_deg:
                continue

            # 高度差约束
            dz = terrain_h - cz
            if abs(dz) > self.max_step_height:
                continue

            nz = terrain_h + self.clearance
            foot_yaw = dyaw   # 机器人朝向为0，落脚点朝向即转向变化

            # 成本函数
            dist = np.hypot(nx - target_pos[0], ny - target_pos[1])
            step_penalty = (dy - self.step) ** 2
            angle_penalty = target_dir ** 2   # 机器人朝向为0，直接惩罚目标方向夹角
            cost = self.w_dist * dist + self.w_step * step_penalty + self.w_angle * angle_penalty

            if cost < best_cost:
                best_cost = cost
                best_step = Footstep(x=nx, y=ny, z=nz, yaw=foot_yaw, foot=next_stance)

        # 如果没有可行候选步，则随机选择一个候选步（强制迈步）
        if best_step is None:
            if len(self.candidates) == 0:
                # 极端情况：没有候选步，回退到原地落脚
                terrain_h, _ = self._get_terrain(cx, cy)
                if terrain_h is None:
                    terrain_h = cz
                fallback_z = terrain_h + self.clearance
                best_step = Footstep(x=cx, y=cy, z=fallback_z, yaw=0.0, foot=next_stance)
                print("警告：无候选步，原地落脚")
            else:
                # 随机选择一个候选步
                rand_idx = np.random.randint(len(self.candidates))
                dx_abs, dy, dyaw = self.candidates[rand_idx]
                sign_x = -current_stance
                dx = sign_x * dx_abs
                nx = cx + dx
                ny = cy + dy
                # 边界修正（如果超出边界则 clamp 到边界内）
                nx = np.clip(nx, self.x_edges[0], self.x_edges[-1])
                ny = np.clip(ny, self.y_edges[0], self.y_edges[-1])
                # 查询地形高度（若失败则使用当前脚高度）
                terrain_h, _ = self._get_terrain(nx, ny)
                if terrain_h is None:
                    terrain_h = cz
                nz = terrain_h + self.clearance
                foot_yaw = dyaw
                best_step = Footstep(x=nx, y=ny, z=nz, yaw=foot_yaw, foot=next_stance)
                print("警告：无可行候选步，随机选择了一个候选步")

        return best_step, next_stance