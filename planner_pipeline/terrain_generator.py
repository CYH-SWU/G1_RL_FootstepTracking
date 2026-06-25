import os
import numpy as np
from scipy.signal import convolve2d
import mujoco
from typing import Tuple, Optional


class TerrainGenerator:
    """
    地形生成器：根据指定模式动态生成 MuJoCo 地形 XML,并加载为可仿真的模型。

    支持四种地形模式：
        - flat   : 平地，用于基础行走训练
        - rough  : 随机起伏面，考核连续地形感知与姿态微调
        - slope  : 线性斜坡，考核上下坡质心控制
        - steps  : 连续台阶，考核离散落足点定位与抬腿高度

    所有地形均通过动态构建 XML 字符串加载。
    """

    def __init__(self, robot_xml_path: str, mesh_dir: Optional[str] = None):
        """
        初始化地形生成器。

        :param robot_xml_path: G1 机器人主 XML 文件路径
        :param mesh_dir: STL 网格文件根目录。
        """
        # 转为绝对路径
        self.robot_xml_path = os.path.abspath(robot_xml_path)

        if mesh_dir is None:
            self.mesh_dir = os.path.dirname(self.robot_xml_path)
        else:
            self.mesh_dir = os.path.abspath(mesh_dir)

    def generate(self, mode: str, difficulty: float = 1.0,
                 goal_pos: Tuple[float, float] = (7.5, 0.0)) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """
        生成指定地形并加载为 MuJoCo 模型。

        流程：
            1. 根据 mode 调用对应的地形构建方法，获得完整 XML 字符串
            2. 使用 mujoco.MjModel.from_xml_string 从内存加载
            3. 创建对应的 MjData 并返回

        :param mode: 地形类型，支持 'flat' | 'rough' | 'slope' | 'steps'
        :param difficulty: 难度因子，取值范围 [0, 1]
        :param goal_pos: 终点在世界坐标系下的 (x, y) 坐标，用于放置红色标记柱
        :return: (model, data) 元组
        :raises ValueError: 传入未知 mode 时抛出
        """
        # 防止难度因子越界
        difficulty = np.clip(difficulty, 0.0, 1.0)
        if difficulty < 1e-6:
            mode = "flat"

        # 根据模式分发构建
        if mode == "flat":
            xml_str = self._build_flat_terrain(goal_pos)
        elif mode == "rough":
            xml_str = self._build_rough_terrain(difficulty, goal_pos)
        elif mode == "slope":
            xml_str = self._build_slope_terrain(difficulty, goal_pos)
        elif mode == "steps":
            xml_str = self._build_steps_terrain(difficulty, goal_pos)
        else:
            raise ValueError(f"未知地形模式: {mode}，可选 'flat', 'rough', 'slope', 'steps'")

        # 从 XML 字符串直接构建模型
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        return model, data


    def _build_flat_terrain(self, goal_pos: Tuple[float, float]) -> str:
        """
        构建平地形 XML。
        """
        return self._make_xml_template(
            terrain_geom='<geom type="plane" size="20 20 0.1" pos="0 0 0" material="groundplane"/>',
            goal_pos=goal_pos,
            extra_assets=''  
        )

    def _build_rough_terrain(self, difficulty: float, goal_pos: Tuple[float, float]) -> str:
        """
        构建随机起伏面地形（高度场）。
        """
        k = 4
        nrow, ncol = 30 * k, 30 * k          # 网格分辨率
        x_len, y_len = 5.0 * k, 5.0 * k      # 地形物理尺寸
        z_max = 0.6 * difficulty             # 最大起伏高度

        # 生成随机高度场
        hf = np.random.randn(nrow, ncol).astype(np.float32)
        # 5x5 均值滤波平滑，消除高频毛刺
        kernel = np.ones((5, 5), dtype=np.float32) / 25
        hf = convolve2d(hf, kernel, mode='same')
        # 归一化到 [0, z_max]
        hf = (hf - hf.min()) / (hf.max() - hf.min() + 1e-8) * z_max

        # 转换为 MuJoCo elevation 字符串(0~65535 整数序列)
        rel_h = hf / z_max if z_max > 0 else np.zeros_like(hf)
        elev_int = (rel_h * 65535).astype(np.uint32)
        elev_str = ' '.join(elev_int.flatten('C').astype(str))

        # 高度场参数
        sx = x_len / 2
        sy = y_len / 2
        sz_half = z_max / 2
        z_mean = z_max / 2

        hfield_def = f'''
            <hfield name="ground" size="{sx} {sy} {sz_half} {z_mean}"
                    nrow="{nrow}" ncol="{ncol}"
                    elevation="{elev_str}"/>
        '''
        terrain_geom = f'<geom type="hfield" hfield="ground" material="groundplane" rgba="0.6 0.8 1.0 1"/>'
        return self._make_xml_template(
            terrain_geom=terrain_geom,
            goal_pos=goal_pos,
            extra_assets=hfield_def
        )

    def _build_slope_terrain(self, difficulty: float, goal_pos: Tuple[float, float]) -> str:
        """
        构建斜坡地形(沿世界坐标系X轴方向线性升高的高度场)。
        """
        k = 4
        nrow, ncol = 100 * k, 100 * k        
        x_len, y_len = 5.0 * k, 5.0 * k      
        z_min, z_max = 0.0, 9.0 * difficulty  

        x = np.linspace(-x_len/2, x_len/2, nrow)
        y = np.linspace(-y_len/2, y_len/2, ncol)
        Y, X = np.meshgrid(x, y, indexing='ij')
        
        t = (X - X.min()) / (X.max() - X.min() + 1e-8)
        hf = z_min + t * (z_max - z_min)
        hf = hf.astype(np.float32)

        # 编码为 MuJoCo elevation 格式
        rel_h = (hf - z_min) / (z_max - z_min + 1e-8)
        elev_int = (rel_h * 65535).astype(np.uint32)
        elev_str = ' '.join(elev_int.flatten('C').astype(str))

        sx = x_len / 2
        sy = y_len / 2
        sz_half = (z_max - z_min) / 2
        z_mean = (z_min + z_max) / 2

        hfield_def = f'''
            <hfield name="ground" size="{sx} {sy} {sz_half} {z_mean}"
                    nrow="{nrow}" ncol="{ncol}"
                    elevation="{elev_str}"/>
        '''
        terrain_geom = f'<geom type="hfield" hfield="ground" material="groundplane" rgba="0.6 0.8 1.0 1"/>'
        return self._make_xml_template(
            terrain_geom=terrain_geom,
            goal_pos=goal_pos,
            extra_assets=hfield_def
        )

    def _build_steps_terrain(self, difficulty: float, goal_pos: Tuple[float, float]) -> str:
        """
        构建台阶地形。
        """
        step_height = 0.1 * difficulty       # 每级台阶高度
        step_width = 0.3                     # 台阶宽度
        step_depth = 2.0                     # 台阶长度
        step_gap = 0.3                       # 台阶中心间距
        num_steps = 25                       # 总级数

        steps_geoms = []
        for i in range(1, num_steps + 1):
            z_top = i * step_height          
            shift_x = 0.05                   # 整体水平偏移
            x_center = i * step_gap + shift_x  # 台阶中心 X 坐标
            step_thickness = 0.1             
            z_pos = z_top - step_thickness / 2  
            steps_geoms.append(
                f'<geom type="box" size="{step_width/2} {step_depth/2} {step_thickness/2}" '
                f'pos="{x_center} 0 {z_pos}" material="step_mat" rgba="0.8 0.6 0.4 1"/>'
            )
        steps_str = '\n'.join(steps_geoms)

        # 平地基底 + 台阶序列
        terrain_geom = f'''
            <geom type="plane" size="20 20 0.1" pos="0 0 0" material="groundplane"/>
            {steps_str}
        '''
        extra_assets = '''
            <material name="step_mat" rgba="0.8 0.6 0.4 1" reflectance="0.3"/>
        '''
        return self._make_xml_template(
            terrain_geom=terrain_geom,
            goal_pos=goal_pos,
            extra_assets=extra_assets
        )

    

    def _make_xml_template(self, terrain_geom: str, goal_pos: Tuple[float, float],
                           extra_assets: str) -> str:
        """
        XML构建通用模板
        构建完整的 MuJoCo XML 字符串。

        该模板统一封装以下通用元素：
            - 引入机器人模型
            - 设置网格文件搜索路径
            - 定义棋盘纹理
            - 添加环境光源
            - 嵌入地形几何体
            - 在终点位置添加红色标识柱

        :param terrain_geom: 地形 <geom> 定义字符串
        :param goal_pos: 终点 (x, y) 世界坐标
        :param extra_assets: 额外的 <asset> 内容
        :return: 完整的 MuJoCo XML 字符串
        """
        goal_x, goal_y = goal_pos
        goal_marker = f'<geom type="cylinder" pos="{goal_x} {goal_y} 7.0" size="0.08 7" rgba="1 0 0 0.9"/>'

        # 统一路径分隔符为正斜杠，确保跨平台（Windows/Linux）兼容
        robot_abs_path = self.robot_xml_path.replace('\\', '/')

        # 棋盘纹理定义
        tex_asset = '''
            <texture name="ground_tex" type="2d" builtin="checker"
                     rgb1="0.2 0.3 0.4" rgb2="0.6 0.7 0.8"
                     width="300" height="300" mark="edge" random="0.01"/>
            <material name="groundplane" texture="ground_tex" texrepeat="4 4"
                      texuniform="true" reflectance="0.2"/>
        '''

        # 组装最终 XML
        xml_str = f'''<mujoco model="terrain_with_g1">
            <!-- 引入外部机器人模型 -->
            <include file="{robot_abs_path}"/>

            <!-- 指定网格文件搜索根目录 -->
            <compiler meshdir="{self.mesh_dir}"/>

            <asset>
                {tex_asset}
                {extra_assets}
            </asset>

            <worldbody>
                <!-- 平行光，从上方垂直照射 -->
                <light pos="0 0 3" dir="0 0 -1" directional="true"/>
                <!-- 地形主体 -->
                {terrain_geom}
                <!-- 终点标识柱 -->
                {goal_marker}
            </worldbody>
        </mujoco>'''
        return xml_str