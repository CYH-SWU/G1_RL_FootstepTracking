"""
生成斜面（斜坡）地形（高度场）并使用 MuJoCo 可视化。
高度沿 X 方向线性增加，从 z_min 到 z_max。
地形表面带有棋盘格纹理，便于观察坡度。
"""

import numpy as np
import mujoco
import mujoco.viewer

print(f"MuJoCo version: {mujoco.__version__}")

# 地形参数
nrow, ncol = 100, 100        # 网格分辨率 (X方向行数, Y方向列数)
x_len, y_len = 5.0, 5.0      # 地形物理尺寸（米）
z_min, z_max = 0.0, 0.8      # 斜面最低点和最高点高度（米）→ 坡度更明显

# 1. 生成斜面高度场
# 创建网格坐标：X 从 -x_len/2 到 x_len/2，Y 从 -y_len/2 到 y_len/2
x = np.linspace(-x_len/2, x_len/2, nrow)
y = np.linspace(-y_len/2, y_len/2, ncol)
X, Y = np.meshgrid(x, y, indexing='ij')  # shape (nrow, ncol)

# 高度随 X 线性变化：在 X 最小处为 z_min，X 最大处为 z_max
t = (X - X.min()) / (X.max() - X.min())   # 0 到 1
hf = z_min + t * (z_max - z_min)          # 线性斜坡
hf = hf.astype(np.float32)

# 2. 将高度转换为 elevation 整数格式 (0-65535)
rel_h = (hf - z_min) / (z_max - z_min)   # 0..1
elev_int = (rel_h * 65535).astype(np.uint32)
# 使用 C-order (行主序) 展平
elev_str = ' '.join(elev_int.flatten('C').astype(str))

# 3. 构建 XML 字符串，包含棋盘纹理材质和高度场
sx = x_len / 2
sy = y_len / 2
sz_half = (z_max - z_min) / 2
z_mean = (z_min + z_max) / 2

xml = f'''<mujoco model="ramp_terrain">
  <asset>
    <!-- 棋盘纹理 -->
    <texture name="ground_tex" type="2d" builtin="checker" 
             rgb1="0.2 0.3 0.4" rgb2="0.6 0.7 0.8" 
             width="300" height="300" mark="edge" random="0.01"/>
    <material name="groundplane" texture="ground_tex" texrepeat="2 2" 
              texuniform="true" reflectance="0.2"/>
    
    <!-- 高度场 -->
    <hfield name="ground" size="{sx} {sy} {sz_half} {z_mean}" 
            nrow="{nrow}" ncol="{ncol}" 
            elevation="{elev_str}"/>
  </asset>
  
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <!-- 高度场几何体，应用带纹理的材质 -->
    <geom type="hfield" hfield="ground" material="groundplane" rgba="0.6 0.8 1.0 1"/>
    
    <!-- 添加一个参考小球，置于斜坡最高点上方 -->
    <geom type="sphere" pos="{x_len/2 - 0.2} 0 {z_max + 0.1}" size="0.05" rgba="1 0 0 1"/>
  </worldbody>
</mujoco>'''

print("XML 长度:", len(xml), "字符")
try:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    print("模型加载成功！启动可视化窗口...")
    mujoco.viewer.launch(model, data)
except Exception as e:
    print("加载失败:", e)