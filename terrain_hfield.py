"""
生成随机起伏地形（高度场）并使用 MuJoCo 可视化。
使用 elevation 属性内嵌数据，无需外部二进制文件。
地形表面带有棋盘格纹理，便于观察起伏。
"""

import numpy as np
import mujoco
import mujoco.viewer
from scipy.signal import convolve2d  # 需要 scipy

print(f"MuJoCo version: {mujoco.__version__}")

# 地形参数
nrow, ncol = 80, 80               # 网格分辨率（不要太大，否则 XML 字符串过长）
x_len, y_len = 5.0, 5.0           # 地形物理尺寸（米）
z_min, z_max = 0.0, 0.2           # 高度范围（米）

# 1. 生成随机平滑高度场
hf = np.random.randn(nrow, ncol).astype(np.float32)
# 平滑（5x5 均值滤波）
kernel = np.ones((4,4), dtype=np.float32) / 16
hf = convolve2d(hf, kernel, mode='same')
# 归一化到 [0,1]
rel_h = (hf - hf.min()) / (hf.max() - hf.min())
# 2. 将高度转换为 elevation 整数格式 (0-65535)
elev_int = (rel_h * 65535).astype(np.uint32)
elev_str = ' '.join(elev_int.flatten('C').astype(str))

# 3. 构建 XML 字符串，包含棋盘纹理材质和高度场
sx = x_len / 2
sy = y_len / 2
sz_half = (z_max - z_min) / 2
z_mean = (z_min + z_max) / 2

xml = f'''<mujoco model="rough_terrain">
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