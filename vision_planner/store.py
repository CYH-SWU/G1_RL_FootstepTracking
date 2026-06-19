# ==================== 1. 生成随机起伏地形（高度场） ====================
print("生成随机起伏地形...")
k = 4
nrow, ncol = 60*k, 60*k               # 网格分辨率
x_len, y_len = 5.0*k, 5.0*k           # 地形物理尺寸（米）
z_min, z_max = 0.0, 0.3              # 高度范围（米）

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

# ==================== 2. 构建完整 XML（起伏地形 + 机器人） ====================
asset_dir = PROJECT_ROOT / "asset"
scene_xml_path = asset_dir / "scene_with_robot.xml"
robot_rel_path = os.path.relpath(processed_xml, start=asset_dir)

xml_content = f'''<mujoco model="rough_terrain_with_g1">
  <!-- 1. 引入机器人模型 -->
  <include file="{robot_rel_path}"/>
  
  <!-- 2. 覆盖资源路径，使STL从 robot/assets 加载 -->
  <compiler meshdir="../robot/assets"/>
  
  <!-- 3. 定义场景的纹理和高度场 -->
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
    <!-- 起伏地面（高度场） -->
    <geom type="hfield" hfield="ground" material="groundplane" rgba="0.6 0.8 1.0 1"/>
    <geom type="sphere" pos="7.5 0 2" size="0.05" rgba="1 0 0 1"/>
  </worldbody>
</mujoco>'''

with open(scene_xml_path, 'w') as f:
    f.write(xml_content)
print(f"场景 XML 已保存至: {scene_xml_path}")





# 台阶定义（使用材质以应用纹理）
steps = '''
    <!-- 台阶1: 上表面 Z=0.10，中心 X=0.75 -->
    <geom type="box" size="0.25 1.0 0.05" pos="0.75 0 0.05" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶2: 上表面 Z=0.20，中心 X=1.25 -->
    <geom type="box" size="0.25 1.0 0.05" pos="1.25 0 0.15" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶3: 上表面 Z=0.30，中心 X=1.75 -->
    <geom type="box" size="0.25 1.0 0.05" pos="1.75 0 0.25" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶4: 上表面 Z=0.40，中心 X=2.25 -->
    <geom type="box" size="0.25 1.0 0.05" pos="2.25 0 0.35" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶5: 上表面 Z=0.50，中心 X=2.75 -->
    <geom type="box" size="0.25 1.0 0.05" pos="2.75 0 0.45" material="step_mat" rgba="0.8 0.6 0.4 1"/>
    <!-- 台阶6: 上表面 Z=0.60? 但原定义是0.55，我们保持0.55 -->
    <geom type="box" size="0.25 1.0 0.05" pos="3.25 0 0.55" material="step_mat" rgba="0.8 0.6 0.4 1"/>
'''

xml_content = f'''<mujoco model="g1_with_steps">
  <!-- 1. 引入机器人模型 -->
  <include file="{robot_rel_path}"/>
  
  <!-- 2. 覆盖资源路径，使STL从 robot/assets 加载 -->
  <compiler meshdir="../robot/assets"/>
  
  <!-- 3. 定义场景的纹理和材质 -->
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
    <!-- 地面（带棋盘纹理） -->
    <geom type="plane" size="20 20 0.1" pos="0 0 0" material="groundplane"/>
    <!-- 台阶 -->
    {steps}
    <geom type="sphere" pos="7.5 0 2" size="0.05" rgba="1 0 0 1"/>
  </worldbody>
</mujoco>'''

with open(scene_xml_path, 'w') as f:
    f.write(xml_content)
print(f"场景 XML 已保存至: {scene_xml_path}")






# ==================== 1. 生成斜坡地形（高度场） ====================
print("生成斜坡地形...")
k = 4
nrow, ncol = 100*k, 100*k        # 网格分辨率
x_len, y_len = 5.0*k, 5.0*k      # 地形物理尺寸（米）
z_min, z_max = 0.0, 1.6          # 斜坡最低/最高高度（米）

# 创建网格坐标（X 方向为斜坡方向）
x = np.linspace(-x_len/2, x_len/2, nrow)
y = np.linspace(-y_len/2, y_len/2, ncol)
X, Y = np.meshgrid(x, y, indexing='ij')

# 高度沿 X 线性变化：从 z_min 到 z_max
t = (X - X.min()) / (X.max() - X.min())
hf = z_min + t * (z_max - z_min)
hf = hf.astype(np.float32)

# 转换为 elevation 整数格式 (0-65535)
rel_h = (hf - z_min) / (z_max - z_min)
elev_int = (rel_h * 65535).astype(np.uint32)
elev_str = ' '.join(elev_int.flatten('C').astype(str))

sx = x_len / 2
sy = y_len / 2
sz_half = (z_max - z_min) / 2
z_mean = (z_min + z_max) / 2

print(f"sx={sx}, sy={sy}, sz_half={sz_half}, z_mean={z_mean}")

# ==================== 2. 构建完整 XML（斜坡地形 + 机器人） ====================
asset_dir = PROJECT_ROOT / "asset"
scene_xml_path = asset_dir / "scene_with_robot.xml"
robot_rel_path = os.path.relpath(processed_xml, start=asset_dir)

xml_content = f'''<mujoco model="ramp_terrain_with_g1">
  <!-- 1. 引入机器人模型 -->
  <include file="{robot_rel_path}"/>
  
  <!-- 2. 覆盖资源路径，使STL从 robot/assets 加载 -->
  <compiler meshdir="../robot/assets"/>
  
  <!-- 3. 定义场景的纹理和高度场 -->
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
    <!-- 斜坡地面（高度场） -->
    <geom type="hfield" hfield="ground" material="groundplane" rgba="0.6 0.8 1.0 1"/>
    <!-- 可选：参考小球标记斜坡最高点 -->
    <geom type="sphere" pos="7.5 0 {z_max + 0.1}" size="0.05" rgba="1 0 0 1"/>
  </worldbody>
</mujoco>'''

with open(scene_xml_path, 'w') as f:
    f.write(xml_content)
print(f"场景 XML 已保存至: {scene_xml_path}")