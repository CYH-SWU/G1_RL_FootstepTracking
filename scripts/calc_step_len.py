import numpy as np
import mujoco
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MODEL_PATH = PROJECT_ROOT / "robot" / "g1_processed.xml"

# 加载模型
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# 获取关节索引
hip_joint_name = "left_hip_pitch_joint"
knee_joint_name = "left_knee_joint"
hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, hip_joint_name)
knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, knee_joint_name)

# 获取关节 qpos 地址
hip_qpos_adr = model.joint(hip_id).qposadr[0]
knee_qpos_adr = model.joint(knee_id).qposadr[0]

# 获取踝关节 body id
ankle_body = model.body("left_ankle_roll_link").id

# 标称姿态
nominal_hip = -0.5236
nominal_knee = 0.8727
action_scale = 0.3

# 有效范围
hip_min = nominal_hip - action_scale
hip_max = nominal_hip + action_scale
knee_min = nominal_knee - action_scale
knee_max = nominal_knee + action_scale

# 进行网格搜索（分辨率 0.01 rad）
max_x = -np.inf
best_hip = best_knee = None
for hip in np.arange(hip_min, hip_max, 0.01):
    for knee in np.arange(knee_min, knee_max, 0.01):
        # 设置关节角度
        data.qpos[hip_qpos_adr] = hip
        data.qpos[knee_qpos_adr] = knee
        mujoco.mj_forward(model, data)
        # 获取踝关节位置（相对于骨盆）
        ankle_pos = data.xpos[ankle_body].copy()
        pelvis_pos = data.xpos[model.body("pelvis").id].copy()
        rel_x = ankle_pos[0] - pelvis_pos[0]
        if rel_x > max_x:
            max_x = rel_x
            best_hip = hip
            best_knee = knee

print(f"最大前向步幅: {max_x:.3f} m")
print(f"对应髋角度: {best_hip:.3f} rad")
print(f"对应膝角度: {best_knee:.3f} rad")