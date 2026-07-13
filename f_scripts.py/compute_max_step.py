#!/usr/bin/env python3
"""
计算 G1 机器人腿部在标称姿态附近的最大前向步幅。
从 f_env.config 读取标称角度和 action_scale，保持与训练环境一致。
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

# 将项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from f_env.utils.config import G1EnvConfig  # 重构后的配置

# 加载配置
config = G1EnvConfig()

# 模型路径（使用默认 XML）
MODEL_PATH = PROJECT_ROOT / "robot" / "g1_processed.xml"

# 加载模型
model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# 关节名称（固定，与配置中的 nominal_angles 顺序一致）
hip_joint_name = "left_hip_pitch_joint"
knee_joint_name = "left_knee_joint"

# 获取关节 ID 和 qpos 地址
hip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, hip_joint_name)
knee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, knee_joint_name)
hip_qpos_adr = model.joint(hip_id).qposadr[0]
knee_qpos_adr = model.joint(knee_id).qposadr[0]

# 获取踝关节 body id（用于测量脚位置）
ankle_body = model.body("left_ankle_roll_link").id
pelvis_body = model.body("pelvis").id

# 从配置读取标称角度（左髋俯仰和左膝）
# nominal_angles 顺序：hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll, ...
nominal_hip = config.nominal_angles[0]
nominal_knee = config.nominal_angles[3]

# 从配置读取动作缩放
action_scale = config.action_scale

# 有效搜索范围（标称 ± action_scale）
hip_min = nominal_hip - action_scale
hip_max = nominal_hip + action_scale
knee_min = nominal_knee - action_scale
knee_max = nominal_knee + action_scale

print(f"标称髋角度: {nominal_hip:.4f} rad")
print(f"标称膝角度: {nominal_knee:.4f} rad")
print(f"动作缩放: {action_scale}")
print(f"搜索范围: 髋 [{hip_min:.4f}, {hip_max:.4f}], 膝 [{knee_min:.4f}, {knee_max:.4f}]")

# 网格搜索（分辨率 0.01 rad）
max_x = -np.inf
best_hip = best_knee = None

for hip in np.arange(hip_min, hip_max, 0.01):
    for knee in np.arange(knee_min, knee_max, 0.01):
        data.qpos[hip_qpos_adr] = hip
        data.qpos[knee_qpos_adr] = knee
        mujoco.mj_forward(model, data)

        ankle_pos = data.xpos[ankle_body].copy()
        pelvis_pos = data.xpos[pelvis_body].copy()
        rel_x = ankle_pos[0] - pelvis_pos[0]  # 前向位移

        if rel_x > max_x:
            max_x = rel_x
            best_hip = hip
            best_knee = knee

print(f"\n最大前向步幅: {max_x:.3f} m")
print(f"对应髋角度: {best_hip:.3f} rad")
print(f"对应膝角度: {best_knee:.3f} rad")