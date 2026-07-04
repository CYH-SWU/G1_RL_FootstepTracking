#!/usr/bin/env python3
"""
检查机器人运动空间是否支持目标步幅（直接使用 env 接口）
用法: python scripts/check_workspace.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mujoco

# 导入你的环境
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from env.g1_env import G1TerrainEnv

# 配置路径
ROBOT_XML = project_root / "robot" / "g1_processed.xml"
MESH_DIR = project_root / "robot" / "assets"

# 目标步幅
TARGET_STEP = 0.25
ACTION_SCALE = 0.25

def main():
    print("=" * 60)
    print("G1 机器人运动空间检查")
    print("=" * 60)

    # 1. 创建环境实例（只初始化，不 reset 也可以访问 model/data）
    env = G1TerrainEnv(
        robot_xml_path=str(ROBOT_XML),
        mesh_dir=str(MESH_DIR),
        max_episode_steps=2000,
    )
    # 执行一次 reset，确保模型已加载且数据有效
    env.reset()
    model = env.model
    data = env.data

    # 2. 关节限位检查（使用 env 中的标称姿态）
    nominal_angles = env.nominal_angles
    joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
    ]

    print("\n=== 关节限位检查 ===")
    all_ok = True
    for i, name in enumerate(joint_names):
        joint = model.joint(name)
        
        joint_id = joint.id
        low, high = model.jnt_range[joint_id]   # ✅ 正确
        nom = nominal_angles[i]
        min_act = nom - ACTION_SCALE
        max_act = nom + ACTION_SCALE
        low_ok = min_act >= low
        high_ok = max_act <= high
        status = "✓" if (low_ok and high_ok) else "✗"
        if not (low_ok and high_ok):
            all_ok = False
        print(f"{name:30s} | 标称:{nom:7.3f} | 动作范围:[{min_act:7.3f}, {max_act:7.3f}] | 限位:[{low:7.3f}, {high:7.3f}] | {status}")

    if not all_ok:
        print("\n⚠️ 警告：部分关节的动作范围超出限位，请调整标称姿态或 action_scale。")
    else:
        print("\n✅ 所有关节均在限位内。")

    # 3. 脚部可达范围分析（使用 env 的模型计算）
    print("\n=== 脚部可达范围（左腿） ===")

    # 获取左髋和左膝的 qpos 地址
    hip_joint = model.joint("left_hip_pitch_joint")
    knee_joint = model.joint("left_knee_joint")
    hip_addr = hip_joint.qposadr[0]
    knee_addr = knee_joint.qposadr[0]

    # 记录脚部相对于骨盆的位移
    foot_dx_list = []
    foot_dz_list = []

    # 在标称基础上扫描髋、膝偏移
    offsets = np.linspace(-ACTION_SCALE, ACTION_SCALE, 11)  # -0.25 ~ 0.25

    for hip_off in offsets:
        for knee_off in offsets:
            # 设置左髋、左膝角度（其他关节保持标称）
            env.data.qpos[hip_addr] = nominal_angles[0] + hip_off
            env.data.qpos[knee_addr] = nominal_angles[3] + knee_off
            # 其他关节维持标称
            for i, name in enumerate(joint_names):
                if i not in [0, 3]:
                    addr = model.joint(name).qposadr[0]
                    env.data.qpos[addr] = nominal_angles[i]

            # 前向动力学，更新 xpos
            mujoco.mj_forward(env.model, env.data)

            # 获取骨盆和左脚位置
            pelvis_pos = env.data.xpos[model.body("pelvis").id]
            left_foot_pos = env.data.xpos[model.body("left_ankle_roll_link").id]
            rel_pos = left_foot_pos - pelvis_pos
            foot_dx_list.append(rel_pos[0])
            foot_dz_list.append(rel_pos[2])

    foot_dx = np.array(foot_dx_list)
    foot_dz = np.array(foot_dz_list)

    max_dx = np.max(foot_dx)
    min_dx = np.min(foot_dx)
    print(f"脚部前向位移范围: [{min_dx:.3f}, {max_dx:.3f}] m")
    print(f"目标步幅: {TARGET_STEP:.3f} m")

    if max_dx >= TARGET_STEP:
        print(f"✅ 可达最大前向位移 {max_dx:.3f} m >= {TARGET_STEP:.3f}m，支持目标步幅。")
    else:
        print(f"❌ 最大前向位移 {max_dx:.3f} m < {TARGET_STEP:.3f}m，无法达到目标步幅。")
        print("   建议：增大 action_scale，或调整标称姿态（增加髋伸展/膝伸展的标称值）。")

    # 4. 可视化脚部可达区域（可选）
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(foot_dx, foot_dz, c=np.sqrt(foot_dx**2 + foot_dz**2), cmap='viridis')
    ax.axvline(x=TARGET_STEP, color='r', linestyle='--', label=f'目标步幅 {TARGET_STEP}m')
    ax.set_xlabel('前向位移 (m)')
    ax.set_ylabel('垂向位移 (m)')
    ax.set_title('左脚可达空间（相对骨盆）')
    ax.legend()
    ax.grid(True)
    plt.colorbar(scatter, label='到原点距离 (m)')
    plt.tight_layout()
    plt.savefig(project_root / "scripts" / "workspace_plot.png", dpi=150)
    print(f"\n✅ 可视化图表已保存到: {project_root / 'scripts' / 'workspace_plot.png'}")
    plt.show()

if __name__ == "__main__":
    main()