#!/usr/bin/env python3
"""
测试脚本：零动作输入，观察 G1 机器人站立情况。
用于调试环境和视觉流程。
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from env.g1_terrain_env import G1TerrainEnv

def main():
    robot_xml = project_root / "robot" / "g1_processed.xml"
    mesh_dir = project_root / "robot" / "assets"

    # 创建环境（单环境，不并行）
    env = G1TerrainEnv(
        robot_xml_path=str(robot_xml),
        mesh_dir=str(mesh_dir),
        max_episode_steps=2000,
        total_timesteps_for_max=11000*1500  # 不影响测试
    )

    # 重置环境
    obs, info = env.reset()
    print(f"地形模式: {env.terrain_mode}, 难度: {env.difficulty:.2f}")
    print(f"终点位置: {env.goal_pos}")

    # 启动查看器
    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    step = 0
    done = False
    total_reward = 0.0

    # 零动作（13个关节）
    zero_action = np.zeros(13, dtype=np.float32)

    while viewer.is_running() and not done:
        step_start = time.time()

        # 执行一步（零动作）
        obs, reward, terminated, truncated, info = env.step(zero_action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        # 同步渲染
        viewer.sync()

        # 控制实时性
        elapsed = time.time() - step_start
        time_to_sleep = env.control_dt - elapsed
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

        # 每50步打印状态
        if step % 5 == 0:
            pelvis_z = env.data.qpos[2]
            stance_foot = env.left_foot_id if env.current_stance == -1 else env.right_foot_id
            foot_z = env.data.xpos[stance_foot][2]
            height = pelvis_z - foot_z
            print(f"步数: {step:4d} | 奖励: {reward:6.2f} | 总奖励: {total_reward:8.2f} | "
                  f"骨盆高: {height:.3f} m | 距终点: {np.linalg.norm(env.data.xpos[env.pelvis_id][:2] - env.goal_pos[:2]):.3f} m")

        if done:
            print(f"回合结束，原因: {'到达终点' if terminated else '超时' if truncated else '未知'}")
            break

    viewer.close()
    env.close()
    print(f"测试结束，总步数: {step}, 总奖励: {total_reward:.2f}")

if __name__ == "__main__":
    main()