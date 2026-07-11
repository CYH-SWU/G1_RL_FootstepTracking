#!/usr/bin/env python3
"""
测试 G1 步点跟踪环境，使用零动作输入，打开 MuJoCo 查看器。
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import mujoco
import mujoco.viewer

# 导入环境（根据实际文件名调整）
from env.g1_test import G1TerrainEnv  # 假设环境类在 env/g1_lhw_env.py

def main():
    # 配置环境参数
    robot_xml = project_root / "robot" / "g1_processed.xml"
    mesh_dir = project_root / "robot" / "assets"

    # 创建环境
    env = G1TerrainEnv(
        robot_xml_path=str(robot_xml),
        mesh_dir=str(mesh_dir),
        max_episode_steps=2000,
        control_dt=0.01,
        physics_dt=0.005,
    )
    env.model.opt.gravity = np.zeros(3)

    # 重置环境
    obs, info = env.reset()
    print(f"模式: {info['mode']}, 难度: {info['difficulty']:.2f}")
    print(f"步点数量: {len(env.sequence)}")

    # 启动查看器
    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    step = 0
    total_reward = 0.0
    done = False
    zero_action = np.zeros(12, dtype=np.float32)  # 动作空间为12维
    #zero_action = np.array([-1,0,0,-1,1,1,0,0,0,0,0,0])

    print("开始零动作测试，按 Esc 退出...")
    while viewer.is_running() and not done and step < 2000:
        step_start = time.time()

        # 执行一步（零动作）
        obs, reward, terminated, truncated, info = env.step(zero_action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        # 同步渲染
        viewer.sync()

        # 控制实时速度
        elapsed = time.time() - step_start
        time_to_sleep = env.control_dt - elapsed
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

        # 每50步打印状态
        if step % 50 == 0:
            pelvis_z = env.data.qpos[2]
            foot_z = min(env.data.xpos[env.left_foot_id][2], env.data.xpos[env.right_foot_id][2]) - env.foot_ankle_offset
            height = pelvis_z - foot_z
            print(f"步数: {step:4d} | 奖励: {reward:6.2f} | 总奖励: {total_reward:8.2f} | "
                  f"骨盆高: {height:.3f} m | t1: {env.t1}, t2: {env.t2}")

        if done:
            print(f"回合结束，原因: {'终止' if terminated else '截断'}")
            break

    viewer.close()
    env.close()
    print(f"测试结束，总步数: {step}, 总奖励: {total_reward:.2f}")

if __name__ == "__main__":
    main()