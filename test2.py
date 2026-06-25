#!/usr/bin/env python3
"""
评估训练好的 PPO 模型，加载 VecNormalize 参数，循环 10 个回合。
用法: python evaluate.py
"""

import os
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from env.g1_terrain_env import G1TerrainEnv


def main():
    # 参数配置
    model_path = project_root / "checkpoints" / "ppo_g1_final.zip"
    norm_path = project_root / "checkpoints" / "vec_normalize_final.pkl"
    robot_xml = project_root / "robot" / "g1_processed.xml"
    mesh_dir = project_root / "robot" / "assets"
    num_episodes = 10
    max_steps_per_episode = 2000

    # 检查文件是否存在
    if not model_path.exists():
        print(f"错误：模型文件不存在: {model_path}")
        return
    if not norm_path.exists():
        print(f"警告：归一化文件不存在: {norm_path}，将尝试仅加载模型（可能失败）")

    # 1. 创建基础环境（单环境）
    base_env = G1TerrainEnv(
        robot_xml_path=str(robot_xml),
        mesh_dir=str(mesh_dir),
        max_episode_steps=max_steps_per_episode,
        total_timesteps_for_max=11_000_000  # 不影响评估
    )

    # 2. 包装为 VecEnv（因为 VecNormalize 需要 VecEnv）
    vec_env = DummyVecEnv([lambda: base_env])

    # 3. 加载归一化参数
    if norm_path.exists():
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        print("已加载 VecNormalize 统计量")
    else:
        print("未找到归一化文件，假设环境未归一化。")

    # 4. 加载模型
    model = PPO.load(str(model_path))
    print("模型加载成功")

    # 5. 提取原始环境（以便直接访问 model/data 用于渲染）
    if hasattr(vec_env, 'venv'):
        raw_env = vec_env.venv.envs[0]
    else:
        raw_env = vec_env.envs[0]

    # 6. 重置 vec_env（获取初始观测）
    obs = vec_env.reset()
    print("环境已重置，开始评估...")

    # 7. 启动 MuJoCo 查看器
    viewer = mujoco.viewer.launch_passive(raw_env.model, raw_env.data)
    print("按 Esc 或关闭窗口可提前退出评估。")

    episode = 0
    total_steps = 0
    episode_rewards = []

    while episode < num_episodes and viewer.is_running():
        episode += 1
        step = 0
        episode_reward = 0.0
        done = False

        # 获取当前地形信息
        terrain = raw_env.terrain_mode
        print(f"\n=== Episode {episode}/{num_episodes} ===")
        print(f"地形: {terrain}, 难度: {raw_env.difficulty:.2f}")
        print(f"终点: ({raw_env.goal_pos[0]:.2f}, {raw_env.goal_pos[1]:.2f})")

        while not done and viewer.is_running() and step < max_steps_per_episode:
            step_start = time.time()

            # 策略推理（确定性）
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = vec_env.step(action)

            done = terminated[0] or truncated[0]
            episode_reward += reward[0]
            step += 1
            total_steps += 1

            # 同步渲染
            viewer.sync()

            # 控制实时速度
            elapsed = time.time() - step_start
            time_to_sleep = raw_env.control_dt - elapsed
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

            # 每 100 步打印状态
            if step % 100 == 0:
                pelvis_z = raw_env.data.qpos[2]
                stance_foot = raw_env.left_foot_id if raw_env.current_stance == -1 else raw_env.right_foot_id
                foot_z = raw_env.data.xpos[stance_foot][2]
                height = pelvis_z - foot_z
                dist = np.linalg.norm(raw_env.data.xpos[raw_env.pelvis_id][:2] - raw_env.goal_pos[:2])
                print(f"  步数 {step:4d}: 奖励={reward[0]:6.2f}, 总奖励={episode_reward:7.2f}, "
                      f"骨盆高={height:.3f}m, 距终点={dist:.3f}m")

        episode_rewards.append(episode_reward)
        print(f"Episode {episode} 结束: 总步数={step}, 总奖励={episode_reward:.2f}")

        # 如果是因为到达终点而结束，打印成功信息
        if done and terminated[0]:
            dist = np.linalg.norm(raw_env.data.xpos[raw_env.pelvis_id][:2] - raw_env.goal_pos[:2])
            if dist < 0.5:
                print("  → 成功到达终点！")
            else:
                print("  → 摔倒终止。")
        elif done and truncated[0]:
            print("  → 超时截断。")

        # 若查看器已关闭，退出循环
        if not viewer.is_running():
            break

        # 重置环境进行下一个 episode
        obs = vec_env.reset()

    viewer.close()
    raw_env.close()

    # 输出整体统计
    print("\n=== 评估完成 ===")
    print(f"总回合数: {episode}")
    print(f"总步数: {total_steps}")
    if episode_rewards:
        print(f"平均回合奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"成功率: {sum(1 for r in episode_rewards if r > 0) / len(episode_rewards) * 100:.1f}%")


if __name__ == "__main__":
    main()