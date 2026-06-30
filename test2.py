#!/usr/bin/env python3
"""
评估训练好的 PPO 模型，加载 VecNormalize 参数，循环 10 个回合。
用法: python evaluate.py
"""

import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# 项目根目录
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from env.g1_anti_env import G1TerrainEnv

def main():
    # 文件路径
    model_path = project_root / "checkpoints" / "ppo_g1_final.zip"
    norm_path = project_root / "checkpoints" / "vec_normalize_final.pkl"
    
    robot_xml = project_root / "robot" / "g1_processed.xml"
    mesh_dir = project_root / "robot" / "assets"

    # 评估参数
    num_episodes = 10
    max_steps_per_episode = 2000

    # 检查文件
    if not model_path.exists():
        print(f"错误：模型文件不存在: {model_path}")
        return

    # 1. 创建基础环境（确保与训练时的参数一致）
    base_env = G1TerrainEnv(
        robot_xml_path=str(robot_xml),
        mesh_dir=str(mesh_dir),
        max_episode_steps=max_steps_per_episode,
        total_timesteps_for_max=11_000_000  # 不影响评估
    )

    # 2. 包装为 VecEnv（单环境）
    vec_env = DummyVecEnv([lambda: base_env])

    # 3. 加载归一化参数
    if norm_path.exists():
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        # ★★★ 关键修复：冻结归一化统计量的更新 ★★★
        vec_env.training = False
        print("已加载 VecNormalize 统计量，并设置为评估模式（training=False）")
    else:
        print("警告：未找到归一化文件，假设环境未归一化。")

    # 4. 加载模型
    model = PPO.load(str(model_path))
    print("模型加载成功")

    # 5. 获取原始环境（用于渲染）
    # VecNormalize 包装了 DummyVecEnv，DummyVecEnv 内部有 envs 列表
    if hasattr(vec_env, 'venv'):
        raw_env = vec_env.venv.envs[0]
    else:
        raw_env = vec_env.envs[0]

    # 6. 重置环境（确保归一化统计量生效）
    obs = vec_env.reset()
    print("环境已重置，开始评估...")

    # 7. 启动查看器
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
            obs, rewards, dones, infos = vec_env.step(action)

            # 单环境，取第一个元素
            reward = rewards[0]
            done = dones[0]
            info = infos[0]

            episode_reward += reward
            step += 1
            total_steps += 1

            # 同步渲染
            viewer.sync()

            # 控制实时速度
            elapsed = time.time() - step_start
            time_to_sleep = raw_env.control_dt - elapsed
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

            # 每100步打印状态
            if step % 100 == 0:
                pelvis_z = raw_env.data.qpos[2]
                foot_z = min(raw_env.data.xpos[raw_env.left_foot_id][2],
                             raw_env.data.xpos[raw_env.right_foot_id][2]) - raw_env.foot_ankle_offset
                height = pelvis_z - foot_z
                dist = np.linalg.norm(raw_env.data.xpos[raw_env.pelvis_id][:2] - raw_env.goal_pos[:2])
                print(f"  步数 {step:4d}: 奖励={reward:6.2f}, 总奖励={episode_reward:7.2f}, "
                      f"骨盆高={height:.3f}m, 距终点={dist:.3f}m")

        episode_rewards.append(episode_reward)
        print(f"Episode {episode} 结束: 总步数={step}, 总奖励={episode_reward:.2f}")

        # 判断终止原因
        if done:
            if info.get('terminated', False):
                dist = np.linalg.norm(raw_env.data.xpos[raw_env.pelvis_id][:2] - raw_env.goal_pos[:2])
                if dist < 0.5:
                    print("  → 成功到达终点！")
                else:
                    print("  → 摔倒终止。")
            elif info.get('truncated', False):
                print("  → 超时截断。")

        # 如果查看器已关闭，退出循环
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
        success_count = sum(1 for r in episode_rewards if r > 0)
        print(f"成功率: {success_count / len(episode_rewards) * 100:.1f}%")


if __name__ == "__main__":
    main()