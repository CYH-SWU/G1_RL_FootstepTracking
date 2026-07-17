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

from g_env.g1_test import G1TerrainEnv

def main():
    # 文件路径
    model_path = project_root / "checkpoints" / "ppo_g1_83200000_steps.zip"
    norm_path = project_root / "checkpoints" / "ppo_g1_vecnormalize_83200000_steps.pkl"
    
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

    # 5. 获取原始环境（用于渲染和状态读取）
    # VecNormalize 包装了 DummyVecEnv，DummyVecEnv 内部有 envs 列表
    if hasattr(vec_env, 'venv'):
        raw_env = vec_env.venv.envs[0]
    else:
        raw_env = vec_env.envs[0]

    # 6. 重置环境
    obs = vec_env.reset()
    print("环境已重置，开始评估...")

    # 7. 启动查看器
    viewer = mujoco.viewer.launch_passive(raw_env.model, raw_env.data)
    print("按 Esc 或关闭窗口可提前退出评估。")

    episode = 0
    total_steps = 0
    episode_rewards = []
    success_count = 0

    while episode < num_episodes and viewer.is_running():
        episode += 1
        step = 0
        episode_reward = 0.0
        done = False
        truncated = False

        print(f"\n=== Episode {episode}/{num_episodes} ===")

        while not done and viewer.is_running() and step < max_steps_per_episode:
            step_start = time.time()

            # 策略推理（确定性）
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)

            reward = rewards[0]
            done = dones[0]
            info = infos[0]

            episode_reward += reward
            step += 1
            total_steps += 1

            # 同步渲染
            viewer.sync()

            # 控制实时速度（模拟 real-time）
            elapsed = time.time() - step_start
            time_to_sleep = raw_env.control_dt - elapsed
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

            # 每100步打印状态（打印更多有用信息）
            if step % 100 == 0:
                # 骨盆高度
                pelvis_z = raw_env.data.qpos[2]
                foot_z = min(raw_env.data.xpos[raw_env.left_foot_id][2],
                             raw_env.data.xpos[raw_env.right_foot_id][2]) - raw_env.foot_ankle_offset
                height = pelvis_z - foot_z
                # 当前步点索引和相位
                t1 = raw_env.t1
                t2 = raw_env.t2
                total_steps_seq = len(raw_env.sequence)
                phase = raw_env.phase
                # 是否已踩中目标
                target_reached = raw_env.target_reached
                # 左脚/右脚足底力
                left_frc = raw_env.data.cfrc_ext[raw_env.left_foot_id][2]
                right_frc = raw_env.data.cfrc_ext[raw_env.right_foot_id][2]
                print(f"  步数 {step:4d}: 奖励={reward:6.2f}, 总奖励={episode_reward:7.2f}, "
                      f"身高={height:.3f}m, 相位={phase:.3f}, "
                      f"步点进度={t1}/{total_steps_seq-1}, 目标踩中={target_reached}, "
                      f"足力 L={left_frc:5.1f} R={right_frc:5.1f} N")

        episode_rewards.append(episode_reward)
        print(f"Episode {episode} 结束: 步数={step}, 总奖励={episode_reward:.2f}")

        # 判断终止原因
        if done:
            # 检查是否由于摔倒
            height = raw_env.data.qpos[2] - (min(raw_env.data.xpos[raw_env.left_foot_id][2],
                                                 raw_env.data.xpos[raw_env.right_foot_id][2]) - raw_env.foot_ankle_offset)
            if height < raw_env.fall_height_threshold:
                print("  → 摔倒终止。")
            else:
                # 可能因为步点序列完成？环境不会因为完成序列而终止，只有摔倒或超时
                print("  → 终止（未知原因）。")
        elif step >= max_steps_per_episode:
            truncated = True
            print("  → 超时截断。")

        # 判断是否成功完成步点序列（到达最后一个步点）
        if raw_env.t1 >= len(raw_env.sequence) - 1:
            success_count += 1
            print("  → 成功完成步点序列！")

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
        print(f"成功完成步点序列的回合数: {success_count}/{episode} ({success_count/episode*100:.1f}%)")
    else:
        print("无有效回合数据。")

if __name__ == "__main__":
    main()