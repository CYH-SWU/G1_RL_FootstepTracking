#!/usr/bin/env python3
"""
使用 Optuna 对 SB3 PPO 进行超参数调优。
所有参数在代码中直接配置，无需命令行参数。
用法: python scripts/tune_ppo.py
"""

import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from g_env.g1_test import G1TerrainEnv
from g_env.mirrorwrapper import MirrorWrapper

# ========== 固定参数（可在此修改） ==========
N_TRIALS = 20                # 调参试验次数（建议减少，因单次试验较耗时）
N_ITER = 1200                 # 每次试验训练迭代次数（600轮足够让机器人学会站立和初迈步）
N_ENVS = 16                   # 并行环境数量（8 可平衡速度和样本效率）
EVAL_EPISODES = 20            # 评估时的 episode 数
STUDY_NAME = "ppo_tuning"    # 研究名称
STORAGE = "sqlite:///optuna.db"   # 持久化存储（SQLite 文件）
# ===========================================


def create_env():
    """创建单个环境（带镜像包装）"""
    env = G1TerrainEnv(
        robot_xml_path=str(PROJECT_ROOT / "robot" / "g1_processed.xml")
    )
    env = MirrorWrapper(env, mirror_prob=0.5)
    return Monitor(env)


def objective(trial):
    """
    Optuna 目标函数。
    返回平均评估奖励（越高越好）。
    """
    # ---- 建议超参数 ----
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    n_steps = trial.suggest_int("n_steps", 64, 1024, step=64)
    batch_size = trial.suggest_int("batch_size", 32, 256, step=32)
    n_epochs = trial.suggest_int("n_epochs", 2, 10)
    gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.01, log=True)  # 放宽范围
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.2, 1.0)

    # ---- 固定网络结构 ----
    net_arch = [256, 256]
    policy_kwargs = dict(
        net_arch=dict(pi=net_arch, vf=net_arch),
        activation_fn=torch.nn.ReLU,
    )

    # ---- 创建并行环境 ----
    vec_env = make_vec_env(
        create_env,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"},  # Linux 推荐 fork
    )
    # 归一化（只归一化 actor_obs）
    vec_env = VecNormalize(
        venv=vec_env,
        norm_obs=True,
        norm_obs_keys=["actor_obs"],
        norm_reward=False,
        clip_obs=10.0,
        gamma=gamma,
    )

    # ---- 创建 PPO 模型 ----
    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        learning_rate=learning_rate,
        tensorboard_log=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # ---- 训练 ----
    total_timesteps = N_ITER * n_steps * N_ENVS
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    # ---- 评估 ----
    eval_env = G1TerrainEnv(
        robot_xml_path=str(PROJECT_ROOT / "robot" / "g1_processed.xml")
    )
    eval_env = MirrorWrapper(eval_env, mirror_prob=0.0)  # 评估禁用镜像

    rewards = []
    for _ in range(EVAL_EPISODES):
        obs, _ = eval_env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_rew += reward
            done = terminated or truncated
        rewards.append(ep_rew)

    avg_reward = np.mean(rewards)
    eval_env.close()
    vec_env.close()
    return avg_reward


def main():
    # 创建研究（如果 SQLite 文件已存在，会自动加载）
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    # 运行优化
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=1,  # 若想并行可设为 >1，注意资源
        show_progress_bar=True,
    )

    # 输出最佳结果
    print("\n===== Best Trial =====")
    print(f"Best value: {study.best_value:.2f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 保存最佳参数
    best_params_path = PROJECT_ROOT / "scripts" / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Best parameters saved to: {best_params_path}")

    # 保存全部试验数据
    df = study.trials_dataframe()
    df.to_csv(PROJECT_ROOT / "scripts" / "optuna_results.csv", index=False)
    print("All trials data saved to: scripts/optuna_results.csv")


if __name__ == "__main__":
    main()