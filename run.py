#!/usr/bin/env python3
"""
人形机器人复杂地形行走训练脚本
使用 Stable-Baselines3 PPO + 非对称 Actor-Critic (MultiInputPolicy)
采用 SubprocVecEnv 多进程并行训练，大幅提升采样效率。
"""

import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from g_env.mirrorwrapper import MirrorWrapper
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from rl.callbacks import AdaptiveLRScheduleCallback, CurriculumCallback
from rl.policy import policy_kwargs


project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from g_env.g1_test import G1TerrainEnv

# -------------------- 配置参数 --------------------
ROBOT_XML = project_root / "robot" / "g1_processed.xml"
MESH_DIR = project_root / "robot" / "assets"
CHECKPOINT_DIR = project_root / "checkpoints"
LOG_DIR = project_root / "logs"

CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

ITERATION = 11000
N_ENVS = 16  
N_STEPS = 800 
                       
TOTAL_TIMESTEPS = ITERATION * N_ENVS * N_STEPS
TOTAL_TIMESTEPS_FOR_MAX = 11000 * N_ENVS * N_STEPS

# -------------------- 环境创建函数（必须可 pickle） --------------------
def make_env():
    """工厂函数：创建单个 G1 环境实例"""
    env = G1TerrainEnv(
        robot_xml_path=str(ROBOT_XML),
    )
    env = MirrorWrapper(env, mirror_prob=0.5)
    return Monitor(env)

# -------------------- 创建 SubprocVecEnv（多进程并行） --------------------
# 注意：必须在 `if __name__ == "__main__":` 块内调用，否则 Windows 下会出错
vec_env = make_vec_env(
    make_env,
    n_envs=N_ENVS,
    vec_env_cls=SubprocVecEnv,
    vec_env_kwargs={"start_method": "fork"}   # Linux 推荐 fork，Windows 可用 spawn
)

# ★ 关键：只归一化 "actor_obs"，特权观测不参与 VecNormalize
vec_env = VecNormalize(
    venv=vec_env,
    norm_obs=True,
    norm_obs_keys=["actor_obs"],
    norm_reward=False,
    clip_obs=10.0,
    gamma=0.99,
)

checkpoint_callback = CheckpointCallback(
    save_freq=(TOTAL_TIMESTEPS // N_ENVS) // 12 ,  # 调整保存频率
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ppo_g1",
    save_replay_buffer=False,
    save_vecnormalize=True,
)

adaptive_lr_callback = AdaptiveLRScheduleCallback(
    patience=5,          # 连续 5 次评估无改进则降学习率
    factor=0.95,          # 减半
    eval_freq=N_ENVS * N_STEPS * 16,      # 每 2000 步检查一次
    min_lr=1e-7,         # 最低学习率
    verbose=1
)

model = PPO(
    policy="MultiInputPolicy",
    env=vec_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    # --- 采样参数 ---
    n_steps=N_STEPS,                     # 每个环境每次更新采集的步数（与 LHW 的 max_traj_len=400 对齐）
    # --- 优化参数 ---
    learning_rate=1e-4,              # 与 LHW 默认 lr 一致
    batch_size=64,                   # 与 LHW 默认 minibatch_size 一致
    n_epochs=3,                      # 与 LHW 默认 epochs 一致
    # --- 折扣与优势估计 ---
    gamma=0.99,                      # 与 LHW 默认 gamma 一致
    gae_lambda=0.95,                 # 与 LHW 默认 lam 一致
    # --- PPO 裁剪 ---
    clip_range=0.15,                  # 与 LHW 默认 clip 一致
    # --- 熵与探索 ---
    ent_coef=0.001,                    # LHW 默认熵系数为 0（不鼓励额外探索）
    max_grad_norm=0.5,               # 与 LHW 默认 grad norm 一致（SB3 默认也是 0.5）
    tensorboard_log=str(LOG_DIR),
    device='cuda',
)

# -------------------- 训练主入口（必须包含 if __name__） --------------------
if __name__ == "__main__":
    print("开始训练（非对称 Actor-Critic + SubprocVecEnv）...")
    print(f"总步数: {TOTAL_TIMESTEPS}, 并行环境数: {N_ENVS}")

    curriculum_callback = CurriculumCallback(TOTAL_TIMESTEPS_FOR_MAX)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[curriculum_callback, checkpoint_callback, adaptive_lr_callback],
        progress_bar=True,
    )

    print("=============正在保存模型=============")
    model.save(str(CHECKPOINT_DIR / "ppo_g1_final.zip"))
    vec_env.save(str(CHECKPOINT_DIR / "vec_normalize_final.pkl"))
    print("✅ 训练完成！模型和归一化参数已保存。")