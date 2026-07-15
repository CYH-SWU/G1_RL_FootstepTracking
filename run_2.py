#!/usr/bin/env python3
"""
人形机器人复杂地形行走训练脚本
使用 Stable-Baselines3 PPO + 真正的非对称 Actor-Critic (自定义策略)
Actor 仅使用 actor_obs，Critic 仅使用 critic_obs。
采用 SubprocVecEnv 多进程并行训练。
"""

import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor
from gymnasium import spaces
from typing import Dict, Callable, Optional, List, Tuple, Type, Union

from g_env.mirrorwrapper import MirrorWrapper

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from g_env.g1_test import G1TerrainEnv  # 你的环境
from rl.policy import AsymmetricPolicy


# ==================== 训练脚本 ====================
# -------------------- 配置参数 --------------------
ROBOT_XML = project_root / "robot" / "g1_processed.xml"
CHECKPOINT_DIR = project_root / "checkpoints"
LOG_DIR = project_root / "logs"

CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

ITERATION = 1500 * 2
N_ENVS = 16
TOTAL_TIMESTEPS = ITERATION * N_ENVS * 400
TOTAL_TIMESTEPS_FOR_MAX = 11000 * N_ENVS * 400

# -------------------- 环境创建函数 --------------------
def make_env():
    env = G1TerrainEnv(robot_xml_path=str(ROBOT_XML))
    env = MirrorWrapper(env, mirror_prob=0.5)
    return Monitor(env)

# -------------------- 创建并行环境 --------------------
vec_env = make_vec_env(
    make_env,
    n_envs=N_ENVS,
    vec_env_cls=SubprocVecEnv,
    vec_env_kwargs={"start_method": "fork"}   # Linux 推荐 fork
)

# 只归一化 actor_obs
vec_env = VecNormalize(
    venv=vec_env,
    norm_obs=True,
    norm_obs_keys=["actor_obs"],
    norm_reward=False,
    clip_obs=10.0,
    gamma=0.99,
)

# -------------------- 回调 --------------------
class CurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps_for_max: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps_for_max = total_timesteps_for_max

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps_for_max)
        self.training_env.env_method("set_difficulty", progress)
        return True

checkpoint_callback = CheckpointCallback(
    save_freq=(TOTAL_TIMESTEPS // N_ENVS) // 16,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ppo_g1",
    save_replay_buffer=False,
    save_vecnormalize=True,
)

# -------------------- 策略参数 --------------------
policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),  # 虽然自定义策略未使用 net_arch，但保留不影响
    activation_fn=torch.nn.ReLU,
)

# -------------------- 创建 PPO 模型（使用自定义非对称策略） --------------------
model = PPO(
    policy=AsymmetricPolicy,          # ★ 使用自定义非对称策略
    env=vec_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=400,
    learning_rate=1e-4,
    batch_size=64,
    n_epochs=3,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,
    max_grad_norm=0.5,
    tensorboard_log=str(LOG_DIR),
    device='cuda',
)

# -------------------- 训练主入口 --------------------
if __name__ == "__main__":
    print("开始训练（真正的非对称 Actor-Critic + SubprocVecEnv）...")
    print(f"总步数: {TOTAL_TIMESTEPS}, 并行环境数: {N_ENVS}")

    curriculum_callback = CurriculumCallback(TOTAL_TIMESTEPS_FOR_MAX)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[curriculum_callback, checkpoint_callback],
        progress_bar=True,
    )

    print("=============正在保存模型=============")
    model.save(str(CHECKPOINT_DIR / "ppo_g1_final.zip"))
    vec_env.save(str(CHECKPOINT_DIR / "vec_normalize_final.pkl"))
    print("✅ 训练完成！模型和归一化参数已保存。")