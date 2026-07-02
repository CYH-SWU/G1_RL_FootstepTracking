#!/usr/bin/env python3
"""
人形机器人复杂地形行走训练脚本
使用 Stable-Baselines3 RecurrentPPO + 非对称 Actor-Critic (MultiInputLstmPolicy)
"""

import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

# ★ 导入 sb3_contrib 的循环 PPO 和策略
from sb3_contrib import RecurrentPPO


project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from env.g1_env import G1TerrainEnv

# -------------------- 配置参数 --------------------
ROBOT_XML = project_root / "robot" / "g1_processed.xml"
MESH_DIR = project_root / "robot" / "assets"
CHECKPOINT_DIR = project_root / "checkpoints"
LOG_DIR = project_root / "logs"

CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

N_ENVS = 8
TOTAL_TIMESTEPS = 6 * 200 * 1500
MAX_EPISODE_STEPS = 2000
TOTAL_TIMESTEPS_FOR_MAX = 11000 * 1500

# -------------------- 环境创建 --------------------
def make_env():
    env = G1TerrainEnv(
        robot_xml_path=str(ROBOT_XML),
        mesh_dir=str(MESH_DIR),
        max_episode_steps=MAX_EPISODE_STEPS,
        total_timesteps_for_max=TOTAL_TIMESTEPS_FOR_MAX,
    )
    return Monitor(env)

vec_env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)

# ★ 关键：只归一化 "actor_obs"，特权观测不参与 VecNormalize
vec_env = VecNormalize(
    venv=vec_env,
    norm_obs=True,
    norm_obs_keys=["actor_obs"],   # 仅归一化 actor 观测
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
    save_freq=(TOTAL_TIMESTEPS / N_ENVS) // 1,  # 按需调整
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ppo_g1",
    save_replay_buffer=False,
    save_vecnormalize=True,
)

# -------------------- 创建 LSTM + 非对称策略 --------------------
policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 128],     # Actor 网络结构（使用 actor_obs）
        vf=[256, 128]      # Critic 网络结构（使用 critic_obs）
    ),
    activation_fn=torch.nn.ReLU,
    lstm_hidden_size=64,          # LSTM 记忆单元维度（宇树常用 64）
    n_lstm_layers=1,              # 单层 LSTM
    enable_critic_lstm=True,      # Critic 也使用 LSTM，提升价值估计稳定性
)

# ★ 使用 RecurrentPPO 和 MultiInputLstmPolicy
model = RecurrentPPO(
    policy="MultiInputLstmPolicy",
    env=vec_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=1024,                 # 建议增大序列长度（原500，现1024）
    tensorboard_log=str(LOG_DIR),
    device='cuda',
    ent_coef=0.01,
    # 其他参数保持 SB3/RecurrentPPO 默认值
)

# -------------------- 训练 --------------------
print("开始训练（LSTM + 非对称 Actor-Critic）...")
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