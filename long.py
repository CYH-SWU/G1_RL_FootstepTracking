#!/usr/bin/env python3
"""
顺序训练：先训练前馈网络（PPO），再训练 LSTM 网络（RecurrentPPO）
两个任务完全独立，不继承训练状态。
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
from env.mirrorwrapper import MirrorWrapper
from sb3_contrib import RecurrentPPO

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from env.g1_test import G1TerrainEnv

# -------------------- 通用配置 --------------------
ROBOT_XML = project_root / "robot" / "g1_processed.xml"
MESH_DIR = project_root / "robot" / "assets"
CHECKPOINT_DIR = project_root / "checkpoints"
LOG_DIR = project_root / "logs"
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
N_ENVS = 16

# -------------------- 环境创建函数 --------------------
def make_env():
    env = G1TerrainEnv(robot_xml_path=str(ROBOT_XML))
    env = MirrorWrapper(env, mirror_prob=0.5)
    return Monitor(env)

def create_vec_env():
    vec_env = make_vec_env(
        make_env,
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"}
    )
    vec_env = VecNormalize(
        venv=vec_env,
        norm_obs=True,
        norm_obs_keys=["actor_obs"],
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )
    return vec_env

# -------------------- 课程学习回调 --------------------
class CurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps_for_max: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps_for_max = total_timesteps_for_max

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps_for_max)
        self.training_env.env_method("set_difficulty", progress)
        return True

# ==================== 任务1：前馈网络 ====================
def train_ff():
    print("\n" + "="*60)
    print("开始训练：前馈网络 (PPO + MultiInputPolicy)")
    print("="*60)

    ITERATION = 3000
    TOTAL_TIMESTEPS = ITERATION * N_ENVS * 400
    TOTAL_TIMESTEPS_FOR_MAX = 11000 * N_ENVS * 400

    vec_env = create_vec_env()

    checkpoint_callback = CheckpointCallback(
        save_freq=(TOTAL_TIMESTEPS // N_ENVS) // 12,
        save_path=str(CHECKPOINT_DIR / "ff"),
        name_prefix="ppo_g1_ff",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=400,
        learning_rate=3e-4,
        batch_size=64,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        max_grad_norm=0.5,
        tensorboard_log=str(LOG_DIR / "ff"),
        device='cuda',
    )

    curriculum_callback = CurriculumCallback(TOTAL_TIMESTEPS_FOR_MAX)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[curriculum_callback, checkpoint_callback],
        progress_bar=True,
    )

    # 保存最终模型
    model.save(str(CHECKPOINT_DIR / "ff" / "ppo_g1_ff_final.zip"))
    vec_env.save(str(CHECKPOINT_DIR / "ff" / "vec_normalize_ff_final.pkl"))
    print("✅ 前馈网络训练完成！")
    return model, vec_env

# ==================== 任务2：LSTM网络 ====================
def train_lstm():
    print("\n" + "="*60)
    print("开始训练：LSTM网络 (RecurrentPPO + MultiInputLstmPolicy)")
    print("="*60)

    ITERATION = 1500
    TOTAL_TIMESTEPS = ITERATION * N_ENVS * 800
    TOTAL_TIMESTEPS_FOR_MAX = 11000 * N_ENVS * 400

    vec_env = create_vec_env()

    checkpoint_callback = CheckpointCallback(
        save_freq=(TOTAL_TIMESTEPS // N_ENVS) // 12,
        save_path=str(CHECKPOINT_DIR / "lstm"),
        name_prefix="ppo_g1_lstm",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
        activation_fn=torch.nn.ReLU,
        lstm_hidden_size=64,
        n_lstm_layers=1,
        shared_lstm=False,
    )

    model = RecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=400,
        learning_rate=3e-4,
        batch_size=64,
        n_epochs=3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        max_grad_norm=0.5,
        tensorboard_log=str(LOG_DIR / "lstm"),
        device='cuda',
    )

    curriculum_callback = CurriculumCallback(TOTAL_TIMESTEPS_FOR_MAX)

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[curriculum_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save(str(CHECKPOINT_DIR / "lstm" / "ppo_g1_lstm_final.zip"))
    vec_env.save(str(CHECKPOINT_DIR / "lstm" / "vec_normalize_lstm_final.pkl"))
    print("✅ LSTM网络训练完成！")
    return model, vec_env

# ==================== 主入口 ====================
if __name__ == "__main__":
    # 先训练前馈网络
    train_ff()
    # 再训练LSTM网络
    train_lstm()
    print("\n" + "="*60)
    print("所有训练任务已完成！")
    print("模型保存在 checkpoints/ff 和 checkpoints/lstm 目录下。")
    print("="*60)