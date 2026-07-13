#!/usr/bin/env python3
"""
人形机器人复杂地形行走训练脚本
使用 Stable-Baselines3 PPO + 非对称 Actor-Critic (MultiInputPolicy)
采用 SubprocVecEnv 多进程并行训练，并堆叠 3 帧观测以提供历史信息。
"""

import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from env.mirrorwrapper import MirrorWrapper

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from env.g1_test import G1TerrainEnv

# -------------------- 配置参数 --------------------
ROBOT_XML = project_root / "robot" / "g1_processed.xml"
MESH_DIR = project_root / "robot" / "assets"
CHECKPOINT_DIR = project_root / "checkpoints"
LOG_DIR = project_root / "logs"

CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

ITERATION = 1500 * 2
N_ENVS = 16   
                       
TOTAL_TIMESTEPS = ITERATION * N_ENVS * 400
TOTAL_TIMESTEPS_FOR_MAX = 11000 * N_ENVS * 400

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

vec_env = VecFrameStack(vec_env, n_stack=2)

# ★ 2. 再进行观测归一化（只归一化 "actor_obs"）
vec_env = VecNormalize(
    venv=vec_env,
    norm_obs=True,
    norm_obs_keys=["actor_obs"],   # VecFrameStack 后 actor_obs 维度自动扩展，VecNormalize 会适应
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
    save_freq=(TOTAL_TIMESTEPS // N_ENVS) // 12 ,  # 调整保存频率
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ppo_g1",
    save_replay_buffer=False,
    save_vecnormalize=True,
)

# -------------------- 创建非对称策略 --------------------
policy_kwargs = dict(
    net_arch=dict(
        pi=[256, 256],
        vf=[256, 256]
    ),
    activation_fn=torch.nn.ReLU,
)


model = PPO(
    policy="MultiInputPolicy",
    env=vec_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    # --- 采样参数 ---
    n_steps=400,                     # 每个环境每次更新采集的步数（与 LHW 的 max_traj_len=400 对齐）
    # --- 优化参数 ---
    learning_rate=3e-4,              # 与 LHW 默认 lr 一致
    batch_size=64,                   # 与 LHW 默认 minibatch_size 一致
    n_epochs=3,                      # 与 LHW 默认 epochs 一致
    # --- 折扣与优势估计 ---
    gamma=0.99,                      # 与 LHW 默认 gamma 一致
    gae_lambda=0.95,                 # 与 LHW 默认 lam 一致
    # --- PPO 裁剪 ---
    clip_range=0.2,                  # 与 LHW 默认 clip 一致
    # --- 熵与探索 ---
    ent_coef=0.001,                  # 适度鼓励探索
    max_grad_norm=0.5,               # 与 LHW 默认 grad norm 一致（SB3 默认也是 0.5）
    tensorboard_log=str(LOG_DIR),
    device='cuda',
)

# -------------------- 训练主入口（必须包含 if __name__） --------------------
if __name__ == "__main__":
    print("开始训练（非对称 Actor-Critic + SubprocVecEnv + 帧堆叠）...")
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