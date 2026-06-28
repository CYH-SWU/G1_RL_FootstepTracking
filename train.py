#!/usr/bin/env python3
"""
人形机器人复杂地形行走训练脚本
使用 Stable-Baselines3 PPO 算法，支持 DummyVecEnv 并行训练，
带 VecNormalize 观测归一化和课程学习。
"""

import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from policies.sb3_lhw_policy import LHWPolicy

# 将项目根目录加入 Python 路径（确保能导入自定义模块）
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from env.LHW_env import G1TerrainEnv

# -------------------- 配置参数 --------------------
ROBOT_XML = project_root / "robot" / "g1_processed.xml"
MESH_DIR = project_root / "robot" / "assets"
CHECKPOINT_DIR = project_root / "checkpoints"
LOG_DIR = project_root / "logs"

# 创建必要的目录
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# 训练超参数（仅保留环境相关，PPO算法参数使用默认值）
N_ENVS = 8                    # 并行环境数量
TOTAL_TIMESTEPS = 200 * 1500       # 总训练步数（可根据需要调整）
MAX_EPISODE_STEPS = 2000       # 单回合最大步数

# 课程学习：达到最大难度所需的总步数（通常与总步数一致）
TOTAL_TIMESTEPS_FOR_MAX = 11000 * 1500

# -------------------- 环境创建 --------------------
def make_env():
    """工厂函数：创建单个 G1 环境实例"""
    env = G1TerrainEnv(
        robot_xml_path=str(ROBOT_XML),
        mesh_dir=str(MESH_DIR),
        max_episode_steps=MAX_EPISODE_STEPS,
        total_timesteps_for_max=TOTAL_TIMESTEPS_FOR_MAX,
    )
    return Monitor(env)

# 创建并行环境（DummyVecEnv）
vec_env = make_vec_env(
    make_env,
    n_envs=N_ENVS,
    vec_env_cls=DummyVecEnv,
)

# 包装 VecNormalize（观测归一化）
vec_env = VecNormalize(
    venv=vec_env,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,
    gamma=0.99,  # 默认gamma也是0.99，但显式保留以避免歧义
)

# -------------------- 课程学习回调 --------------------
class CurriculumCallback(BaseCallback):
    """
    课程学习回调：根据当前总步数更新所有环境的难度因子。
    """
    def __init__(self, total_timesteps_for_max: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps_for_max = total_timesteps_for_max

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps_for_max)
        self.training_env.env_method("set_difficulty", progress)
        return True

# -------------------- 检查点回调 --------------------
save_freq = TOTAL_TIMESTEPS / 1
checkpoint_callback = CheckpointCallback(
    save_freq=save_freq/N_ENVS,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ppo_g1",
    save_replay_buffer=False,
    save_vecnormalize=True,
)

# -------------------- 创建模型（使用PPO默认参数，仅指定网络结构） --------------------


from stable_baselines3 import PPO
from policies.sb3_lhw_policy import LHWPolicy  # 你的自定义策略

policy_kwargs = dict(
    net_arch=dict(pi=[256, 128], vf=[256, 128]),
    activation_fn=torch.nn.ReLU,
)

model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=500,    
    tensorboard_log=str(LOG_DIR),
    device='cpu',
    ent_coef=0.05
)

# -------------------- 训练 --------------------
print("开始训练...")
print(f"总步数: {TOTAL_TIMESTEPS}, 并行环境数: {N_ENVS}")

# 课程回调
curriculum_callback = CurriculumCallback(TOTAL_TIMESTEPS_FOR_MAX)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[curriculum_callback, checkpoint_callback],
    progress_bar=True,
)

print("=============正在保存模型=============")
model.save(str(CHECKPOINT_DIR / "ppo_g1_final.zip"))
vec_env.save(str(CHECKPOINT_DIR / "vec_normalize_final.pkl"))