#!/usr/bin/env python3
"""
人形机器人复杂地形行走训练脚本
使用 Stable-Baselines3 PPO + 非对称 Actor-Critic (MultiInputPolicy)
支持手动指定检查点恢复，确保训练可中断续跑。
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

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from HumanoidRLTerrain.env.old import G1TerrainEnv

# -------------------- 配置参数 --------------------
ROBOT_XML = project_root / "robot" / "g1_processed.xml"
MESH_DIR = project_root / "robot" / "assets"
CHECKPOINT_DIR = project_root / "checkpoints"
LOG_DIR = project_root / "logs"

CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

N_ENVS = 8
TOTAL_TIMESTEPS = 6 * 200 * 1500           # 总训练步数（约180万）
MAX_EPISODE_STEPS = 2000
TOTAL_TIMESTEPS_FOR_MAX = 11000 * 1500     # 课程学习最大难度步数


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
    save_freq=(TOTAL_TIMESTEPS / N_ENVS) / 1,  # 每约25%总步数保存一次
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ppo_g1",
    save_replay_buffer=False,
    save_vecnormalize=True,
)

# -------------------- 检查点恢复（完全手动指定） --------------------
model = None
ckpt_zip = CHECKPOINT_DIR / f"ppo_g1_4.zip"
ckpt_norm = CHECKPOINT_DIR / f"vec_4.pkl"
if ckpt_zip.exists() and ckpt_norm.exists():
    print(f"✅ 检测到检查点，正在恢复模型: {ckpt_zip}")
    print(f"✅ 恢复归一化统计量: {ckpt_norm}")
    # 加载模型（传入当前 env，用于继续训练）
    model = PPO.load(ckpt_zip, env=vec_env)
    # 加载 VecNormalize 统计量（替换原有的归一化包装）
    vec_env = VecNormalize.load(ckpt_norm, vec_env)
    print(f"✅ 模型已训练步数: {model.num_timesteps}")
else:
    print(f"⚠️ 未找到检查点文件（{ckpt_zip} 或 {ckpt_norm}），将从头开始训练。")


# -------------------- 创建模型（如果未恢复） --------------------
if model is None:
    print("🚀 创建新模型...")
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 128],
            vf=[256, 128]
        ),
        activation_fn=torch.nn.ReLU,
    )
    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=500,
        tensorboard_log=str(LOG_DIR),
        device='cpu',
        ent_coef=0.01,
    )

# -------------------- 训练 --------------------
print("="*60)
print("开始训练（非对称 Actor-Critic）...")
print(f"目标总步数: {TOTAL_TIMESTEPS}")
print(f"并行环境数: {N_ENVS}")
print(f"当前已训练步数: {model.num_timesteps}")
print("="*60)

curriculum_callback = CurriculumCallback(TOTAL_TIMESTEPS_FOR_MAX)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[curriculum_callback, checkpoint_callback],
    progress_bar=True,
)

print("=============正在保存最终模型=============")
model.save(str(CHECKPOINT_DIR / "ppo_g1_final.zip"))
vec_env.save(str(CHECKPOINT_DIR / "vec_normalize_final.pkl"))
print("✅ 训练完成！模型和归一化参数已保存。")