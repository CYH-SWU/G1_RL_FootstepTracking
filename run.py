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


class AdaptiveLRScheduleCallback(BaseCallback):
    """
    自适应学习率回调：当性能停滞时降低学习率。
    
    :param patience: 允许性能无提升的评估次数（步数 / eval_freq）
    :param factor: 学习率衰减因子（如 0.5 表示减半）
    :param eval_freq: 评估间隔（步数）
    :param min_lr: 最小学习率，避免降得过低
    :param verbose: 是否打印日志
    """
    def __init__(self, patience: int = 5, factor: float = 0.5, 
                 eval_freq: int = 1000, min_lr: float = 1e-7, verbose: int = 1):
        super().__init__(verbose)
        self.patience = patience
        self.factor = factor
        self.eval_freq = eval_freq
        self.min_lr = min_lr

        self.best_mean_reward = -np.inf
        self.wait = 0  # 连续无改进的次数
        self.current_lr = None  # 会在 _on_training_start 中初始化

    def _on_training_start(self) -> None:
        """训练开始时初始化当前学习率"""
        # 获取当前学习率（如果是调度函数，取初始值）
        if callable(self.model.learning_rate):
            # 取 progress_remaining=1 时的值（初始值）
            self.current_lr = self.model.learning_rate(1.0)
        else:
            self.current_lr = self.model.learning_rate

    def _on_step(self) -> bool:
        """每步调用，但只在达到评估间隔时执行检查"""
        # 只在达到评估间隔时执行（num_timesteps 是全局总步数）
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            # 获取最近的平均 episode 奖励
            mean_reward = self._get_mean_reward()
            if mean_reward is None:
                return True  # 没有足够的 episode 数据，跳过

            # 检查是否改进
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.wait = 0
                if self.verbose > 0:
                    print(f"[{self.num_timesteps}] 性能提升: {mean_reward:.2f} (最佳 {self.best_mean_reward:.2f})")
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    # 性能停滞 → 降低学习率
                    new_lr = max(self.current_lr * self.factor, self.min_lr)
                    if new_lr < self.current_lr:
                        self.current_lr = new_lr
                        # ★★★ 关键：修改模型的学习率 ★★★
                        if callable(self.model.learning_rate):
                            # 如果是调度函数，替换为一个返回新固定值的函数（或继续调度但重置）
                            # 简单方法：直接用固定学习率（也可继续调度但调整初始值）
                            self.model.learning_rate = lambda _: self.current_lr
                        else:
                            self.model.learning_rate = self.current_lr
                        # 必须调用此方法使优化器更新
                        self.model._setup_lr_schedule()
                        self.wait = 0
                        if self.verbose > 0:
                            print(f"[{self.num_timesteps}] 性能停滞，降低学习率至 {self.current_lr:.2e}")
                    else:
                        if self.verbose > 0:
                            print(f"[{self.num_timesteps}] 学习率已到下限 {self.min_lr:.2e}，不再降低")
        return True

    def _get_mean_reward(self) -> Optional[float]:
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            recent = min(10, len(self.model.ep_info_buffer))
            # 转换为 list 后切片
            rewards = [ep_info['r'] for ep_info in list(self.model.ep_info_buffer)[-recent:]]
            return float(np.mean(rewards))
        return None

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

ITERATION = 1500 * 1.5
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


adaptive_lr_callback = AdaptiveLRScheduleCallback(
    patience=5,          # 连续 5 次评估无改进则降学习率
    factor=0.9,          # 减半
    eval_freq=N_ENVS * 400 * 16,      # 每 2000 步检查一次
    min_lr=1e-7,         # 最低学习率
    verbose=1
)

model = PPO(
    policy="MultiInputPolicy",
    env=vec_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    # --- 采样参数 ---
    n_steps=400,                     # 每个环境每次更新采集的步数（与 LHW 的 max_traj_len=400 对齐）
    # --- 优化参数 ---
    learning_rate=1e-4,              # 与 LHW 默认 lr 一致
    batch_size=64,                   # 与 LHW 默认 minibatch_size 一致
    n_epochs=3,                      # 与 LHW 默认 epochs 一致
    # --- 折扣与优势估计 ---
    gamma=0.99,                      # 与 LHW 默认 gamma 一致
    gae_lambda=0.95,                 # 与 LHW 默认 lam 一致
    # --- PPO 裁剪 ---
    clip_range=0.2,                  # 与 LHW 默认 clip 一致
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