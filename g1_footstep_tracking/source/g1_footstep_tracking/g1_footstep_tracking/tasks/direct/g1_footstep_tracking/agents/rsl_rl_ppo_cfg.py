# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO 训练配置（适配大规模并行环境）"""

    # ===== 训练步数配置 =====
    # 原项目：14 环境 × 2048 步 = 28672 步/迭代
    # 当前：4096 环境 × 24 步 ≈ 98304 步/迭代（总步数约为原项目的 3.4 倍）
    # 总目标步数 ~2.8 亿，迭代次数约为 2.8e8 / 98304 ≈ 2850 次
    num_steps_per_env = 400           # 每个环境每次迭代的步数
    max_iterations = 3000            # 总迭代次数（可根据需要调整）
    save_interval = 100              # 每 100 次迭代保存一次模型
    experiment_name = "g1_footstep_tracking_direct"

    # ===== Actor 网络配置 =====
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],   # 与 AsymmetricPolicy 的 [512,512] 相近，更深
        activation="elu",
        obs_normalization=False,        # 观测已在环境层归一化
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
        ),
    )

    # ===== Critic 网络配置 =====
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )

    # ===== PPO 算法参数 =====
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,                # 价值损失权重
        use_clipped_value_loss=True,        # 使用裁剪的价值损失
        clip_param=0.2,                     # PPO clip range
        entropy_coef=0.0005,                # 熵系数（原项目 0.0005）
        num_learning_epochs=3,              # 每次迭代更新轮数（原项目 3）
        num_mini_batches=256,               # mini-batch 数量（batch_size 由总步数 / mini_batches 决定）
        learning_rate=1.0e-4,               # 学习率（原项目 1e-4）
        schedule="adaptive",                # 自适应学习率调度（基于 KL 散度）
        gamma=0.99,                         # 折扣因子
        lam=0.95,                           # GAE lambda
        desired_kl=0.01,                    # 目标 KL 散度
        max_grad_norm=1.0,                  # 梯度裁剪阈值
    )
