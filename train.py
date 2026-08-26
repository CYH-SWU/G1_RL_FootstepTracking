#!/usr/bin/env python3
"""
G1 humanoid robot footstep tracking training script.

Supports both fresh training and resuming from a checkpoint.
When resuming, both the model and the VecNormalize statistics must be loaded
to ensure consistent observation normalization.

Usage:
  uv run python train.py                                       # Fresh training (default 10000 iters)

  uv run python train.py -i 1100 -s 110 -e 20                  # Custom iterations, save/eval intervals

  uv run python train.py -i 7000 -s 700 -e 20 \
    --batch_size 64 \
    --ent_coef 0.0001 \
    --policy "multi" \
    --lr_callback "pl" \
    --no-domain-rand                                           # Baseline

  uv run python train.py \
    --model checkpoints/ppo_g1_xxx.zip \
    --norm checkpoints/vec_normalize_final.pkl                 # Resume from checkpoint

  uv run python train.py --lr 3e-4 --n_steps 512               # Adjust PPO hyperparameters

  uv run python train.py --no-domain-rand -i 1100              # No domain rand
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from env.g1_env import G1Env
from env_utils.mirrorwrapper import MirrorWrapper
from rl.callbacks import CurriculumCallback, KLAdaptiveLRCallback, PlateauLRCallback
from rl.policy import AsymmetricPolicy

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Project paths
project_root = Path(__file__).parent.absolute()
ROBOT_XML = project_root / "robot" / "g1_processed.xml"
CHECKPOINT_DIR = project_root / "checkpoints"
LOG_DIR = project_root / "logs"
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# Environment factory
def make_env(enable_domain_rand: bool = True):
    env = G1Env(robot_xml_path=str(ROBOT_XML), enable_domain_randomization=enable_domain_rand)
    env = MirrorWrapper(env, mirror_prob=0.5)
    return Monitor(env)


def create_vec_env(n_envs: int, norm_path: str = None, enable_domain_rand: bool = True):
    """Create vectorized environment with VecNormalize. If norm_path is provided, load stats."""
    vec_env = make_vec_env(
        lambda: make_env(enable_domain_rand),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "forkserver"} if sys.platform != "win32" else {},
    )
    vec_env = VecNormalize(
        venv=vec_env,
        norm_obs=True,
        norm_obs_keys=["actor_obs"],
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )
    if norm_path is not None and Path(norm_path).exists():
        vec_env = VecNormalize.load(str(norm_path), vec_env)
        vec_env.training = True  # Keep updating statistics during training
        print(f"Loaded VecNormalize stats from {norm_path}")
    return vec_env


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="G1 RL training script")

    # Iterations
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=10000,
        help="Total number of training iterations (including previous if resuming)",
    )

    # Save interval in iterations
    parser.add_argument(
        "--save_interval", "-s", type=int, default=1000, help="Iteration interval for saving model checkpoints"
    )

    # Evaluation interval in iterations
    parser.add_argument(
        "--eval_interval",
        "-e",
        type=int,
        default=20,
        help="Iteration interval for evaluating and saving the best model",
    )

    # Model checkpoint to resume from (optional)
    parser.add_argument(
        "--model", type=str, default=None, help="Path to a pre-trained model checkpoint to resume training from"
    )

    # NORM PARAMETER ADDED HERE
    parser.add_argument(
        "--norm", type=str, default=None, help="Path to VecNormalize statistics file (.pkl) to load when resuming"
    )

    # PPO training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=2048, help="Steps per environment per rollout")
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of update epochs per rollout")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE smoothing parameter")
    parser.add_argument("--clip_range", type=float, default=0.20, help="PPO clipping range")
    parser.add_argument("--ent_coef", type=float, default=0.0005, help="Entropy coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Gradient clipping threshold")

    # Number of parallel environments
    parser.add_argument("--n_envs", type=int, default=14, help="Number of parallel environments")

    parser.add_argument(
        "--policy",
        type=str,
        default="asym",
        choices=["multi", "asym"],
        help="Policy type: 'multi' for MultiInputPolicy (shared obs), 'asym' for AsymmetricPolicy (default)",
    )

    parser.add_argument(
        "--curriculum",
        action="store_true",
        default=False,
        help="Enable curriculum learning (increases step height over time)",
    )

    parser.add_argument(
        "--lr_callback",
        type=str,
        default="kl",
        choices=["pl", "kl"],
        help="LR callback: 'pl' for PlateauLRCallback, 'kl' for KLAdaptiveLRCallback (default)",
    )

    parser.add_argument(
        "--no-domain-rand",
        action="store_true",
        default=False,
        help="Disable domain randomization (mass, friction, sensor noise) during training",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create training environment (load normalization stats if provided)
    vec_env = create_vec_env(
        args.n_envs,
        args.norm,
        enable_domain_rand=not args.no_domain_rand,
    )

    # Steps per iteration (total across all envs).
    steps_per_iter = args.n_steps * args.n_envs

    # Total timesteps for the entire training (including already trained if resuming).
    total_timesteps = args.iterations * steps_per_iter

    # Callback setup
    callbacks = []

    # Curriculum callback.
    if args.curriculum:
        total_timesteps_for_max = 5000 * args.n_steps * args.n_envs
        curriculum_callback = CurriculumCallback(total_timesteps_for_max)
        callbacks.append(curriculum_callback)
        print("Curriculum learning enabled")
    else:
        print("Curriculum learning disabled")

    # Adaptive learning rate callback.
    if args.lr_callback == "pl":
        lr_callback = PlateauLRCallback(
            patience=3,
            factor=0.98,
            eval_freq=16 * args.n_steps * args.n_envs,
            min_lr=5e-6,
            verbose=1,
        )
        callbacks.append(lr_callback)
        print("Using PlateauLRCallback")
    else:  # "kl"
        kl_callback = KLAdaptiveLRCallback(
            target_kl=0.022,
            factor=0.02,
            min_lr=5e-6,
            max_lr=2e-4,
            verbose=0,
        )
        callbacks.append(kl_callback)
        print("Using KLAdaptiveLRCallback")

    # Evaluation environment (must share same normalization as training)
    eval_env = make_vec_env(
        lambda: make_env(enable_domain_rand=False),
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "forkserver"} if sys.platform != "win32" else {},
    )
    eval_env = VecNormalize(
        venv=eval_env,
        norm_obs=True,
        norm_obs_keys=["actor_obs"],
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )
    if args.norm is not None:
        norm_path = Path(args.norm)
        if norm_path.exists():
            eval_env = VecNormalize.load(str(norm_path), eval_env)
            eval_env.training = False  # Freeze statistics for evaluation
            print(f"Loaded VecNormalize stats for evaluation from {norm_path}")
        else:
            raise FileNotFoundError(f"Normalization file not found: {norm_path}")

    # Best model saving (EvalCallback)
    eval_freq_steps = args.eval_interval * args.n_steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(CHECKPOINT_DIR / "best_model"),
        log_path=str(LOG_DIR / "eval"),
        eval_freq=eval_freq_steps,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1,
    )
    callbacks.append(eval_callback)
    print(
        f"Best model saving enabled (EvalCallback), "
        f"evaluating every {args.eval_interval} iterations "
        f"(i.e., every {eval_freq_steps:,} timesteps)"
    )

    # Periodic model checkpoint (CheckpointCallback)
    save_freq = args.save_interval * args.n_steps
    if save_freq < 1:
        raise ValueError(
            f"Computed save_freq={save_freq} (per-env steps) is < 1. Increase --save_interval or adjust --n_steps."
        )
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_g1",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    print(
        f"Periodic checkpoint saving enabled (CheckpointCallback), "
        f"saving every {args.save_interval} iterations (i.e., every {save_freq * args.n_envs:,} timesteps)"
    )

    # Create or load model
    if args.model is not None:
        # Resume from checkpoint
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {model_path}")
        model = PPO.load(str(model_path), env=vec_env)
        print(f"Resumed from checkpoint: {model_path}")
        # Determine already trained timesteps.
        already_trained = model.num_timesteps
        remaining_timesteps = total_timesteps - already_trained
        if remaining_timesteps <= 0:
            print(f"Model already reached target timesteps ({total_timesteps}). Skipping training.")
            vec_env.save(str(CHECKPOINT_DIR / "vec_normalize_final.pkl"))
            return
        print(f"Already trained: {already_trained:,} timesteps")
        print(f"Remaining to train: {remaining_timesteps:,} timesteps")
        train_timesteps = remaining_timesteps
        reset_num = False
    else:
        if args.policy == "multi":
            policy = "MultiInputPolicy"
            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                activation_fn=torch.nn.ReLU,
            )
            print("Using MultiInputPolicy")
        else:  # "asym"
            policy = AsymmetricPolicy
            policy_kwargs = dict(
                net_arch=dict(pi=[512, 512], vf=[512, 512]),
                activation_fn=torch.nn.ReLU,
            )
            print("Using AsymmetricPolicy")

        model = PPO(
            policy=policy,
            env=vec_env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=args.n_steps,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=str(LOG_DIR),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        train_timesteps = total_timesteps
        reset_num = True
        print("Fresh training started.")

    # Training
    print("\nStarting training")
    print(f"  Total iterations (target): {args.iterations}")
    print(f"  Parallel environments: {args.n_envs}")
    print(f"  Steps per environment per rollout: {args.n_steps}")
    print(f"  Total training timesteps: {total_timesteps:,}")
    print(f"  Each iteration = {steps_per_iter:,} timesteps")
    print(f"  Learning rate: {args.lr}")
    print(f"  PPO clip range: {args.clip_range}")
    print(f"  Entropy coefficient: {args.ent_coef}")
    if args.model:
        print(f"  Resuming from previous checkpoint, training {train_timesteps:,} additional timesteps")
    print()

    if args.no_domain_rand:
        print("Domain randomization: DISABLED")
    else:
        print("Domain randomization: ENABLED (mass ±10%, friction ±20%, noise σ=0.01)")

    model.learn(
        total_timesteps=train_timesteps,
        reset_num_timesteps=reset_num,  # False when resuming to keep timestep counting continuous
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model (overwrite if resuming)
    final_model_path = CHECKPOINT_DIR / "ppo_g1_final.zip"
    model.save(str(final_model_path))
    vec_env.save(str(CHECKPOINT_DIR / "vec_normalize_final.pkl"))
    print(f"\nTraining completed! Final model saved to: {final_model_path}")
    print(f"Normalization parameters saved to: {CHECKPOINT_DIR / 'vec_normalize_final.pkl'}")


if __name__ == "__main__":
    main()
