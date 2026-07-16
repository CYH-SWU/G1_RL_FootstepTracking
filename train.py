#!/usr/bin/env python3
"""
G1 humanoid robot footstep tracking training script.

Training iterations are specified by --iterations (default 11000).
All model saving and evaluation callbacks are based on iteration count:
- --save-interval: save a checkpoint every N iterations (CheckpointCallback)
- --eval-interval: evaluate and save the best model every N iterations (EvalCallback)

Underlying SB3 uses total timesteps as the time unit, but this script automatically
converts iteration intervals to timesteps:
- Total timesteps = iterations * n_steps * n_envs
- Per-env steps (save_freq) = save_interval * n_steps
- Evaluation timesteps (eval_freq) = eval_interval * n_steps

Usage:
  python train.py                                    # Default training (11000 iterations)
  python train.py -i 5000 -s 100 -e 200              # Custom iterations, save interval, eval interval
  python train.py --lr 3e-4 --n-steps 512            # Adjust PPO hyperparameters
  python train.py --lr-patience 10 --lr-factor 0.9   # Adjust learning rate callback parameters
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from env_utils.mirrorwrapper import MirrorWrapper
from env.g1_env import G1Env
from rl.callbacks import AdaptiveLRScheduleCallback, CurriculumCallback
from rl.policy import policy_kwargs

# Project paths
project_root = Path(__file__).parent.absolute()
ROBOT_XML = project_root / "robot" / "g1_processed.xml"
CHECKPOINT_DIR = project_root / "checkpoints"
LOG_DIR = project_root / "logs"
CHECKPOINT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Environment factory 
def make_env():
    env = G1Env(robot_xml_path=str(ROBOT_XML))
    env = MirrorWrapper(env, mirror_prob=0.5)
    return Monitor(env)

def create_vec_env(n_envs: int):
    vec_env = make_vec_env(
        make_env,
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"} if sys.platform != "win32" else {}
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

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="G1 RL training script")
    
    # Iterations (default 11000)
    parser.add_argument(
        "--iterations", "-i", type=int, default=11000,
        help="Total number of training iterations"
    )
    
    # Save interval in iterations
    parser.add_argument(
        "--save-interval", type=int, default=500,
        help="Iteration interval for saving model checkpoints"
    )
    
    # Evaluation interval in iterations
    parser.add_argument(
        "--eval-interval", type=int, default=500,
        help="Iteration interval for evaluating and saving the best model"
    )
    
    # PPO training parameters
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=800, help="Steps per environment per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--n-epochs", type=int, default=3, help="Number of update epochs per rollout")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE smoothing parameter")
    parser.add_argument("--clip-range", type=float, default=0.15, help="PPO clipping range")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping threshold")
    
    # Learning rate callback parameters
    parser.add_argument("--lr-patience", type=int, default=5, help="Patience for performance plateau")
    parser.add_argument("--lr-factor", type=float, default=0.95, help="Learning rate decay factor")
    parser.add_argument("--lr-min", type=float, default=1e-7, help="Minimum learning rate")
    parser.add_argument("--lr-eval-freq", type=int, default=None,
                        help="Evaluation frequency for LR callback (in timesteps)")
    
    # Number of parallel environments
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel environments")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Fixed parameters for curriculum learning.
    TOTAL_TIMESTEPS_FOR_MAX = 11000 * args.n_envs * args.n_steps
    
    # Create vectorized environment.
    vec_env = create_vec_env(args.n_envs)
    
    # Steps per iteration (total across all envs).
    steps_per_iter = args.n_steps * args.n_envs

    # Total timesteps for SB3 learn.
    total_timesteps = args.iterations * steps_per_iter
    
    # -------------------- Callback setup --------------------
    callbacks = []
    
    # Curriculum callback.
    callbacks.append(CurriculumCallback(TOTAL_TIMESTEPS_FOR_MAX))
    
    # Adaptive learning rate callback.
    lr_eval_freq = args.lr_eval_freq if args.lr_eval_freq is not None else (16 * steps_per_iter)
    lr_callback = AdaptiveLRScheduleCallback(
        patience=args.lr_patience,
        factor=args.lr_factor,
        eval_freq=lr_eval_freq,
        min_lr=args.lr_min,
        verbose=1
    )
    callbacks.append(lr_callback)
    
    # Best model saving (EvalCallback) based on iteration interval.
    eval_freq_steps = args.eval_interval * args.n_steps
    eval_env = make_vec_env(
        make_env,
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"} if sys.platform != "win32" else {}
    )
    eval_env = VecNormalize(
        venv=eval_env,
        norm_obs=True,
        norm_obs_keys=["actor_obs"],
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )
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
    
    # Periodic model checkpoint (CheckpointCallback).
    save_freq = args.save_interval * args.n_steps
    if save_freq < 1:
        raise ValueError(f"Computed save_freq={save_freq} (per-env steps) is < 1. \
                         Increase --save-interval or adjust --n-steps.")
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="ppo_g1",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    print(f"Periodic checkpoint saving enabled (CheckpointCallback), "
          f"saving every {args.save_interval} iterations (i.e., every {save_freq * args.n_envs:,} timesteps)")
    
    # -------------------- Create model --------------------
    model = PPO(
        policy="MultiInputPolicy",
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
    
    # -------------------- Training --------------------
    print(f"\nStarting training")
    print(f"  Total iterations: {args.iterations}")
    print(f"  Parallel environments: {args.n_envs}")
    print(f"  Steps per environment per rollout: {args.n_steps}")
    print(f"  Total training timesteps: {total_timesteps:,}")
    print(f"  Each iteration = {steps_per_iter:,} total timesteps")
    print(f"  Learning rate: {args.lr}")
    print(f"  PPO clip range: {args.clip_range}")
    print(f"  Entropy coefficient: {args.ent_coef}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model 
    final_model_path = CHECKPOINT_DIR / "ppo_g1_final.zip"
    model.save(str(final_model_path))
    vec_env.save(str(CHECKPOINT_DIR / "vec_normalize_final.pkl"))
    print(f"\nTraining completed! Final model saved to: {final_model_path}")
    print(f"Normalization parameters saved to: {CHECKPOINT_DIR / 'vec_normalize_final.pkl'}")

if __name__ == "__main__":
    main()