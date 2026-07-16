#!/usr/bin/env python3
"""
G1 footstep tracking model evaluation script.

Loads a trained model, runs visual evaluation in MuJoCo, and outputs statistics
including average reward and success rate.

Usage:
    python test.py                                  # Load best model, default 10 episodes
    python test.py --model path/to/model.zip        # Specify model path
    python test.py --norm path/to/norm.pkl          # Specify normalization file
    python test.py --episodes 20 --max-steps 3000   # Custom episodes and steps
    python test.py --difficulty 0.5                 # Set difficulty (0~1)
    python test.py --no-render                      # Disable rendering

Auto-loading:
    - Model: checkpoints/best_model/best_model.zip
    - Normalization: same directory as model or latest vec_normalize_*.pkl in checkpoints/
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Import project environment.
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from env_utils.mirrorwrapper import MirrorWrapper
from env.g1_env import G1Env

CHECKPOINT_DIR = project_root / "checkpoints"
ROBOT_XML = project_root / "robot" / "g1_processed.xml"


def find_best_model(checkpoint_dir: Path) -> Path:
    """Locate the best model (checkpoints/best_model/best_model.zip)."""
    best_path = checkpoint_dir / "best_model" / "best_model.zip"
    if best_path.exists():
        return best_path
    # Fallback: any best_model*.zip (backward compatibility).
    candidates = list(checkpoint_dir.glob("best_model*.zip"))
    if candidates:
        return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
    raise FileNotFoundError(f"Best model not found in {checkpoint_dir}")


def find_norm_file(checkpoint_dir: Path, model_path: Path = None) -> Path:
    """Automatically locate the normalization file."""
    if model_path is not None:
        model_dir = model_path.parent
        candidates = list(model_dir.glob("vec_normalize_*.pkl"))
        if candidates:
            return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
    candidates = list(checkpoint_dir.glob("vec_normalize_*.pkl")) + \
                 list(checkpoint_dir.glob("ppo_g1_*_vecnormalize.pkl"))
    if candidates:
        return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
    raise FileNotFoundError("Normalization file not found. Please specify with --norm.")


def create_eval_env(difficulty: float = 1.0):
    """Create evaluation environment (single env, no mirror augmentation)."""
    env = G1Env(robot_xml_path=str(ROBOT_XML))
    env.set_difficulty(difficulty)
    env = MirrorWrapper(env, mirror_prob=0.0)  # Disable mirroring for evaluation.
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser(description="G1 footstep tracking evaluation")
    parser.add_argument("--model", type=str, default=None,
                        help="Model file path (.zip). If not provided, auto-load best model.")
    parser.add_argument("--norm", type=str, default=None,
                        help="Normalization parameter file (.pkl). Auto-detected if omitted.")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Maximum steps per episode.")
    parser.add_argument("--difficulty", type=float, default=1.0,
                        help="Curriculum difficulty in [0,1].")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering (only output stats).")
    args = parser.parse_args()

    # Determine model path.
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
    else:
        model_path = find_best_model(CHECKPOINT_DIR)
        print(f"Auto-loaded best model: {model_path}")

    # Determine normalization file.
    if args.norm:
        norm_path = Path(args.norm)
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalization file not found: {norm_path}")
    else:
        norm_path = find_norm_file(CHECKPOINT_DIR, model_path)
        print(f"Auto-loaded normalization parameters: {norm_path}")

    # Create evaluation environment.
    raw_env = create_eval_env(args.difficulty)
    vec_env = DummyVecEnv([lambda: raw_env])

    # Load VecNormalize.
    vec_env = VecNormalize.load(str(norm_path), vec_env)
    vec_env.training = False  # Freeze statistics.
    vec_env.norm_obs = True   # Keep normalization active.

    # Load model.
    model = PPO.load(str(model_path))
    print("Model loaded successfully.")

    # Unwrap the environment to get the underlying G1Env for rendering and access.
    inner_env = raw_env
    while hasattr(inner_env, 'env'):
        inner_env = inner_env.env
    # Now inner_env is the G1Env instance.

    # Launch viewer if rendering is enabled.
    viewer = None
    if not args.no_render:
        viewer = mujoco.viewer.launch_passive(inner_env.model, inner_env.data)
        print("Press Esc or close the window to exit early.")

    # Evaluation loop.
    episode_rewards = []
    episode_lengths = []
    successes = 0

    for ep in range(args.episodes):
        obs = vec_env.reset()
        done = False
        truncated = False
        step = 0
        ep_reward = 0.0

        while not (done or truncated) and step < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += reward[0]
            step += 1

            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(inner_env.config.control_dt)  # Maintain real-time speed.
            elif viewer is not None and not viewer.is_running():
                break

            # Check if the full footstep sequence was successfully completed.
            if done and inner_env.t1 >= len(inner_env.sequence) - 1:
                successes += 1

        episode_rewards.append(ep_reward)
        episode_lengths.append(step)

        if viewer is not None and not viewer.is_running():
            break

        print(f"Episode {ep+1}/{args.episodes} completed: steps={step}, reward={ep_reward:.2f}")

    if viewer is not None:
        viewer.close()

    # Statistics output.
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Mean steps: {np.mean(episode_lengths):.1f}")
    print(f"Success rate (completed sequence): {successes}/{len(episode_rewards)} ({successes/len(episode_rewards)*100:.1f}%)")
    print(f"Difficulty: {args.difficulty}")
    print("="*50)

    raw_env.close()
    vec_env.close()


if __name__ == "__main__":
    main()