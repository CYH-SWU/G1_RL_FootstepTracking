#!/usr/bin/env python3
"""
Inspect SB3 model: policy, spaces, networks, log_std, optimizer.

Usage:
    uv run scripts/inspect_model.py

    uv run scripts/inspect_model.py --model pretrained_models/ppo_G1_FootstepTracking.zip
"""

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO


def inspect(model_path: str):
    path = Path(model_path)
    if not path.exists():
        print(f"Error: model file not found: {path}")
        return

    print(f"Loading: {path}")
    model = PPO.load(str(path), device="cpu")
    policy = model.policy

    print("\n" + "=" * 60)
    print(f"Policy class: {type(policy).__name__}")
    print("=" * 60)

    # Observation space
    obs = policy.observation_space
    print("\nObservation space:")
    if hasattr(obs, "spaces"):
        for k, v in obs.spaces.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"  shape={obs.shape}, dtype={obs.dtype}")

    # Action space
    act = policy.action_space
    print(f"\nAction space: shape={act.shape}, dtype={act.dtype}")

    # Actor network
    print("\n--- Actor (action_net) ---")
    if hasattr(policy, "action_net"):
        print(policy.action_net)
        n = sum(p.numel() for p in policy.action_net.parameters())
        print(f"  Params: {n:,}")
    else:
        print("  (absent)")

    # Critic network
    print("\n--- Critic (value_net) ---")
    if hasattr(policy, "value_net"):
        print(policy.value_net)
        n = sum(p.numel() for p in policy.value_net.parameters())
        print(f"  Params: {n:,}")
    else:
        print("  (absent)")

    # Optional components
    if hasattr(policy, "value_baseline"):
        print(f"\nvalue_baseline: {policy.value_baseline.item():.4f}")

    if hasattr(policy, "log_std"):
        log_std = policy.log_std.detach().numpy()
        std = torch.exp(policy.log_std).detach().numpy()
        print(f"\nlog_std: {log_std}")
        print(f"std: {std}")

    if hasattr(policy, "features_extractor"):
        fe = policy.features_extractor
        print(f"\nFeatures extractor: {type(fe).__name__}")
        if hasattr(fe, "features_dim"):
            print(f"  Output dim: {fe.features_dim}")

    # AsymmetricPolicy specific
    if hasattr(policy, "actor_flatten"):
        print(f"\nactor_flatten: {type(policy.actor_flatten).__name__}, dim={policy.actor_flatten.features_dim}")
    if hasattr(policy, "critic_flatten"):
        print(f"\ncritic_flatten: {type(policy.critic_flatten).__name__}, dim={policy.critic_flatten.features_dim}")

    # Optimizer
    if hasattr(policy, "optimizer"):
        opt = policy.optimizer
        print(f"\nOptimizer: {type(opt).__name__}")
        print(f"  lr: {opt.param_groups[0]['lr']}")
        if "weight_decay" in opt.param_groups[0]:
            print(f"  weight_decay: {opt.param_groups[0]['weight_decay']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect SB3 model architecture and parameters.")
    parser.add_argument(
        "--model",
        type=str,
        default="pretrained_models/ppo_G1_FootstepTracking.zip",
        help="Path to the model .zip file (default: pretrained_models/ppo_G1_FootstepTracking.zip)",
    )
    args = parser.parse_args()
    inspect(args.model)
