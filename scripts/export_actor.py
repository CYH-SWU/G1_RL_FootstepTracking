"""
Usage:
    python scripts/export_actor.py \
        --model xxx \
        --out deploy/actor.pt \
        --obs-dim 41

    python scripts/export_actor.py \
        --model xxx \
        --vecnorm xxx \
        --out deploy/actor.pt \
        --obs-dim 41
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO


def build_actor_module(policy):
    has_actor_flatten = hasattr(policy, "actor_flatten")
    mlp_is_none = getattr(policy, "mlp_extractor", "missing") is None

    if has_actor_flatten and mlp_is_none:
        print("[export] Detected AsymmetricPolicy (actor_flatten + action_net)")

        class AsymmetricActor(nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.actor_flatten = policy.actor_flatten
                self.action_net = policy.action_net

            def forward(self, obs):
                features = self.actor_flatten(obs)
                return self.action_net(features)

        return AsymmetricActor(policy)

    print("[export] Detected standard ActorCriticPolicy")
    has_features_extractor = hasattr(policy, "features_extractor")
    has_mlp_extractor = hasattr(policy, "mlp_extractor") and policy.mlp_extractor is not None

    if not (has_features_extractor and has_mlp_extractor):
        raise RuntimeError("no actor need actor_flatten+action_net，or features_extractor+mlp_extractor+action_net。")

    class StandardActor(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.features_extractor = policy.features_extractor
            self.mlp_extractor = policy.mlp_extractor
            self.action_net = policy.action_net

        def forward(self, obs):
            features = self.features_extractor(obs)
            latent_pi = self.mlp_extractor.forward_actor(features)
            return self.action_net(latent_pi)

    return StandardActor(policy)


def wrap_with_vecnorm(actor, mean, var, eps=1e-8, clip=10.0):
    class ActorWithNorm(nn.Module):
        def __init__(self, actor, mean, var, eps, clip):
            super().__init__()
            self.actor = actor
            self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("var", torch.tensor(var, dtype=torch.float32))
            self.eps = eps
            self.clip = clip

        def forward(self, obs):
            norm_obs = (obs - self.mean) / torch.sqrt(self.var + self.eps)
            norm_obs = torch.clamp(norm_obs, -self.clip, self.clip)
            return self.actor(norm_obs)

    return ActorWithNorm(actor, mean, var, eps, clip)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="SB3 .zip")
    parser.add_argument("--vecnorm", default=None, help="VecNormalize .pkl")
    parser.add_argument("--out", default="deploy/actor.pt", help="out route")
    parser.add_argument("--obs-dim", type=int, default=41, help="actor obs")
    args = parser.parse_args()

    print(f"[export] Loading SB3 model: {args.model}")
    model = PPO.load(args.model, device="cpu")
    policy = model.policy
    print(f"[export] Policy class: {type(policy).__name__}")
    print(f"[export] Obs space: {policy.observation_space}")
    print(f"[export] Action space: {policy.action_space}")

    actor = build_actor_module(policy).eval()

    if args.vecnorm:
        print(f"[export] Loading VecNormalize: {args.vecnorm}")
        with open(args.vecnorm, "rb") as f:
            vec = pickle.load(f)
        mean = np.asarray(vec.obs_rms.mean, dtype=np.float32)
        var = np.asarray(vec.obs_rms.var, dtype=np.float32)
        print(f"[export] VecNormalize mean shape: {mean.shape}")
        print(f"[export] VecNormalize var shape: {var.shape}")
        actor = wrap_with_vecnorm(actor, mean, var).eval()

    dummy = torch.zeros(1, args.obs_dim, dtype=torch.float32)
    with torch.no_grad():
        out = actor(dummy)
    print(f"[export] Dummy output shape: {out.shape}")

    if out.shape[-1] != policy.action_space.shape[0]:
        raise RuntimeError(f"dim {out.shape[-1]} and action_space {policy.action_space.shape[0]} mismatch!")

    traced = torch.jit.trace(actor, dummy)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    print(f"[export] Saved TorchScript actor: {out_path}")

    reloaded = torch.jit.load(str(out_path))
    with torch.no_grad():
        out2 = reloaded(dummy)
    diff = float(torch.max(torch.abs(out - out2)).item())
    print(f"[export] Reload check max diff: {diff:.2e}")
    assert diff < 1e-5, "TorchScript reload mismatch"

    print()
    print("[export] Verifying against SB3 policy._predict ...")
    n_check = 5
    max_diff_sb3 = 0.0
    for i in range(n_check):
        test_obs = torch.randn(1, args.obs_dim, dtype=torch.float32)
        # TorchScript actor
        with torch.no_grad():
            ts_out = actor(test_obs).squeeze(0).numpy()
        # SB3 policy._predict
        with torch.no_grad():
            sb3_out = policy._predict(test_obs.squeeze(0).numpy(), deterministic=True)
        d = float(np.max(np.abs(ts_out - sb3_out)))
        max_diff_sb3 = max(max_diff_sb3, d)
        if i < 2:
            print(f"  sample {i}: ts={ts_out[:4].round(4)}  sb3={sb3_out[:4].round(4)}  diff={d:.2e}")

    print(f"[export] Max diff vs SB3 _predict: {max_diff_sb3:.2e}")
    if max_diff_sb3 > 1e-4:
        print("[export] WARNING: out is different with sb3 largely,check VecNormalize")
    else:
        print("[export] OK")


if __name__ == "__main__":
    main()
