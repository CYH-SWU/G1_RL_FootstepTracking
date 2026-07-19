import numpy as np
import torch
from gymnasium import spaces

from rl.policy import AsymmetricPolicy


def test_policy_forward():
    obs_space = spaces.Dict(
        {
            "actor_obs": spaces.Box(low=-1, high=1, shape=(41,), dtype=np.float32),
            "critic_obs": spaces.Box(low=-1, high=1, shape=(58,), dtype=np.float32),
        }
    )
    action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)

    policy = AsymmetricPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda x: 1e-4,
        net_arch=[64, 64],
    )

    # Convert inputs to tensors (required for torch.nn.Flatten)
    dummy_obs = {
        "actor_obs": torch.from_numpy(np.random.randn(1, 41).astype(np.float32)),
        "critic_obs": torch.from_numpy(np.random.randn(1, 58).astype(np.float32)),
    }

    actions, values, log_prob = policy.forward(dummy_obs)
    assert actions.shape == (1, 12)
    assert values.shape == (1,)
    assert log_prob.shape == (1,)

    values = policy.predict_values(dummy_obs)
    assert values.shape == (1,)


def test_policy_device():
    # Use CUDA if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_space = spaces.Dict(
        {
            "actor_obs": spaces.Box(low=-1, high=1, shape=(41,), dtype=np.float32),
            "critic_obs": spaces.Box(low=-1, high=1, shape=(58,), dtype=np.float32),
        }
    )
    action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)

    policy = AsymmetricPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda x: 1e-4,
        net_arch=[64, 64],
    )
    policy.to(device)

    param_device = next(policy.parameters()).device
    assert param_device.type == device.type
