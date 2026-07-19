from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor


class AsymmetricPolicy(ActorCriticPolicy):
    """
    Asymmetric Actor-Critic policy: Actor uses actor_obs, Critic uses critic_obs.
    Supports net_arch as list or dict (e.g., {'pi': [256, 256], 'vf': [256, 256]}).
    Activation function is configurable (default ReLU). Output layer is linear (unbounded),
    with learnable log_std for the action distribution.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        lr_schedule: Callable[[float], float],
        net_arch: list[int] | dict[str, list[int]] | None = [256, 256],
        activation_fn: type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):
        assert isinstance(observation_space, spaces.Dict), "Observation space must be Dict"
        self.actor_obs_key = "actor_obs"
        self.critic_obs_key = "critic_obs"
        actor_shape = observation_space[self.actor_obs_key].shape
        critic_shape = observation_space[self.critic_obs_key].shape
        self.actor_features_dim = int(np.prod(actor_shape))
        self.critic_features_dim = int(np.prod(critic_shape))

        # Parse net_arch: separate architectures for policy and value.
        if isinstance(net_arch, dict):
            actor_arch = net_arch.get("pi", [256, 256])
            critic_arch = net_arch.get("vf", [256, 256])
        else:
            actor_arch = net_arch
            critic_arch = net_arch
        if not isinstance(actor_arch, list):
            actor_arch = [actor_arch]
        if not isinstance(critic_arch, list):
            critic_arch = [critic_arch]

        # Call parent with empty net_arch to prevent default network creation.
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )

        # Feature flatteners for each observation stream.
        self.actor_flatten = FlattenExtractor(observation_space.spaces[self.actor_obs_key])
        self.critic_flatten = FlattenExtractor(observation_space.spaces[self.critic_obs_key])

        # Build Actor network (outputs mean actions, no tanh).
        layers = []
        prev_dim = self.actor_features_dim
        for h_dim in actor_arch:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation_fn())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, action_space.shape[0]))
        self.action_net = nn.Sequential(*layers)

        # Build Critic network.
        layers_c = []
        prev_dim = self.critic_features_dim
        for h_dim in critic_arch:
            layers_c.append(nn.Linear(prev_dim, h_dim))
            layers_c.append(activation_fn())
            prev_dim = h_dim
        layers_c.append(nn.Linear(prev_dim, 1))
        self.value_net = nn.Sequential(*layers_c)

        # Learnable log_std for Gaussian actions.
        self.log_std = nn.Parameter(torch.zeros(action_space.shape[0]))

        # Initialize weights with orthogonal initialization.
        self._initialize_weights()

        # Disable parent's mlp_extractor.
        self.mlp_extractor = None

        # Recreate optimizer to include all custom parameters.
        self.optimizer = torch.optim.Adam(
            list(self.action_net.parameters()) + list(self.value_net.parameters()) + [self.log_std],
            lr=lr_schedule(1),
            eps=1e-5,
        )

    def _initialize_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.action_net.apply(init_weights)
        self.value_net.apply(init_weights)
        # Scale output layer for smaller initial actions.
        if isinstance(self.action_net[-1], nn.Linear):
            nn.init.orthogonal_(self.action_net[-1].weight, gain=0.01)

    def _get_features(self, obs, key, flatten):
        return flatten(obs[key])

    def forward(self, obs, deterministic=False):
        features_actor = self._get_features(obs, self.actor_obs_key, self.actor_flatten)
        features_critic = self._get_features(obs, self.critic_obs_key, self.critic_flatten)
        mean_actions = self.action_net(features_actor)
        values = self.value_net(features_critic).flatten()
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean_actions, std)
        if deterministic:
            actions = mean_actions
        else:
            actions = dist.sample()
        log_prob = dist.log_prob(actions).sum(dim=-1)
        return actions, values, log_prob

    def _predict(self, observation, deterministic=False):
        features_actor = self._get_features(observation, self.actor_obs_key, self.actor_flatten)
        return self.action_net(features_actor)

    def extract_features(self, obs):
        return self._get_features(obs, self.actor_obs_key, self.actor_flatten)

    def evaluate_actions(self, obs, actions):
        features_actor = self._get_features(obs, self.actor_obs_key, self.actor_flatten)
        features_critic = self._get_features(obs, self.critic_obs_key, self.critic_flatten)
        mean_actions = self.action_net(features_actor)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean_actions, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.value_net(features_critic).flatten()
        return values, log_prob, entropy

    def get_distribution(self, obs):
        features_actor = self._get_features(obs, self.actor_obs_key, self.actor_flatten)
        mean_actions = self.action_net(features_actor)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean_actions, std)

    def predict_values(self, obs):
        features_critic = self._get_features(obs, self.critic_obs_key, self.critic_flatten)
        return self.value_net(features_critic).flatten()


policy_kwargs = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    activation_fn=torch.nn.ReLU,
)
