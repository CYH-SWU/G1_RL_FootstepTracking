import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor
from gymnasium import spaces
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

class LHWAsymmetricPolicy(ActorCriticPolicy):
    """
    非对称策略，适用于字典观测（actor_obs / critic_obs）。
    - Actor 和 Critic 使用不同的观测输入和独立的网络结构。
    - 网络结构与 LHW 默认规格完全一致：
        - 两层隐藏层，每层 256 个神经元，ReLU 激活。
        - 输出层无 tanh 约束（bounded=False）。
        - 正交初始化，输出层权重乘以 0.01。
        - 标准差固定为 0.223（不可学习）。
    """
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):
        # 提取观测空间各个键的维度
        assert isinstance(observation_space, spaces.Dict), "观测空间必须是 Dict"
        self.actor_obs_key = "actor_obs"
        self.critic_obs_key = "critic_obs"
        actor_shape = observation_space[self.actor_obs_key].shape
        critic_shape = observation_space[self.critic_obs_key].shape
        self.actor_features_dim = int(np.prod(actor_shape))
        self.critic_features_dim = int(np.prod(critic_shape))

        # 调用父类，传入空的 net_arch，禁用 SB3 默认的 MLP 构建
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],  # 不使用默认 MLP
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )

        # 创建独立的特征展平器
        self.actor_flatten = FlattenExtractor(observation_space.spaces[self.actor_obs_key])
        self.critic_flatten = FlattenExtractor(observation_space.spaces[self.critic_obs_key])

        # 构建 Actor 网络（均值输出，无激活）
        self.policy_net = nn.Sequential(
            nn.Linear(self.actor_features_dim, 256),
            activation_fn(),
            nn.Linear(256, 256),
            activation_fn(),
            nn.Linear(256, action_space.shape[0])   # 无激活函数，输出关节增量
        )

        # 构建 Critic 网络（价值输出）
        self.value_net = nn.Sequential(
            nn.Linear(self.critic_features_dim, 256),
            activation_fn(),
            nn.Linear(256, 256),
            activation_fn(),
            nn.Linear(256, 1)                       # 输出标量价值
        )

        # 应用 LHW 风格初始化
        self._initialize_weights()

        # ★★★ 关键修正：重新创建优化器，使其包含自定义网络参数 ★★★
        # 同时覆盖父类可能使用的旧属性
        self.action_net = self.policy_net          # 覆盖 SB3 内部使用的 action_net
        self.mlp_extractor = None                  # 禁用父类的 mlp_extractor
        self._update_optimizer(lr_schedule)        # 重新创建优化器

    def _update_optimizer(self, lr_schedule):
        """创建包含当前网络参数的优化器"""
        params = list(self.policy_net.parameters()) + list(self.value_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr_schedule(1), eps=1e-5)

    def _initialize_weights(self):
        """正交初始化 + 输出层缩放 0.01"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.policy_net.apply(init_weights)
        self.value_net.apply(init_weights)

        # Actor 输出层额外缩放
        if isinstance(self.policy_net[-1], nn.Linear):
            nn.init.orthogonal_(self.policy_net[-1].weight, gain=0.01)
        # Critic 输出层额外缩放
        if isinstance(self.value_net[-1], nn.Linear):
            nn.init.orthogonal_(self.value_net[-1].weight, gain=0.01)

    def _get_features(self, obs: Dict[str, torch.Tensor], key: str, flatten):
        """提取指定键的观测并展平"""
        return flatten(obs[key])

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        """
        SB3 标准 forward，返回动作、价值、对数概率。
        """
        features_actor = self._get_features(obs, self.actor_obs_key, self.actor_flatten)
        features_critic = self._get_features(obs, self.critic_obs_key, self.critic_flatten)
        mean_actions = self.policy_net(features_actor)
        values = self.value_net(features_critic).flatten()
        log_std = torch.full_like(mean_actions, 0.223).log()
        dist = torch.distributions.Normal(mean_actions, log_std.exp())
        if deterministic:
            actions = mean_actions
        else:
            actions = dist.sample()
        log_prob = dist.log_prob(actions).sum(dim=-1)
        return actions, values, log_prob

    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """仅预测动作（用于确定性评估）"""
        features_actor = self._get_features(observation, self.actor_obs_key, self.actor_flatten)
        return self.policy_net(features_actor)

    def extract_features(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """提取 Actor 特征（SB3 接口要求）"""
        return self._get_features(obs, self.actor_obs_key, self.actor_flatten)

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor):
        """评估给定动作的对数概率、熵和价值"""
        features_actor = self._get_features(obs, self.actor_obs_key, self.actor_flatten)
        features_critic = self._get_features(obs, self.critic_obs_key, self.critic_flatten)

        mean_actions = self.policy_net(features_actor)
        log_std = torch.full_like(mean_actions, 0.223).log()   # 固定标准差
        dist = torch.distributions.Normal(mean_actions, log_std.exp())
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.value_net(features_critic).flatten()
        return values, log_prob, entropy

    def get_distribution(self, obs: Dict[str, torch.Tensor]):
        """返回动作分布（供 SB3 内部使用）"""
        features_actor = self._get_features(obs, self.actor_obs_key, self.actor_flatten)
        mean_actions = self.policy_net(features_actor)
        log_std = torch.full_like(mean_actions, 0.223).log()
        return torch.distributions.Normal(mean_actions, log_std.exp())

    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测状态价值"""
        features_critic = self._get_features(obs, self.critic_obs_key, self.critic_flatten)
        return self.value_net(features_critic).flatten()