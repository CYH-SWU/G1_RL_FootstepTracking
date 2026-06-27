import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution

# 导入 LHW 的网络类（根据你的实际路径调整）
from policies.actor import Gaussian_FF_Actor
from policies.critic import FF_V


class LHWPolicy(ActorCriticPolicy):
    """
    自定义策略类，使用 LHW 的 Gaussian_FF_Actor 和 FF_V 网络，
    完全兼容 SB3 的 PPO 训练流程。
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        layers=(256, 256),
        init_std=0.2,
        learn_std=True,
        bounded=False,
        **kwargs,
    ):
        # 必须调用父类初始化，传入必要的参数
        # 我们不需要 SB3 默认的 MLP extractor，所以将 net_arch 设为空
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=[],          # 不使用默认的 MLP 架构
            activation_fn=nn.ReLU, # 占位，实际由我们的网络控制
            **kwargs,
        )

        # 保存自定义参数
        self.layers = layers
        self.init_std = init_std
        self.learn_std = learn_std
        self.bounded = bounded

        # 构建 LHW 风格的网络
        self._build_net()

        # 设置观测归一化参数（外部 VecNormalize 已处理，内部不做归一化）
        self.actor.obs_mean = 0.0
        self.actor.obs_std = 1.0
        self.critic.obs_mean = 0.0
        self.critic.obs_std = 1.0

        # 重要：SB3 会通过 get_distribution 获取分布，需要保证该方法返回正确的分布
        # 我们已经重写了 get_distribution，见下文

    def _build_net(self):
        """构建 Actor 和 Critic 网络"""
        state_dim = self.observation_space.shape[0]
        action_dim = self.action_space.shape[0]

        # Actor: 高斯分布输出
        self.actor = Gaussian_FF_Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            layers=self.layers,
            init_std=self.init_std,
            learn_std=self.learn_std,
            bounded=self.bounded,
            normc_init=True,
        )

        # Critic: 价值输出
        self.critic = FF_V(
            state_dim=state_dim,
            layers=self.layers,
            normc_init=True,
        )

    def _get_action_dist_from_latent(self, latent_pi, latent_vf=None):
        """
        SB3 会调用此方法从潜在特征（latent）构建动作分布。
        由于我们绕过了 SB3 的 feature extractor，此方法不会被真正用到，
        但为了接口完整性，我们仍然实现它。
        注意：此方法返回的分布必须与 get_distribution 一致。
        """
        # 这里我们不应该直接使用 latent_pi，因为我们的 actor 期望接收原始观测。
        # 但在 SB3 的调用链中，get_distribution 会优先被调用，所以此方法通常不被调用。
        # 为了安全，我们返回一个默认分布（实际不会被使用）
        return self.actor.distribution(self.obs_from_latent(latent_pi))

    def get_distribution(self, obs):
        """
        SB3 在 collect_rollouts 中会调用此方法来获取动作分布。
        这是关键接口，必须返回一个 SB3 兼容的分布对象。
        """
        return self.actor.distribution(obs)

    def evaluate_actions(self, obs, actions):
        """
        评估给定动作的对数概率、熵和价值。
        SB3 在更新时会调用此方法。
        """
        # 获取分布
        dist = self.actor.distribution(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().mean()
        value = self.critic(obs)
        return value, log_prob, entropy

    def forward(self, obs, deterministic=False):
        """
        前向传播，返回动作、价值、对数概率。
        SB3 在 predict 时会调用此方法。
        """
        dist = self.actor.distribution(obs)
        if deterministic:
            action = dist.mode()   # 使用均值
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs)
        return action, value, log_prob

    def predict_values(self, obs):
        """
        预测状态价值，SB3 会在计算优势时调用。
        """
        return self.critic(obs)

    def _predict(self, observation, deterministic=False):
        """
        底层预测方法，SB3 在 predict 中最终调用此方法。
        """
        action, _, _ = self.forward(observation, deterministic)
        return action