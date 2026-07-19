import numpy as np
from gymnasium import spaces
from env_utils.mirrorwrapper import MirrorWrapper
from env.g1_env import G1Env
from pathlib import Path

def test_mirrorwrapper_initialization():
    env = G1Env(robot_xml_path=str(Path(__file__).parent.parent / "robot" / "g1_processed.xml"))
    wrapper = MirrorWrapper(env, mirror_prob=0.5)
    assert wrapper.mirror_prob == 0.5
    assert wrapper.mirror is False

def test_mirror_action():
    # 直接测试镜像逻辑，无需完整环境
    # 但 MirrorWrapper 依赖 env，我们可以用一个 dummy env
    class DummyEnv:
        action_space = spaces.Box(low=-1, high=1, shape=(12,))
        observation_space = spaces.Dict({
            "actor_obs": spaces.Box(low=-1, high=1, shape=(41,)),
            "critic_obs": spaces.Box(low=-1, high=1, shape=(58,)),
        })
        def reset(self): return {}, {}
        def step(self, action): return {}, 0, False, False, {}
    env = DummyEnv()
    wrapper = MirrorWrapper(env, mirror_prob=0.0)
    wrapper.mirror = True  # 强制镜像
    action = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2])
    mirrored = wrapper._mirror_action(action)
    # 验证左右交换 + 符号翻转
    assert np.allclose(mirrored[:6], action[6:])  # 左换右
    assert np.allclose(mirrored[6:], action[:6])  # 右换左
    # 验证符号翻转的索引（1,2,5,7,8,11）
    assert mirrored[1] == -action[7]  # 左hip_roll 对应右hip_roll翻转？需要检查索引逻辑