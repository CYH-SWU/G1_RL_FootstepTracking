from pathlib import Path

import numpy as np
import pytest

from env.g1_env import G1Env

# 获取 robot XML 路径
project_root = Path(__file__).parent.parent
ROBOT_XML = project_root / "robot" / "g1_processed.xml"


@pytest.fixture
def env():
    """创建环境实例，若 XML 不存在则跳过测试"""
    if not ROBOT_XML.exists():
        pytest.skip("robot XML not found (run robot/gen_xml.py first)")
    return G1Env(robot_xml_path=str(ROBOT_XML))


def test_env_reset(env):
    """测试环境重置和观测空间形状"""
    obs, info = env.reset(seed=42)
    assert "actor_obs" in obs
    assert "critic_obs" in obs
    assert obs["actor_obs"].shape == (41,)
    assert obs["critic_obs"].shape == (58,)
    assert "mode" in info


def test_env_step(env):
    """测试单步执行（零动作）"""
    env.reset()
    action = np.zeros(12, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float) or isinstance(reward, np.floating)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, bool)


def test_env_random_steps(env):
    """测试多步随机动作不崩溃"""
    env.reset()
    for _ in range(50):
        action = np.random.uniform(-0.5, 0.5, 12).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert True


def test_env_difficulty(env):
    """测试难度设置是否生效"""
    env.set_difficulty(0.5)
    assert env.difficulty == 0.5
    env.set_difficulty(1.2)
    assert env.difficulty == 1.0  # 应被裁剪到 [0,1]
