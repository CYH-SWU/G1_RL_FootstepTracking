from pathlib import Path

import numpy as np
import pytest

from env.g1_env import G1Env

project_root = Path(__file__).parent.parent
ROBOT_XML = project_root / "robot" / "g1_processed.xml"


@pytest.fixture
def env():
    """Create environment instance; skip if XML is missing."""
    if not ROBOT_XML.exists():
        pytest.skip("robot XML not found (run robot/gen_xml.py first)")
    return G1Env(robot_xml_path=str(ROBOT_XML))


def test_env_reset(env):
    """Reset and verify observation space shapes."""
    obs, info = env.reset(seed=42)
    assert "actor_obs" in obs
    assert "critic_obs" in obs
    assert obs["actor_obs"].shape == (41,)
    assert obs["critic_obs"].shape == (58,)
    assert "mode" in info


def test_env_step(env):
    """Step with zero action and check return types."""
    env.reset()
    action = np.zeros(12, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float) or isinstance(reward, np.floating)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, bool)


def test_env_random_steps(env):
    """Run random actions for 50 steps without crashing."""
    env.reset()
    for _ in range(50):
        action = np.random.uniform(-0.5, 0.5, 12).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    assert True


def test_env_difficulty(env):
    """Set difficulty and verify clipping to [0, 1]."""
    env.set_difficulty(0.5)
    assert env.difficulty == 0.5
    env.set_difficulty(1.2)
    assert env.difficulty == 1.0
