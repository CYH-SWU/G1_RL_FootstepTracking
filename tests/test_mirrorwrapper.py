import numpy as np
from pathlib import Path
import pytest

from env.g1_env import G1Env
from env_utils.mirrorwrapper import MirrorWrapper


def test_mirrorwrapper_initialization():
    xml_path = Path(__file__).parent.parent / "robot" / "g1_processed.xml"
    if not xml_path.exists():
        pytest.skip("robot XML not found")
    env = G1Env(robot_xml_path=str(xml_path))
    wrapper = MirrorWrapper(env, mirror_prob=0.5)
    assert wrapper.mirror_prob == 0.5
    assert wrapper.mirror is False


def test_mirror_action():
    xml_path = Path(__file__).parent.parent / "robot" / "g1_processed.xml"
    if not xml_path.exists():
        pytest.skip("robot XML not found")
    env = G1Env(robot_xml_path=str(xml_path))
    wrapper = MirrorWrapper(env, mirror_prob=1.0)
    wrapper.mirror = True

    # 生成随机动作
    action = np.random.uniform(-1, 1, 12).astype(np.float32)
    mirrored_once = wrapper._mirror_action(action)
    mirrored_twice = wrapper._mirror_action(mirrored_once)

    # 两次镜像应还原为原始动作
    assert np.allclose(mirrored_twice, action, atol=1e-6)