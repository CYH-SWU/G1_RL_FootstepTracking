from pathlib import Path

import numpy as np
import pytest

from env.g1_env import G1Env
from env_utils.mirrorwrapper import MirrorWrapper


def test_mirrorwrapper_initialization():
    """Verify that MirrorWrapper initializes with correct mirror probability."""
    xml_path = Path(__file__).parent.parent / "robot" / "g1_processed.xml"
    if not xml_path.exists():
        pytest.skip("robot XML not found")
    env = G1Env(robot_xml_path=str(xml_path))
    wrapper = MirrorWrapper(env, mirror_prob=0.5)
    assert wrapper.mirror_prob == 0.5
    assert wrapper.mirror is False


def test_mirror_action():
    """Test that applying mirror twice restores the original action."""
    xml_path = Path(__file__).parent.parent / "robot" / "g1_processed.xml"
    if not xml_path.exists():
        pytest.skip("robot XML not found")
    env = G1Env(robot_xml_path=str(xml_path))
    wrapper = MirrorWrapper(env, mirror_prob=1.0)
    wrapper.mirror = True

    action = np.random.uniform(-1, 1, 12).astype(np.float32)
    mirrored_once = wrapper._mirror_action(action)
    mirrored_twice = wrapper._mirror_action(mirrored_once)

    # Double mirroring should recover the original action.
    assert np.allclose(mirrored_twice, action, atol=1e-6)
