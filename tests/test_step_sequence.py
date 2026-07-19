import numpy as np

from env.utils.step_sequence import StepSequenceGenerator, WalkModes


def test_step_sequence_all_modes():
    """Generate sequences for all walking modes and verify shape and length."""
    gen = StepSequenceGenerator(step_length=0.2, step_width=0.237, total_duration=1.3, swing_duration=0.85, stance_duration=0.45)
    for mode in [WalkModes.FORWARD, WalkModes.BACKWARD, WalkModes.LATERAL, WalkModes.CURVED, WalkModes.INPLACE, WalkModes.STANDING]:
        seq = gen.generate(mode, phase=0.0, num_steps=5)
        assert len(seq) == (1 if mode == WalkModes.STANDING else 5)
        assert seq.shape[1] == 4  # x, y, z, theta
        # Verify phase sensitivity for CURVED mode.
        if mode == WalkModes.CURVED:
            seq2 = gen.generate(mode, phase=0.5, num_steps=5)
            assert not np.allclose(seq, seq2)  # Different phase yields different sequence.


def test_transform_to_world():
    """Transform a local step sequence to world coordinates."""
    gen = StepSequenceGenerator(step_length=0.2, step_width=0.237, total_duration=1.3, swing_duration=0.85, stance_duration=0.45)
    seq = np.array([[0.1, 0.2, 0.0, 0.3]])
    left_foot = np.array([1.0, 2.0, 0.0])
    right_foot = np.array([1.2, 2.1, 0.0])
    root_yaw = 0.5
    world = gen.transform_to_world(seq, left_foot, right_foot, root_yaw)
    assert world.shape == (1, 4)
    # Check world X coordinate calculation.
    mid = (left_foot + right_foot) / 2
    expected_x = mid[0] + 0.1 * np.cos(0.5) - 0.2 * np.sin(0.5)
    assert np.isclose(world[0, 0], expected_x)
