"""Test that all key modules can be imported correctly."""

from env.g1_env import G1Env
from env.utils.config import G1EnvConfig
from rl.policy import AsymmetricPolicy
from rl.callbacks import AdaptiveLRScheduleCallback, CurriculumCallback
from env.utils.step_sequence import WalkModes, StepSequenceGenerator
from env.utils.observation_builder import ObservationBuilder
from env.utils.reward_calculator import RewardCalculator
from env.utils.terrain_generator import TerrainGenerator


def test_import_env():
    """Test env module imports."""
    assert G1Env is not None
    assert G1EnvConfig is not None


def test_import_rl():
    """Test rl module imports."""
    assert AsymmetricPolicy is not None
    assert AdaptiveLRScheduleCallback is not None
    assert CurriculumCallback is not None


def test_import_utils():
    """Test env.utils module imports."""
    assert WalkModes is not None
    assert StepSequenceGenerator is not None
    assert ObservationBuilder is not None
    assert RewardCalculator is not None
    assert TerrainGenerator is not None
