import gymnasium as gym

from . import agents

gym.register(
    id="G1FootstepTracking-v0",
    entry_point="g1_footstep_tracking.tasks.direct.g1_footstep_tracking.g1_footstep_tracking_env:G1FootstepTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "g1_footstep_tracking.tasks.direct.g1_footstep_tracking.g1_footstep_tracking_env_cfg:G1FootstepTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)