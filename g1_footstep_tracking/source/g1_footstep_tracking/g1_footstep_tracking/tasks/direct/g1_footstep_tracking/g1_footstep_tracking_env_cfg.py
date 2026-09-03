# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab_assets import G1_MINIMAL_CFG
from isaaclab.sensors import ContactSensorCfg
import isaaclab.envs.mdp as mdp

JOINT_NAMES = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

@configclass
class G1FootstepTrackingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))
    robot: ArticulationCfg = G1_MINIMAL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.75),
            joint_pos={
                "left_hip_yaw_joint": 0.0,
                "left_hip_roll_joint": 0.0,
                "left_hip_pitch_joint": -0.5235987756,
                "left_knee_joint": 0.872664626,
                "left_ankle_pitch_joint": -0.34906585,
                "left_ankle_roll_joint": 0.0,
                "right_hip_yaw_joint": 0.0,
                "right_hip_roll_joint": 0.0,
                "right_hip_pitch_joint": -0.5235987756,
                "right_knee_joint": 0.872664626,
                "right_ankle_pitch_joint": -0.34906585,
                "right_ankle_roll_joint": 0.0,
                "torso_joint": 0.0,
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": 0.0,
                "left_elbow_roll_joint": 0.0,
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": 0.0,
                "right_elbow_roll_joint": 0.0,
                "left_five_joint": 0.0,
                "left_three_joint": 0.0,
                "left_zero_joint": 0.0,
                "left_six_joint": 0.0,
                "left_four_joint": 0.0,
                "left_one_joint": 0.0,
                "left_two_joint": 0.0,
                "right_five_joint": 0.0,
                "right_three_joint": 0.0,
                "right_zero_joint": 0.0,
                "right_six_joint": 0.0,
                "right_four_joint": 0.0,
                "right_one_joint": 0.0,
                "right_two_joint": 0.0,
            }
        ),
        actuators={
            "hip_yaw": ImplicitActuatorCfg(joint_names_expr=[".*_hip_yaw_joint"], stiffness=115.0, damping=14.0, effort_limit=88.0, velocity_limit=32.0),
            "hip_roll": ImplicitActuatorCfg(joint_names_expr=[".*_hip_roll_joint"], stiffness=115.0, damping=14.0, effort_limit=139.0, velocity_limit=32.0),
            "hip_pitch": ImplicitActuatorCfg(joint_names_expr=[".*_hip_pitch_joint"], stiffness=115.0, damping=14.0, effort_limit=88.0, velocity_limit=32.0),
            "knee": ImplicitActuatorCfg(joint_names_expr=[".*_knee_joint"], stiffness=172.0, damping=14.0, effort_limit=139.0, velocity_limit=32.0),
            "feet": ImplicitActuatorCfg(joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"], stiffness=46.0, damping=5.0, effort_limit=50.0, velocity_limit=53.0),
            "arms": ImplicitActuatorCfg(joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_pitch_joint", ".*_elbow_roll_joint", ".*_five_joint", ".*_three_joint", ".*_six_joint", ".*_four_joint", ".*_zero_joint", ".*_one_joint", ".*_two_joint"], stiffness=1000.0, damping=100.0, effort_limit=300.0, velocity_limit=23.0),
        }
    )
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0))

    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        history_length=1,
        track_air_time=False,
    )

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale=0.25,
        offset={
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.5235987756,
            "left_knee_joint": 0.872664626,
            "left_ankle_pitch_joint": -0.34906585,
            "left_ankle_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.5235987756,
            "right_knee_joint": 0.872664626,
            "right_ankle_pitch_joint": -0.34906585,
            "right_ankle_roll_joint": 0.0,
        },
    )

@configclass
class G1FootstepTrackingEnvCfg(DirectRLEnvCfg):
    scene: G1FootstepTrackingSceneCfg = G1FootstepTrackingSceneCfg(num_envs=1, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()

    observation_space = gym.spaces.Dict({
        "policy": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(40,), dtype=float),
        "critic": gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(57,), dtype=float),
    })
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=float)

    def __post_init__(self):
        self.decimation = 3
        self.episode_length_s = 22.5
        self.viewer.eye = (2.0, 0.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.viewer.origin_type = "world"
        self.viewer.resolution = (1280, 720)
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
