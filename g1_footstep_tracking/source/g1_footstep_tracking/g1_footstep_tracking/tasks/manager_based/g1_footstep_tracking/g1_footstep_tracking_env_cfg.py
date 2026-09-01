# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets import G1_MINIMAL_CFG
import isaaclab.envs.mdp as mdp

# ============ 导入原项目 config ============
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(project_root))
from env.utils.config import G1EnvConfig
_G1_CFG = G1EnvConfig()

# ============ 关节名称（与 G1_MINIMAL_CFG 一致） ============
JOINT_NAMES = [
    "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

# ============ 场景配置 ============
@configclass
class G1FootstepTrackingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))
    
    robot: ArticulationCfg = G1_MINIMAL_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.75),  # 骨盆高度
            joint_pos={
                # ===== 腿部（原项目标称角度） =====
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
                # ===== 躯干 =====
                "torso_joint": 0.0,
                # 所有手臂关节归零
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
                # 手部也归零（也可直接省略，默认就是0）
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
            # ===== 髋关节 =====
            "hip_yaw": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_yaw_joint"],
                stiffness=115.0,
                damping=14.0,
                effort_limit=88.0,
                velocity_limit=32.0,
            ),
            "hip_roll": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_roll_joint"],
                stiffness=115.0,
                damping=14.0,
                effort_limit=139.0,
                velocity_limit=32.0,
            ),
            "hip_pitch": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_pitch_joint"],
                stiffness=115.0,
                damping=14.0,
                effort_limit=88.0,
                velocity_limit=32.0,
            ),
            # ===== 膝关节 =====
            "knee": ImplicitActuatorCfg(
                joint_names_expr=[".*_knee_joint"],
                stiffness=172.0,
                damping=14.0,
                effort_limit=139.0,
                velocity_limit=32.0,
            ),
            # ===== 踝关节 =====
            "feet": ImplicitActuatorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                stiffness=46.0,
                damping=5.0,
                effort_limit=50.0,
                velocity_limit=53.0,
            ),
            # ===== 手臂固定（高刚度、高阻尼，无 target） =====
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint", ".*_elbow_roll_joint",
                    ".*_five_joint", ".*_three_joint", ".*_six_joint",
                    ".*_four_joint", ".*_zero_joint", ".*_one_joint", ".*_two_joint"
                ],
                stiffness=1000.0,   # 更高刚度
                damping=100.0,      # 更高阻尼
                effort_limit=300.0,
                velocity_limit=23.0,
            ),
        }
    )
    
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0))

# ============ 动作配置 ============
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale=_G1_CFG.action_scale,
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

# ============ 观测配置 ============
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)}
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

# ============ 奖励配置 ============
@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

# ============ 终止配置 ============
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

# ============ 主环境配置 ============
@configclass
class G1FootstepTrackingEnvCfg(ManagerBasedRLEnvCfg):
    scene = G1FootstepTrackingSceneCfg(num_envs=1, env_spacing=4.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()

    def __post_init__(self):
        self.decimation = int(_G1_CFG.control_dt / _G1_CFG.physics_dt)
        self.episode_length_s = _G1_CFG.max_episode_steps * _G1_CFG.control_dt
        self.viewer.eye = (2.0, 0.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
        self.viewer.origin_type = "world"
        self.viewer.resolution = (1280, 720)
        self.sim.dt = _G1_CFG.physics_dt
        self.sim.render_interval = self.decimation