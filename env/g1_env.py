import os
import sys
from pathlib import Path

# Add project root to sys.path to ensure f_env module can be imported.
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
import random

from env.utils.config import G1EnvConfig
from env.utils.step_sequence import WalkModes, StepSequenceGenerator
from env.utils.observation_builder import ObservationBuilder
from env.utils.reward_calculator import RewardCalculator
from env.utils.terrain_generator import TerrainGenerator

class G1Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, robot_xml_path, config=None):
        super().__init__()
        self.config = config or G1EnvConfig()
        self.robot_xml_path = os.path.abspath(robot_xml_path)

        # Load model and cache body/joint IDs.
        self.terrain_gen = TerrainGenerator(robot_xml_path, self.config.max_boxes)
        self.model, self.data = self.terrain_gen.load_model()
        self._cache_ids()

        # Initialize helper components.
        self.step_gen = StepSequenceGenerator(
            self.config.step_length, self.config.step_width,
            self.config.total_duration, self.config.swing_duration, self.config.stance_duration
        )
        self.obs_builder = ObservationBuilder(
            self.config, self.joint_indices, self.joint_vel_indices,
            self.actuator_indices, self.max_torques
        )
        self.reward_calc = RewardCalculator(self.config)

        # Define action and observation spaces.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        actor_dim = 12 + 12 + 1 + 8 + 2 + 3 + 3   # 41
        priv_dim = 2 + 3 + 12                     # 17
        self.observation_space = spaces.Dict({
            "actor_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(actor_dim,), dtype=np.float32),
            "critic_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(actor_dim + priv_dim,), dtype=np.float32),
        })

        # Internal state variables.
        self.step_counter = 0
        self.phase = 0.0
        self.mode = None
        self.sequence = []
        self.t1 = 0
        self.t2 = 1
        self.target_reached = False
        self.target_reached_frames = 0
        self.difficulty = 0
        self.last_action = None
        self.last_torque = None
        self.smooth_target = np.zeros(12)

    def _cache_ids(self):
        model = self.model
        self.pelvis_id = model.body("pelvis").id
        self.left_foot_id = model.body("left_ankle_roll_link").id
        self.right_foot_id = model.body("right_ankle_roll_link").id
        self.torso_id = model.body("torso_link").id
        self.head_id = self.torso_id

        joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"
        ]
        self.joint_indices = []
        self.joint_vel_indices = []
        for name in joint_names:
            joint = model.joint(name)
            self.joint_indices.append(joint.qposadr[0])
            self.joint_vel_indices.append(joint.dofadr[0])

        self.actuator_indices = []
        for name in joint_names:
            self.actuator_indices.append(model.actuator(name).id)

        self.max_torques = np.array([
            88, 139, 88, 139, 50, 50,
            88, 139, 88, 139, 50, 50
        ])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.target_reached = False
        self.target_reached_frames = 0
        self.last_action = None
        self.last_torque = None
        self.smooth_target = np.zeros(12)

        # Sample walking mode from config probabilities.
        self.mode = np.random.choice(
            [WalkModes.STANDING, WalkModes.CURVED, WalkModes.BACKWARD,
             WalkModes.LATERAL, WalkModes.FORWARD],
            p=self.config.mode_probs
        )

        # Determine step height variation for forward mode.
        step_height = 0.0
        if self.mode == WalkModes.FORWARD:
            max_h = 0.1 * max(0.0, (self.difficulty - 0.273) / (1.0 - 0.273))
            step_height = np.random.choice([-max_h, max_h])

        num_steps = 20
        if self.mode == WalkModes.STANDING:
            num_steps = 1
        elif self.mode == WalkModes.CURVED:
            num_steps = 25

        # Reset simulation to stand keyframe.
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            self.data.ctrl[:] = self.data.qpos[self.actuator_indices]
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Generate footstep sequence.
        self.phase = random.choice([0.0, 0.5])
        local_seq = self.step_gen.generate(self.mode, self.phase, num_steps, step_height)
        left_pos = self.data.xpos[self.left_foot_id]
        right_pos = self.data.xpos[self.right_foot_id]
        root_yaw = self._get_pelvis_yaw()
        self.sequence = self.step_gen.transform_to_world(local_seq, left_pos, right_pos, root_yaw)

        # Update visual stepping stones.
        self.terrain_gen.update_boxes(self.model, self.data, self.sequence)

        # Initialize pelvis position offset behind the first step.
        if len(self.sequence) > 0:
            offset_back = -0.12
            first_step = self.sequence[0]
            self.data.qpos[0] = first_step[0] + offset_back
            self.data.qpos[1] = first_step[1]
            self.data.qpos[2] = first_step[2] + self.config.nominal_pelvis_height
            yaw = first_step[3]
            quat = R.from_euler('z', yaw).as_quat()
            self.data.qpos[3] = quat[3]
            self.data.qpos[4] = quat[0]
            self.data.qpos[5] = quat[1]
            self.data.qpos[6] = quat[2]

        mujoco.mj_forward(self.model, self.data)

        # Reset footstep indices.
        self.t1 = 0
        self.t2 = min(1, len(self.sequence)-1) if len(self.sequence) > 1 else 0
        self.step_counter = 0

        obs = self._get_obs()
        info = {"mode": self.mode.name, "difficulty": self.difficulty}
        return obs, info

    def step(self, action):
        self._apply_action(action)

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.step_counter += 1

        self.phase = (self.step_counter * self.config.control_dt % self.config.total_duration) / self.config.total_duration

        # Check if any foot reaches the current target.
        if self.mode != WalkModes.STANDING and len(self.sequence) > 0 and self.t1 < len(self.sequence):
            target_pos = self.sequence[self.t1][:3]
            left_pos = self.data.xpos[self.left_foot_id]
            right_pos = self.data.xpos[self.right_foot_id]
            l_dist = np.linalg.norm(left_pos - target_pos)
            r_dist = np.linalg.norm(right_pos - target_pos)
            if l_dist < self.config.target_radius or r_dist < self.config.target_radius:
                self.target_reached = True
                self.target_reached_frames += 1
            else:
                self.target_reached = False
                self.target_reached_frames = 0

            # Advance to next target after holding for delay_frames.
            if self.target_reached and self.target_reached_frames >= self.delay_frames:
                self.t1 = self.t2
                self.t2 = min(self.t2 + 1, len(self.sequence) - 1)
                self.target_reached = False
                self.target_reached_frames = 0

        # Compute reward.
        self.reward_calc.set_target_reached(self.target_reached)
        reward = self.reward_calc.compute_reward(
            self.model, self.data, self.pelvis_id, self.left_foot_id, self.right_foot_id, self.head_id,
            self.joint_indices, self.actuator_indices, self.mode, self.phase, self.sequence, self.t1, action
        )

        obs = self._get_obs()

        terminated = self._check_termination()
        truncated = self.step_counter >= self.config.max_episode_steps

        info = {}
        return obs, reward, terminated, truncated, info

    def _apply_action(self, action):
        raw_target = self.config.nominal_angles + action * self.config.action_scale
        smooth = self.config.action_smoothing
        self.smooth_target = smooth * raw_target + (1 - smooth) * self.smooth_target
        target_qpos = self.smooth_target

        for i, idx in enumerate(self.actuator_indices):
            low, high = self.model.actuator_ctrlrange[idx]
            target_qpos[i] = np.clip(target_qpos[i], low, high)
        self.data.ctrl[self.actuator_indices] = target_qpos

    def _get_obs(self):
        actor_obs = self.obs_builder.get_actor_obs(
            self.model, self.data, self.pelvis_id, self.left_foot_id, self.right_foot_id,
            self.sequence, self.t1, self.t2, self.phase
        )
        critic_obs = self.obs_builder.get_critic_obs(
            self.model, self.data, self.pelvis_id, self.left_foot_id, self.right_foot_id,
            self.sequence, self.t1, self.t2, self.phase, actor_obs
        )
        return {"actor_obs": actor_obs, "critic_obs": critic_obs}

    def _check_termination(self):
        pelvis_z = self.data.qpos[2]
        foot_z = min(self.data.xpos[self.left_foot_id][2], self.data.xpos[self.right_foot_id][2]) - self.config.foot_ankle_offset
        height = pelvis_z - foot_z
        return height < self.config.fall_height_threshold

    def _get_pelvis_yaw(self):
        quat = self.data.xquat[self.pelvis_id].copy()
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        return r.as_euler('xyz')[2]

    def set_difficulty(self, progress: float):
        self.difficulty = np.clip(progress, 0.0, 1.0)

    def render(self):
        raise NotImplementedError("This environment does not support real-time rendering. Use a separate evaluation script.")

    def close(self):
        pass

    @property
    def n_substeps(self):
        return int(self.config.control_dt / self.config.physics_dt)

    @property
    def delay_frames(self):
        return int(np.floor(self.config.swing_duration / self.config.control_dt))
    

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import mujoco.viewer

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xml",
        type=str,
        default=str(Path(__file__).parent.parent / "robot" / "g1_processed.xml"),
        help="Path to robot XML file"
    )
    parser.add_argument("--steps", type=int, default=1500, help="Max steps to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    xml_path = args.xml
    print(f"Using XML file: {xml_path}")

    env = G1Env(robot_xml_path=xml_path)
    env.set_difficulty(1.0)

    obs, info = env.reset(seed=args.seed)
    print("Environment reset.")

    viewer = mujoco.viewer.launch_passive(env.model, env.data)

    step = 0
    max_steps = args.steps

    while step < max_steps and viewer.is_running():
        action = np.zeros(12, dtype=np.float32)
        obs, _, terminated, truncated, _ = env.step(action)
        step += 1
        viewer.sync()

    print(f"Test completed. Total steps: {step}")
    viewer.close()
    env.close()