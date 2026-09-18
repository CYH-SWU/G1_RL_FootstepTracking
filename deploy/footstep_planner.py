import sys
from pathlib import Path

import numpy as np

from env.utils.step_sequence import StepSequenceGenerator, WalkModes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FootstepPlanner:
    def __init__(self, config, mode=WalkModes.FORWARD):
        self.config = config
        self.mode = mode

        self.generator = StepSequenceGenerator(
            config.step_length,
            config.step_width,
            config.total_duration,
            config.swing_duration,
            config.stance_duration,
        )

        self.sequence_world = None
        self.t1 = 0
        self.t2 = 1
        self.reached_frames = 0

        self.delay_frames = int(np.floor(config.swing_duration / config.control_dt))
        self.target_radius = config.target_radius

    def reset(self, left_foot_pos, right_foot_pos, root_yaw, phase=0.0):
        num_steps = 20
        if self.mode == WalkModes.STANDING:
            num_steps = 1
        elif self.mode == WalkModes.CURVED:
            num_steps = 25

        local_seq = self.generator.generate(self.mode, phase, num_steps)
        self.sequence_world = self.generator.transform_to_world(
            local_seq,
            np.asarray(left_foot_pos, dtype=np.float64),
            np.asarray(right_foot_pos, dtype=np.float64),
            float(root_yaw),
        )
        self.t1 = 0
        self.t2 = min(1, len(self.sequence_world) - 1)
        self.reached_frames = 0

    def update(self, left_foot_pos, right_foot_pos):
        if self.sequence_world is None:
            return
        if self.mode == WalkModes.STANDING:
            return
        if self.t1 >= len(self.sequence_world):
            return

        target_pos = self.sequence_world[self.t1][:3]
        l_dist = np.linalg.norm(np.asarray(left_foot_pos, dtype=np.float64) - target_pos)
        r_dist = np.linalg.norm(np.asarray(right_foot_pos, dtype=np.float64) - target_pos)

        if l_dist < self.target_radius or r_dist < self.target_radius:
            self.reached_frames += 1
        else:
            self.reached_frames = 0

        if self.reached_frames >= self.delay_frames:
            self.t1 = self.t2
            self.t2 = min(self.t2 + 1, len(self.sequence_world) - 1)
            self.reached_frames = 0

    def get_t1_t2_world(self):
        if self.sequence_world is None:
            return None, None
        return self.sequence_world[self.t1], self.sequence_world[self.t2]

    @property
    def is_finished(self):
        if self.sequence_world is None:
            return False
        return self.t1 >= len(self.sequence_world) - 1
