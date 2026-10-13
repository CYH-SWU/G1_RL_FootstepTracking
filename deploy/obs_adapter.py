"""Observation adapters for deployment.

build_actor_obs  : assemble the 41-dim raw actor_obs from robot state.
StudentObsPipeline: convert a stream of raw actor_obs into the flat history
                    window that the distilled student expects
                    (normalize -> drop pelvis-height channel -> stack).
"""

from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R

from deploy.joint_map import JOINT_MAP

REAL_IDX = list(JOINT_MAP)


def _wrap_to_pi(x):
    return np.arctan2(np.sin(x), np.cos(x))


def _rpy_from_quat_wxyz(quat_wxyz):
    """(w, x, y, z) -> (roll, pitch, yaw)."""
    w, x, y, z = quat_wxyz
    r = R.from_quat([x, y, z, w])
    return r.as_euler("xyz")


def build_actor_obs(
    lowstate,
    t1_local_pos,
    t2_local_pos,
    foot_yaw_rel,
    next_yaw_rel,
    phase,
    pelvis_height,
):
    """Assemble the raw 41-dim actor_obs (same layout as training)."""
    obs = np.zeros(41, dtype=np.float32)

    for i, real_idx in enumerate(REAL_IDX):
        obs[i] = float(lowstate.motor_state[real_idx].q)

    for i, real_idx in enumerate(REAL_IDX):
        obs[12 + i] = float(lowstate.motor_state[real_idx].dq)

    obs[24] = float(pelvis_height)

    obs[25:28] = np.asarray(t1_local_pos, dtype=np.float32)
    obs[28:31] = np.asarray(t2_local_pos, dtype=np.float32)

    obs[31] = _wrap_to_pi(float(foot_yaw_rel))
    obs[32] = _wrap_to_pi(float(next_yaw_rel))

    obs[33] = np.sin(2.0 * np.pi * phase)
    obs[34] = np.cos(2.0 * np.pi * phase)

    quat = lowstate.imu_state.quaternion  # [w, x, y, z]
    roll, pitch, yaw = _rpy_from_quat_wxyz(quat)
    obs[35] = roll
    obs[36] = pitch
    obs[37] = yaw

    obs[38:41] = np.asarray(lowstate.imu_state.gyroscope, dtype=np.float32)

    return obs


class StudentObsPipeline:
    """Turn a stream of raw 41-dim actor_obs into the student's flat input.

    Per frame: normalize with the frozen training statistics, drop the
    pelvis-height channel, push into a history deque, then flatten
    oldest -> newest into a (base_dim * history_len) vector.

    Must match the distillation-time processing exactly.
    """

    def __init__(self, meta, eps=1e-8, clip=10.0):
        self.norm_mean = np.asarray(meta["norm_mean"], dtype=np.float32)
        self.norm_var = np.asarray(meta["norm_var"], dtype=np.float32)
        self.eps = eps
        self.clip = clip
        self.pelvis_height_idx = int(meta["pelvis_height_idx"])
        self.base_dim = int(meta["base_dim"])
        self.history_len = int(meta["history_len"])
        self.buf = deque(maxlen=self.history_len)
        self.reset()

    def reset(self):
        """Zero-fill the history window (matches HistoryWindow.reset at training)."""
        self.buf.clear()
        for _ in range(self.history_len):
            self.buf.append(np.zeros(self.base_dim, dtype=np.float32))

    def _normalize_and_strip(self, actor_obs):
        x = (np.asarray(actor_obs, dtype=np.float32) - self.norm_mean) / np.sqrt(self.norm_var + self.eps)
        x = np.clip(x, -self.clip, self.clip)
        idx = self.pelvis_height_idx
        base = np.concatenate([x[:idx], x[idx + 1 :]])
        return base.astype(np.float32)

    def update(self, actor_obs):
        """Push one raw frame and return the flattened history window."""
        self.buf.append(self._normalize_and_strip(actor_obs))
        return np.concatenate(self.buf).astype(np.float32)
