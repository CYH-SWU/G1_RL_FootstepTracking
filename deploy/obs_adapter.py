import numpy as np
from scipy.spatial.transform import Rotation as R

from deploy.joint_map import JOINT_MAP

REAL_IDX = list(JOINT_MAP)


def _wrap_to_pi(x):
    return np.arctan2(np.sin(x), np.cos(x))


def _rpy_from_quat_wxyz(quat_wxyz):
    """(w, x, y, z) -> (roll, pitch, yaw)。"""
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
