import numpy as np
from scipy.spatial.transform import Rotation as R


def world_to_pelvis(world_pos, pelvis_pos, pelvis_quat):
    w, x, y, z = pelvis_quat
    R_inv = R.from_quat([x, y, z, w]).inv().as_matrix()
    return R_inv @ (np.asarray(world_pos, dtype=np.float64) - np.asarray(pelvis_pos, dtype=np.float64))


def world_yaw_to_pelvis_rel(world_yaw, pelvis_yaw):
    diff = float(world_yaw) - float(pelvis_yaw)
    return np.arctan2(np.sin(diff), np.cos(diff))


def quat_to_yaw(quat_wxyz):
    w, x, y, z = quat_wxyz
    return R.from_quat([x, y, z, w]).as_euler("xyz")[2]
