from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parent.parent


LEG_JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]


class ForwardKinematics:
    def __init__(self, xml_path="robot/g1_processed.xml"):
        xml_full = PROJECT_ROOT / xml_path
        if not xml_full.exists():
            raise FileNotFoundError(f"XML not found: {xml_full}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_full))
        self.data = mujoco.MjData(self.model)

        self.pelvis_id = self.model.body("pelvis").id
        self.left_foot_id = self.model.body("left_ankle_roll_link").id
        self.right_foot_id = self.model.body("right_ankle_roll_link").id

        self.joint_qpos_adr = []
        for name in LEG_JOINTS:
            j = self.model.joint(name)
            self.joint_qpos_adr.append(j.qposadr[0])

    def compute(self, joint_angles, pelvis_pos, pelvis_quat):
        """
        Args:
            joint_angles: (12,)
            pelvis_pos:   (3,)
            pelvis_quat:  (4,)

        Returns:
            left_foot_world:
            right_foot_world:
            left_foot_local:
            right_foot_local:
        """
        joint_angles = np.asarray(joint_angles, dtype=np.float64)
        pelvis_pos = np.asarray(pelvis_pos, dtype=np.float64)
        pelvis_quat = np.asarray(pelvis_quat, dtype=np.float64)

        self.data.qpos[0:3] = pelvis_pos
        self.data.qpos[3:7] = pelvis_quat

        for i, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = joint_angles[i]

        mujoco.mj_forward(self.model, self.data)

        left_world = self.data.xpos[self.left_foot_id].copy()
        right_world = self.data.xpos[self.right_foot_id].copy()

        w, x, y, z = pelvis_quat
        R_inv = R.from_quat([x, y, z, w]).inv().as_matrix()
        left_local = R_inv @ (left_world - pelvis_pos)
        right_local = R_inv @ (right_world - pelvis_pos)

        return left_world, right_world, left_local, right_local
