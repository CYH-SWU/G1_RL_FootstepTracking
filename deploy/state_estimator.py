import numpy as np
from scipy.spatial.transform import Rotation as R

NOMINAL_PELVIS_HEIGHT = 0.7268
FOOT_ANKLE_OFFSET = 0.0331

ANKLE_TAU_THRESHOLD = 3.0
ANKLE_VEL_THRESHOLD = 2.0

LEFT_ANKLE_IDX = [4, 5]
RIGHT_ANKLE_IDX = [10, 11]

HEIGHT_FILTER_ALPHA = 0.2
POS_XY_FILTER_ALPHA = 0.5


class StateEstimator:
    def __init__(self):
        self.pelvis_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.pelvis_pos = np.array([0.0, 0.0, NOMINAL_PELVIS_HEIGHT], dtype=np.float64)

        self.pelvis_height = NOMINAL_PELVIS_HEIGHT
        self._height_initialized = False

        self.pelvis_lin_vel = np.zeros(3, dtype=np.float64)

        self.left_contact = False
        self.right_contact = False
        self.left_contact_prev = False
        self.right_contact_prev = False

        self.left_foot_world_locked = None
        self.right_foot_world_locked = None

    def update_attitude(self, lowstate):
        self.pelvis_quat = np.array(lowstate.imu_state.quaternion, dtype=np.float64)

    def detect_contact(self, lowstate):
        self.left_contact = self._check_foot_contact(lowstate, LEFT_ANKLE_IDX)
        self.right_contact = self._check_foot_contact(lowstate, RIGHT_ANKLE_IDX)

    @staticmethod
    def _check_foot_contact(lowstate, ankle_idx):
        max_tau = 0.0
        max_vel = 0.0
        for i in ankle_idx:
            max_tau = max(max_tau, abs(float(lowstate.motor_state[i].tau_est)))
            max_vel = max(max_vel, abs(float(lowstate.motor_state[i].dq)))
        return max_tau > ANKLE_TAU_THRESHOLD and max_vel < ANKLE_VEL_THRESHOLD

    def update_height(self, left_foot_local, right_foot_local):
        if self.left_contact and self.right_contact:
            foot_local = 0.5 * (
                np.asarray(left_foot_local, dtype=np.float64) + np.asarray(right_foot_local, dtype=np.float64)
            )
        elif self.left_contact:
            foot_local = np.asarray(left_foot_local, dtype=np.float64)
        elif self.right_contact:
            foot_local = np.asarray(right_foot_local, dtype=np.float64)
        else:
            return

        w, x, y, z = self.pelvis_quat
        R_world = R.from_quat([x, y, z, w]).as_matrix()
        foot_world_offset = R_world @ foot_local

        height_raw = float(-foot_world_offset[2] + FOOT_ANKLE_OFFSET)

        if height_raw < 0.3 or height_raw > 1.2:
            return

        if not self._height_initialized:
            self.pelvis_height = height_raw
            self._height_initialized = True
        else:
            a = HEIGHT_FILTER_ALPHA
            self.pelvis_height = a * height_raw + (1.0 - a) * self.pelvis_height

    def update_position(self, left_foot_local, right_foot_local):
        w, x, y, z = self.pelvis_quat
        R_world = R.from_quat([x, y, z, w]).as_matrix()

        left_local = np.asarray(left_foot_local, dtype=np.float64)
        right_local = np.asarray(right_foot_local, dtype=np.float64)

        self.pelvis_pos[2] = self.pelvis_height

        left_just = self.left_contact and not self.left_contact_prev
        right_just = self.right_contact and not self.right_contact_prev

        if left_just:
            self.left_foot_world_locked = self.pelvis_pos + R_world @ left_local
        if right_just:
            self.right_foot_world_locked = self.pelvis_pos + R_world @ right_local

        new_xy = None
        if self.left_contact and self.left_foot_world_locked is not None:
            new_pos = self.left_foot_world_locked - R_world @ left_local
            new_xy = new_pos[:2]
        elif self.right_contact and self.right_foot_world_locked is not None:
            new_pos = self.right_foot_world_locked - R_world @ right_local
            new_xy = new_pos[:2]

        if new_xy is not None:
            a = POS_XY_FILTER_ALPHA
            self.pelvis_pos[0] = a * new_xy[0] + (1.0 - a) * self.pelvis_pos[0]
            self.pelvis_pos[1] = a * new_xy[1] + (1.0 - a) * self.pelvis_pos[1]
        self.left_contact_prev = self.left_contact
        self.right_contact_prev = self.right_contact

    @property
    def pelvis_yaw(self):
        w, x, y, z = self.pelvis_quat
        return float(R.from_quat([x, y, z, w]).as_euler("xyz")[2])

    @property
    def contact_str(self):
        return f"L={int(self.left_contact)} R={int(self.right_contact)}"
