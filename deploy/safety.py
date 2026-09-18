import time
from collections import deque
from enum import Enum

import numpy as np


class SafetyState(Enum):
    OK = 0
    WARN = 1
    HOLD = 2
    STOP = 3


class SafetyMonitor:
    def __init__(self, config):
        self.lowstate_timeout = float(config.get("lowstate_timeout", 0.1))
        self.max_tilt_deg = float(config.get("max_tilt_deg", 45.0))
        self.max_loop_dt = float(config.get("max_loop_dt", 0.05))
        self.max_joint_dev = float(config.get("max_joint_deviation", 0.6))
        self.max_consecutive_warn = int(config.get("max_consecutive_warn", 20))

        self.last_lowstate_t = None
        self.last_loop_t = None
        self.loop_dt_history = deque(maxlen=100)
        self.consecutive_warn = 0
        self.total_warn = 0

        self.joint_lower = np.array(
            config.get(
                "joint_lower",
                [
                    -2.53,
                    -0.52,
                    -2.76,
                    -0.09,
                    -0.87,
                    -0.26,
                    -2.53,
                    -2.97,
                    -2.76,
                    -0.09,
                    -0.87,
                    -0.26,
                ],
            ),
            dtype=np.float64,
        )
        self.joint_upper = np.array(
            config.get(
                "joint_upper",
                [
                    2.88,
                    2.97,
                    2.76,
                    2.88,
                    0.52,
                    0.26,
                    2.88,
                    0.52,
                    2.76,
                    2.88,
                    0.52,
                    0.26,
                ],
            ),
            dtype=np.float64,
        )

    def on_lowstate(self):
        self.last_lowstate_t = time.time()

    def on_loop(self):
        now = time.time()
        if self.last_loop_t is not None:
            dt = now - self.last_loop_t
            self.loop_dt_history.append(dt)
        self.last_loop_t = now

    def check_lowstate_timeout(self) -> bool:
        if self.last_lowstate_t is None:
            return False
        return (time.time() - self.last_lowstate_t) > self.lowstate_timeout

    def check_finite(self, *arrays) -> bool:
        for arr in arrays:
            if arr is None:
                continue
            if not np.all(np.isfinite(arr)):
                return False
        return True

    def check_tilt(self, quat_wxyz) -> bool:
        from scipy.spatial.transform import Rotation as R

        w, x, y, z = quat_wxyz
        roll, pitch, _ = R.from_quat([x, y, z, w]).as_euler("xyz")

        limit = np.deg2rad(self.max_tilt_deg)
        return abs(roll) < limit and abs(pitch) < limit

    def check_loop_freq(self) -> bool:
        if len(self.loop_dt_history) < 10:
            return True
        mean_dt = np.mean(self.loop_dt_history)
        return mean_dt < self.max_loop_dt

    def clamp_joints(self, targets, nominal):
        targets = np.asarray(targets, dtype=np.float64)
        clamped = np.clip(targets, self.joint_lower, self.joint_upper)
        num_clamped = int(np.sum(np.abs(clamped - targets) > 1e-6))
        return clamped, num_clamped

    def evaluate(self, lowstate, action, targets, nominal) -> SafetyState:
        self.on_loop()

        if self.check_lowstate_timeout():
            print("[SAFETY] LowState timeout")
            return SafetyState.STOP

        if not self.check_finite(action, targets):
            print("[SAFETY] NaN/Inf in action or targets")
            self._register_warn()
            return SafetyState.HOLD

        try:
            quat = np.array(lowstate.imu_state.quaternion, dtype=np.float64)
            if not self.check_tilt(quat):
                print("[SAFETY] Robot tilted beyond limit")
                return SafetyState.STOP
        except Exception as e:
            print(f"[SAFETY] Tilt check failed: {e}")

        if not self.check_loop_freq():
            if len(self.loop_dt_history) >= 10:
                mean_dt = np.mean(self.loop_dt_history)
                print(f"[SAFETY] Loop slow: {1.0 / mean_dt:.1f}Hz")
            self._register_warn()
            return SafetyState.WARN

        clamped, n_clamped = self.clamp_joints(targets, nominal)
        if n_clamped > 0:
            print(f"[SAFETY] {n_clamped} joints clamped to range")
            self._register_warn()
            return SafetyState.WARN

        self.consecutive_warn = 0
        return SafetyState.OK

    def _register_warn(self):
        self.consecutive_warn += 1
        self.total_warn += 1

    @property
    def stats_str(self):
        if len(self.loop_dt_history) < 2:
            return "freq=N/A"
        mean_dt = np.mean(self.loop_dt_history)
        return f"freq={1.0 / mean_dt:.1f}Hz warns={self.total_warn}"
