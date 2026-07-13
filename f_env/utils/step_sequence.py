import numpy as np
import random
from enum import Enum, auto

class WalkModes(Enum):
    STANDING = auto()
    CURVED = auto()
    FORWARD = auto()
    BACKWARD = auto()
    INPLACE = auto()
    LATERAL = auto()

class StepSequenceGenerator:
    def __init__(self, step_length, step_width, total_duration, swing_duration, stance_duration):
        self.step_length = step_length
        self.step_width = step_width
        self.total_duration = total_duration
        self.swing_duration = swing_duration
        self.stance_duration = stance_duration

    def generate(self, mode, phase, num_steps=20, step_height=0.0, plans=None):
        """
        生成步点序列（局部坐标系）
        :param mode: WalkModes
        :param phase: 当前步态相位 (0~1)
        :param num_steps: 步点数量
        :param step_height: 台阶高度变化
        :param plans: 用于 CURVED 模式的预定义计划（可选）
        :return: list of [x, y, z, theta]
        """
        if mode == WalkModes.CURVED:
            seq = []
            first_shift = np.random.uniform(0.100, 0.125)
            if np.isclose(phase, 0.0):
                seq.append([0.0, -first_shift, 0.0, 0.0])
                initial_y_sign = -1
                curve_dir = -1
            else:
                seq.append([0.0, first_shift, 0.0, 0.0])
                initial_y_sign = 1
                curve_dir = 1

            R = np.random.uniform(2.5, 4.0)
            y0 = initial_y_sign * first_shift
            cy = y0 - curve_dir * R
            total_angle = (num_steps - 1) * (self.step_length - 0.025) / R
            total_angle *= np.random.uniform(0.9, 1.1)
            dtheta = total_angle / (num_steps - 1)

            for i in range(1, num_steps):
                theta_i = i * dtheta
                radius_offset = ((-1) ** i) * ((self.step_width - 0.025) / 2)
                R_i = R + radius_offset
                x_local = R_i * np.sin(theta_i)
                y_local = curve_dir * R_i * np.cos(theta_i)
                x_world = x_local
                y_world = cy + y_local
                yaw = -theta_i * curve_dir
                seq.append([x_world, y_world, 0.0, yaw])
            return np.array(seq)

        elif mode == WalkModes.LATERAL:
            seq = []
            y = 0
            c = np.random.choice([-1, 1])
            for i in range(1, num_steps + 1):
                if i % 2:
                    y += self.step_length * 0.8
                else:
                    y -= (2/3) * self.step_length * 0.8
                step = np.array([0, c * y, 0, 0])
                seq.append(step)
            return np.array(seq)

        elif mode == WalkModes.STANDING:
            return np.array([[0.0, 0.0, 0.0, 0.0]])

        elif mode == WalkModes.INPLACE:
            ss = np.random.uniform(-0.05, 0.05)
            seq = []
            for i in range(num_steps):
                x = ss * (i % 2)
                seq.append([x, 0.0, 0.0, 0.0])
            return np.array(seq)

        elif mode == WalkModes.BACKWARD:
            seq = []
            x = 0
            y = self.step_width / 2 * (1 if np.random.rand() > 0.5 else -1)
            for i in range(num_steps):
                x -= 0.1
                y = -y
                seq.append([x, y, 0.0, 0.0])
            return np.array(seq)

        else:  # FORWARD
            seq = []
            if step_height < 0:
                initial_z = -step_height * (num_steps - 1)
            else:
                initial_z = 0

            first_shift = np.random.uniform(0.100, 0.125)

            if np.isclose(phase, 0.0):
                seq.append([0.0, -first_shift, initial_z, 0.0])
                y = -self.step_width / 2
            else:
                seq.append([0.0, first_shift, initial_z, 0.0])
                y = self.step_width / 2

            x = 0
            z = initial_z
            for i in range(1, num_steps):
                x += self.step_length
                y *= -1
                z += step_height
                seq.append([x, y, z, 0.0])
            return np.array(seq)

    def transform_to_world(self, sequence, left_foot_pos, right_foot_pos, root_yaw):
        """
        将局部序列转换为世界坐标
        """
        mid_pt = (left_foot_pos + right_foot_pos) / 2
        cos_y = np.cos(root_yaw)
        sin_y = np.sin(root_yaw)
        world_seq = []
        for x, y, z, theta in sequence:
            x_w = mid_pt[0] + x * cos_y - y * sin_y
            y_w = mid_pt[1] + x * sin_y + y * cos_y
            theta_w = root_yaw + theta
            world_seq.append(np.array([x_w, y_w, z, theta_w]))
        return np.array(world_seq)