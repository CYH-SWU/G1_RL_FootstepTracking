import os
import mujoco
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

class TerrainGenerator:
    """
    Injects visual stepping stone markers into the MuJoCo model for footstep tracking.

    The generator loads the robot XML, resolves the asset paths, and appends a set of
    visual bodies (box, sphere, arrow) for each potential footstep target. The markers
    are initially placed far below the scene and later repositioned via update_boxes()
    to match the current footstep sequence.

    Methods:
        load_model()
            Reads the robot XML, replaces meshdir with an absolute path, appends marker
            bodies for up to max_boxes targets, and loads the model into MuJoCo.
            Returns the model and data instances.

        update_boxes(model, data, sequence)
            Updates the position and orientation of all marker bodies based on the
            provided footstep sequence (list of [x, y, z, yaw]). Targets beyond the
            sequence length are hidden at (0, 0, -10).

    Attributes:
        robot_xml_path: Path to the input robot XML file.
        max_boxes: Maximum number of stepping stones to visualize.
        model, data: MuJoCo model and data instances after loading.
    """
    def __init__(self, robot_xml_path, max_boxes=30):
        self.robot_xml_path = os.path.abspath(robot_xml_path)
        self.max_boxes = max_boxes
        self.model = None
        self.data = None

    def load_model(self):
        """Load robot XML, add stepping stones, and build the model."""
        with open(self.robot_xml_path, 'r') as f:
            robot_xml = f.read()

        # Resolve absolute path for assets directory.
        robot_dir = Path(self.robot_xml_path).parent
        mesh_abs_path = (robot_dir / "assets").as_posix()
        robot_xml = robot_xml.replace('meshdir="assets"', f'meshdir="{mesh_abs_path}"')
        robot_xml = robot_xml.replace("meshdir='assets'", f"meshdir='{mesh_abs_path}'")

        # Append visual markers for each possible footstep target.
        markers_xml = ""
        for i in range(self.max_boxes):
            markers_xml += f'''
            <!-- Stepping stone (box) -->
            <body name="step_{i}" pos="0 0 -10" quat="1 0 0 0">
                <geom type="box" size="0.100 1.0 0.05" rgba="0.8 0.8 0.8 1" group="1"/>
            </body>
            <!-- Center dot (sphere) -->
            <body name="step_dot_{i}" pos="0 0 -10" quat="1 0 0 0">
                <geom type="sphere" size="0.03" rgba="1.0 0.0 0.0 1" group="1"/>
            </body>
            <!-- Direction arrow -->
            <body name="step_arrow_{i}" pos="0 0 -10" quat="1 0 0 0">
                <geom type="box" size="0.08 0.01 0.01" rgba="0.0 0.0 1.0 1" group="1"/>
            </body>
            '''
        full_xml = robot_xml.replace('</worldbody>', markers_xml + '</worldbody>')

        self.model = mujoco.MjModel.from_xml_string(full_xml)
        self.data = mujoco.MjData(self.model)
        return self.model, self.data

    def update_boxes(self, model, data, sequence):
        """Update the position and orientation of stepping stone markers."""
        num_steps = len(sequence)
        for i in range(self.max_boxes):
            box_body_name = f"step_{i}"
            box_body_id = model.body(box_body_name).id
            if i < num_steps:
                x, y, z, theta = sequence[i]
                geom_addr = model.body_geomadr[box_body_id]
                box_h = model.geom_size[geom_addr][2]
                model.body_pos[box_body_id] = [x, y, z - box_h]
                quat_scipy = R.from_euler('z', theta).as_quat()
                quat_mujoco = [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]
                model.body_quat[box_body_id] = quat_mujoco

                # Center dot.
                dot_body_name = f"step_dot_{i}"
                dot_body_id = model.body(dot_body_name).id
                model.body_pos[dot_body_id] = [x, y, z]
                model.body_quat[dot_body_id] = quat_mujoco

                # Direction arrow.
                arrow_body_name = f"step_arrow_{i}"
                arrow_body_id = model.body(arrow_body_name).id
                model.body_pos[arrow_body_id] = [x, y, z]
                model.body_quat[arrow_body_id] = quat_mujoco
            else:
                for body_name in [box_body_name, f"step_dot_{i}", f"step_arrow_{i}"]:
                    body_id = model.body(body_name).id
                    model.body_pos[body_id] = [0, 0, -10]
                    model.body_quat[body_id] = [1, 0, 0, 0]