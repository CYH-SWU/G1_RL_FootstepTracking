import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_INPUT = Path(__file__).parent / "unitree_g1.xml"
DEFAULT_OUTPUT = Path(__file__).parent / "g1_processed.xml"

KEEP_JOINT_KEYWORDS = ["hip", "knee", "ankle", "waist"]


def add_ground_with_texture(root):
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    tex = asset.find("texture[@name='ground_tex']")
    if tex is None:
        tex = ET.SubElement(asset, "texture")
        tex.set("name", "ground_tex")
        tex.set("type", "2d")
        tex.set("builtin", "checker")
        tex.set("rgb1", "0.2 0.3 0.4")
        tex.set("rgb2", "0.6 0.7 0.8")
        tex.set("width", "300")
        tex.set("height", "300")
        tex.set("mark", "edge")
        tex.set("random", "0.01")

    mat = asset.find("material[@name='groundplane']")
    if mat is None:
        mat = ET.SubElement(asset, "material")
        mat.set("name", "groundplane")
        mat.set("texture", "ground_tex")
        mat.set("texrepeat", "2 2")
        mat.set("texuniform", "true")
        mat.set("reflectance", "0.2")

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    for geom in worldbody.findall(".//geom"):
        if geom.get("name") in ("floor", "ground"):
            parent = worldbody.find(f".//body[geom='{geom}']") or worldbody
            parent.remove(geom)
    for body in worldbody.findall("body"):
        if body.get("name") == "floor":
            worldbody.remove(body)

    floor_body = ET.SubElement(worldbody, "body")
    floor_body.set("name", "floor")
    floor_geom = ET.SubElement(floor_body, "geom")
    floor_geom.set("name", "floor")
    floor_geom.set("type", "plane")
    floor_geom.set("size", "0 0 0.25")
    floor_geom.set("material", "groundplane")
    floor_geom.set("pos", "0 0 0")
    floor_geom.set("friction", "0.7 0.005 0.0001")


def process_g1_model(input_path=None, output_path=None, add_ground=True):
    in_path = Path(input_path) if input_path else DEFAULT_INPUT
    out_path = Path(output_path) if output_path else DEFAULT_OUTPUT

    if not in_path.exists():
        print(f"Error: Input model not found: {in_path}")
        return None

    tree = ET.parse(in_path)
    root = tree.getroot()

    actuator_node = root.find(".//actuator")
    if actuator_node is not None:
        kept = 0
        for actuator in list(actuator_node.findall("position")):
            joint_name = actuator.get("joint")
            if joint_name is None:
                continue

            keep = any(kw in joint_name.lower() for kw in KEEP_JOINT_KEYWORDS)

            if keep:
                kept += 1
                if "inheritrange" in actuator.attrib:
                    del actuator.attrib["inheritrange"]
                actuator.set("inheritrange", "0")
                actuator.set("kp", "250")
                actuator.set("dampratio", "1")

                joint = root.find(f".//joint[@name='{joint_name}']")
                if joint is not None:
                    joint_range = joint.get("range")
                    if joint_range:
                        actuator.set("ctrlrange", joint_range)
                    else:
                        print(f"Warning: joint '{joint_name}' has no range, ctrlrange not set.")
                else:
                    print(f"Warning: no joint found for actuator '{joint_name}', ctrlrange not set.")

            else:
                actuator_node.remove(actuator)

                joint = root.find(f".//joint[@name='{joint_name}']")
                if joint is not None:
                    joint.set("type", "hinge")
                    joint.set("range", "0 0")
                    joint.set("damping", "10000")
                    joint.set("armature", "0")

                    if "actuatorfrcrange" in joint.attrib:
                        del joint.attrib["actuatorfrcrange"]
                    if "frictionloss" in joint.attrib:
                        del joint.attrib["frictionloss"]
        print(f"Kept {kept} actuators (legs + waist).")

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    ET.SubElement(contact, "exclude", body1="torso_link", body2="left_shoulder_pitch_link")
    ET.SubElement(contact, "exclude", body1="torso_link", body2="right_shoulder_pitch_link")

    ET.SubElement(contact, "exclude", body1="left_shoulder_pitch_link", body2="left_elbow_link")
    ET.SubElement(contact, "exclude", body1="right_shoulder_pitch_link", body2="right_elbow_link")

    ET.SubElement(contact, "exclude", body1="left_shoulder_pitch_link", body2="pelvis")
    ET.SubElement(contact, "exclude", body1="right_shoulder_pitch_link", body2="pelvis")

    if add_ground:
        add_ground_with_texture(root)

    keyframe = root.find(".//keyframe")
    if keyframe is not None:
        if "ctrl" in keyframe.attrib:
            del keyframe.attrib["ctrl"]
        for key in keyframe.findall("key"):
            if "ctrl" in key.attrib:
                del key.attrib["ctrl"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Processed model saved to: {out_path}")
    return out_path