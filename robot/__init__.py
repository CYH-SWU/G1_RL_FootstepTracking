from pathlib import Path

from .gen_xml import process_g1_model

ROBOT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = ROBOT_DIR.parent.parent
ORIGINAL_XML = ROBOT_DIR / "unitree_g1.xml"
PROCESSED_XML = ROBOT_DIR / "g1_processed.xml"

__all__ = [
    "process_g1_model",
    "ORIGINAL_XML",
    "PROCESSED_XML",
    "ROBOT_DIR",
    "PROJECT_ROOT",
]
