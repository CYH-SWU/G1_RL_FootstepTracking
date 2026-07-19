import sys
from pathlib import Path

# 将项目根目录加入 Python 路径，确保能导入 env、rl 等模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
