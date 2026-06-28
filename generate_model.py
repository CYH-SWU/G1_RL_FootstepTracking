#!/usr/bin/env python3
"""
调用 gen_xml 模块，生成 g1_processed.xml 文件。
"""

import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径，以便导入 robot 包
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from robot.gen_xml import process_g1_model

if __name__ == "__main__":
    output_path = process_g1_model()
    if output_path:
        print(f"✅ 成功生成模型文件: {output_path}")
    else:
        print("❌ 生成失败，请检查输入文件是否存在。")