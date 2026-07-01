#!/usr/bin/env python3
"""
测试 G1 机器人模型加载与关键帧切换。
加载 robot/g1_processed.xml，切换到 stand 关键帧，启动 MuJoCo 查看器。
"""

import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
MODEL_PATH = PROJECT_ROOT / "robot" / "g1_processed.xml"


def main():
    if not MODEL_PATH.exists():
        print(f"错误：模型文件不存在: {MODEL_PATH}")
        print("请先运行 gen_xml.py 生成模型文件。")
        return

    # 加载模型
    print(f"加载模型: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # 切换到 stand 关键帧
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print("已切换到 'stand' 关键帧")
    else:
        print("警告：未找到 'stand' 关键帧，使用默认重置。")
        mujoco.mj_resetData(model, data)

    # 前向计算，更新派生量
    
    mujoco.mj_forward(model, data)

    # 启动查看器
    print("启动 MuJoCo 查看器。按 Esc 或关闭窗口退出。")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            # 推进仿真
            mujoco.mj_step(model, data)
            viewer.sync()
            # 保持实时
            elapsed = time.time() - step_start
            time_to_sleep = model.opt.timestep - elapsed
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)


if __name__ == "__main__":
    main()