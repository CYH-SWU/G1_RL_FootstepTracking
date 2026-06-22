'''
该脚本用于地形可视化

切换地形请手动修改地形标签字符串
为使机器人骨盆高度正常请手动设置机器人的骨盆高度
'''



import os
import sys
import time
import mujoco
import mujoco.viewer

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from planner_pipeline.terrain_generator import TerrainGenerator


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    robot_xml = os.path.join(project_root, "robot", "g1_processed.xml")
    mesh_dir = os.path.join(project_root, "robot", "assets")

    if not os.path.exists(robot_xml):
        print(f"错误:找不到机器人 XML 文件：{robot_xml}")
        return
    if not os.path.exists(mesh_dir):
        print(f"警告:STL 目录不存在：{mesh_dir}，将尝试从 XML 所在目录自动推断")

    print("初始化地形生成器...")
    terrain_gen = TerrainGenerator(robot_xml_path=robot_xml, mesh_dir=mesh_dir)

    print("生成平地地形...")
    model, data = terrain_gen.generate(
        mode="rough",          
        difficulty=1,       
        goal_pos=(7.5, 0.0)   
    )

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        # actuator_qpos_indices = [7,8,9,10,11,12,13,14,15,16,17,18,21]
        # data.ctrl[:] = data.qpos[actuator_qpos_indices]
        data.qpos[2] = 1.00 #手动修改骨盆高度
        print("已重置到 'stand' 关键帧并同步 ctrl。")
    else:
        print("警告: 未找到 'stand' 关键帧，使用默认重置。")
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)  

    print("启动 MuJoCo 仿真窗口...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("仿真运行中。关闭窗口或按 Esc 退出。")
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.time() - step_start
            time_to_sleep = model.opt.timestep - elapsed
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

        print("仿真已结束。")


if __name__ == "__main__":
    main()