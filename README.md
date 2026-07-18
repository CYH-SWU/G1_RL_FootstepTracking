# G1 全向步点跟踪控制

基于深度强化学习的宇树 G1 人形机器人全向步点跟踪控制系统。

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3.0+-green.svg)](https://mujoco.org/)
[![SB3](https://img.shields.io/badge/SB3-1.7.0+-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
---

## 📖 项目简介

本项目为宇树 G1 人形机器人构建了一套全向步点跟踪行走控制系统，在 MuJoCo 物理仿真环境中基于 PPO 算法训练而成。机器人接收预生成的步态序列（含落足位置与朝向），策略网络以本体感知信息（关节角度/速度、IMU 姿态）和任务指令（步点位置/偏航、步态相位）为输入，输出 12 个关节的位置增量指令，驱动双腿精确跟踪每一步，实现稳定、全向的双足行走。


## ⚙️ 环境与训练设置

### 支持的行走模式
- 前进 (FORWARD)
- 后退 (BACKWARD)
- 侧移 (LATERAL)
- 弧形行走 (CURVED)
- 站立 (STANDING)

### 地形支持
平地 + 0.05m 台阶，通过课程学习渐进式引入。

### 机器人模型与关节配置

- **自由度**：宇树 G1 29 自由度，腿部 12 个主动关节（髋×3、膝×1、踝×2）×左右。
- **标称姿态**：
  - 髋俯仰 -0.5236 rad (-30°)
  - 膝俯仰 0.8727 rad (50°)
  - 踝俯仰 -0.3491 rad (-20°)
  - 腰俯仰 0.1500 rad
- **PD 控制器参数**：
  - 髋部：KP=115, dampratio=0.65
  - 膝部：KP=172, dampratio=0.55
  - 踝部：KP=46, dampratio=0.40
- **关节力矩限制**：髋±139/±88 Nm，膝±139 Nm，踝±50 Nm。

### 观测与动作空间

环境返回字典观测，使用 `MultiInputPolicy` 自动拼接 `actor_obs` 和 `critic_obs`。

- **actor_obs (41维)**：
关节角度(12)、关节速度(12)、骨盆高度(1)、当前步点位置(3)、下一步点位置(3)、当前步点偏航(1)、下一步点偏航(1)、步态相位(2)、骨盆欧拉角(3)、骨盆角速度(3)。
- **critic_obs (17 + 41维)**：
基于先验经验归一化的actor_obs(41)、足底力(2)、线速度(3)、关节力矩(12)。
- **动作空间**：
12维连续值，范围[-1,1]，经`action_scale=0.25`映射为关节角度增量。
- **控制周期**：
0.015s (≈66.7Hz)，物理步长 0.005s (200Hz)。

### 步态参数
total_duration      1.30s
swing_duration      0.85s
stance_duration     0.45s

step_length         0.20m
step_width          0.237m
target_radius       0.16m

### 奖励函数设计

步点跟踪奖励（权重 0.45）
核心任务奖励，驱动机器人踩中目标步点。

足底力相位匹配奖励（权重 0.15）
引导策略在支撑相踩实地面、在摆动相抬离脚掌。

足底速度相位匹配奖励（权重 0.15）
引导策略在支撑相保持脚掌静止、在摆动相快速迈步。

躯干姿态奖励（权重 0.05）
鼓励骨盆偏航角与目标步点朝向对齐，确保机器人沿正确方向行走。

骨盆高度奖励（权重 0.05）
鼓励骨盆高度维持在标称值 0.7268m 附近。

上身稳定性奖励（权重 0.05）
鼓励头部与骨盆的 XY 投影距离最小化，保持上身稳定，避免行走过程中躯干过度晃动。

### 课程学习

- 前 3000 次迭代：仅平地
- 3000~11000 次迭代：台阶高度从 0 线性增加至 0.1m
- 11000 次迭代后：保持最大难度
- 台阶高度以 50% 概率取正（上坡）或负（下坡）。

### 训练超参数

n_steps         800
batch_size      64
n_epochs        3
gamma           0.99
gae_lambda      0.95
clip_range      0.15
learning_rate   1e-4
ent_coef        0.001
max_grad_norm   0.5
n_envs          16

learning_rate 训练过程中由性能回调自动调整

### 网络结构

- 策略类：MultiInputPolicy。
- Actor 网络：两层隐藏层，每层 256 个神经元，ReLU 激活。
- Critic 网络：两层隐藏层，每层 256 个神经元，ReLU 激活。
- 网络独立性：Actor 和 Critic 互不共享。
- 权重初始化：正交初始化
- 动作分布：对角高斯分布

### 数据增强与归一化

- `MirrorWrapper`：50% 概率左右镜像翻转观测和动作
- `VecNormalize`：仅对 `actor_obs` 归一化（零均值单位方差），裁剪范围 10.0

---

## 📁 项目结构
```plaintext
G1_RL_FootstepTracking
├── envs/
│   ├── G1FootstepEnv.py                <--- 主环境类
│   └── utils                           <--- 环境模块     
│       ├── config.py
│       ├── observation_builder.py
│       ├── reward_calculator.py
│       ├── step_sequence.py
│       └── terrain_generator.py
├── env_utils/                          <--- 环境工具
│   ├── mirrorwrapper.py
│   └── reward_functions.py
├── rl/                                 <--- 训练自定义模块
│   ├── callbacks.py
│   └── policy.py
├── robot/                              <--- 机器人配置
│   ├── assets/
│   ├── gen_xml.py
│   └── unitree_g1.xml
├── scripts/                            <--- 辅助脚本
│   ├── compute_height.py
│   ├── compute_max_step.py
│   └── test_pose.py
├── train.py                            <--- 训练主入口
└── test.py                             <--- 模型测试
```

# 项目下载
```bash
git clone https://github.com/CYH-SWU/G1_RL_FootstepTracking.git
```

# 安装依赖
```bash
uv sync
```

# 训练模型
生成处理后的G1机器人xml文件
```bash
uv run python robot/gen_xml.py
```

训练
```bash
uv run python train.py -i 20000 --save-interval 500 --eval-interval 500
```

继续训练
```bash
uv run python train.py \
  -i 20000 \
  --model checkpoints/ppo_g1_xxx_steps.zip \
  --norm checkpoints/vec_normalize_final.pkl
```

# 评估与可视化
```bash
uv run python test.py \
  --model checkpoints/ppo_g1_final.zip \
  --norm checkpoints/vec_normalize_final.pkl \
  --episodes 20 \
  --difficulty 1.0
```

# 其他辅助脚本
```bash
uv run python scripts/compute_height.py
```
计算机器人标称姿态下的骨盆高度

```bash
uv run python scripts/compute_max_step.py
```
计算机器人在当前config设置下可达到的最大步幅

```bash
uv run python scripts/test_pose.py
```
可视化查看机器人的标称姿态

## 参考文献
**Learning Humanoid Walking**
R. P. Singh et al., “Learning Bipedal Walking On Planned Footsteps For Humanoid Robots,” in IEEE-RAS Humanoids, 2022.
R. P. Singh et al., “Learning Bipedal Walking for Humanoids with Current Feedback,” arXiv:2303.03724, 2023.
R. P. Singh et al., “Robust Humanoid Walking on Compliant and Uneven Terrain with Deep RL,” IEEE Access, 2024.
GitHub Repository: https://github.com/rohanpsingh/LearningHumanoidWalking

**Unitree RL Gym**
GitHub Repository: https://github.com/unitreerobotics/unitree_rl_gym

