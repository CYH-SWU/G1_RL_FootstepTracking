# G1 Omnidirectional Footstep Tracking Control

Deep reinforcement learning based omnidirectional footstep tracking control system for Unitree G1 humanoid robot.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-2.3.0+-green.svg)](https://mujoco.org/)
[![SB3](https://img.shields.io/badge/SB3-1.7.0+-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/CYH-SWU/G1_RL_FootstepTracking/actions/workflows/ci.yml/badge.svg)](https://github.com/CYH-SWU/G1_RL_FootstepTracking/actions/workflows/ci.yml)
[![Lint](https://github.com/CYH-SWU/G1_RL_FootstepTracking/actions/workflows/lint.yml/badge.svg)](https://github.com/CYH-SWU/G1_RL_FootstepTracking/actions/workflows/lint.yml)

---

## 🤖Project Overview

This project builds an omnidirectional footstep tracking walking control system for the Unitree G1 humanoid robot, trained in the MuJoCo physics simulation environment using the PPO algorithm. The robot receives pre-generated footstep sequences (including foot placement positions and orientations). The policy network takes proprioceptive information (joint angles/velocities, IMU attitude) and task instructions (footstep positions/yaws, gait phase) as input, and outputs 12-dimensional joint position increment commands to drive both legs to accurately track each footstep, achieving stable omnidirectional bipedal walking.


## 🎛️Robot Model and Joint Configuration

- **Degrees of Freedom**: Unitree G1 with 29 DOF, 12 active joints in legs (3x hip, 1x knee, 2x ankle) per side.
- **Nominal Posture**:
  - Hip pitch: -0.5236 rad (-30 deg)
  - Knee pitch: 0.8727 rad (50 deg)
  - Ankle pitch: -0.3491 rad (-20 deg)
  - Waist pitch: 0.1500 rad
- **PD Controller Gains**:

| Joint | KP | Dampratio |
|-------|----|-----------|
| Hip   | 115 | 0.65      |
| Knee  | 172 | 0.55      |
| Ankle | 46  | 0.40      |

- **Joint Torque Limits**: Hip +-139/+-88 Nm, Knee +-139 Nm, Ankle +-50 Nm.



## ⚙️Environment and Training Setup

### Supported Walking Modes
- FORWARD
- BACKWARD
- LATERAL
- INPLACE
- CURVED
- STANDING

### Terrain Support
Flat ground + 0.05m steps, progressively introduced via curriculum learning.

### Observation and Action Space

The environment returns dictionary observations, using `MultiInputPolicy` to automatically concatenate `actor_obs` and `critic_obs`.

- **actor_obs (41 dims)**:
Joint angles (12), joint velocities (12), pelvis height (1), current footstep position (3), next footstep position (3), current footstep yaw (1), next footstep yaw (1), gait phase (2), pelvis Euler angles (3), pelvis angular velocity (3).
- **critic_obs (17 + 41 dims)**:
Normalized actor_obs based on prior experience (41), foot forces (2), linear velocity (3), joint torques (12).
- **Action Space**:
12-dimensional continuous values in range [-1,1], mapped to joint angle increments via `action_scale=0.25`.
- **Control Cycle**:
0.015s (approx 66.7Hz), physics step 0.005s (200Hz).

### Gait Parameters
```bash
total_duration      1.30s
swing_duration      0.85s
stance_duration     0.45s

step_length         0.20m
step_width          0.237m
target_radius       0.16m
```

### 🎯Reward Function Design
```plaintext
Footstep Tracking Reward (weight 0.45):
  Core task reward that drives the robot to step onto target footholds.

Foot Force Phase Matching Reward (weight 0.15):
  Guides the policy to press down firmly during stance phase and lift off during swing phase.

Foot Velocity Phase Matching Reward (weight 0.15):
  Guides the policy to keep feet stationary during stance phase and move quickly during swing phase.

Torso Attitude Reward (weight 0.05):
  Encourages pelvis yaw to align with target footstep yaw, ensuring the robot walks in the correct direction.

Pelvis Height Reward (weight 0.05):
  Encourages pelvis height to be maintained near the nominal value of 0.7268m.

Upper Body Stability Reward (weight 0.05):
  Encourages minimizing the XY distance between head and pelvis to maintain upper body stability and avoid excessive torso swaying during walking.
```

### 📈Curriculum Learning

- First 3000 iterations: flat ground only
- 3000~11000 iterations: step height linearly increases from 0 to 0.05m
- After 11000 iterations: maintains maximum difficulty
- Step height has 50% probability of being positive (upward step) or negative (downward step).

### Training Hyperparameters
```bash
n_steps         800
batch_size      64
n_epochs        3
gamma           0.99
gae_lambda      0.95
clip_range      0.18
learning_rate   1e-4
ent_coef        0.001
max_grad_norm   0.5
n_envs          14

learning_rate is automatically adjusted by the performance callback during training.
```

### Network Architecture

- **Policy Class**: MultiInputPolicy.
- **Actor Network**: Two hidden layers with 256 neurons each, ReLU activation.
- **Critic Network**: Two hidden layers with 256 neurons each, ReLU activation.
- **Network Independence**: Actor and Critic do not share parameters.
- **Weight Initialization**: Orthogonal initialization.
- **Action Distribution**: Diagonal Gaussian distribution.

### Data Augmentation and Normalization

- `MirrorWrapper`: 50% probability of flipping observations and actions left-right.
- `VecNormalize`: Only normalizes `actor_obs` (zero mean, unit variance), clip range 10.0.


## 📂Project Structure

```plaintext
G1_RL_FootstepTracking/
├── env/
│   ├── g1_env.py                       # Main environment class
│   └── utils/                          # Environment modules
│       ├── config.py
│       ├── observation_builder.py
│       ├── reward_calculator.py
│       ├── step_sequence.py
│       └── terrain_generator.py
├── env_utils/                          # Environment utilities
│   ├── mirrorwrapper.py
│   └── reward_functions.py
├── rl/                                 # Training custom modules
│   ├── callbacks.py
│   └── policy.py
├── robot/                              # Robot configuration
│   ├── assets/
│   ├── gen_xml.py
│   └── unitree_g1.xml
├── scripts/                            # Auxiliary scripts
│   ├── compute_height.py
│   ├── compute_max_step.py
│   └── test_pose.py
├── tests/                              # Unit tests (pytest)
│   ├── test_env.py
│   ├── test_imports.py
│   ├── test_mirrorwrapper.py
│   ├── test_policy.py
│   └── test_step_sequence.py
├── train.py                            # Main training entry
└── test.py                             # Model testing entry
```

## 🔁Clone the Repository
```bash
git clone https://github.com/CYH-SWU/G1_RL_FootstepTracking.git
cd G1_RL_FootstepTracking
```

## Install Dependencies
```bash
uv sync --all-extras
```
**Note**: Python 3.12+ is required.


## Train the Model
### Generate the processed G1 robot XML file:
```bash
uv run python robot/gen_xml.py
```
### Start training from scratch:
```bash
uv run python train.py
```
```bash
uv run python train.py \
  ---iterations 20000 \
  --save_interval 500 \
  --eval_interval 500
```
### Resume training from a checkpoint:
```bash
uv run python train.py \
  ---iterations 20000 \
  --model checkpoints/ppo_g1_xxx_steps.zip \
  --norm checkpoints/vec_normalize_final.pkl
```


## Evaluate and Visualize
```bash
uv run python test.py \
  --model checkpoints/ppo_g1_final.zip \
  --norm checkpoints/vec_normalize_final.pkl \
  --episodes 20 \
  --difficulty 1.0
```


## 🛠️Auxiliary Scripts
### Compute the pelvis height under the nominal posture.
```bash
uv run python scripts/compute_height.py
```
### Compute the maximum achievable step length under the current config.
```bash
uv run python scripts/compute_max_step.py
```
### Visualize the robot's nominal posture.
```bash
uv run python scripts/test_pose.py
```


## 🔍Testing
### Run all unit tests with coverage
```bash
uv run pytest tests/ -v --cov=env --cov=rl --cov=env_utils --cov-report=term
```
### Check code style (Ruff)
```bash
uv run ruff check .
uv run ruff format . --check
```
### Auto-fix style issues
```bash
uv run ruff check . --fix && uv run ruff format .
```


## 🧪 CI/CD
This project uses GitHub Actions to automatically run:
- Unit tests (with coverage) on Python 3.12
- Linting and formatting check with Ruff

All CI jobs must pass before merging a pull request.


## 📚References
**Learning Humanoid Walking**

R. P. Singh et al., "Learning Bipedal Walking On Planned Footsteps For Humanoid Robots," in *IEEE-RAS Humanoids*, 2022.
R. P. Singh et al., "Learning Bipedal Walking for Humanoids with Current Feedback," *arXiv:2303.03724*, 2023.
R. P. Singh et al., "Robust Humanoid Walking on Compliant and Uneven Terrain with Deep RL," *IEEE Access*, 2024.
GitHub Repository: [https://github.com/rohanpsingh/LearningHumanoidWalking](https://github.com/rohanpsingh/LearningHumanoidWalking)

**Unitree RL Gym**

GitHub Repository: [https://github.com/unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)


## 🎉Acknowledgments

- This project uses the Unitree G1 robot model, which is Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. and is licensed under the BSD 3-Clause License.
- The footstep tracking framework is inspired by the Learning Humanoid Walking (LHW) project by Rohan P. Singh, licensed under the BSD 2-Clause License.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.