"""DAgger distillation for G1 footstep tracking.

What it does
------------
Trains a student policy to imitate a (privileged) teacher via DAgger:
  1. the student rolls out and visits states,
  2. the teacher labels those states with deterministic actions,
  3. (student_obs, teacher_action) pairs are aggregated into a growing buffer,
  4. the student is trained with MSE, then the loop repeats.

Teacher: the pretrained baseline (MultiInputPolicy). It consumes the full obs
    dict {actor_obs(41), critic_obs(58)}, so its action uses privileged info.
Student: an actor-only MLP. It consumes ONLY actor_obs with the pelvis-height
    channel removed, stacked over a history window. No critic, initialized from
    scratch, and NOT compressed (same width as the teacher).
Normalization: the exact frozen VecNormalize statistics used to train the
    teacher (pretrained_models/vec_normalize.pkl).

Outputs are written to ./distill_dagger/: student checkpoints, run config, and
TensorBoard logs.

Parameters
----------
--rounds N              DAgger rounds (default 600).
--steps-per-round N     Student environment steps collected per round (default 8192).
--history-len N         History window length stacked into the student input (default 12).
--hidden ...            Student hidden layer sizes (default 512 512 256).
--updates-per-round N   Gradient updates per round (default 2000).
--batch-size N          Training mini-batch size (default 512).
--lr F                  Adam learning rate, fixed (default 1e-4).
--teacher-rollout-ratio F  Fraction of rollout steps driven by the teacher
                        action instead of the student (default 0.3).
--buffer-size N         Aggregation buffer capacity (default 2_000_000).
--save-interval N       Save a checkpoint every N rounds (default 200).
--eval-interval N       Evaluate every N rounds (default 5).
--eval-episodes N       Episodes per evaluation (default 3).
--mirror-prob F         Mirror augmentation probability (default 0.5).
--seed N                Random seed (default 42).
--teacher-model PATH    Teacher policy checkpoint (.zip). Default: the baseline
                        pretrained_models/ppo_G1_FootstepTracking.zip.
--teacher-vecnorm PATH  VecNormalize stats for the teacher (and the student env).
                        Default: pretrained_models/vec_normalize.pkl.
--name STR              Suffix for run_config_<name>.json. If omitted, an
                        auto-incrementing index (1, 2, ...) is used.

Usage
-----
# Train with the default baseline teacher:
uv run python distill_dagger.py \
    --rounds 600 \
    --save-interval 200

# Specify a custom teacher model and normalization explicitly:
uv run python distill_dagger.py \
    --teacher-model pretrained_models/ppo_G1_FootstepTracking.zip \
    --teacher-vecnorm pretrained_models/vec_normalize.pkl
"""

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from env.g1_env import G1Env  # noqa: E402
from env_utils.mirrorwrapper import MirrorWrapper  # noqa: E402

ROBOT_XML = PROJECT_ROOT / "robot" / "g1_processed.xml"
TEACHER_MODEL_PATH = PROJECT_ROOT / "pretrained_models" / "ppo_G1_FootstepTracking.zip"
TEACHER_VECNORM_PATH = PROJECT_ROOT / "pretrained_models" / "vec_normalize.pkl"
OUTPUT_DIR = PROJECT_ROOT / "distill_dagger"

# actor_obs layout (see env/utils/observation_builder.py):
# joint_angles(12) | joint_vels(12) | pelvis_height(1) | foot_pos(3) |
# next_pos(3) | foot_yaw(1) | next_yaw(1) | phase(2) | orient(3) | angvel(3)
ACTOR_OBS_DIM = 41
PELVIS_HEIGHT_IDX = 24  # removed for the student (hard to deploy on the real robot)
ACTION_DIM = 12


class StudentPolicy(nn.Module):
    """Actor-only MLP mapping a history-stacked obs window to a 12-dim action."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HistoryWindow:
    """Keeps the last `history_len` student-base frames, oldest -> newest."""

    def __init__(self, history_len: int, base_dim: int):
        self.history_len = history_len
        self.base_dim = base_dim
        self.buf = deque(maxlen=history_len)
        self.reset()

    def reset(self):
        # Zero-fill so the mirrored/unnormalized semantics stay self-consistent.
        self.buf.clear()
        for _ in range(self.history_len):
            self.buf.append(np.zeros(self.base_dim, dtype=np.float32))

    def push(self, base_obs: np.ndarray):
        self.buf.append(base_obs.astype(np.float32))

    def get(self) -> np.ndarray:
        return np.concatenate(self.buf).astype(np.float32)


def extract_student_base(actor_obs: np.ndarray) -> np.ndarray:
    """Remove the pelvis-height channel (index PELVIS_HEIGHT_IDX) from actor_obs."""
    return np.concatenate([actor_obs[:PELVIS_HEIGHT_IDX], actor_obs[PELVIS_HEIGHT_IDX + 1 :]]).astype(np.float32)


class DaggerBuffer:
    """Growing replay buffer; overwrites the oldest sample once full."""

    def __init__(self, obs_dim: int, act_dim: int, max_size: int):
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act = np.zeros((max_size, act_dim), dtype=np.float32)
        self.max_size = max_size
        self.size = 0
        self.ptr = 0

    def add(self, obs: np.ndarray, act: np.ndarray):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, n: int):
        idx = np.random.randint(0, self.size, n)
        return self.obs[idx], self.act[idx]

    def __len__(self):
        return self.size


def build_env(n_envs: int, mirror_prob: float, vecnorm_path):
    """Env with no DR and no obs noise, wrapped in DummyVecEnv + frozen VecNormalize."""

    def _init():
        env = G1Env(robot_xml_path=str(ROBOT_XML), enable_domain_randomization=False)
        env.obs_builder.training = False  # no observation noise
        return MirrorWrapper(env, mirror_prob=mirror_prob)

    vec_env = DummyVecEnv([_init for _ in range(n_envs)])
    vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
    vec_env.training = False  # freeze running statistics
    vec_env.norm_reward = False
    return vec_env


def get_actor_norm_stats(vec_env):
    """Extract the frozen (mean, var) used to normalize actor_obs."""
    rms = vec_env.obs_rms
    if isinstance(rms, dict):
        rms = rms["actor_obs"]
    return np.asarray(rms.mean, dtype=np.float32), np.asarray(rms.var, dtype=np.float32)


def collect(vec_env, student, teacher, hist, buffer, steps, device, teacher_rollout_ratio=0.0):
    """Run `steps` with mixed rollout (teacher_rollout_ratio of steps driven by
    the teacher action), label each visited state with the teacher, and store
    (student_input, teacher_action) in the buffer."""
    student.eval()
    obs = vec_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    hist.reset()

    ep_rewards, ep_reward = [], 0.0
    action_diffs = []

    for _ in range(steps):
        actor_obs = obs["actor_obs"][0]  # already normalized by VecNormalize
        hist.push(extract_student_base(actor_obs))
        student_input = hist.get()

        with torch.no_grad():
            student_action = (
                student(torch.as_tensor(student_input, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
            )

        # Teacher labels the same state (it sees the privileged obs dict).
        teacher_action = teacher.predict(obs, deterministic=True)[0][0]

        buffer.add(student_input, teacher_action)
        action_diffs.append(float(np.mean(np.abs(student_action - teacher_action))))

        # Drive the env with the teacher action for a fraction of the rollout,
        # so the buffer also covers the teacher's own state distribution.
        step_action = teacher_action if np.random.random() < teacher_rollout_ratio else student_action

        obs, reward, done, _ = vec_env.step(step_action[None, :])
        ep_reward += float(reward[0])
        if done[0]:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            hist.reset()

    return {
        "ep_rewards": ep_rewards,
        "mean_action_diff": float(np.mean(action_diffs)) if action_diffs else 0.0,
    }


def train_student(student, buffer, optimizer, updates, batch_size, device):
    """Fit the student to teacher labels with MSE. Returns mean loss."""
    student.train()
    losses = []
    for _ in range(updates):
        obs_b, act_b = buffer.sample(batch_size)
        obs_b = torch.as_tensor(obs_b, device=device)
        act_b = torch.as_tensor(act_b, device=device)
        pred = student(obs_b)
        loss = functional.mse_loss(pred, act_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate(vec_env, student, hist, episodes, device, max_steps=2000):
    """Run the student headless and return the mean episode reward."""
    student.eval()
    obs = vec_env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    hist.reset()

    ep_rewards, ep_reward, steps = [], 0.0, 0
    while len(ep_rewards) < episodes and steps < max_steps * episodes:
        actor_obs = obs["actor_obs"][0]
        hist.push(extract_student_base(actor_obs))
        student_input = hist.get()
        with torch.no_grad():
            action = student(torch.as_tensor(student_input, device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
        obs, reward, done, _ = vec_env.step(action[None, :])
        ep_reward += float(reward[0])
        steps += 1
        if done[0]:
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
            hist.reset()
    return float(np.mean(ep_rewards)) if ep_rewards else float("nan")


def save_student(path, student, meta):
    """Save the student state_dict together with its metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": student.state_dict(), **meta}, str(path))


def main():
    parser = argparse.ArgumentParser(description="DAgger distillation for G1 footstep tracking")
    parser.add_argument("--rounds", type=int, default=600)
    parser.add_argument("--steps-per-round", type=int, default=8192)
    parser.add_argument("--history-len", type=int, default=12)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 256])
    parser.add_argument("--updates-per-round", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--teacher-rollout-ratio", type=float, default=0.3)
    parser.add_argument("--buffer-size", type=int, default=2_000_000)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--mirror-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--teacher-model", type=str, default=str(TEACHER_MODEL_PATH), help="Teacher policy checkpoint (.zip)."
    )
    parser.add_argument(
        "--teacher-vecnorm", type=str, default=str(TEACHER_VECNORM_PATH), help="VecNormalize stats for the teacher."
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Suffix for run_config_<name>.json. If omitted, an auto-incrementing index is used.",
    )
    args = parser.parse_args()

    teacher_model_path = Path(args.teacher_model)
    teacher_vecnorm_path = Path(args.teacher_vecnorm)
    for p in (ROBOT_XML, teacher_model_path, teacher_vecnorm_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(OUTPUT_DIR / "tb"))

    vec_env = build_env(n_envs=1, mirror_prob=args.mirror_prob, vecnorm_path=teacher_vecnorm_path)
    mean, var = get_actor_norm_stats(vec_env)

    teacher = PPO.load(str(teacher_model_path), device=device)
    teacher.policy.eval()
    print(f"[teacher] loaded {teacher_model_path.name} | policy={type(teacher.policy).__name__}")

    base_dim = ACTOR_OBS_DIM - 1  # pelvis height removed
    student_obs_dim = base_dim * args.history_len
    student = StudentPolicy(student_obs_dim, ACTION_DIM, tuple(args.hidden)).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    print(f"[student] obs_dim={student_obs_dim} (base={base_dim} x history={args.history_len}) | hidden={args.hidden}")

    buffer = DaggerBuffer(student_obs_dim, ACTION_DIM, args.buffer_size)
    hist = HistoryWindow(args.history_len, base_dim)

    meta = {
        "student_obs_dim": student_obs_dim,
        "base_dim": base_dim,
        "history_len": args.history_len,
        "pelvis_height_idx": PELVIS_HEIGHT_IDX,
        "action_dim": ACTION_DIM,
        "hidden": list(args.hidden),
        "norm_mean": mean.tolist(),
        "norm_var": var.tolist(),
        "mirror_prob": args.mirror_prob,
    }
    # run_config file: run_config_<name>.json if --name is given, otherwise
    # run_config_<n>.json with the smallest unused n.
    if args.name:
        config_path = OUTPUT_DIR / f"run_config_{args.name}.json"
    else:
        n = 1
        while (OUTPUT_DIR / f"run_config_{n}.json").exists():
            n += 1
        config_path = OUTPUT_DIR / f"run_config_{n}.json"
    with open(config_path, "w") as f:
        json.dump({**vars(args), **meta}, f, indent=2)
    print(f"[config] saved to {config_path}")

    best_eval = -np.inf
    for rnd in range(args.rounds):
        stats = collect(
            vec_env,
            student,
            teacher,
            hist,
            buffer,
            args.steps_per_round,
            device,
            args.teacher_rollout_ratio,
        )
        loss = train_student(student, buffer, optimizer, args.updates_per_round, args.batch_size, device)

        writer.add_scalar("train/loss", loss, rnd)
        writer.add_scalar("train/buffer_size", len(buffer), rnd)
        writer.add_scalar("collect/mean_action_diff", stats["mean_action_diff"], rnd)
        if stats["ep_rewards"]:
            writer.add_scalar("collect/ep_reward", float(np.mean(stats["ep_rewards"])), rnd)

        msg = (
            f"[round {rnd + 1}/{args.rounds}] loss={loss:.4f} | "
            f"buf={len(buffer)} | act_diff={stats['mean_action_diff']:.4f}"
        )
        if stats["ep_rewards"]:
            msg += f" | collect_rew={np.mean(stats['ep_rewards']):.2f}"
        print(msg)

        if (rnd + 1) % args.save_interval == 0:
            save_student(OUTPUT_DIR / f"student_round{rnd + 1}.pt", student, meta)

        if (rnd + 1) % args.eval_interval == 0 or rnd == args.rounds - 1:
            eval_rew = evaluate(vec_env, student, hist, args.eval_episodes, device)
            writer.add_scalar("eval/reward", eval_rew, rnd)
            print(f"    [eval] mean_reward={eval_rew:.2f}")
            if eval_rew > best_eval:
                best_eval = eval_rew
                save_student(OUTPUT_DIR / "student_best.pt", student, meta)

    save_student(OUTPUT_DIR / "student_final.pt", student, meta)
    writer.close()
    vec_env.close()
    print(f"\n[done] artifacts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
