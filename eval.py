"""Unified evaluation / visualization script for G1 footstep tracking.

What it does
------------
Loads either a trained RL policy (PPO, .zip) or a distilled student policy (.pt),
runs it in the G1 footstep-tracking environment, and either:
  * renders the simulation in real time (default), or
  * runs headless and writes aggregated metrics to eval_results/*.json (--eval).

If no --mode is given, all modes are evaluated sequentially:
    FORWARD, BACKWARD, LATERAL, CURVED, STANDING

--episodes means EPISODES PER MODE. With a single --mode it equals the total.
The policy type is auto-detected from the model extension (.zip -> rl, .pt -> student).

Parameters
----------
--model PATH        Policy file (.zip for RL, .pt for student). Auto-detected if omitted.
--norm PATH         VecNormalize statistics (.pkl). Auto-detected if omitted.
--policy-type       auto | rl | student. 'auto' infers from the model extension.
--mode NAME         Fixed mode. If omitted, all modes are swept in order.
--episodes N        Episodes per mode (default 100).
--env-max-steps N   Episode truncation length passed to the env (default 2500).
--max-steps N       Safety cap for the eval loop, must be >= --env-max-steps (default 3000).
--difficulty F      Curriculum difficulty in [0, 1] (default 0.0).
--device            cpu | cuda (default cpu).
--eval              Headless fast evaluation + JSON export. Omit to render.

Usage
-----
# 1. Teacher (RL): render
uv run python eval.py \
    --model pretrained_models/ppo_G1_FootstepTracking.zip \
    --norm pretrained_models/vec_normalize.pkl \
    --mode FORWARD

# 2. Teacher (RL): evaluate
uv run python eval.py \
    --eval \
    --model pretrained_models/ppo_G1_FootstepTracking.zip \
    --norm pretrained_models/vec_normalize.pkl \
    --episodes 100

# 3. Student: render
uv run python eval.py \
    --model distill_dagger/student_best.pt \
    --norm pretrained_models/vec_normalize.pkl \
    --mode FORWARD

# 4. Student: evaluate
uv run python eval.py \
    --eval \
    --model distill_dagger/student_best.pt \
    --norm pretrained_models/vec_normalize.pkl \
    --episodes 100
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from distill_dagger import HistoryWindow, StudentPolicy, extract_student_base  # noqa: E402
from env.g1_env import G1Env  # noqa: E402
from env.utils.step_sequence import WalkModes  # noqa: E402
from env_utils.mirrorwrapper import MirrorWrapper  # noqa: E402

ROBOT_XML = PROJECT_ROOT / "robot" / "g1_processed.xml"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "eval_results"

ALL_MODES = ["FORWARD", "BACKWARD", "LATERAL", "CURVED", "STANDING"]


# Policy adapters: unified interface reset() + act(obs) -> (12,)
class RLPolicyAdapter:
    """Stable-Baselines3 PPO policy (teacher). Stateless."""

    def __init__(self, model_path, device):
        self.model = PPO.load(str(model_path), device=device)
        self.device = device

    def reset(self):
        pass

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action[0], dtype=np.float32)


class StudentPolicyAdapter:
    """Distilled student policy with a history window (actor_obs only)."""

    def __init__(self, model_path, device):
        ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
        meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
        self.student = StudentPolicy(meta["student_obs_dim"], meta["action_dim"], tuple(meta["hidden"])).to(device)
        self.student.load_state_dict(ckpt["state_dict"])
        self.student.eval()
        self.device = device
        self.hist = HistoryWindow(int(meta["history_len"]), int(meta["base_dim"]))
        print(
            f"[student] obs_dim={meta['student_obs_dim']} "
            f"(base={meta['base_dim']} x history={meta['history_len']}) | hidden={meta['hidden']}"
        )

    def reset(self):
        self.hist.reset()

    def act(self, obs):
        actor_obs = obs["actor_obs"][0]  # already normalized by VecNormalize
        self.hist.push(extract_student_base(actor_obs))
        student_input = self.hist.get()
        with torch.no_grad():
            action = (
                self.student(torch.as_tensor(student_input, device=self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
            )
        return np.asarray(action, dtype=np.float32)


class EvalWrapper(gym.Wrapper):
    """Records per-episode completed / terminated / truncated and footfall hits.

    Does NOT modify the env. A hit is logged when t1 advances, using the target
    index BEFORE the advance.
    """

    def __init__(self, env):
        super().__init__(env)
        self.hits = []
        self.completed = False
        self.terminated = False
        self.truncated = False
        self.is_standing = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.hits = []
        self.completed = False
        self.terminated = False
        self.truncated = False
        self.is_standing = self.env.mode == WalkModes.STANDING
        return obs, info

    def step(self, action):
        t1_before = self.env.t1
        obs, reward, terminated, truncated, info = self.env.step(action)
        e = self.env

        # Completion latch is meaningless for STANDING (1-element sequence).
        if not self.is_standing and len(e.sequence) > 0 and e.t1 >= len(e.sequence) - 1:
            self.completed = True

        if e.t1 != t1_before and 0 <= t1_before < len(e.sequence):
            tgt = e.sequence[t1_before][:3]
            l_pos = e.data.xpos[e.left_foot_id]
            r_pos = e.data.xpos[e.right_foot_id]
            l_d = float(np.linalg.norm(l_pos[:2] - tgt[:2]))
            r_d = float(np.linalg.norm(r_pos[:2] - tgt[:2]))
            if l_d <= r_d:
                foot, pos, d_xy = "left", l_pos, l_d
            else:
                foot, pos, d_xy = "right", r_pos, r_d
            d_3d = float(np.linalg.norm(pos - tgt))

            target_yaw = e.sequence[t1_before][3]
            pelvis_yaw = e._get_pelvis_yaw()
            dd = pelvis_yaw - target_yaw
            angle_diff = float(np.arctan2(np.sin(dd), np.cos(dd)))

            self.hits.append(
                {
                    "target_idx": int(t1_before),
                    "foot": foot,
                    "dist_xy": d_xy,
                    "dist_3d": d_3d,
                    "angle_diff": angle_diff,
                }
            )

        self.terminated = self.terminated or bool(terminated)
        self.truncated = self.truncated or bool(truncated)

        info = dict(info) if isinstance(info, dict) else {}
        info["eval_terminated"] = bool(terminated)
        info["eval_truncated"] = bool(truncated)
        info["eval_completed"] = bool(self.completed)
        info["eval_is_standing"] = bool(self.is_standing)
        info["eval_hits"] = list(self.hits)
        return obs, reward, terminated, truncated, info


def create_env(norm_path, mode, difficulty, env_max_steps):
    g1 = G1Env(robot_xml_path=str(ROBOT_XML))
    g1.enable_domain_randomization = False
    g1.obs_builder.training = False
    g1.set_difficulty(difficulty)
    g1.config.max_episode_steps = env_max_steps
    if mode is not None:
        g1.set_mode(getattr(WalkModes, mode))

    mirror = MirrorWrapper(EvalWrapper(g1), mirror_prob=0.0)
    vec_env = DummyVecEnv([lambda: mirror])
    vec_env = VecNormalize.load(str(norm_path), vec_env)
    vec_env.training = False
    vec_env.norm_obs = True
    vec_env.norm_reward = False
    return vec_env, g1


# Auto-discovery fallbacks (used only when --model / --norm are omitted).
def find_best_model(checkpoint_dir: Path) -> Path:
    best_path = checkpoint_dir / "best_model" / "best_model.zip"
    if best_path.exists():
        return best_path
    candidates = list(checkpoint_dir.glob("best_model*.zip"))
    if candidates:
        return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
    raise FileNotFoundError(f"Best model not found in {checkpoint_dir}")


def find_norm_file(checkpoint_dir: Path, model_path: Path = None) -> Path:
    if model_path is not None:
        candidates = list(model_path.parent.glob("vec_normalize_*.pkl"))
        if candidates:
            return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
    candidates = list(checkpoint_dir.glob("vec_normalize_*.pkl")) + list(
        checkpoint_dir.glob("ppo_g1_*_vecnormalize.pkl")
    )
    if candidates:
        return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
    raise FileNotFoundError("Normalization file not found. Please specify with --norm.")


def run_episodes(adapter, vec_env, g1, episodes, max_steps, mode, viewer):
    results = []
    aborted = False
    for ep in range(episodes):
        obs = vec_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        adapter.reset()
        done = False
        steps = 0
        info = [{}]
        while not done and steps < max_steps:
            action = adapter.act(obs)
            obs, _reward, done_arr, info = vec_env.step(action[None, :])
            steps += 1
            done = bool(done_arr[0])
            if viewer is not None:
                if viewer.is_running():
                    viewer.sync()
                    time.sleep(g1.config.control_dt)
                else:
                    aborted = True
                    break

        d = info[0] if isinstance(info[0], dict) else {}
        r = {
            "terminated": bool(d.get("eval_terminated", False)),
            "truncated": bool(d.get("eval_truncated", False)),
            "completed": bool(d.get("eval_completed", False)),
            "is_standing": bool(d.get("eval_is_standing", False)),
            "hits": d.get("eval_hits", []),
            "steps": steps,
        }
        results.append(r)
        print(
            f"ep {ep + 1:3d}/{episodes} | mode={mode:<8} steps={r['steps']:4d} | "
            f"complete={int(r['completed'])} fall={int(r['terminated'])} timeout={int(r['truncated'])} | "
            f"hits={len(r['hits'])}"
        )
        if aborted:
            break
    return results


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None, None
    return float(arr.mean()), float(arr.std())


def aggregate(results):
    """3-state outcome (success > timeout > fall) plus footfall metrics."""
    n = len(results)
    is_standing = all(r.get("is_standing", False) for r in results)

    all_hits = [h for r in results for h in r["hits"]]
    n_hits = len(all_hits)
    steps_mean, steps_std = _mean_std([r["steps"] for r in results])
    hits_mean, hits_std = _mean_std([len(r["hits"]) for r in results])

    n_success = sum(1 for r in results if r["completed"])
    n_timeout = sum(1 for r in results if not r["completed"] and r["truncated"])
    n_fall = sum(1 for r in results if not r["completed"] and not r["truncated"] and r["terminated"])

    agg = {
        "episodes": n,
        "success_rate": None if is_standing else float(n_success / n),
        "timeout_rate": None if is_standing else float(n_timeout / n),
        "fall_rate": None if is_standing else float(n_fall / n),
        "steps_mean": steps_mean,
        "steps_std": steps_std,
        "judged_hits_total": int(n_hits),
        "hits_per_episode_mean": hits_mean,
        "hits_per_episode_std": hits_std,
    }

    if n_hits:
        dxy = np.array([h["dist_xy"] for h in all_hits])
        d3d = np.array([h["dist_3d"] for h in all_hits])
        ang = np.array([h["angle_diff"] for h in all_hits])
        agg.update(
            {
                "hit_dist_xy_mean": float(dxy.mean()),
                "hit_dist_xy_std": float(dxy.std()),
                "hit_dist_xy_median": float(np.median(dxy)),
                "hit_dist_xy_p95": float(np.percentile(dxy, 95)),
                "hit_dist_xy_max": float(dxy.max()),
                "hit_dist_3d_mean": float(d3d.mean()),
                "hit_dist_3d_std": float(d3d.std()),
                "body_orient_dev_mean_abs": float(np.abs(ang).mean()),
                "body_orient_dev_std": float(ang.std()),
                "body_orient_dev_max_abs": float(np.abs(ang).max()),
            }
        )
    else:
        agg.update(
            {
                "hit_dist_xy_mean": None,
                "hit_dist_xy_std": None,
                "hit_dist_xy_median": None,
                "hit_dist_xy_p95": None,
                "hit_dist_xy_max": None,
                "hit_dist_3d_mean": None,
                "hit_dist_3d_std": None,
                "body_orient_dev_mean_abs": None,
                "body_orient_dev_std": None,
                "body_orient_dev_max_abs": None,
            }
        )
    return agg


def print_summary(mode, agg):
    print(f"--- Summary [{mode}] ---")
    print(f"Episodes                : {agg['episodes']}")
    if agg["success_rate"] is None:
        print("Success rate            : N/A (STANDING)")
        print("Timeout rate            : N/A (STANDING)")
        print("Fall rate               : N/A (STANDING)")
    else:
        print(f"Success rate            : {agg['success_rate'] * 100:.1f}%")
        print(f"Timeout rate            : {agg['timeout_rate'] * 100:.1f}%")
        print(f"Fall rate               : {agg['fall_rate'] * 100:.1f}%")
    print(f"Steps (mean +/- std)    : {agg['steps_mean']:.1f} +/- {agg['steps_std']:.1f}")
    print(f"Judged hits (total)     : {agg['judged_hits_total']}")
    print(f"Hits/episode (mean+/-std): {agg['hits_per_episode_mean']:.1f} +/- {agg['hits_per_episode_std']:.1f}")
    if agg["hit_dist_xy_mean"] is not None:
        print(f"Hit dist XY (mean+/-std): {agg['hit_dist_xy_mean']:.4f} +/- {agg['hit_dist_xy_std']:.4f} m")
        print(f"Hit dist XY (p95/max)   : {agg['hit_dist_xy_p95']:.4f} / {agg['hit_dist_xy_max']:.4f} m")
        print(f"Body-orient dev (mabs)  : {agg['body_orient_dev_mean_abs']:.4f} rad")


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation / visualization for G1 footstep tracking")
    parser.add_argument("--model", type=str, default=None, help="Model path (.zip for RL, .pt for student).")
    parser.add_argument("--norm", type=str, default=None, help="VecNormalize stats (.pkl). Auto-detected if omitted.")
    parser.add_argument(
        "--policy-type",
        type=str,
        default="auto",
        choices=["auto", "rl", "student"],
        help="Policy type. 'auto' detects from the model extension.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["FORWARD", "BACKWARD", "LATERAL", "CURVED", "STANDING"],
        help="Fixed walking mode. If omitted, all modes are evaluated sequentially.",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Episodes PER MODE.")
    parser.add_argument("--env-max-steps", type=int, default=2500, help="Episode truncation length (env, eval only).")
    parser.add_argument("--max-steps", type=int, default=3000, help="Safety cap for the loop; >= env-max-steps.")
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Enable evaluation mode (no rendering, fast, export JSON). Default: render.",
    )
    args = parser.parse_args()

    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
    else:
        model_path = find_best_model(CHECKPOINT_DIR)
        print(f"Auto-loaded best model: {model_path}")

    if args.norm:
        norm_path = Path(args.norm)
        if not norm_path.exists():
            raise FileNotFoundError(f"Normalization file not found: {norm_path}")
    else:
        norm_path = find_norm_file(CHECKPOINT_DIR, model_path)
        print(f"Auto-loaded normalization: {norm_path}")

    if not ROBOT_XML.exists():
        raise FileNotFoundError(f"Robot XML not found: {ROBOT_XML}")

    if args.policy_type == "auto":
        ptype = "student" if str(model_path).endswith(".pt") else "rl"
    else:
        ptype = args.policy_type

    render = not args.eval
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    modes = [args.mode] if args.mode is not None else list(ALL_MODES)

    print(f"[eval] model={model_path}")
    print(f"[eval] norm ={norm_path}")
    print(f"[eval] policy_type={ptype} | device={device}")
    print(f"[eval] modes={modes}")
    print(f"[eval] episodes/mode={args.episodes} | env_max_steps={args.env_max_steps}")
    print(f"[eval] render={render}")
    print("-" * 70)

    adapter = RLPolicyAdapter(model_path, device) if ptype == "rl" else StudentPolicyAdapter(model_path, device)

    per_mode = {}
    for mode in modes:
        print(f"\n### Mode: {mode} ###")
        vec_env, g1 = create_env(norm_path, mode, args.difficulty, args.env_max_steps)

        viewer = mujoco.viewer.launch_passive(g1.model, g1.data) if render else None
        results = run_episodes(adapter, vec_env, g1, args.episodes, args.max_steps, mode, viewer)
        agg = aggregate(results)
        per_mode[mode] = agg
        print_summary(mode, agg)

        if viewer is not None:
            viewer.close()
        vec_env.close()

        if render and viewer is not None and not viewer.is_running():
            break

    if args.eval:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = "teacher" if ptype == "rl" else "student"
        out_path = RESULTS_DIR / f"{label}__{stamp}.json"

        payload = {
            "model_path": str(Path(model_path).resolve()),
            "norm_path": str(Path(norm_path).resolve()),
            "policy_type": ptype,
            "label": label,
            "modes": modes,
            "episodes_per_mode": args.episodes,
            "env_max_steps": args.env_max_steps,
            "difficulty": args.difficulty,
            "device": device,
            "timestamp": stamp,
            "results": per_mode,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[eval] results saved to: {out_path}")


if __name__ == "__main__":
    main()
