import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

from deploy.footstep_planner import FootstepPlanner
from deploy.forward_kinematics import ForwardKinematics
from deploy.obs_adapter import build_actor_obs
from deploy.safety import SafetyMonitor, SafetyState
from deploy.state_estimator import StateEstimator
from deploy.transform import world_to_pelvis, world_yaw_to_pelvis_rel
from env.utils.config import G1EnvConfig
from env.utils.step_sequence import WalkModes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ACTOR_PATH = "deploy/actor.pt"
DEFAULT_XML_PATH = "robot/g1_processed.xml"


class PolicyRunner:
    def __init__(self, config_path, actor_path=None):
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.real_index = list(self.cfg["joints"]["real_index"])
        self.kp = np.array(self.cfg["pd_gains"]["kp"], dtype=np.float32)
        self.kd = np.array(self.cfg["pd_gains"]["kd"], dtype=np.float32)
        self.nominal = np.array(self.cfg["nominal_angles"], dtype=np.float32)
        self.action_scale = float(self.cfg["action_scale"])
        self.control_dt = float(self.cfg["control"]["control_dt"])
        self.dryrun = bool(self.cfg["safety"]["dryrun"])
        self.startup_delay = float(self.cfg["safety"]["startup_delay_s"])
        self.max_dev = float(self.cfg["safety"]["max_joint_deviation"])
        self.total_duration = float(self.cfg["gait"]["total_duration"])

        self.safety = SafetyMonitor(self.cfg["safety"])
        print("[PolicyRunner] SafetyMonitor initialized")

        if actor_path is None:
            actor_path = self.cfg["policy"].get("actor_path", DEFAULT_ACTOR_PATH)

        actor_full_path = Path(actor_path)
        if not actor_full_path.is_absolute():
            actor_full_path = PROJECT_ROOT / actor_full_path

        print(f"[PolicyRunner] Loading TorchScript actor: {actor_full_path}")
        if not actor_full_path.exists():
            raise FileNotFoundError(f"actor.pt not found: {actor_full_path}. Please run scripts/export_actor.py")
        self.actor = torch.jit.load(str(actor_full_path))
        self.actor.eval()
        print(f"[PolicyRunner] Actor loaded: {type(self.actor).__name__}")

        with torch.no_grad():
            dummy = torch.zeros(1, 41, dtype=torch.float32)
            out = self.actor(dummy)
        self.actor_out_dim = int(out.shape[-1])
        print(f"[PolicyRunner] Actor output dim: {self.actor_out_dim}")

        self.env_cfg = G1EnvConfig()

        print("[PolicyRunner] Initializing state estimator, FK, planner")
        self.state_est = StateEstimator()
        self.fk = ForwardKinematics(DEFAULT_XML_PATH)
        self.planner = FootstepPlanner(self.env_cfg, mode=WalkModes.FORWARD)
        self._planner_initialized = False

        self.initial_phase = random.choice([0.0, 0.5])
        print(f"[PolicyRunner] Initial phase: {self.initial_phase}")
        self.phase = self.initial_phase
        print("[PolicyRunner] Submodules ready")

        self.lowstate = None
        self.t_start = None
        self.last_policy_t = 0.0

        self.sub = None
        self.pub = None

    def setup_comm(self, domain_id, interface):
        ChannelFactoryInitialize(domain_id, interface)
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._cb_lowstate, 10)
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        print(f"[PolicyRunner] DDS: domain={domain_id}, iface={interface}")

    def _cb_lowstate(self, msg: LowState_):
        self.lowstate = msg
        self.safety.on_lowstate()

    def _infer_action(self, actor_obs):
        with torch.no_grad():
            obs_t = torch.from_numpy(actor_obs).unsqueeze(0).float()
            action_t = self.actor(obs_t)
            action = action_t.squeeze(0).cpu().numpy()
        action = np.clip(action, -1.0, 1.0)
        return action.astype(np.float32)

    def _pack_lowcmd(self, targets):
        cmd = LowCmd_()
        cmd.mode_pr = 0
        cmd.mode_machine = 0

        for i in range(35):
            cmd.motor_cmd[i].mode = 0
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].kd = 0.0

        for sim_idx in range(12):
            real_idx = self.real_index[sim_idx]
            cmd.motor_cmd[real_idx].mode = 1
            cmd.motor_cmd[real_idx].q = float(targets[sim_idx])
            cmd.motor_cmd[real_idx].dq = 0.0
            cmd.motor_cmd[real_idx].tau = 0.0
            cmd.motor_cmd[real_idx].kp = float(self.kp[sim_idx])
            cmd.motor_cmd[real_idx].kd = float(self.kd[sim_idx])

        return cmd

    def _send_nominal(self, reason=""):
        cmd = self._pack_lowcmd(self.nominal)
        if self.dryrun:
            print(f"[SAFETY-HOLD] {reason} -> sending nominal")
        else:
            self.pub.Write(cmd)

    def run(self):
        print("[PolicyRunner] Running. Ctrl-C to stop.")
        self.t_start = time.time()

        while True:
            now = time.time()

            if now - self.t_start < self.startup_delay:
                time.sleep(0.01)
                continue

            if self.lowstate is None:
                time.sleep(0.005)
                continue

            if now - self.last_policy_t < self.control_dt:
                time.sleep(0.001)
                continue
            self.last_policy_t = now

            try:
                self.state_est.update_attitude(self.lowstate)
            except Exception as e:
                print(f"[ERROR] update_attitude: {e}")
                continue

            try:
                joint_angles = np.array(
                    [self.lowstate.motor_state[i].q for i in range(12)],
                    dtype=np.float64,
                )
                left_world, right_world, left_local, right_local = self.fk.compute(
                    joint_angles,
                    self.state_est.pelvis_pos,
                    self.state_est.pelvis_quat,
                )
            except Exception as e:
                print(f"[ERROR] FK: {e}")
                continue

            try:
                self.state_est.detect_contact(self.lowstate)
                self.state_est.update_height(left_local, right_local)
            except Exception as e:
                print(f"[ERROR] height est: {e}")
                continue

            if not self._planner_initialized:
                self.planner.reset(
                    left_foot_pos=left_world,
                    right_foot_pos=right_world,
                    root_yaw=self.state_est.pelvis_yaw,
                    phase=self.initial_phase,
                )
                self._planner_initialized = True
                print(
                    f"[PolicyRunner] Planner initialized: "
                    f"num_steps={len(self.planner.sequence_world)}, "
                    f"initial_phase={self.initial_phase}"
                )

            self.planner.update(left_world, right_world)

            t1_w, t2_w = self.planner.get_t1_t2_world()
            if t1_w is None:
                continue

            t1_local = world_to_pelvis(t1_w[:3], self.state_est.pelvis_pos, self.state_est.pelvis_quat)
            t2_local = world_to_pelvis(t2_w[:3], self.state_est.pelvis_pos, self.state_est.pelvis_quat)
            t1_yaw_rel = world_yaw_to_pelvis_rel(t1_w[3], self.state_est.pelvis_yaw)
            t2_yaw_rel = world_yaw_to_pelvis_rel(t2_w[3], self.state_est.pelvis_yaw)

            try:
                actor_obs = build_actor_obs(
                    self.lowstate,
                    t1_local,
                    t2_local,
                    t1_yaw_rel,
                    t2_yaw_rel,
                    self.phase,
                    pelvis_height=self.state_est.pelvis_height,
                )
            except Exception as e:
                print(f"[ERROR] obs build: {e}")
                continue

            try:
                action = self._infer_action(actor_obs)
            except Exception as e:
                print(f"[ERROR] inference: {e}")
                continue

            targets = self.nominal + self.action_scale * action
            dev = np.clip(targets - self.nominal, -self.max_dev, self.max_dev)
            targets = self.nominal + dev

            state = self.safety.evaluate(self.lowstate, action, targets, self.nominal)

            if state == SafetyState.STOP:
                print("[SAFETY] STOP triggered, sending nominal and exiting")
                self._send_nominal("STOP")
                break

            if state == SafetyState.HOLD:
                self._send_nominal("HOLD")
                continue

            if state == SafetyState.WARN:
                targets, _ = self.safety.clamp_joints(targets, self.nominal)

            cmd = self._pack_lowcmd(targets)
            if self.dryrun:
                print(
                    f"[DRY] t={now - self.t_start:5.2f}s  "
                    f"phase={self.phase:.3f}  "
                    f"t1={self.planner.t1:2d} t2={self.planner.t2:2d}  "
                    f"contact={self.state_est.contact_str}  "
                    f"height={self.state_est.pelvis_height:.4f}  "
                    f"|action|max={np.abs(action).max():.3f}  "
                    f"|dev|max={np.abs(dev).max():.3f}  "
                    f"{self.safety.stats_str}"
                )
            else:
                self.pub.Write(cmd)

            self.phase = (self.phase + self.control_dt / self.total_duration) % 1.0

        print("[PolicyRunner] Sending final nominal command")
        self._send_nominal("final")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="deploy/config.yaml")
    parser.add_argument(
        "--actor",
        default=None,
        help=f"TorchScript actor .pt route (default config.policy.actor_path, or {DEFAULT_ACTOR_PATH})",
    )
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--no-dryrun", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"[main] Random seed set to {args.seed}")

    runner = PolicyRunner(args.config, actor_path=args.actor)

    if args.dryrun:
        runner.dryrun = True
    if args.no_dryrun:
        runner.dryrun = False

    runner.setup_comm(
        int(runner.cfg["dds"]["domain_id"]),
        str(runner.cfg["dds"]["interface"]),
    )

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n[PolicyRunner] Stopped by user")
        runner._send_nominal("interrupt")


if __name__ == "__main__":
    main()
