"""Export a distilled student policy to TorchScript for deployment.

The student is an actor-only MLP that consumes a history window of frames.
It is not an SB3 policy, so it is rebuilt directly from the distillation
checkpoint. Normalization and history stacking are done on the deployment side
(deploy/obs_adapter.py); the exported TorchScript is a stateless MLP mapping a
flat (base_dim * history_len) input to a 12-dim action.

A companion JSON with the deployment metadata (history_len, pelvis_height_idx,
norm_mean, norm_var, ...) is written alongside the .pt.

Usage:
    python scripts/export_actor.py \
        --student distill_dagger/student_best.pt \
        --out deploy/student.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from distill_dagger import StudentPolicy  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Export a distilled student policy to TorchScript")
    parser.add_argument("--student", required=True, help="Student checkpoint (.pt) produced by distill_dagger.py")
    parser.add_argument("--out", default="deploy/student.pt", help="Output TorchScript path")
    parser.add_argument("--meta-out", default=None, help="Output metadata JSON path (default: same stem as --out)")
    args = parser.parse_args()

    student_path = Path(args.student)
    if not student_path.exists():
        raise FileNotFoundError(f"student checkpoint not found: {student_path}")

    ckpt = torch.load(str(student_path), map_location="cpu", weights_only=False)
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}

    obs_dim = int(meta["student_obs_dim"])
    action_dim = int(meta["action_dim"])
    hidden = tuple(meta["hidden"])

    print(f"[export] student: {student_path}")
    print(f"[export] obs_dim={obs_dim} action_dim={action_dim} hidden={hidden}")
    print(
        f"[export] base_dim={meta['base_dim']} history_len={meta['history_len']} "
        f"pelvis_height_idx={meta['pelvis_height_idx']}"
    )

    student = StudentPolicy(obs_dim, action_dim, hidden)
    student.load_state_dict(ckpt["state_dict"])
    student.eval()

    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        out = student(dummy)
    if out.shape[-1] != action_dim:
        raise RuntimeError(f"output dim {out.shape[-1]} != action_dim {action_dim}")
    print(f"[export] dummy output shape: {tuple(out.shape)}")

    traced = torch.jit.trace(student, dummy)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    print(f"[export] saved TorchScript student: {out_path}")

    reloaded = torch.jit.load(str(out_path))
    with torch.no_grad():
        out2 = reloaded(dummy)
    diff = float(torch.max(torch.abs(out - out2)).item())
    print(f"[export] reload check max diff: {diff:.2e}")
    assert diff < 1e-5, "TorchScript reload mismatch"

    deploy_meta = {
        "student_obs_dim": obs_dim,
        "base_dim": int(meta["base_dim"]),
        "history_len": int(meta["history_len"]),
        "pelvis_height_idx": int(meta["pelvis_height_idx"]),
        "action_dim": action_dim,
        "hidden": list(hidden),
        "norm_mean": list(meta["norm_mean"]),
        "norm_var": list(meta["norm_var"]),
    }
    meta_path = Path(args.meta_out) if args.meta_out else out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(deploy_meta, f, indent=2)
    print(f"[export] saved deployment metadata: {meta_path}")


if __name__ == "__main__":
    main()
