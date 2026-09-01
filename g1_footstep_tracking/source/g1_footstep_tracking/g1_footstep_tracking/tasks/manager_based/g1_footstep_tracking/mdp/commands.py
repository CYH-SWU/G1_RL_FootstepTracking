# commands.py - 足迹命令（从原项目 config 读取参数）
import torch
import numpy as np
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass
from .step_sequence import StepSequenceGenerator, WalkModes


@configclass
class FootstepCommandCfg(CommandTermCfg):
    """足迹命令配置 - 继承 CommandTermCfg"""
    mode: WalkModes = WalkModes.FORWARD
    num_steps: int = 20
    step_height: float = 0.0
    resampling_time_range: tuple[float, float] = (0.0, 0.0)
    debug_vis: bool = False
    cmd_kind: str = "footstep"
    element_names: list[str] = ["x", "y", "z", "yaw"]


class FootstepCommand(CommandTerm):
    """足迹命令 - 实现所有抽象方法"""

    def __init__(self, cfg: FootstepCommandCfg, env):
        super().__init__(cfg, env)
        # 从原项目 config 读取步态参数
        import sys
        from pathlib import Path
        project_root = Path("/home/chen/my_project/G1_RL_FootstepTracking")
        sys.path.insert(0, str(project_root))
        from env.utils.config import G1EnvConfig
        _G1_CFG = G1EnvConfig()

        self._planner = StepSequenceGenerator(
            step_length=_G1_CFG.step_length,
            step_width=_G1_CFG.step_width,
            total_duration=_G1_CFG.total_duration,
            swing_duration=_G1_CFG.swing_duration,
            stance_duration=_G1_CFG.stance_duration,
        )
        self._num_steps = cfg.num_steps
        self._step_height = cfg.step_height
        self._mode = cfg.mode
        self._footsteps = None
        self._command = None
        self._env = env  # 保存 env 引用，用于 reset 中更新命令

        # 立即生成足迹序列（但不调用 _update_command，因为 command_manager 还未创建）
        self._reset_footsteps(env.num_envs)

    # ============ 必须实现的抽象方法 ============

    @property
    def command(self):
        return self._command

    def _update_command(self, env):
        """更新命令（每步调用）"""
        if self._footsteps is None:
            return
        # 从环境读取 t1/t2（由外部维护）
        t1 = getattr(env, 't1', 0)
        t2 = getattr(env, 't2', 1)
        num_steps = self._footsteps.shape[1]
        current_step = self._footsteps[:, t1 % num_steps, :]
        next_step = self._footsteps[:, t2 % num_steps, :]

        self._command = torch.cat([current_step, next_step], dim=-1)
        # 注意：此时 env.command_manager 应该已经存在
        env.command_manager.set_command("footstep_current", current_step)
        env.command_manager.set_command("footstep_next", next_step)

    def _resample_command(self, env):
        pass  # 不重新规划

    def _update_metrics(self):
        pass

    # ============ 辅助方法 ============

    def _reset_footsteps(self, num_envs, env_ids=None):
        device = self.device
        if env_ids is None:
            env_ids = range(num_envs)
        if self._footsteps is None:
            self._footsteps = torch.zeros(num_envs, self._num_steps, 4, device=device)
        for i in env_ids:
            phase = 0.0 if i % 2 == 0 else 0.5
            seq = self._planner.generate(
                mode=self._mode,
                phase=phase,
                num_steps=self._num_steps,
                step_height=self._step_height,
            )
            self._footsteps[i] = torch.tensor(seq, dtype=torch.float32, device=device)

    def reset(self, env_ids):
        """重置足迹，并立即更新命令（此时 command_manager 已存在）"""
        if self._footsteps is None:
            return
        self._reset_footsteps(self._footsteps.shape[0], env_ids)
        # 更新命令，确保观测管理器可以获取
        self._update_command(self._env)