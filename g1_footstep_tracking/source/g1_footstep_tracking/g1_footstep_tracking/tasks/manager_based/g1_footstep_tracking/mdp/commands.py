# commands.py
import torch
import numpy as np
from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from .step_sequence import StepSequenceGenerator, WalkModes


@configclass
class FootstepCommandCfg:
    """足迹命令配置 - 步态参数从环境配置读取"""
    mode: WalkModes = WalkModes.FORWARD
    num_steps: int = 20
    step_height: float = 0.0


class FootstepCommand(CommandTerm):
    """
    足迹命令 - 完全复现原项目逻辑
    
    关键设计：
    1. reset 时一次性生成完整序列，整个 episode 不重新规划
    2. 不管理 phase 和步进索引，这些由环境主循环驱动
    3. 只负责根据当前 t1/t2 提供当前和下一步足迹目标
    """

    def __init__(self, cfg: FootstepCommandCfg, env):
        super().__init__(cfg, env)
        # 从环境配置读取步态参数
        gait_cfg = env.cfg.gait
        self._planner = StepSequenceGenerator(
            step_length=gait_cfg.step_length,
            step_width=gait_cfg.step_width,
            total_duration=gait_cfg.total_duration,
            swing_duration=gait_cfg.swing_duration,
            stance_duration=gait_cfg.stance_duration,
        )
        self._num_steps = cfg.num_steps
        self._step_height = cfg.step_height
        self._mode = cfg.mode

        # 每个环境独立的足迹序列
        self._footsteps = None  # (num_envs, num_steps, 4)

    def _update_commands(self, env):
        """每步更新：从环境读取 t1/t2，提供对应的足迹目标"""
        if self._footsteps is None:
            return

        # 从环境读取当前的 t1, t2 索引（由 phase 驱动）
        # 注意：这里假设 env 有 t1, t2 属性，需要在环境配置中暴露
        t1 = env.t1 if hasattr(env, 't1') else 0
        t2 = env.t2 if hasattr(env, 't2') else 1

        # 获取当前和下一步足迹（循环使用）
        num_steps = self._footsteps.shape[1]
        current_step = self._footsteps[:, t1 % num_steps, :]   # (num_envs, 4)
        next_step = self._footsteps[:, t2 % num_steps, :]      # (num_envs, 4)

        # 写入命令管理器
        env.command_manager.set_command("footstep_current", current_step)
        env.command_manager.set_command("footstep_next", next_step)

    def _reset_footsteps(self, num_envs, env_ids=None):
        """为指定环境生成足迹序列（reset 时调用）"""
        device = self.device
        if env_ids is None:
            env_ids = range(num_envs)

        for i in env_ids:
            # 随机初始相位：0 左腿先，0.5 右腿先
            phase = 0.0 if i % 2 == 0 else 0.5
            seq = self._planner.generate(
                mode=self._mode,
                phase=phase,
                num_steps=self._num_steps,
                step_height=self._step_height,
            )
            self._footsteps[i] = torch.tensor(seq, dtype=torch.float32, device=device)

    def reset(self, env_ids):
        """环境重置时重新生成足迹"""
        if self._footsteps is None:
            return
        self._reset_footsteps(self._footsteps.shape[0], env_ids)