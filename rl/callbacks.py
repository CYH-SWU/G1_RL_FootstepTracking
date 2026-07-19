
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class AdaptiveLRScheduleCallback(BaseCallback):
    """
    Adaptive learning rate callback: reduces learning rate when performance plateaus.

    :param patience: Number of evaluations with no improvement before reducing LR.
    :param factor: Multiplicative factor for LR decay .
    :param eval_freq: Evaluation interval in steps.
    :param min_lr: Minimum learning rate to avoid excessive decay.
    :param verbose: Verbosity level.
    """
    def __init__(self, patience: int = 5, factor: float = 0.90,
                 eval_freq: int = 16 * 800 * 14, min_lr: float = 1e-7, verbose: int = 1):
        super().__init__(verbose)
        self.patience = patience
        self.factor = factor
        self.eval_freq = eval_freq
        self.min_lr = min_lr

        self.best_mean_reward = -np.inf
        self.wait = 0
        self.current_lr = None

    def _on_training_start(self) -> None:
        """Initialize current learning rate at the start of training."""
        if callable(self.model.learning_rate):
            self.current_lr = self.model.learning_rate(1.0)
        else:
            self.current_lr = self.model.learning_rate

    def _on_step(self) -> bool:
        """Check performance at evaluation intervals and adjust LR if needed."""
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            mean_reward = self._get_mean_reward()
            if mean_reward is None:
                return True

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.wait = 0
                if self.verbose > 0:
                    print(f"[{self.num_timesteps}] Performance improved: {mean_reward:.2f} (best {self.best_mean_reward:.2f})")
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    new_lr = max(self.current_lr * self.factor, self.min_lr)
                    if new_lr < self.current_lr:
                        self.current_lr = new_lr
                        # Update the model's learning rate.
                        if callable(self.model.learning_rate):
                            self.model.learning_rate = lambda _: self.current_lr
                        else:
                            self.model.learning_rate = self.current_lr
                        self.model._setup_lr_schedule()
                        self.wait = 0
                        if self.verbose > 0:
                            print(f"[{self.num_timesteps}] Performance plateau, reducing LR to {self.current_lr:.2e}")
                    else:
                        if self.verbose > 0:
                            print(f"[{self.num_timesteps}] LR already at minimum {self.min_lr:.2e}, no further reduction.")
        return True

    def _get_mean_reward(self) -> float | None:
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            recent = min(10, len(self.model.ep_info_buffer))
            rewards = [ep_info['r'] for ep_info in list(self.model.ep_info_buffer)[-recent:]]
            return float(np.mean(rewards))
        return None


class CurriculumCallback(BaseCallback):
    """
    Curriculum callback that progressively increases task difficulty over time.
    """
    def __init__(self, total_timesteps_for_max: int, verbose=0):
        super().__init__(verbose)
        self.total_timesteps_for_max = total_timesteps_for_max

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_timesteps_for_max)
        self.training_env.env_method("set_difficulty", progress)
        return True
