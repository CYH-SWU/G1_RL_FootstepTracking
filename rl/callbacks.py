import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class AdaptiveLRScheduleCallback(BaseCallback):
    def __init__(
        self,
        patience: int = 3,
        factor: float = 0.98,
        eval_freq: int = 16 * 2048 * 14,
        min_lr: float = 5e-6,
        verbose: int = 1,
    ):
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
                    print(
                        f"[{self.num_timesteps}] Performance improved: {mean_reward:.2f} (best {self.best_mean_reward:.2f})"
                    )
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
                            print(
                                f"[{self.num_timesteps}] LR already at minimum {self.min_lr:.2e}, no further reduction."
                            )
        return True

    def _get_mean_reward(self) -> float | None:
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            recent = min(10, len(self.model.ep_info_buffer))
            rewards = [ep_info["r"] for ep_info in list(self.model.ep_info_buffer)[-recent:]]
            return float(np.mean(rewards))
        return None


class KLAdaptiveLRCallback(BaseCallback):
    def __init__(
        self,
        target_kl: float = 0.022,
        factor: float = 0.02,
        min_lr: float = 5e-6,
        max_lr: float = 3e-4,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.target_kl = target_kl
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self._last_iter_index = -1

    def _on_training_start(self) -> None:
        self.current_lr = self._get_current_lr()
        if self.current_lr is None:
            if callable(self.model.learning_rate):
                self.current_lr = self.model.learning_rate(1.0)
            else:
                self.current_lr = self.model.learning_rate

        n_epochs = getattr(self.model, "n_epochs", 1)
        self._last_iter_index = self.model._n_updates // n_epochs

    def _on_step(self) -> bool:
        if not hasattr(self.model, "_n_updates"):
            return True

        n_epochs = getattr(self.model, "n_epochs", 1)
        current_iter_index = self.model._n_updates // n_epochs

        if current_iter_index == self._last_iter_index:
            return True

        kl_value = self._get_kl_from_logger()
        if kl_value is None:
            return True

        new_lr = self._adjust_lr(self.current_lr, kl_value)
        self._set_lr(new_lr)
        self.current_lr = new_lr

        if hasattr(self.model, "_setup_lr_schedule"):
            self.model._setup_lr_schedule()

        self._last_iter_index = current_iter_index
        return True

    def _get_kl_from_logger(self) -> float | None:
        if not hasattr(self.model.logger, "name_to_value"):
            return None

        possible_keys = ["approx_kl", "train/approx_kl"]
        for key in possible_keys:
            if key in self.model.logger.name_to_value:
                return float(self.model.logger.name_to_value[key])

        for name, value in self.model.logger.name_to_value.items():
            if "approx_kl" in name.lower():
                return float(value)
        return None

    def _get_current_lr(self) -> float | None:
        if hasattr(self.model.policy, "optimizer"):
            return float(self.model.policy.optimizer.param_groups[0]["lr"])
        return None

    def _adjust_lr(self, current_lr: float, kl_value: float) -> float:
        if kl_value > self.target_kl:
            new_lr = current_lr * (1.0 - self.factor)
        else:
            new_lr = current_lr * (1.0 + self.factor)
        return float(np.clip(new_lr, self.min_lr, self.max_lr))

    def _set_lr(self, new_lr: float) -> None:
        if hasattr(self.model.policy, "optimizer"):
            for param_group in self.model.policy.optimizer.param_groups:
                param_group["lr"] = new_lr

        if hasattr(self.model, "learning_rate"):
            if callable(self.model.learning_rate):
                self.model.learning_rate = lambda _: new_lr
            else:
                self.model.learning_rate = new_lr


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
