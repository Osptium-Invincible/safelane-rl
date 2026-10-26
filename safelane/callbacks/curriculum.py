"""Traffic-density curriculum: promote to the next level when the agent is reliable.

Works with the DomainRandomization wrapper's ``curriculum`` list. The level goes up when
the success rate over the last ``window`` training episodes reaches ``threshold``.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    def __init__(self, n_levels: int, threshold: float = 0.85, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.n_levels = n_levels
        self.threshold = threshold
        self.window = window
        self.level = 0
        self.success: deque = deque(maxlen=window)
        self.promotions: list[tuple[int, int]] = []

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_curriculum_level", 0)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            m = info.get("episode_metrics")
            if m is not None:
                self.success.append(m["success"])
        if (
            self.level < self.n_levels - 1
            and len(self.success) == self.window
            and np.mean(self.success) >= self.threshold
        ):
            self.level += 1
            self.training_env.env_method("set_curriculum_level", self.level)
            self.promotions.append((self.num_timesteps, self.level))
            self.success.clear()
            if self.verbose:
                print(f"[curriculum] level -> {self.level} at {self.num_timesteps} steps")
        self.logger.record("curriculum/level", self.level)
        return True
