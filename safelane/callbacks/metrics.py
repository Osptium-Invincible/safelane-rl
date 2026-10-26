"""Log driving metrics (collision rate, speed, success, cost, ...) during training.

Writes them to TensorBoard under ``driving/`` and appends rows to ``metrics.csv`` in the
run directory, which ``scripts/make_figures.py`` turns into learning curves.
"""

from __future__ import annotations

import csv
import os
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

KEYS = ("return", "crashed", "success", "avg_speed", "cost", "length", "rss_violation_frac", "shield_frac")


class DrivingMetricsCallback(BaseCallback):
    def __init__(self, csv_path: str | None = None, window: int = 100, log_every: int = 2000, lagrange=None):
        super().__init__()
        self.csv_path = csv_path
        self.window = window
        self.log_every = log_every
        self.lagrange = lagrange  # optional LagrangianRewardWrapper, to log lambda
        self.buffer: deque = deque(maxlen=window)
        self._last_log = 0
        self._writer = None
        self._file = None

    def _on_training_start(self) -> None:
        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            self._file = open(self.csv_path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._writer.writerow(["timesteps", "episodes", *KEYS, "lambda"])

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            m = info.get("episode_metrics")
            if m is not None:
                self.buffer.append(m)
        if self.num_timesteps - self._last_log >= self.log_every and self.buffer:
            self._last_log = self.num_timesteps
            self._flush()
        return True

    def _flush(self) -> None:
        means = {k: float(np.mean([m[k] for m in self.buffer])) for k in KEYS}
        for k, v in means.items():
            self.logger.record(f"driving/{k}", v)
        lam = self.lagrange.lam if self.lagrange is not None else float("nan")
        if self._writer:
            self._writer.writerow([self.num_timesteps, len(self.buffer), *[means[k] for k in KEYS], lam])
            self._file.flush()

    def _on_training_end(self) -> None:
        if self.buffer:
            self._flush()
        if self._file:
            self._file.close()
