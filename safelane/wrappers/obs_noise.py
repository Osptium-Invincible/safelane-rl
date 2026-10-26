"""Imperfect perception for robustness tests on Kinematics observations.

* Gaussian noise (std ``sigma`` in normalised units) on every feature except ``presence``
* Dropout: each non-ego vehicle row is erased with probability ``dropout``
* Delay: the agent sees the observation from ``delay`` decisions ago
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np


class ObservationNoise(gym.Wrapper):
    def __init__(self, env: gym.Env, sigma: float = 0.0, dropout: float = 0.0, delay: int = 0):
        super().__init__(env)
        self.sigma, self.dropout, self.delay = sigma, dropout, int(delay)
        features = env.unwrapped.config["observation"].get("features", ["presence", "x", "y", "vx", "vy"])
        self.presence_col = features.index("presence") if "presence" in features else None
        self._rng = np.random.default_rng()
        self._buffer: deque = deque(maxlen=self.delay + 1)

    def _corrupt(self, obs: np.ndarray) -> np.ndarray:
        obs = np.array(obs, dtype=np.float32, copy=True)
        if self.sigma > 0:
            noise = self._rng.normal(0.0, self.sigma, size=obs.shape).astype(np.float32)
            if self.presence_col is not None:
                noise[:, self.presence_col] = 0.0
                noise *= obs[:, [self.presence_col]]  # no noise on empty rows
            obs = np.clip(obs + noise, -1.0, 1.0)
        if self.dropout > 0 and obs.shape[0] > 1:
            drop = self._rng.random(obs.shape[0]) < self.dropout
            drop[0] = False  # never drop the ego row
            obs[drop] = 0.0
        return obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed + 7919)
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._corrupt(obs)
        self._buffer.clear()
        for _ in range(self.delay + 1):
            self._buffer.append(obs)
        return self._buffer[0], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buffer.append(self._corrupt(obs))
        return self._buffer[0], reward, terminated, truncated, info
