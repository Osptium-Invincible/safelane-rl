"""Per-episode traffic randomisation with an optional curriculum.

Parameter spec examples::

    vehicles_density:    {uniform: [0.5, 2.0]}
    other_vehicles_type: {choice: [idm, randomized, aggressive, cautious]}
    vehicles_count:      {randint: [15, 30]}
    lanes_count:         3            # constant

A curriculum is a list of partial specs; level k overrides the base spec with
``curriculum[k]``. A callback raises the level through ``set_curriculum_level``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from safelane.envs.vehicles import resolve_vehicle_type


def sample_value(spec, rng: np.random.Generator):
    if not isinstance(spec, dict):
        return spec
    if "uniform" in spec:
        lo, hi = spec["uniform"]
        return float(rng.uniform(lo, hi))
    if "randint" in spec:
        lo, hi = spec["randint"]
        return int(rng.integers(lo, hi + 1))
    if "choice" in spec:
        options = spec["choice"]
        return options[int(rng.integers(len(options)))]
    raise ValueError(f"Unknown randomisation spec: {spec}")


class DomainRandomization(gym.Wrapper):
    def __init__(self, env: gym.Env, params: dict, curriculum: list[dict] | None = None):
        super().__init__(env)
        self.params = dict(params)
        self.curriculum = curriculum or []
        self.level = 0
        self._rng = np.random.default_rng()
        self.last_sample: dict = {}

    def set_curriculum_level(self, level: int) -> int:
        if self.curriculum:
            self.level = int(np.clip(level, 0, len(self.curriculum) - 1))
        return self.level

    def get_curriculum_level(self) -> int:
        return self.level

    def _active_spec(self) -> dict:
        spec = dict(self.params)
        if self.curriculum:
            spec.update(self.curriculum[self.level])
        return spec

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        sample = {k: sample_value(v, self._rng) for k, v in self._active_spec().items()}
        if "other_vehicles_type" in sample:
            sample["other_vehicles_type"] = resolve_vehicle_type(sample["other_vehicles_type"])
        self.env.unwrapped.configure(sample)
        self.last_sample = sample
        obs, info = self.env.reset(seed=seed, options=options)
        info["randomization"] = sample
        return obs, info
