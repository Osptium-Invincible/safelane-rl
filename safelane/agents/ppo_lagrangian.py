"""PPO-Lagrangian for the constrained MDP   max J_r  s.t.  J_c <= d.

Implementation (RCPO-style, Tessler et al. 2019, with PID dual updates, Stooke et al. 2020):

* ``LagrangianRewardWrapper`` (VecEnvWrapper, lives in the main process) replaces each
  reward by (r - lambda * c) / (1 + lambda), reading c from ``info["cost"]``, and records
  per-episode costs.
* ``LagrangeMultiplierCallback`` updates lambda after every rollout from the mean episode
  cost of that rollout:
        e_k = J_c - d
        lambda = max(0, Kp * e_k + Ki * sum(e) + Kd * max(0, e_k - e_{k-1}))
  With Kp = Kd = 0 this is plain gradient ascent on the dual variable.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class LagrangianRewardWrapper(VecEnvWrapper):
    def __init__(self, venv: VecEnv, lam: float = 0.0, normalize: bool = True):
        super().__init__(venv)
        self.lam = float(lam)
        self.normalize = normalize
        self._running_cost = np.zeros(self.num_envs)
        self.finished_episode_costs: list[float] = []

    def reset(self):
        self._running_cost[:] = 0.0
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        costs = np.array([info.get("cost", 0.0) for info in infos], dtype=np.float32)
        self._running_cost += costs
        for i, done in enumerate(dones):
            if done:
                self.finished_episode_costs.append(float(self._running_cost[i]))
                self._running_cost[i] = 0.0
        shaped = rewards - self.lam * costs
        if self.normalize:
            shaped = shaped / (1.0 + self.lam)
        return obs, shaped.astype(np.float32), dones, infos


class LagrangeMultiplierCallback(BaseCallback):
    def __init__(
        self,
        wrapper: LagrangianRewardWrapper,
        cost_limit: float,
        kp: float = 0.05,
        ki: float = 0.01,
        kd: float = 0.0,
        lam_max: float = 50.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.wrapper = wrapper
        self.cost_limit = cost_limit
        self.kp, self.ki, self.kd = kp, ki, kd
        self.lam_max = lam_max
        self._integral = 0.0
        self._prev_error = 0.0
        self.history: list[tuple[int, float, float]] = []  # (timesteps, J_c, lambda)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        costs = self.wrapper.finished_episode_costs
        if not costs:
            return
        jc = float(np.mean(costs))
        self.wrapper.finished_episode_costs = []
        error = jc - self.cost_limit
        self._integral = max(0.0, self._integral + error)
        derivative = max(0.0, error - self._prev_error)
        self._prev_error = error
        lam = self.kp * error + self.ki * self._integral + self.kd * derivative
        self.wrapper.lam = float(np.clip(lam, 0.0, self.lam_max))
        self.history.append((self.num_timesteps, jc, self.wrapper.lam))
        self.logger.record("safety/episode_cost", jc)
        self.logger.record("safety/lambda", self.wrapper.lam)
        self.logger.record("safety/cost_limit", self.cost_limit)
