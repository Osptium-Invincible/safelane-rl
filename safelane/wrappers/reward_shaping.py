"""Extra reward terms on top of the scenario's native reward.

    r' = r + w_lc * 1[lane change] - w_jerk * |jerk| / 10 - w_ttc * max(0, (tau - TTC) / tau)
           + w_headway * (gamma * Phi(s') - Phi(s))

with the potential Phi(s) = -max(0, (h0 - headway) / h0) ("2-second rule"). The potential
term is policy-invariant (Ng et al., 1999); the others change the optimum on purpose.
Needs SafetyMonitor underneath (reads ``info["safety"]``).
"""

from __future__ import annotations

import gymnasium as gym


class RewardShaping(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        lane_change: float = 0.0,
        jerk: float = 0.0,
        ttc: float = 0.0,
        ttc_tau: float = 3.0,
        headway: float = 0.0,
        headway_target: float = 2.0,
        gamma: float = 0.9,
    ):
        super().__init__(env)
        self.w_lc, self.w_jerk, self.w_ttc, self.w_headway = lane_change, jerk, ttc, headway
        self.ttc_tau, self.h0, self.gamma = ttc_tau, headway_target, gamma
        self._phi = 0.0

    def _potential(self, safety: dict) -> float:
        return -max(0.0, (self.h0 - safety["headway"]) / self.h0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi = self._potential(info["safety"])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        s = info["safety"]
        phi_next = 0.0 if terminated else self._potential(s)
        bonus = (
            self.w_lc * float(s["lane_change"])
            - self.w_jerk * abs(s["jerk"]) / 10.0
            - self.w_ttc * max(0.0, (self.ttc_tau - s["ttc"]) / self.ttc_tau)
            + self.w_headway * (self.gamma * phi_next - self._phi)
        )
        self._phi = phi_next
        info["shaping_bonus"] = bonus
        return obs, float(reward) + bonus, terminated, truncated, info
