"""Name -> algorithm class, plus model loading and a uniform ``policy(obs, env) -> action``."""

from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib import QRDQN, MaskablePPO
from stable_baselines3 import DQN, PPO

from safelane.agents.d3qn import D3QN

ALGOS = {
    "dqn": DQN,
    "d3qn": D3QN,
    "qrdqn": QRDQN,
    "ppo": PPO,
    "ppo_lag": PPO,
    "maskable_ppo": MaskablePPO,
}


def load_model(algo: str, path: str, device: str = "cpu"):
    return ALGOS[algo].load(path, device=device)


class ModelPolicy:
    """Wraps a trained SB3 model as ``policy(obs, env) -> int``.

    * MaskablePPO receives the shield mask from the env.
    * QR-DQN can act risk-averse with ``cvar_alpha`` < 1: choose the action maximising the
      mean of the worst alpha-fraction of return quantiles (CVaR) instead of the mean.
    """

    def __init__(self, model, algo: str, deterministic: bool = True, cvar_alpha: float | None = None):
        self.model, self.algo = model, algo
        self.deterministic = deterministic
        self.cvar_alpha = cvar_alpha
        self.name = algo

    def __call__(self, obs, env) -> int:
        if self.algo == "maskable_ppo":
            masks = env.get_wrapper_attr("action_masks")()
            action, _ = self.model.predict(obs, deterministic=self.deterministic, action_masks=masks)
            return int(action)
        if self.algo == "qrdqn" and self.cvar_alpha is not None and self.cvar_alpha < 1.0:
            return self._cvar_action(obs)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action)

    def _cvar_action(self, obs) -> int:
        policy = self.model.policy
        obs_t, _ = policy.obs_to_tensor(obs)
        with th.no_grad():
            quantiles = policy.quantile_net(obs_t)  # (1, n_quantiles, n_actions)
        q = th.sort(quantiles, dim=1).values
        k = max(1, int(np.ceil(self.cvar_alpha * q.shape[1])))
        return int(q[:, :k, :].mean(dim=1).argmax(dim=1).item())
