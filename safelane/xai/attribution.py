"""Integrated Gradients (Sundararajan et al., 2017) for SB3 driving policies.

IG_i(x) = (x_i - x'_i) * mean_{k=1..m} dF(x' + k/m (x - x')) / dx_i

F is the score of the action the policy picks (PPO: logit, DQN family: Q-value), the
baseline x' is an empty road (all zeros except the ego row). Returns attributions with
the observation's shape (vehicles x features), so they can be averaged per vehicle slot
and per feature.
"""

from __future__ import annotations

import numpy as np
import torch as th


def action_scores(model, obs: th.Tensor) -> th.Tensor:
    policy = model.policy
    if hasattr(policy, "q_net"):  # DQN / D3QN
        return policy.q_net(obs)
    if hasattr(policy, "quantile_net"):  # QR-DQN: mean over quantiles
        return policy.quantile_net(obs).mean(dim=1)
    features = policy.extract_features(obs, policy.pi_features_extractor)
    latent_pi = policy.mlp_extractor.forward_actor(features)
    return policy.action_net(latent_pi)


def integrated_gradients(model, observations: np.ndarray, steps: int = 32) -> np.ndarray:
    obs = th.as_tensor(np.asarray(observations), dtype=th.float32)
    baseline = th.zeros_like(obs)
    baseline[:, 0] = obs[:, 0]  # keep the ego row: attribute only to the surrounding traffic
    with th.no_grad():
        target = action_scores(model, obs).argmax(dim=1)
    total = th.zeros_like(obs)
    for k in range(1, steps + 1):
        x = (baseline + k / steps * (obs - baseline)).requires_grad_(True)
        score = action_scores(model, x).gather(1, target[:, None]).sum()
        (grad,) = th.autograd.grad(score, x)
        total += grad
    return ((obs - baseline) * total / steps).numpy()
