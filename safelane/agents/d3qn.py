"""Double + Dueling DQN on top of Stable-Baselines3's (vanilla) DQN.

* Dueling head (Wang et al., 2016):  Q(s,a) = V(s) + A(s,a) - mean_a' A(s,a')
* Double-Q target (van Hasselt et al., 2016): a* = argmax_a Q_online(s',a), y = r + gamma * Q_target(s',a*)
* n-step returns come from SB3 >= 2.7 (``n_steps`` argument -> NStepReplayBuffer)
"""

from __future__ import annotations

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from torch import nn


class DuelingHead(nn.Module):
    def __init__(self, features_dim: int, n_actions: int, net_arch: list[int], activation_fn):
        super().__init__()
        # shared trunk = all but the last layer; the last layer size is used by each stream
        trunk_arch, hidden = list(net_arch[:-1]), (net_arch[-1] if net_arch else 64)
        self.trunk = nn.Sequential(*create_mlp(features_dim, -1, trunk_arch, activation_fn))
        hidden_in = trunk_arch[-1] if trunk_arch else features_dim
        self.value = nn.Sequential(nn.Linear(hidden_in, hidden), activation_fn(), nn.Linear(hidden, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden_in, hidden), activation_fn(), nn.Linear(hidden, n_actions))

    def forward(self, features: th.Tensor) -> th.Tensor:
        h = self.trunk(features)
        adv = self.advantage(h)
        return self.value(h) + adv - adv.mean(dim=1, keepdim=True)


class DuelingQNetwork(QNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_net = DuelingHead(self.features_dim, int(self.action_space.n), self.net_arch, self.activation_fn)


class DuelingDQNPolicy(DQNPolicy):
    def make_q_net(self) -> DuelingQNetwork:
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return DuelingQNetwork(**net_args).to(self.device)


class D3QN(DQN):
    """DQN with optional Double-Q targets and a dueling architecture (``MlpPolicy`` alias)."""

    policy_aliases = {"MlpPolicy": DuelingDQNPolicy, "DuelingMlpPolicy": DuelingDQNPolicy}

    def __init__(self, *args, double_q: bool = True, **kwargs):
        self.double_q = double_q
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = batch.discounts if batch.discounts is not None else self.gamma

            with th.no_grad():
                next_q_target = self.q_net_target(batch.next_observations)
                if self.double_q:
                    best = self.q_net(batch.next_observations).argmax(dim=1, keepdim=True)
                    next_q = th.gather(next_q_target, dim=1, index=best)
                else:
                    next_q = next_q_target.max(dim=1, keepdim=True).values
                target = batch.rewards + (1 - batch.dones) * discounts * next_q

            current = th.gather(self.q_net(batch.observations), dim=1, index=batch.actions.long())
            loss = F.smooth_l1_loss(current, target)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
