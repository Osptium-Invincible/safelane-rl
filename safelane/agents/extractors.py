"""Ego-attention feature extractor for Kinematics observations (SB3 BaseFeaturesExtractor).

Each observed vehicle (row) is embedded by a shared MLP; the ego embedding queries all
vehicles with multi-head attention (keys/values = all present vehicles). The result is
permutation-invariant w.r.t. the other vehicles and ignores empty (presence = 0) rows.
In the spirit of Leurent & Mercat (2019), written from scratch for this project.

The latest attention weights are kept in ``last_attention`` (batch, heads, vehicles) for
explainability overlays.
"""

from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class EgoAttentionExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        embed_dim: int = 64,
        n_heads: int = 2,
        features_dim: int = 64,
        presence_index: int | None = 0,
    ):
        super().__init__(observation_space, features_dim)
        n_vehicles, n_features = observation_space.shape
        self.n_vehicles = n_vehicles
        self.presence_index = presence_index
        self.embed = nn.Sequential(
            nn.Linear(n_features, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Linear(2 * embed_dim, features_dim), nn.ReLU())
        self.last_attention: th.Tensor | None = None

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = obs.reshape(obs.shape[0], self.n_vehicles, -1)
        tokens = self.embed(x)  # (B, V, E)
        ego = tokens[:, :1, :]  # (B, 1, E)
        pad_mask = None
        if self.presence_index is not None:
            pad_mask = x[..., self.presence_index] < 0.5  # True = ignore
            pad_mask[:, 0] = False  # ego is always attended
        attended, weights = self.attention(
            ego, tokens, tokens, key_padding_mask=pad_mask, need_weights=True, average_attn_weights=False
        )
        self.last_attention = weights.detach()[:, :, 0, :]  # (B, heads, V)
        fused = self.norm(ego + attended).squeeze(1)
        return self.head(th.cat([fused, ego.squeeze(1)], dim=-1))
