import numpy as np
import torch as th
from gymnasium import spaces

from safelane.agents import D3QN, EgoAttentionExtractor
from safelane.agents.d3qn import DuelingHead
from safelane.agents.ppo_lagrangian import LagrangeMultiplierCallback, LagrangianRewardWrapper
from safelane.config import build_env_spec, load_named
from safelane.envs import make_env, make_vec_env


def test_dueling_head_mean_zero_advantage():
    head = DuelingHead(8, 5, [32, 32], th.nn.ReLU)
    x = th.randn(4, 8)
    q = head(x)
    h = head.trunk(x)
    v = head.value(h)
    assert q.shape == (4, 5)
    assert th.allclose(q.mean(dim=1, keepdim=True), v, atol=1e-5)


def test_attention_is_permutation_invariant_and_ignores_empty_rows():
    space = spaces.Box(-1, 1, shape=(6, 5), dtype=np.float32)
    ext = EgoAttentionExtractor(space, embed_dim=16, n_heads=2, features_dim=16).eval()
    obs = th.rand(2, 6, 5)
    obs[:, :, 0] = 1.0
    obs[:, 4:, :] = 0.0  # two empty slots
    perm = obs[:, [0, 3, 1, 2, 4, 5], :]
    out = ext(obs)
    assert th.allclose(out, ext(perm), atol=1e-5)
    junk = obs.clone()
    junk[:, 4:, 1:] = 0.7  # garbage features in empty rows must not matter
    assert th.allclose(out, ext(junk), atol=1e-5)
    assert ext.last_attention.shape == (2, 2, 6)


def test_d3qn_learns_a_few_steps():
    spec = build_env_spec(load_named("env", "highway_fast"), load_named("algo", "d3qn"))
    model = D3QN("MlpPolicy", make_env(spec), learning_starts=10, buffer_size=500, n_steps=3, device="cpu")
    model.learn(60)
    assert model.predict(np.zeros((5, 5), dtype=np.float32))[0] in range(5)


def test_lagrange_multiplier_rises_when_cost_exceeds_budget():
    spec = build_env_spec(load_named("env", "highway_fast"), load_named("algo", "ppo"))
    venv = LagrangianRewardWrapper(make_vec_env(spec, n_envs=1, subprocess=False))
    cb = LagrangeMultiplierCallback(venv, cost_limit=0.0, kp=0.1, ki=0.1)
    venv.finished_episode_costs = [5.0, 7.0]

    class _Logger:
        def record(self, *a, **k):
            pass

    class _Model:
        logger = _Logger()

    cb.model = _Model()
    cb.num_timesteps = 0
    cb._on_rollout_end()
    assert venv.lam > 0
    venv.close()
