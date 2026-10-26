import gymnasium as gym
import numpy as np
import pytest
from highway_env.envs.common.observation import KinematicObservation

import safelane  # noqa: F401  (installs the fast observation)
from safelane.envs.fast_observation import FastKinematicObservation

CONFIGS = [
    {"type": "Kinematics"},
    {"type": "Kinematics", "vehicles_count": 15, "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]},
    {"type": "Kinematics", "absolute": True, "vehicles_count": 8},
    {"type": "Kinematics", "features": ["presence", "x", "y", "heading", "long_off", "lat_off"]},
]


@pytest.mark.parametrize("env_id", ["highway-fast-v0", "RampMerge-v0", "merge-v1"])
@pytest.mark.parametrize("obs_cfg", CONFIGS)
def test_fast_kinematics_matches_stock(env_id, obs_cfg):
    env = gym.make(env_id, config={"observation": obs_cfg})
    rng = np.random.default_rng(0)
    for seed in range(3):
        obs, _ = env.reset(seed=seed)
        u = env.unwrapped
        assert isinstance(u.observation_type, FastKinematicObservation)
        stock = KinematicObservation(u, **obs_cfg)
        for _ in range(15):
            stock.features_range = u.observation_type.features_range
            assert np.allclose(u.observation_type.observe(), stock.observe(), atol=1e-6, equal_nan=True)
            _, _, term, trunc, _ = env.step(int(rng.integers(5)))
            if term or trunc:
                break
