import gymnasium as gym
import numpy as np
import pytest

import safelane  # noqa: F401
from safelane.baselines import IDMMobilPolicy
from safelane.config import build_env_spec, load_named
from safelane.envs import make_env
from safelane.envs.geometric_road import GeometricRoad

ACTION = {"type": "DiscreteMetaAction", "target_speeds": [10, 15, 20, 25, 30]}


def test_ramp_merge_registered_and_ego_starts_on_ramp():
    env = gym.make("RampMerge-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == (5, 5)
    assert env.unwrapped.vehicle.lane_index[:2] == ("j", "k")
    assert info["is_success"] is False and info["merged"] is False


def test_ramp_merge_success_requires_merge():
    env = make_env({"id": "RampMerge-v0", "config": {"action": ACTION, "vehicles_density": 0.3}})
    policy = IDMMobilPolicy()
    successes = []
    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs, env))
            done = term or trunc
        m = info["episode_metrics"]
        if m["success"]:
            assert m["merged"] == 1.0 and m["crashed"] == 0.0
        successes.append(m["success"])
    assert np.mean(successes) > 0.5  # light traffic: the rule-based driver merges


def test_geometric_road_links_ramp_to_its_merge_lane():
    env = gym.make("RampMerge-v0")
    env.reset(seed=0)
    road = env.unwrapped.road
    assert isinstance(road, GeometricRoad)
    linked = [lane for lane, _ in road._connected_lanes(("k", "b", 0))]
    assert road.network.get_lane(("b", "c", 2)) in linked
    assert road.network.get_lane(("b", "c", 0)) not in linked


@pytest.mark.parametrize("env_name", ["highway_fast", "ramp_merge", "merge"])
def test_spec_builds_and_steps(env_name):
    algo = load_named("algo", "ppo_attn")
    env = make_env(build_env_spec(load_named("env", env_name), algo))
    obs, info = env.reset(seed=1)
    assert obs.shape == (15, 7)
    assert "safety" in info
    obs, r, term, trunc, info = env.step(1)
    assert {"cost", "safety"} <= set(info)


def test_domain_randomization_changes_traffic():
    env = make_env(build_env_spec(load_named("env", "highway_dr"), load_named("algo", "dqn")))
    samples = set()
    for seed in range(6):
        _, info = env.reset(seed=seed)
        samples.add(round(info["randomization"]["vehicles_density"], 3))
    assert len(samples) > 1


def test_observation_scale_is_fixed_on_every_road():
    """Lateral normalisation must not depend on the lanes around the ego at reset (ramp vs highway)."""
    algo = load_named("algo", "ppo_attn")
    for env_name in ["ramp_merge", "highway_fast", "highway", "merge"]:
        env = make_env(build_env_spec(load_named("env", env_name), algo))
        env.reset(seed=0)
        assert env.unwrapped.observation_type.features_range["y"] == [-16, 16]
    env = make_env(build_env_spec(load_named("env", "ramp_merge"), algo))
    obs, _ = env.reset(seed=0)
    assert abs(obs[0, 2]) < 1.0  # ego on the ramp (y = 14.5 m) is not clipped


def test_driver_speed_factor_applied_once():
    from safelane.envs.vehicles import AggressiveIDMVehicle, CautiousIDMVehicle

    for name, cls, (lo, hi) in [("aggressive", AggressiveIDMVehicle, (1.05, 1.2)),
                                ("cautious", CautiousIDMVehicle, (0.8, 0.95))]:
        spec = build_env_spec(load_named("env", "highway_fast"), load_named("algo", "dqn"))
        spec["config"]["other_vehicles_type"] = name
        env = make_env(spec)
        env.reset(seed=1)
        ratios = [v.target_speed / v.speed for v in env.unwrapped.road.vehicles if isinstance(v, cls)]
        assert ratios and lo - 1e-6 <= min(ratios) and max(ratios) <= hi + 1e-6
