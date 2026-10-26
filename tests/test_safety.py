import math

import numpy as np

from safelane.config import build_env_spec, load_named
from safelane.envs import make_env
from safelane.wrappers.safety import rss_safe_distance, time_headway, time_to_collision


def test_ttc_and_headway():
    assert time_to_collision(20.0, 25.0, 15.0) == 2.0
    assert math.isinf(time_to_collision(20.0, 15.0, 25.0))  # opening gap
    assert time_to_collision(-1.0, 20.0, 20.0) == 0.0  # overlap
    assert time_headway(30.0, 15.0) == 2.0


def test_rss_distance_grows_with_speed():
    assert rss_safe_distance(30.0, 30.0) > rss_safe_distance(20.0, 20.0) > 0
    assert rss_safe_distance(0.0, 30.0) == 0.0


def test_shield_mask_never_empty_and_legal():
    env = make_env(build_env_spec(load_named("env", "ramp_merge"), load_named("algo", "maskable_ppo")))
    for seed in range(3):
        env.reset(seed=seed)
        done = False
        while not done:
            mask = env.get_wrapper_attr("action_masks")()
            legal = env.get_wrapper_attr("legal_mask")()
            assert mask.any()
            assert not (mask & ~legal).any()
            action = int(np.flatnonzero(mask)[0])
            _, _, term, trunc, _ = env.step(action)
            done = term or trunc


def test_episode_metrics_present_and_consistent():
    env = make_env(build_env_spec(load_named("env", "highway_fast"), load_named("algo", "dqn")))
    env.reset(seed=0)
    done = False
    while not done:
        _, _, term, trunc, info = env.step(1)
        done = term or trunc
    m = info["episode_metrics"]
    assert m["crashed"] in (0.0, 1.0)
    assert m["success"] == 1.0 - m["crashed"]
    assert m["length"] > 0 and m["avg_speed"] > 0


def test_observation_noise_keeps_shape_and_ego_row():
    algo = load_named("algo", "dqn")
    spec = build_env_spec(load_named("env", "highway_fast"), algo)
    clean = make_env(spec).reset(seed=3)[0]
    noisy_spec = dict(spec, obs_noise={"sigma": 0.1, "dropout": 1.0})
    noisy = make_env(noisy_spec).reset(seed=3)[0]
    assert noisy.shape == clean.shape
    assert np.all(noisy[1:] == 0.0)  # dropout=1 erases every other vehicle
    assert noisy[0, 0] == 1.0  # ego presence untouched


def test_iqm_and_stratified_bootstrap():
    from safelane.eval.stats import iqm, stratified_bootstrap

    scores = np.arange(1, 9, dtype=float).reshape(4, 2)  # values 1..8
    assert iqm(scores) == np.mean([3, 4, 5, 6])  # middle 50%
    point, lo, hi = stratified_bootstrap(scores, iqm, reps=500)
    assert lo <= point <= hi
    const = np.full((3, 2), 0.7)
    assert stratified_bootstrap(const, iqm, reps=50) == (0.7, 0.7, 0.7)


def test_barrier_counts_for_ttc_but_not_headway_or_rss():
    env = make_env({"id": "RampMerge-v0", "config": {"vehicles_density": 0.01}})
    env.reset(seed=0)
    for _ in range(25):
        _, _, term, trunc, info = env.step(1)  # stay on the ramp, drive towards the barrier
        if term or trunc:
            break
        s = info["safety"]
        if s["ttc"] < 99:  # the barrier is ahead and we are closing in
            assert s["headway"] == 99.0 and s["rss_violation"] is False
