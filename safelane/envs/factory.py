"""Build (vectorised) environments from a plain-dict spec.

Spec keys (all optional except ``id``)::

    id: highway-fast-v0
    config: {...}                      # passed to highway-env (incl. observation / action)
    max_episode_steps: 60              # gymnasium TimeLimit
    randomization: {params: {...}, curriculum: [...]}
    shield: {ttc_threshold: 2.5, enforce: false}
    safety: {ttc_threshold: 2.0, headway_threshold: 1.0, crash_cost: 10.0}
    reward_shaping: {lane_change: -0.05, ...}
    obs_noise: {sigma: 0.1, dropout: 0.0, delay: 0}

Wrapper order (inner -> outer):
    env -> DomainRandomization -> SafetyShield -> SafetyMonitor -> RewardShaping -> ObservationNoise
"""

from __future__ import annotations

import copy
import multiprocessing
import os
from functools import partial

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from safelane.envs.vehicles import resolve_vehicle_type


def _prepare_config(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg or {})
    if "other_vehicles_type" in cfg:
        cfg["other_vehicles_type"] = resolve_vehicle_type(cfg["other_vehicles_type"])
    return cfg


def make_env(spec: dict, render_mode: str | None = None) -> gym.Env:
    import safelane  # noqa: F401  (registers envs inside subprocess workers)
    from safelane.wrappers import (
        DomainRandomization,
        ObservationNoise,
        RewardShaping,
        SafetyMonitor,
        SafetyShield,
    )

    kwargs = {}
    if spec.get("max_episode_steps"):
        kwargs["max_episode_steps"] = int(spec["max_episode_steps"])
    env = gym.make(spec["id"], config=_prepare_config(spec.get("config")), render_mode=render_mode, **kwargs)

    if spec.get("randomization"):
        r = spec["randomization"]
        env = DomainRandomization(env, r.get("params", {}), r.get("curriculum"))
    if spec.get("shield"):
        env = SafetyShield(env, **spec["shield"])
    env = SafetyMonitor(env, **(spec.get("safety") or {}))
    if spec.get("reward_shaping"):
        env = RewardShaping(env, **spec["reward_shaping"])
    if spec.get("obs_noise"):
        env = ObservationNoise(env, **spec["obs_noise"])
    return env


def _make_monitored(spec: dict, monitor_file: str | None) -> gym.Env:
    env = make_env(spec)
    return Monitor(env, filename=monitor_file)


def make_vec_env(
    specs: dict | list[dict],
    n_envs: int = 1,
    seed: int = 0,
    monitor_dir: str | None = None,
    subprocess: bool = True,
) -> VecEnv:
    """Create ``n_envs`` workers; a list of specs is assigned round-robin (mixed-scenario training)."""
    specs = specs if isinstance(specs, list) else [specs]
    if monitor_dir:
        os.makedirs(monitor_dir, exist_ok=True)
    fns = [
        partial(
            _make_monitored,
            specs[i % len(specs)],
            os.path.join(monitor_dir, f"worker{i}") if monitor_dir else None,
        )
        for i in range(n_envs)
    ]
    if subprocess and n_envs > 1:
        # workers fork from a server that already imported the heavy modules -> shared memory
        multiprocessing.set_forkserver_preload(["torch", "stable_baselines3", "highway_env", "safelane"])
        venv = SubprocVecEnv(fns, start_method="forkserver")
    else:
        venv = DummyVecEnv(fns)
    venv.seed(seed)
    return venv
