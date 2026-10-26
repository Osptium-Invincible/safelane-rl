"""Training entry point shared by ``scripts/train.py`` and ``scripts/run_experiments.py``."""

from __future__ import annotations

import copy
import json
import os
import time

import torch as th
import yaml
from stable_baselines3.common.callbacks import CallbackList
from torch import nn

from safelane.agents import EgoAttentionExtractor, LagrangeMultiplierCallback, LagrangianRewardWrapper
from safelane.agents.registry import ALGOS
from safelane.callbacks import CurriculumCallback, DrivingMetricsCallback
from safelane.config import build_env_spec, load_named
from safelane.envs import make_vec_env

EXTRACTORS = {"ego_attention": EgoAttentionExtractor}
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


def resolve_policy_kwargs(raw: dict | None) -> dict:
    kwargs = copy.deepcopy(raw or {})
    if isinstance(kwargs.get("features_extractor_class"), str):
        kwargs["features_extractor_class"] = EXTRACTORS[kwargs["features_extractor_class"]]
    if isinstance(kwargs.get("activation_fn"), str):
        kwargs["activation_fn"] = ACTIVATIONS[kwargs["activation_fn"]]
    return kwargs


def resolve_env_cfgs(env_field) -> list[dict]:
    """``env`` may be a config name, an inline dict, or a list of either (mixed training)."""
    items = env_field if isinstance(env_field, list) else [env_field]
    return [load_named("env", e) if isinstance(e, str) else e for e in items]


def train(env_cfgs: list[dict], algo_cfg: dict, seed: int, run_dir: str, verbose: int = 0) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    th.set_num_threads(int(algo_cfg.get("torch_threads", 1)))
    specs = [build_env_spec(e, algo_cfg) for e in env_cfgs]
    algo = algo_cfg["algo"]
    n_envs = int(algo_cfg.get("n_envs", 1))

    venv = make_vec_env(specs, n_envs=n_envs, seed=seed, monitor_dir=os.path.join(run_dir, "monitor"))
    callbacks = []
    lagrange = None
    lag_cfg = algo_cfg.get("lagrangian")
    if lag_cfg:
        venv = lagrange = LagrangianRewardWrapper(venv, lam=lag_cfg.get("lam_init", 0.0),
                                                  normalize=lag_cfg.get("normalize", True))
        if not lag_cfg.get("fixed", False):
            callbacks.append(
                LagrangeMultiplierCallback(
                    lagrange,
                    cost_limit=lag_cfg["cost_limit"],
                    kp=lag_cfg.get("kp", 0.05),
                    ki=lag_cfg.get("ki", 0.01),
                    kd=lag_cfg.get("kd", 0.0),
                )
            )
    callbacks.append(DrivingMetricsCallback(os.path.join(run_dir, "metrics.csv"), lagrange=lagrange))

    curriculum = None
    for spec in specs:
        levels = (spec.get("randomization") or {}).get("curriculum")
        if levels:
            curriculum = CurriculumCallback(n_levels=len(levels), **algo_cfg.get("curriculum", {}))
            callbacks.append(curriculum)
            break

    model = ALGOS[algo](
        algo_cfg.get("policy", "MlpPolicy"),
        venv,
        seed=seed,
        device=algo_cfg.get("device", "cpu"),
        tensorboard_log=os.path.join(run_dir, "tb"),
        policy_kwargs=resolve_policy_kwargs(algo_cfg.get("policy_kwargs")),
        verbose=verbose,
        **algo_cfg.get("hyperparams", {}),
    )

    start = time.time()
    model.learn(total_timesteps=int(algo_cfg["total_timesteps"]), callback=CallbackList(callbacks))
    wall = time.time() - start
    model.save(os.path.join(run_dir, "model.zip"))
    venv.close()

    summary = {
        "algo": algo,
        "seed": seed,
        "total_timesteps": int(algo_cfg["total_timesteps"]),
        "train_seconds": round(wall, 1),
        "env_names": [e.get("name", e.get("id")) for e in env_cfgs],
    }
    with open(os.path.join(run_dir, "run_config.yaml"), "w") as f:
        yaml.safe_dump({"summary": summary, "algo": algo_cfg, "env_specs": specs}, f, sort_keys=False)
    for cb in callbacks:
        if isinstance(cb, LagrangeMultiplierCallback):
            with open(os.path.join(run_dir, "lagrange.json"), "w") as f:
                json.dump(cb.history, f)
    if curriculum is not None:
        with open(os.path.join(run_dir, "curriculum.json"), "w") as f:
            json.dump(curriculum.promotions, f)
    return summary
