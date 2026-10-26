#!/usr/bin/env python
"""Optuna hyperparameter search (TPE sampler + median pruning on intermediate evaluations).

    python scripts/tune.py algo=ppo env=highway_dr trials=30 steps=40000
    python scripts/tune.py algo=d3qn env=ramp_merge_dr trials=20 steps=30000

Objective: mean evaluation score = success - collisions (in [-1, 1]) over held-out seeds.
Best parameters are written to results/tuning/<algo>_<env>.yaml.
"""

from __future__ import annotations

import sys

import numpy as np
import optuna
import torch as th
import yaml
from stable_baselines3.common.callbacks import BaseCallback

from safelane.agents.registry import ALGOS, ModelPolicy
from safelane.config import ROOT, build_env_spec, deep_update, load_named, split_overrides
from safelane.envs import make_env, make_vec_env
from safelane.train import resolve_policy_kwargs


def sample(trial: optuna.Trial, algo: str) -> dict:
    hp = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 2e-3, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.8, 0.85, 0.9, 0.95, 0.99]),
    }
    if algo.startswith("ppo") or algo == "maskable_ppo":
        hp.update(
            n_steps=trial.suggest_categorical("n_steps", [64, 128, 256]),
            batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
            ent_coef=trial.suggest_float("ent_coef", 1e-4, 0.05, log=True),
            clip_range=trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
            gae_lambda=trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98]),
        )
    else:
        hp.update(
            buffer_size=trial.suggest_categorical("buffer_size", [15_000, 50_000]),
            target_update_interval=trial.suggest_categorical("target_update_interval", [50, 250, 1000]),
            exploration_fraction=trial.suggest_float("exploration_fraction", 0.1, 0.5),
        )
    return hp


def score(model, algo: str, spec: dict, episodes: int = 10) -> float:
    env, policy, vals = make_env(spec), ModelPolicy(model, algo), []
    for ep in range(episodes):
        obs, _ = env.reset(seed=50_000 + ep)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs, env))
            done = terminated or truncated
        m = info["episode_metrics"]
        vals.append(m["success"] - m["crashed"])
    return float(np.mean(vals))


class PruningCallback(BaseCallback):
    def __init__(self, trial, algo, spec, every):
        super().__init__()
        self.trial, self.algo, self.spec, self.every = trial, algo, spec, every
        self.step_idx, self.pruned = 0, False

    def _on_step(self) -> bool:
        if self.num_timesteps // self.every > self.step_idx:
            self.step_idx += 1
            self.trial.report(score(self.model, self.algo, self.spec, 5), self.step_idx)
            if self.trial.should_prune():
                self.pruned = True
                return False
        return True


def main(argv: list[str]) -> None:
    top, algo_over, env_over = split_overrides(argv)
    algo_name, env_name = top.get("algo", "ppo"), top.get("env", "highway_dr")
    trials, steps = int(top.get("trials", 20)), int(top.get("steps", 40_000))
    base = deep_update(load_named("algo", algo_name), algo_over)
    env_cfg = deep_update(load_named("env", env_name), env_over)
    spec = build_env_spec(env_cfg, base)
    th.set_num_threads(1)

    def objective(trial):
        cfg = deep_update(base, {"hyperparams": sample(trial, base["algo"])})
        venv = make_vec_env(spec, n_envs=int(cfg.get("n_envs", 1)), seed=trial.number)
        model = ALGOS[cfg["algo"]](cfg.get("policy", "MlpPolicy"), venv, seed=trial.number, device="cpu",
                                   policy_kwargs=resolve_policy_kwargs(cfg.get("policy_kwargs")),
                                   **cfg["hyperparams"])
        cb = PruningCallback(trial, cfg["algo"], spec, every=max(steps // 4, 1))
        try:
            model.learn(steps, callback=cb)
        finally:
            venv.close()
        if cb.pruned:
            raise optuna.TrialPruned()
        return score(model, cfg["algo"], spec, 20)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(n_startup_trials=5),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    study.optimize(objective, n_trials=trials)
    out = ROOT / "results" / "tuning" / f"{algo_name}_{env_name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump({"best_value": study.best_value, "best_params": study.best_params}))
    print("best", study.best_value, study.best_params, "->", out)


if __name__ == "__main__":
    main(sys.argv[1:])
