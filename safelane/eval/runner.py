"""Parallel, resumable evaluation of trained agents and baselines across scenarios.

A *candidate* is a policy to evaluate (a trained run or a rule-based baseline); a
*scenario* is a named env config. Every (candidate, scenario) job runs N episodes on
fixed evaluation seeds (EVAL_SEED_BASE + i, never used in training) and its per-episode
metrics are cached as one parquet file, so an interrupted evaluation resumes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from safelane.config import build_env_spec, deep_update, load_named

EVAL_SEED_BASE = 10_000


@dataclass
class Candidate:
    method: str  # unique id (experiment run name, or base_<baseline>)
    kind: str  # "model" or "baseline"
    algo: str  # SB3 algo name or baseline name
    seed: int = 0
    run_dir: str | None = None
    train_env: str = "-"
    algo_cfg: dict = field(default_factory=dict)  # observation / action / shield config
    policy_kwargs: dict = field(default_factory=dict)  # e.g. {"cvar_alpha": 0.25}
    label: str = ""  # display name in tables / plots
    group: str = ""  # experiment group (highway / ramp / mixed / pareto / baseline)


def candidate_from_run(run_dir: str, method: str, policy_kwargs: dict | None = None, label: str = "",
                       group: str = "") -> Candidate:
    with open(Path(run_dir) / "run_config.yaml") as f:
        rc = yaml.safe_load(f)
    s = rc["summary"]
    return Candidate(
        method=method,
        kind="model",
        algo=s["algo"],
        seed=int(s["seed"]),
        run_dir=str(run_dir),
        train_env="+".join(s["env_names"]),
        algo_cfg=rc["algo"],
        policy_kwargs=policy_kwargs or {},
        label=label or method,
        group=group,
    )


def baseline_candidate(name: str, label: str) -> Candidate:
    reference = load_named("algo", "dqn")  # baselines share the agents' action space
    cfg = {"observation": reference["observation"], "action": reference["action"]}
    return Candidate(method=f"base_{name}", kind="baseline", algo=name, algo_cfg=cfg, label=label,
                     group="baseline")


def scenario_spec(scenario: dict, algo_cfg: dict) -> dict:
    """Scenario = base env config name + overrides; the candidate supplies obs/action."""
    scenario = copy.deepcopy(scenario)
    env_cfg = load_named("env", scenario.pop("env"))
    env_cfg = deep_update(env_cfg, scenario)
    spec = build_env_spec(env_cfg, algo_cfg)
    if "shield" in scenario:  # scenario-level shield (post-hoc shielding test) wins
        spec["shield"] = scenario["shield"]
    return spec


# ----------------------------------------------------------------------------- worker side
_MODEL_CACHE: dict = {}


def make_policy(c: Candidate):
    if c.kind == "baseline":
        from safelane.baselines import BASELINES

        return BASELINES[c.algo]()
    from safelane.agents.registry import ModelPolicy, load_model

    key = c.run_dir
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_model(c.algo, os.path.join(c.run_dir, "model.zip"))
    return ModelPolicy(_MODEL_CACHE[key], c.algo, **c.policy_kwargs)


def run_episodes(c: Candidate, scenario_name: str, scenario: dict, episodes: int) -> list[dict]:
    import torch as th

    from safelane.envs import make_env

    th.set_num_threads(1)
    policy = make_policy(c)
    env = make_env(scenario_spec(scenario, c.algo_cfg))
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + ep)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs, env))
            done = terminated or truncated
        row = dict(info["episode_metrics"])
        row.update(
            method=c.method,
            label=c.label,
            group=c.group,
            algo=c.algo,
            kind=c.kind,
            seed=c.seed,
            train_env=c.train_env,
            scenario=scenario_name,
            episode=ep,
        )
        rows.append(row)
    env.close()
    return rows


# ----------------------------------------------------------------------------- driver side
def _job_key(c: Candidate, scenario_name: str, scenario: dict, episodes: int) -> str:
    model_mtime = os.path.getmtime(os.path.join(c.run_dir, "model.zip")) if c.run_dir else 0
    payload = json.dumps(
        [c.method, c.algo, c.seed, c.run_dir, c.policy_kwargs, c.algo_cfg, model_mtime, scenario_name, scenario,
         episodes],
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def _job(args) -> tuple[str, list[dict]]:
    c_dict, name, scenario, episodes, out = args
    rows = run_episodes(Candidate(**c_dict), name, scenario, episodes)
    pd.DataFrame(rows).to_parquet(out)
    return out, rows


def evaluate(
    candidates: list[Candidate],
    scenarios: dict[str, dict],
    episodes: int,
    cache_dir: str,
    workers: int = 8,
    only: dict[str, list[str]] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Evaluate every candidate on every scenario (or on ``only[method]`` scenarios)."""
    os.makedirs(cache_dir, exist_ok=True)
    jobs, done_files = [], []
    for c in candidates:
        for name, scenario in scenarios.items():
            if only and c.method in only and name not in only[c.method]:
                continue
            out = os.path.join(cache_dir, f"{_job_key(c, name, scenario, episodes)}.parquet")
            if os.path.exists(out):
                done_files.append(out)
            else:
                jobs.append((asdict(c), name, scenario, episodes, out))
    if verbose:
        print(f"[eval] {len(done_files)} cached jobs, {len(jobs)} to run with {workers} workers")
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_job, j) for j in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                out, _ = fut.result()
                done_files.append(out)
                if verbose and (i % 10 == 0 or i == len(jobs)):
                    print(f"[eval] {i}/{len(jobs)} jobs done")
    return pd.concat([pd.read_parquet(f) for f in done_files], ignore_index=True)
