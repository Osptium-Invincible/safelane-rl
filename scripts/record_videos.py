#!/usr/bin/env python
"""Record README / demo GIFs for trained agents and baselines.

    python scripts/record_videos.py                       # default showcase set -> videos/
    python scripts/record_videos.py --run runs/ramp_ppo_lag/seed0 --scenario ramp_merge --seed 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

from safelane.agents.registry import ModelPolicy, load_model
from safelane.baselines import BASELINES
from safelane.config import ROOT, load_yaml
from safelane.envs import make_env
from safelane.eval.runner import baseline_candidate, candidate_from_run, scenario_spec
from safelane.xai import record_gif

SHOWCASE = [
    # (output name, run dir or baseline name, scenario, episode seed, caption)
    ("ramp_safelane", "runs/ext_ramp_ppo_lag/seed0", "ramp_merge", 10_004, "SafeLane (PPO-Lag + attention)"),
    ("ramp_ppo", "runs/ramp_ppo/seed0", "ramp_merge", 10_004, "PPO (MLP)"),
    ("ramp_idm", "idm_mobil", "ramp_merge", 10_004, "IDM+MOBIL rule-based"),
    ("highway_safelane", "runs/ext_hw_ppo_lag/seed0", "highway", 10_001, "SafeLane (PPO-Lag + attention)"),
    ("highway_dqn", "runs/hw_dqn/seed0", "highway", 10_001, "DQN"),
    ("highway_dense_safelane", "runs/ext_hw_ppo_lag/seed0", "hw_density_2.0", 10_002, "SafeLane, density x2"),
    ("highway_sparse_ppo", "runs/hw_ppo/seed0", "hw_density_0.5", 10_000, "PPO (MLP), sparse traffic"),
    ("merge_mixed", "runs/ext_mixed_ppo_lag/seed0", "merge", 10_000, "SafeLane-Mixed on stock merge-v1"),
]


def record(source: str, scenario_name: str, seed: int, caption: str, out: Path) -> dict:
    scenarios = load_yaml(ROOT / "configs" / "experiment" / "eval.yaml")["scenarios"]
    if source in BASELINES:
        cand = baseline_candidate(source, caption)
        policy, model = BASELINES[source](), None
    else:
        cand = candidate_from_run(str(ROOT / source), Path(source).parent.name)
        model = load_model(cand.algo, str(ROOT / source / "model.zip"))
        policy = ModelPolicy(model, cand.algo)
    env = make_env(scenario_spec(scenarios[scenario_name], cand.algo_cfg), render_mode="rgb_array")
    metrics = record_gif(policy, env, str(out), seed=seed, model=model, caption=caption)
    env.close()
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", help="run dir (e.g. runs/ramp_ppo_lag/seed0) or baseline name")
    p.add_argument("--scenario", default="ramp_merge")
    p.add_argument("--seed", type=int, default=10_000)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out_dir = ROOT / "videos"
    out_dir.mkdir(exist_ok=True)
    jobs = SHOWCASE if not args.run else [
        (Path(args.out).stem if args.out else f"{Path(args.run).parent.name}_{args.scenario}", args.run,
         args.scenario, args.seed, Path(args.run).parent.name)]
    for name, source, scenario, seed, caption in jobs:
        if source not in BASELINES and not (ROOT / source / "model.zip").exists():
            print(f"skip {name}: {source} not trained yet")
            continue
        m = record(source, scenario, seed, caption, out_dir / f"{name}.gif")
        print(f"{name}.gif  crashed={m.get('crashed')} success={m.get('success')} speed={m.get('avg_speed', 0):.1f}")


if __name__ == "__main__":
    main()
