#!/usr/bin/env python
"""Train one agent.

Examples:
    python scripts/train.py env=highway_fast algo=dqn seed=0
    python scripts/train.py env=ramp_merge algo=ppo_lag seed=1 algo.lagrangian.cost_limit=2
    python scripts/train.py env=[highway_dr,ramp_merge_dr] algo=ppo_attn algo.total_timesteps=300000
    python scripts/train.py env=highway_fast algo=ppo env.config.vehicles_density=1.5
"""

from __future__ import annotations

import sys

from safelane.config import ROOT, deep_update, load_named, split_overrides
from safelane.train import resolve_env_cfgs, train


def main(argv: list[str]) -> None:
    top, algo_over, env_over = split_overrides(argv)
    seed = int(top.get("seed", 0))
    algo_cfg = deep_update(load_named("algo", top.get("algo", "dqn")), algo_over)
    env_cfgs = [deep_update(e, env_over) for e in resolve_env_cfgs(top.get("env", "highway_fast"))]
    env_tag = "+".join(e["name"] for e in env_cfgs)
    default_dir = ROOT / "runs" / "manual" / f"{algo_cfg['name']}_{env_tag}" / f"seed{seed}"
    run_dir = str(top.get("run_dir", default_dir))
    summary = train(env_cfgs, algo_cfg, seed, run_dir, verbose=int(top.get("verbose", 1)))
    print(summary)
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
