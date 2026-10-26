#!/usr/bin/env python
"""Evaluate all finished runs of an experiment plus the rule-based baselines.

    python scripts/evaluate.py                                  # main.yaml + eval.yaml
    python scripts/evaluate.py --episodes 20 --workers 6        # quicker
    python scripts/evaluate.py --groups ramp,baseline

Writes results/eval/episodes.parquet (one row per episode) and results/eval/summary.csv
(mean + 95% hierarchical-bootstrap CI per method x scenario). Cached per job, resumable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from safelane.config import ROOT, load_yaml
from safelane.eval import baseline_candidate, candidate_from_run, evaluate
from safelane.eval.stats import summarize

BASELINES = {"random": "Random", "constant_speed": "Constant speed", "idm_mobil": "IDM+MOBIL (rule-based)"}
SUMMARY_METRICS = [
    "crashed", "success", "avg_speed", "return", "cost", "near_miss", "ttc_below_1_5_frac",
    "headway_below_1_frac", "rms_jerk", "lane_changes_per_km", "rss_violation_frac", "shield_frac", "length",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="configs/experiment/main.yaml,configs/experiment/extended.yaml",
                   help="comma-separated experiment files")
    p.add_argument("--eval-config", default="configs/experiment/eval.yaml")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--groups", type=str, default=None, help="comma-separated groups to evaluate")
    p.add_argument("--out", default=str(ROOT / "results" / "eval"))
    args = p.parse_args()

    ev = load_yaml(args.eval_config)
    exp = {"runs": [], "eval_sets": {}}
    for path in args.experiment.split(","):  # several experiment files can be evaluated together
        part = load_yaml(path)
        exp["runs"] += part["runs"]
        exp["eval_sets"].update(part["eval_sets"])
    episodes = args.episodes or ev["episodes"]
    groups = set(args.groups.split(",")) if args.groups else None
    set_names = exp["eval_sets"]

    candidates, only = [], {}

    def add(c, group):
        scenario_names = sorted({s for set_name in set_names[group] for s in ev["sets"][set_name]})
        only[c.method] = scenario_names
        candidates.append(c)

    for run in exp["runs"]:
        group = run["group"]
        if groups and group not in groups:
            continue
        for seed_dir in sorted((ROOT / "runs" / run["name"]).glob("seed*")):
            if not (seed_dir / "model.zip").exists():
                continue
            add(candidate_from_run(str(seed_dir), run["name"], label=run["label"], group=group), group)
            for v in run.get("variants", []):
                method = f"{run['name']}__{v['label']}"
                add(candidate_from_run(str(seed_dir), method, v.get("policy_kwargs"), v["label"], group), group)
    if not groups or "baseline" in groups:
        for name, label in BASELINES.items():
            add(baseline_candidate(name, label), "baseline")

    print(f"{len(candidates)} candidates, {episodes} episodes per scenario")
    if not candidates:
        print("nothing to evaluate (no finished runs in the selected groups)")
        return
    df = evaluate(candidates, ev["scenarios"], episodes, cache_dir=str(Path(args.out) / "cache"),
                  workers=args.workers, only=only)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    df.to_parquet(Path(args.out) / "episodes.parquet")
    summary = summarize(df, SUMMARY_METRICS, by=("method", "label", "group", "scenario"))
    summary.to_csv(Path(args.out) / "summary.csv", index=False)
    print(f"wrote {len(df)} episodes -> {args.out}")


if __name__ == "__main__":
    main()
