#!/usr/bin/env python
"""Train every (run, seed) of an experiment file; finished runs are skipped (resumable).

Multi-env (PPO-family) jobs are memory-heavy, so at most ``--heavy`` run at once;
single-env DQN-family jobs use up to ``--light`` extra slots.

    python scripts/run_experiments.py configs/experiment/main.yaml --heavy 1 --light 2
    python scripts/run_experiments.py configs/experiment/main.yaml --only hw_dqn,hw_ppo --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

from safelane.config import ROOT, deep_update, load_named, load_yaml


def flatten(prefix: str, d: dict) -> list[str]:
    out = []
    for k, v in d.items():
        key = f"{prefix}.{k}"
        if isinstance(v, dict):
            out += flatten(key, v)
        else:
            out.append(f"{key}={yaml.safe_dump(v, default_flow_style=True).strip().removesuffix('...').strip()}")
    return out


def build_jobs(exp: dict, only: set[str] | None) -> list[dict]:
    jobs = []
    for run in exp["runs"]:
        if only and run["name"] not in only:
            continue
        algo_cfg = deep_update(load_named("algo", run["algo"]), run.get("overrides", {}))
        env = run["env"]
        env_arg = f"[{','.join(env)}]" if isinstance(env, list) else env
        for seed in run.get("seeds", exp["seeds"]):
            run_dir = ROOT / "runs" / run["name"] / f"seed{seed}"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "train.py"),
                f"env={env_arg}",
                f"algo={run['algo']}",
                f"seed={seed}",
                f"run_dir={run_dir}",
                "verbose=0",
                *flatten("algo", run.get("overrides", {})),
            ]
            jobs.append(
                {
                    "name": f"{run['name']}/seed{seed}",
                    "cmd": cmd,
                    "run_dir": run_dir,
                    "heavy": int(algo_cfg.get("n_envs", 1)) > 1,
                    "done": (run_dir / "model.zip").exists(),
                }
            )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", default="configs/experiment/main.yaml", nargs="?")
    parser.add_argument("--heavy", type=int, default=1, help="concurrent multi-env jobs")
    parser.add_argument("--light", type=int, default=2, help="concurrent single-env jobs")
    parser.add_argument("--only", type=str, default=None, help="comma-separated run names")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    exp = load_yaml(args.experiment)
    jobs = build_jobs(exp, set(args.only.split(",")) if args.only else None)
    pending = [j for j in jobs if not j["done"]]
    print(f"{len(jobs)} jobs, {len(jobs) - len(pending)} already finished, {len(pending)} to run")
    if args.dry_run:
        for j in pending:
            print(("H " if j["heavy"] else "L ") + " ".join(j["cmd"][1:]))
        return

    running: list[tuple[dict, subprocess.Popen, float]] = []
    t0 = time.time()
    while pending or running:
        for item in list(running):
            job, proc, start = item
            if proc.poll() is not None:
                running.remove(item)
                status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
                print(f"[{time.time() - t0:7.0f}s] {status:>12} {job['name']} ({time.time() - start:.0f}s)",
                      flush=True)
        n_heavy = sum(j["heavy"] for j, _, _ in running)
        n_light = len(running) - n_heavy
        for job in list(pending):
            if (job["heavy"] and n_heavy < args.heavy) or (not job["heavy"] and n_light < args.light):
                Path(job["run_dir"]).mkdir(parents=True, exist_ok=True)
                log = open(Path(job["run_dir"]) / "train.log", "w")
                proc = subprocess.Popen(job["cmd"], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                running.append((job, proc, time.time()))
                pending.remove(job)
                n_heavy += job["heavy"]
                n_light += not job["heavy"]
                print(f"[{time.time() - t0:7.0f}s] {'started':>12} {job['name']}", flush=True)
        time.sleep(5)
    print(f"all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
