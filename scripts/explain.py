#!/usr/bin/env python
"""Explainability figures for a trained agent.

1. Integrated-Gradients attribution per observation slot (vehicle) x feature, averaged
   over states visited on evaluation episodes -> figures/ig_<run>.png
2. Average ego-attention per vehicle slot (nearest-first) for attention models
   -> figures/attention_<run>.png

    python scripts/explain.py --run runs/ramp_ppo_lag/seed0 --scenario ramp_merge
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch as th  # noqa: E402

from safelane.agents.registry import ModelPolicy, load_model  # noqa: E402
from safelane.config import ROOT, load_yaml  # noqa: E402
from safelane.envs import make_env  # noqa: E402
from safelane.eval.runner import candidate_from_run, scenario_spec  # noqa: E402
from safelane.xai import integrated_gradients  # noqa: E402
from safelane.xai.video import attention_extractor  # noqa: E402

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def collect(run: str, scenario: str, episodes: int):
    scenarios = load_yaml(ROOT / "configs" / "experiment" / "eval.yaml")["scenarios"]
    cand = candidate_from_run(str(ROOT / run), Path(run).parent.name)
    model = load_model(cand.algo, str(ROOT / run / "model.zip"))
    policy = ModelPolicy(model, cand.algo)
    env = make_env(scenario_spec(scenarios[scenario], cand.algo_cfg))
    ext = attention_extractor(model)
    observations, attention = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=20_000 + ep)
        done = False
        while not done:
            observations.append(obs)
            action = policy(obs, env)
            if ext is not None and ext.last_attention is not None:
                attention.append(ext.last_attention[0].mean(0).numpy())
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    features = cand.algo_cfg["observation"]["features"]
    return model, np.array(observations), np.array(attention), features


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/ramp_ppo_lag/seed0")
    p.add_argument("--scenario", default="ramp_merge")
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()
    th.set_num_threads(2)
    model, obs, att, features = collect(args.run, args.scenario, args.episodes)
    tag = Path(args.run).parent.name

    # the ego row is part of the IG baseline, so its attribution is zero by construction: drop it
    ig = np.abs(integrated_gradients(model, obs)).mean(axis=0)[1:]  # (V-1, F)
    ig = ig / ig.sum()
    fig, ax = plt.subplots(figsize=(4.8, 0.28 * ig.shape[0] + 1.3), facecolor=SURFACE)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    im = ax.imshow(ig, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(features)), features, fontsize=8)
    ax.set_yticks(range(ig.shape[0]), [f"car {i}" for i in range(1, ig.shape[0] + 1)], fontsize=7)
    ax.set_title(f"Integrated Gradients share: {tag}\n(cars sorted nearest first)", loc="left", fontsize=9,
                 color=INK)
    fig.colorbar(im, ax=ax, fraction=0.05)
    out = ROOT / "figures" / f"ig_{tag}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("figure:", out)

    if len(att):
        mean_att = att.mean(axis=0)
        fig, ax = plt.subplots(figsize=(4.8, 2.6), facecolor=SURFACE)
        ax.bar(range(len(mean_att)), mean_att, color="#2a78d6", width=0.6)
        ax.set_xticks(range(len(mean_att)), ["ego"] + [str(i) for i in range(1, len(mean_att))], fontsize=7)
        ax.set_xlabel("observation slot (cars sorted nearest first)", color=INK2)
        ax.set_ylabel("mean attention weight", color=INK2)
        ax.set_title(f"Where the ego-attention looks: {tag}", loc="left", fontsize=9, color=INK)
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        out = ROOT / "figures" / f"attention_{tag}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        print("figure:", out)


if __name__ == "__main__":
    main()
