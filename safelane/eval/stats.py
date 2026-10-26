"""Statistics for RL evaluation (Agarwal et al., 2021).

* ``summarize``: per (method, scenario) mean over seeds of per-seed episode means, with a
  95% hierarchical bootstrap CI (resample seeds, then episodes within each seed).
* ``iqm_with_ci``: interquartile mean (IQM) with stratified-bootstrap CIs over a
  (runs x tasks) score matrix, where tasks = scenarios.
* ``probability_of_improvement``: P(X > Y) averaged over tasks (Mann-Whitney statistic).

The last two follow the definitions of Agarwal et al. and their ``rliable`` library; they are
implemented here directly because rliable's ``arch`` dependency is incompatible with pandas 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import trim_mean


def hierarchical_bootstrap_ci(groups: list[np.ndarray], reps: int = 2000, alpha: float = 0.05, seed: int = 0):
    if len(groups) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(np.mean([g.mean() for g in groups]))
    stats = np.empty(reps)
    for r in range(reps):
        picked = rng.integers(len(groups), size=len(groups))
        stats[r] = np.mean([rng.choice(groups[i], size=len(groups[i])).mean() for i in picked])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def summarize(df: pd.DataFrame, metrics: list[str], by=("method", "scenario"), reps: int = 1000) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(list(by)):
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row["n_seeds"] = g["seed"].nunique()
        row["n_episodes"] = len(g)
        for m in metrics:
            groups = [s[m].to_numpy(dtype=float) for _, s in g.groupby("seed")]
            point, lo, hi = hierarchical_bootstrap_ci(groups, reps=reps)
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = point, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def score_matrices(df: pd.DataFrame, metric: str, scenarios: list[str]) -> dict[str, np.ndarray]:
    """method -> (n_seeds x n_scenarios) matrix of per-seed means."""
    out = {}
    for method, g in df[df["scenario"].isin(scenarios)].groupby("method"):
        table = g.groupby(["seed", "scenario"])[metric].mean().unstack("scenario")
        table = table.reindex(columns=scenarios)
        if table.isna().any().any():
            continue
        out[method] = table.to_numpy()
    return out


def iqm(scores: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50% of all (run, task) scores."""
    return float(trim_mean(np.asarray(scores).ravel(), 0.25))


def stratified_bootstrap(scores: np.ndarray, stat, reps: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Resample runs with replacement independently within each task (column); percentile CI."""
    rng = np.random.default_rng(seed)
    n_runs, n_tasks = scores.shape
    stats = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(n_runs, size=(n_runs, n_tasks))
        stats[r] = stat(np.take_along_axis(scores, idx, axis=0))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(stat(scores)), float(lo), float(hi)


def iqm_with_ci(df: pd.DataFrame, metric: str, scenarios: list[str], reps: int = 2000) -> pd.DataFrame:
    rows = []
    for method, scores in score_matrices(df, metric, scenarios).items():
        point, lo, hi = stratified_bootstrap(scores, iqm, reps=reps)
        rows.append({"method": method, "iqm": point, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def _poi(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) + 0.5 P(X = Y), averaged over tasks (columns)."""
    per_task = []
    for j in range(x.shape[1]):
        a, b = x[:, j][:, None], y[:, j][None, :]
        per_task.append(np.mean((a > b) + 0.5 * (a == b)))
    return float(np.mean(per_task))


def probability_of_improvement(df: pd.DataFrame, metric: str, scenarios: list[str], x: str, y: str,
                               reps: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    scores = score_matrices(df, metric, scenarios)
    sx, sy = scores[x], scores[y]
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for r in range(reps):
        ix = rng.integers(sx.shape[0], size=sx.shape)
        iy = rng.integers(sy.shape[0], size=sy.shape)
        stats[r] = _poi(np.take_along_axis(sx, ix, 0), np.take_along_axis(sy, iy, 0))
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return _poi(sx, sy), float(lo), float(hi)
