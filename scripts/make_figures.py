#!/usr/bin/env python
"""Turn evaluation results and training logs into report tables and figures.

Inputs:  results/eval/episodes.parquet, runs/<name>/seed*/{metrics.csv,lagrange.json}
Outputs: figures/*.png, results/tables/*.{md,tex}
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from safelane.config import ROOT, load_yaml  # noqa: E402
from safelane.eval.stats import iqm_with_ci, summarize  # noqa: E402

FIG, TAB = ROOT / "figures", ROOT / "results" / "tables"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

# One colour per method identity, identical in every figure (reference categorical order).
COLORS = {
    "DQN": "#2a78d6",
    "D3QN (ours)": "#eb6834",
    "QR-DQN": "#1baf7a",
    "QR-DQN CVaR0.25": "#eda100",
    "PPO": "#e87ba4",
    "PPO-Attn": "#008300",
    "SafeLane (PPO-Lag-Attn)": "#6250d6",
    "MaskablePPO (shield)": "#e34948",
    "PPO-Attn + curriculum": "#1baf7a",
    "PPO-Attn + fixed penalty": "#eda100",
    "SafeLane-Mixed": "#e34948",  # never shares a figure with MaskablePPO (ramp-only ablation)
}
BASE_STYLE = {
    "IDM+MOBIL (rule-based)": dict(color=INK2, ls="--"),
    "Constant speed": dict(color="#9a9893", ls=":"),
    "Random": dict(color="#bdbcb6", ls=":"),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5,
})


LONG = " (long budget)"
LONG_BASE = {"SafeLane": "SafeLane (PPO-Lag-Attn)", "SafeLane-Mixed": "SafeLane-Mixed", "D3QN": "D3QN (ours)"}

# Line charts show a focused subset (<= 6 series) so every line stays identifiable.
FOCUS_HIGHWAY = ["DQN", "PPO", "PPO-Attn", "SafeLane (PPO-Lag-Attn)", "SafeLane (long budget)",
                 "IDM+MOBIL (rule-based)"]
FOCUS_RAMP = ["PPO", "PPO-Attn", "PPO-Attn + fixed penalty", "SafeLane (PPO-Lag-Attn)", "SafeLane (long budget)",
              "IDM+MOBIL (rule-based)"]


def style(label: str) -> dict:
    """Colour follows the method; long-budget runs reuse their method's colour, dashed."""
    if label in BASE_STYLE:
        return dict(BASE_STYLE[label], lw=1.6)
    if label.endswith(LONG):
        base = LONG_BASE.get(label[: -len(LONG)], label[: -len(LONG)])
        return dict(color=COLORS.get(base, "#6d6c67"), ls=(0, (4, 2)))
    return dict(color=COLORS.get(label, "#6d6c67"), ls="-")


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("figure:", FIG / name)


# ----------------------------------------------------------------------------- tables
TABLE_COLS = {
    "highway": [("crashed", "Collision %", 100, "{:.1f}"), ("avg_speed", "Speed (m/s)", 1, "{:.1f}"),
                ("return", "Return", 1, "{:.1f}"), ("ttc_below_1_5_frac", "TTC<1.5s %", 100, "{:.1f}"),
                ("headway_below_1_frac", "Headway<1s %", 100, "{:.1f}"), ("rms_jerk", "RMS jerk", 1, "{:.2f}"),
                ("lane_changes_per_km", "LC/km", 1, "{:.1f}")],
    "ramp_merge": [("success", "Merge success %", 100, "{:.1f}"), ("crashed", "Collision %", 100, "{:.1f}"),
                   ("avg_speed", "Speed (m/s)", 1, "{:.1f}"), ("cost", "Episode cost", 1, "{:.2f}"),
                   ("near_miss", "Near-miss %", 100, "{:.1f}"), ("rss_violation_frac", "RSS viol. %", 100, "{:.1f}")],
}


def training_steps(method: str) -> str:
    """Training budget of a run (e.g. '120k'); '-' for rule-based baselines."""
    cfg = ROOT / "runs" / method.split("__")[0] / "seed0" / "run_config.yaml"
    if not cfg.exists():
        return "-"
    steps = load_yaml(cfg)["summary"]["total_timesteps"]
    return f"{steps // 1000}k"


def results_table(summary: pd.DataFrame, scenario: str, groups: list[str], name: str) -> pd.DataFrame:
    s = summary[(summary.scenario == scenario) & summary.group.isin(groups)].copy()
    cols = TABLE_COLS["highway" if scenario == "highway" else "ramp_merge"]
    rows = []
    for _, r in s.iterrows():
        row = {"Method": r["label"], "Steps": training_steps(r["method"]), "Seeds": int(r["n_seeds"])}
        for key, title, scale, fmt in cols:
            row[title] = (fmt + " [" + fmt + ", " + fmt + "]").format(
                r[key] * scale, r[f"{key}_lo"] * scale, r[f"{key}_hi"] * scale)
        row["_sort"] = r["crashed"]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("_sort").drop(columns="_sort")
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.md").write_text(df.to_markdown(index=False))
    compact = df.copy()
    for c in compact.columns[3:]:
        compact[c] = compact[c].str.split(" ").str[0]
    (TAB / f"{name}.tex").write_text(compact.to_latex(index=False, escape=True))
    print("table:", TAB / f"{name}.md")
    return df


# ----------------------------------------------------------------------------- figures
def load_curves(run_name: str, column: str, grid: np.ndarray) -> np.ndarray | None:
    curves = []
    for f in sorted((ROOT / "runs" / run_name).glob("seed*/metrics.csv")):
        if not (f.parent / "model.zip").exists():  # unfinished run
            continue
        d = pd.read_csv(f)
        if len(d) < 2:
            continue
        curves.append(np.interp(grid, d["timesteps"], d[column], left=np.nan, right=np.nan))
    return np.array(curves) if curves else None


def fig_learning_curves(exp: dict) -> None:
    panels = [("highway", "crashed", "Highway: collision rate (training)"),
              ("ramp", "success", "RampMerge: merge success (training)")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (group, col, title) in zip(axes, panels):
        for run in [r for r in exp["runs"] if r["group"] in (group, f"ext_{group}")]:
            steps = int(load_algo_steps(run))
            grid = np.linspace(2000, steps, 60)
            c = load_curves(run["name"], col, grid)
            if c is None:
                continue
            mean, sd = np.nanmean(c, 0), np.nanstd(c, 0)
            st = style(run["label"])
            ax.plot(grid / 1e3, mean, label=run["label"], color=st["color"], ls=st["ls"])
            ax.fill_between(grid / 1e3, mean - sd, mean + sd, color=st["color"], alpha=0.12, lw=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("environment steps (thousands)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="best")
    save(fig, "learning_curves.png")


def load_algo_steps(run: dict) -> float:
    from safelane.config import deep_update, load_named

    return deep_update(load_named("algo", run["algo"]), run.get("overrides", {}))["total_timesteps"]


def method_order(df: pd.DataFrame, groups: list[str], focus: list[str] | None = None) -> list[tuple[str, str]]:
    sub = df[df.group.isin(groups)][["method", "label"]].drop_duplicates()
    pairs = list(sub.itertuples(index=False, name=None))
    if focus is not None:
        pairs = sorted([p for p in pairs if p[1] in focus], key=lambda p: focus.index(p[1]))
    return pairs


def sweep_plot(ax, summary, methods, scenarios, xs, metric, title, ylabel):
    for method, label in methods:
        ys, lo, hi = [], [], []
        for sc in scenarios:
            r = summary[(summary.method == method) & (summary.scenario == sc)]
            ys.append(r[metric].iloc[0] if len(r) else np.nan)
            lo.append(r[f"{metric}_lo"].iloc[0] if len(r) else np.nan)
            hi.append(r[f"{metric}_hi"].iloc[0] if len(r) else np.nan)
        if np.all(np.isnan(ys)):
            continue
        st = style(label)
        ax.plot(xs, ys, marker="o", label=label, color=st["color"], ls=st["ls"], lw=st.get("lw", 2))
        ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.1, lw=0)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)


def fig_density(summary, df) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    hw_methods = method_order(df, ["highway", "ext_highway", "baseline"], FOCUS_HIGHWAY)
    sweep_plot(axes[0], summary, hw_methods,
               ["hw_density_0.5", "highway_fast", "hw_density_1.5", "hw_density_2.0", "hw_density_2.5"],
               [0.5, 1.0, 1.5, 2.0, 2.5], "crashed", "Highway: collision rate vs traffic density", "collision rate")
    for lo_x, hi_x in [(0.4, 0.6), (2.25, 2.6)]:  # training range was [0.7, 2.0]
        axes[0].axvspan(lo_x, hi_x, color=GRID, alpha=0.6, lw=0)
    ramp_methods = method_order(df, ["ramp", "ext_ramp", "baseline"], FOCUS_RAMP)
    sweep_plot(axes[1], summary, ramp_methods,
               ["ramp_density_0.5", "ramp_density_0.75", "ramp_merge", "ramp_density_1.5", "ramp_density_2.0"],
               [0.5, 0.75, 1.0, 1.5, 2.0], "success", "RampMerge: merge success vs traffic density",
               "merge success rate")
    axes[1].axvspan(1.75, 2.05, color=GRID, alpha=0.6, lw=0)  # training range was [0.5, 1.5]
    axes[0].legend(fontsize=6.5, loc="upper center")
    axes[1].legend(fontsize=6.5, loc="lower left")
    for ax in axes:
        ax.set_xlabel("traffic density (shaded: not seen in training)")
    save(fig, "density_sweep.png")


def fig_robustness(summary, df) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    sweep_plot(axes[0], summary, method_order(df, ["highway", "ext_highway"], FOCUS_HIGHWAY),
               ["highway_fast", "hw_noise_0.05", "hw_noise_0.1", "hw_noise_0.2"], [0, 0.05, 0.1, 0.2],
               "crashed", "Highway: collision rate vs sensor noise", "collision rate")
    sweep_plot(axes[1], summary, method_order(df, ["ramp", "ext_ramp"], FOCUS_RAMP),
               ["ramp_merge", "ramp_noise_0.05", "ramp_noise_0.1", "ramp_noise_0.2"], [0, 0.05, 0.1, 0.2],
               "success", "RampMerge: merge success vs sensor noise", "merge success rate")
    for ax in axes:
        ax.set_xlabel("observation noise std (normalised units)")
        ax.legend(fontsize=6.5, loc="best")
    save(fig, "robustness.png")


def fig_generalisation(summary) -> None:
    scen = ["highway_fast", "ramp_merge", "merge"]
    names = {"highway_fast": "Highway", "ramp_merge": "RampMerge (ours)", "merge": "merge-v1 (stock)"}
    groups = ["highway", "ramp", "mixed", "ext_highway", "ext_ramp", "ext_mixed", "baseline"]
    s = summary[summary.scenario.isin(scen) & summary.group.isin(groups)]
    s = s[~s.method.str.contains("__")]
    rows = s.drop_duplicates("method")[["method", "label", "group"]]
    rows = rows.sort_values(["group", "label"])
    mat = np.full((len(rows), len(scen)), np.nan)
    for i, m in enumerate(rows.method):
        for j, sc in enumerate(scen):
            r = s[(s.method == m) & (s.scenario == sc)]
            if len(r):
                mat[i, j] = r["success"].iloc[0]
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(rows) + 1.2))
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(scen)), [names[c] for c in scen])
    ax.set_yticks(range(len(rows)), [f"{lab}  [{g}]" for lab, g in zip(rows.label, rows.group)], fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > 0.55 else INK)
    ax.set_title("Task success (RampMerge: merged; others: no crash)\nrow = agent [training group], column = test task",
                 loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    save(fig, "generalisation_heatmap.png")


PARETO_SHOWN = {  # methods relevant to the speed/safety trade-off (all others are in the tables)
    "PPO", "PPO-Attn", "PPO-Attn + fixed penalty", "MaskablePPO (shield)", "SafeLane (PPO-Lag-Attn)",
    "SafeLane (long budget)", "SafeLane-Mixed", "IDM+MOBIL (rule-based)",
}
PARETO_LABELS = {  # selective direct labels with hand-placed offsets (points) to avoid collisions
    "SafeLane": (7, 5), "SafeLane (long budget)": (-4, 7), "SafeLane-Mixed": (7, -3), "PPO": (5, 5),
    "PPO-Attn": (5, 5), "PPO-Attn + fixed penalty": (7, -9), "MaskablePPO (shield)": (5, -12),
    "IDM+MOBIL (rule-based)": (5, 6), "PPO-Lag d=0.5": (7, -3), "PPO-Lag d=1": (7, -12), "PPO-Lag d=4": (7, 3),
    "PPO-Lag d=8": (5, 6),
}


def fig_pareto(summary) -> None:
    s = summary[(summary.scenario == "ramp_merge")
                & summary.group.isin(["ramp", "pareto", "mixed", "ext_ramp", "ext_mixed", "baseline"])]
    s = s[s.label.isin(PARETO_SHOWN) | (s.group == "pareto")]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    pts = []
    for _, r in s.iterrows():
        st = style(r["label"])
        marker = "s" if r["group"] == "pareto" else ("D" if r["group"] == "baseline" else "o")
        color = COLORS["SafeLane (PPO-Lag-Attn)"] if r["group"] == "pareto" else st["color"]
        ax.errorbar(r["crashed"], r["avg_speed"],
                    xerr=[[r["crashed"] - r["crashed_lo"]], [r["crashed_hi"] - r["crashed"]]],
                    fmt=marker, color=color, ms=7, elinewidth=1, capsize=0, alpha=0.95,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        name = r["label"].replace(" (PPO-Lag-Attn)", "")
        if name in PARETO_LABELS:
            ax.annotate(name, (r["crashed"], r["avg_speed"]), fontsize=6.5, color=INK2,
                        xytext=PARETO_LABELS[name], textcoords="offset points")
        pts.append((r["crashed"], r["avg_speed"]))
    pts = sorted(pts)
    front, best = [], -np.inf
    for x, y in pts:  # non-dominated: lower collision, higher speed
        if y > best:
            front.append((x, y))
            best = y
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", ls="", color=INK2, label="learned policy (mean of 3 seeds)"),
               Line2D([], [], marker="s", ls="", color=COLORS["SafeLane (PPO-Lag-Attn)"],
                      label="safety-budget sweep d (1 seed)"),
               Line2D([], [], marker="D", ls="", color=INK2, label="rule-based / trivial baseline")]
    if len(front) > 1:
        handles.append(ax.plot(*zip(*front), color=INK2, lw=1, ls="--", zorder=0, label="Pareto front")[0])
    ax.legend(handles=handles, fontsize=6.5, loc="lower right")
    ax.set_xlim(-0.03, max(0.62, float(s["crashed"].max()) + 0.08))
    ax.set_xlabel("collision rate (lower is better)")
    ax.set_ylabel("average speed (m/s)")
    ax.set_title("RampMerge: safety-speed trade-off (squares = budget sweep)", loc="left")
    save(fig, "pareto.png")


def fig_lagrange() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for name, ax_title in [("ramp_ppo_lag", "RampMerge"), ("hw_ppo_lag", "Highway")]:
        files = sorted((ROOT / "runs" / name).glob("seed*/lagrange.json"))
        for k, f in enumerate(files):
            h = np.array(json.loads(f.read_text()))
            if len(h) == 0:
                continue
            color = COLORS["SafeLane (PPO-Lag-Attn)"]  # same method; the task is shown by line style
            ls = "-" if name.startswith("ramp") else (0, (3, 1.5))
            lab = ax_title if k == 0 else None
            axes[0].plot(h[:, 0] / 1e3, h[:, 2], color=color, ls=ls, alpha=0.8, lw=1.4, label=lab)
            axes[1].plot(h[:, 0] / 1e3, pd.Series(h[:, 1]).rolling(5, min_periods=1).mean(), color=color,
                         ls=ls, alpha=0.8, lw=1.4, label=lab)
    axes[1].axhline(2.0, color=INK2, ls="--", lw=1, label="budget d = 2")
    axes[0].set_title("Lagrange multiplier λ during training", loc="left")
    axes[1].set_title("Episode safety cost J_c (rolling mean)", loc="left")
    for ax in axes:
        ax.set_xlabel("environment steps (thousands)")
        ax.legend(fontsize=7)
    save(fig, "lagrange.png")


def fig_iqm(df) -> None:
    sets = load_yaml(ROOT / "configs" / "experiment" / "eval.yaml")["sets"]
    df = df.copy()
    df["safe"] = 1.0 - df["crashed"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"wspace": 0.75})
    for ax, (group_set, metric, title) in zip(axes, [
        (("highway", ["highway", "mixed", "ext_highway", "ext_mixed"]), "safe",
         "Highway family: IQM of (1 - collision rate)"),
        (("ramp", ["ramp", "mixed", "ext_ramp", "ext_mixed"]), "success", "RampMerge family: IQM of merge success"),
    ]):
        set_name, groups = group_set
        sub = df[df.group.isin(groups + ["baseline"]) & ~df.method.str.contains("__")]
        sub = sub[sub.method != "base_random"]
        res = iqm_with_ci(sub, metric, sets[set_name], reps=1000)
        if res.empty:
            continue
        labels = sub.drop_duplicates("method").set_index("method")["label"]
        res["label"] = res.method.map(labels)
        res = res.sort_values("iqm")
        y = np.arange(len(res))
        colors = [INK2 if m.startswith("base_") else "#2a78d6" for m in res.method]
        ax.barh(y, res.iqm, color=colors, height=0.6)
        ax.errorbar(res.iqm, y, xerr=[res.iqm - res.lo, res.hi - res.iqm], fmt="none", ecolor=INK, lw=1)
        ax.set_yticks(y, res.label, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
    save(fig, "iqm.png")


def main() -> None:
    exp = load_yaml(ROOT / "configs" / "experiment" / "main.yaml")
    ext_path = ROOT / "configs" / "experiment" / "extended.yaml"
    if ext_path.exists():
        exp["runs"] += load_yaml(ext_path)["runs"]
    eval_dir = ROOT / "results" / "eval"
    fig_learning_curves(exp)
    fig_lagrange()
    if not (eval_dir / "episodes.parquet").exists():
        print("no evaluation results yet - run scripts/evaluate.py")
        return
    df = pd.read_parquet(eval_dir / "episodes.parquet")
    summary = pd.read_csv(eval_dir / "summary.csv") if (eval_dir / "summary.csv").exists() else summarize(
        df, ["crashed", "success", "avg_speed", "return"], by=("method", "label", "group", "scenario"))
    results_table(summary, "highway", ["highway", "mixed", "ext_highway", "ext_mixed", "baseline"], "table_highway")
    results_table(summary, "ramp_merge", ["ramp", "mixed", "ext_ramp", "ext_mixed", "baseline", "pareto"],
                  "table_ramp")
    fig_density(summary, df)
    fig_robustness(summary, df)
    fig_generalisation(summary)
    fig_pareto(summary)
    fig_iqm(df)


if __name__ == "__main__":
    main()
