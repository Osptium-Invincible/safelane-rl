#!/usr/bin/env python
"""Verify that the numbers quoted in README.md and report/main.tex match results/eval/summary.csv.

    python scripts/check_numbers.py        # exits with status 1 if anything disagrees

Checks every cell of the README results tables (sections C1 and C2) and every row of the report's Table 1.
"""

from __future__ import annotations

import re
import sys

import pandas as pd

from safelane.config import ROOT

ALIASES = {"SafeLane": "SafeLane (PPO-Lag-Attn)", "QR-DQN (CVaR 0.25)": "QR-DQN CVaR0.25"}
README_TABLES = {
    "## C1.": ("highway", ["highway", "ext_highway", "mixed", "ext_mixed", "baseline"],
               [("crashed", 100, 1), ("avg_speed", 1, 1), ("return", 1, 1), ("headway_below_1_frac", 100, 1),
                ("lane_changes_per_km", 1, 1)]),
    "## C2.": ("ramp_merge", ["ramp", "ext_ramp", "mixed", "ext_mixed", "baseline"],
               [("success", 100, 1), ("crashed", 100, 1), ("avg_speed", 1, 1), ("cost", 1, 2), ("near_miss", 100, 1)]),
}
# report Table 1: (row label, highway method, ramp method)
REPORT_ROWS = [
    ("Random", "base_random", "base_random"), ("Constant speed", "base_constant_speed", "base_constant_speed"),
    ("IDM+MOBIL (rules)", "base_idm_mobil", "base_idm_mobil"), ("DQN", "hw_dqn", None),
    ("D3QN (ours)", "hw_d3qn", "ramp_d3qn"), ("QR-DQN", "hw_qrdqn", None), ("PPO", "hw_ppo", "ramp_ppo"),
    ("PPO-Attn", "hw_ppo_attn", "ramp_ppo_attn"), ("PPO-Attn+penalty", None, "ramp_ppo_pen"),
    ("MaskablePPO (shield)", None, "ramp_maskable"), (r"\textbf{SafeLane}", "hw_ppo_lag", "ramp_ppo_lag"),
    (r"\textbf{SafeLane} (long)", "ext_hw_ppo_lag", "ext_ramp_ppo_lag"),
    ("SafeLane-Mixed (long)", "ext_mixed_ppo_lag", "ext_mixed_ppo_lag"),
]


def readme_problems(readme: str, s: pd.DataFrame) -> tuple[int, list[str]]:
    problems, n = [], 0
    for title, (scenario, groups, cols) in README_TABLES.items():
        start = readme.index(title)
        block = readme[start: readme.index("**How to read it:**", start)]
        sub = s[(s.scenario == scenario) & s.group.isin(groups)]
        for line in block.splitlines():
            if not line.startswith("|") or "---" in line or "Method" in line:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            label = cells[0].replace("*", "").strip()
            label = ALIASES.get(label, label)
            rows = sub[sub.label == label]
            if len(rows) != 1:
                problems.append(f"README {title}: no unique result row for '{label}'")
                continue
            r = rows.iloc[0]
            for (col, scale, dp), cell in zip(cols, cells[1:]):
                found = re.findall(r"-?\d+\.?\d*", cell.replace("*", ""))
                if not found:
                    continue
                n += 1
                truth = round(float(r[col]) * scale, dp)
                if abs(float(found[0]) - truth) > 10 ** (-dp) / 2 + 1e-9:
                    problems.append(f"README {title} {label} {col}: says {found[0]}, data {truth}")
    return n, problems


def report_problems(tex: str, s: pd.DataFrame) -> tuple[int, list[str]]:
    idx = s.set_index(["method", "scenario"])

    def cell(method, scenario, col, scale=100):
        if method is None or (method, scenario) not in idx.index:
            return "--"
        return f"{idx.loc[(method, scenario), col] * scale:.1f}"

    problems, n = [], 0
    for label, hw, ramp in REPORT_ROWS:
        expected = " & ".join([label, cell(hw, "highway", "crashed"), cell(hw, "highway", "return", 1),
                               cell(hw, "hw_density_0.5", "crashed"), cell(ramp, "ramp_merge", "success"),
                               cell(ramp, "ramp_merge", "crashed"), cell(ramp, "ramp_density_2.0", "success")])
        n += 1
        if expected + r" \\" not in tex:
            problems.append(f"report Table 1 row differs, expected: {expected}")
    return n, problems


def main() -> int:
    s = pd.read_csv(ROOT / "results" / "eval" / "summary.csv")
    n1, p1 = readme_problems((ROOT / "README.md").read_text(), s)
    n2, p2 = report_problems((ROOT / "report" / "main.tex").read_text(), s)
    for p in p1 + p2:
        print("MISMATCH:", p)
    print(f"checked {n1} README cells and {n2} report rows: {len(p1) + len(p2)} problems")
    return 1 if p1 or p2 else 0


if __name__ == "__main__":
    sys.exit(main())
