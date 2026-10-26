"""Interactive demo: pick a trained agent, a scenario and traffic conditions, watch it drive.

    pip install -e ".[demo]"
    python app/demo.py            # http://localhost:7860
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from safelane.agents.registry import ModelPolicy, load_model
from safelane.baselines import BASELINES
from safelane.config import ROOT, load_yaml
from safelane.envs import make_env
from safelane.eval.runner import baseline_candidate, candidate_from_run, scenario_spec
from safelane.xai import record_gif

SCENARIOS = {"RampMerge (ours)": "ramp_merge", "Highway (highway-v0)": "highway",
             "Highway fast": "highway_fast", "Stock merge-v1": "merge"}
TRAFFIC = {"IDM (default)": "idm", "Randomised drivers": "randomized", "Aggressive": "aggressive",
           "Cautious": "cautious"}


def available_agents() -> dict[str, str]:
    agents = {f"[rule-based] {k}": k for k in BASELINES}
    for model in sorted((ROOT / "runs").glob("*/seed0/model.zip")):
        agents[model.parent.parent.name] = str(model.parent)
    return agents


AGENTS = available_agents()


def run(agent: str, scenario: str, density: float, traffic: str, noise: float, seed: int):
    source = AGENTS[agent]
    base = load_yaml(ROOT / "configs" / "experiment" / "eval.yaml")["scenarios"][SCENARIOS[scenario]]
    sc = dict(base)
    sc["config"] = {**base.get("config", {}), "vehicles_density": density,
                    "other_vehicles_type": TRAFFIC[traffic]}
    if noise > 0:
        sc["obs_noise"] = {"sigma": noise}
    if source in BASELINES:
        cand, policy, model = baseline_candidate(source, agent), BASELINES[source](), None
    else:
        cand = candidate_from_run(source, Path(source).parent.name)
        model = load_model(cand.algo, str(Path(source) / "model.zip"))
        policy = ModelPolicy(model, cand.algo)
    env = make_env(scenario_spec(sc, cand.algo_cfg), render_mode="rgb_array")
    out = Path(tempfile.mkdtemp()) / "episode.gif"
    metrics = record_gif(policy, env, str(out), seed=int(seed), model=model, caption=agent)
    env.close()
    shown = {k: round(float(v), 3) for k, v in metrics.items()
             if k in ("success", "crashed", "avg_speed", "return", "cost", "min_ttc", "lane_changes")}
    return str(out), shown


demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Dropdown(list(AGENTS), value=next(iter(AGENTS)), label="Agent"),
        gr.Dropdown(list(SCENARIOS), value="RampMerge (ours)", label="Scenario"),
        gr.Slider(0.25, 3.0, value=1.0, step=0.25, label="Traffic density"),
        gr.Dropdown(list(TRAFFIC), value="IDM (default)", label="Other drivers"),
        gr.Slider(0.0, 0.3, value=0.0, step=0.05, label="Sensor noise (std)"),
        gr.Number(value=10_000, precision=0, label="Episode seed"),
    ],
    outputs=[gr.Image(label="Episode (attention lines in red for attention agents)"), gr.JSON(label="Metrics")],
    title="SafeLane-RL: safe RL highway driving",
    description="Constrained RL agents for highway driving and on-ramp merging (highway-env).",
)

if __name__ == "__main__":
    demo.launch()
