# SafeLane-RL — Safe, Robust & Explainable Reinforcement Learning for Highway Driving

> **AI Course · Assignment 2 (Mini Project): Real-Life Use Cases of RL** — IIT Jodhpur, 2026
> Team: _<names / roll numbers>_ · Repo: _<github link>_
>
> **Status:** this is the original research & design document written before implementation (21 Oct 2026).
> It is kept for the trade-off analysis and references. The implemented project, its actual commands and
> its results are described in the top-level [README.md](../README.md); where the two differ, the README wins
> (e.g. final training budgets, config system, exact experiment list).

---

## Table of contents

1. [What we submitted vs. what we will build](#1-what-we-submitted-vs-what-we-will-build)
2. [How this project scores on every grading criterion](#2-how-this-project-scores-on-every-grading-criterion)
3. [The upgraded idea — feature tiers](#3-the-upgraded-idea--feature-tiers)
4. [RL formulation (S, A, R, P, γ) — goes straight into the report](#4-rl-formulation-s-a-r-p-γ)
5. [Technology choices and their trade-offs](#5-technology-choices-and-their-trade-offs)
6. [Where to use what (decision map)](#6-where-to-use-what-decision-map)
7. [System architecture & repository layout](#7-system-architecture--repository-layout)
8. [Installation (verified)](#8-installation-verified)
9. [Usage — planned commands](#9-usage--planned-commands)
10. [Evaluation protocol & metrics](#10-evaluation-protocol--metrics)
11. [Measured numbers so far & expected results](#11-measured-numbers-so-far--expected-results)
12. [Experiments matrix (what goes in the report)](#12-experiments-matrix)
13. [Timeline, team split & commit plan](#13-timeline-team-split--commit-plan)
14. [Gotchas we verified (read before coding)](#14-gotchas-we-verified)
15. [Plagiarism-safe rules for this repo](#15-plagiarism-safe-rules)
16. [2-page report skeleton](#16-2-page-report-skeleton)
17. [References](#17-references)

---

## 1. What we submitted vs. what we will build

**Submitted abstract (approved):**
> Develop an autonomous driving agent using Reinforcement Learning algorithms such as PPO or DQN in a
> simulated highway environment. The agent will learn speed control, lane changing, collision avoidance,
> and safe merging. Performance will be evaluated using collision rates, average speed, merging success
> rates, and cumulative rewards across diverse traffic scenarios.

Everything below stays inside this scope. We only go **deeper**; we never change the topic.

**One-line pitch of the upgraded project:**
> *A constrained-RL driving agent that treats safety as a hard budget, not a reward weight. It
> merges from an on-ramp into dense traffic. It reads traffic through an ego-attention network whose
> attention we visualise. It is stress-tested on unseen traffic densities, aggressive drivers and noisy
> sensors, and reported with confidence intervals against PPO, DQN, and a rule-based IDM+MOBIL driver.*

Why this wins relative marks. Most teams will run the stock `highway-env` SB3 example with one seed and
report one number. We differentiate on five axes that graders notice:

| Axis | Typical team | SafeLane-RL |
|---|---|---|
| Problem | stock `highway-v0` | stock highway **+ a custom on-ramp-merge env we wrote** (stock `merge-v0` has no ego-merge and no success flag, see §14) |
| Safety | collision penalty only | **CMDP + PPO-Lagrangian** with a time-to-collision cost, plus an optional action-masking shield ablation |
| Model | MLP on a flattened 5×5 matrix | **Ego-attention (transformer) extractor**, permutation-invariant, with attention-overlay videos |
| Evaluation | 1 seed, mean reward | **5 seeds × 100 episodes, IQM + 95% bootstrap CIs** (rliable), TTC/headway/jerk/RSS metrics, rule-based baseline |
| Generalisation | none | **train→test heat-map** across scenarios and densities, sensor-noise robustness curves |

---

## 2. How this project scores on every grading criterion

Mapped directly from `AI_Assignment2_2026_RL.pdf`:

| Requirement in the PDF | How we satisfy it | Where |
|---|---|---|
| Strictly RL; heuristic or supervised = disqualified | Every controller is learned by RL (DQN family, PPO family, PPO-Lagrangian). IDM+MOBIL appears **only as a comparison baseline**. The safety shield is an optional, clearly separated ablation, and results are reported with **and** without it. No behaviour cloning. | §3, §4 |
| Runnable code, environment setup, **trained agent** in the repo | `models/` ships the final `.zip` checkpoints (small: MLP or attention ≈ 0.1–2 MB). `uv`/`pip` lockfile included. | §7, §8 |
| README with install, execution commands, dependencies, reproduction | This file, plus `make reproduce`, which regenerates every table and figure from fixed seeds | §8, §9 |
| Commit history shows ongoing team effort | Feature branches per person, PRs, weekly tags `v0.1`…`v1.0` (real incremental work, never faked) | §13 |
| 2-page two-column report: S, A, R, environment and algorithm, baselines, graphs | §4 is written to be pasted into the report. §12 lists the exact figures. | §4, §16 |
| GitHub link + team at the top of the report | In the report template | §16 |
| Marks relative to other teams | Stronger baselines + statistics + more scenarios | §10–12 |
| **Bonus: highly unique ideation & exceptional effort** | Custom ramp-merge env, CMDP formulation, attention XAI, RSS formal check, generalisation heat-map, live demo | §3 Tier 2–3 |
| No forking or copying of repos | We use libraries (`pip install`) only. All env subclasses, wrappers, extractors, the Lagrangian and the evaluation code are written by us. | §15 |

---

## 3. The upgraded idea — feature tiers

Effort is in **person-days**. Do Tier 1 completely before touching Tier 2.

### Tier 1 — Core (needed for good marks) · ~8–10 person-days

| # | Feature | What exactly | Library |
|---|---|---|---|
| 1.1 | **Env suite** | `highway-fast-v0` (train), `highway-v0` (eval), `merge-v1`, `roundabout-v1`, `intersection-v2`, `exit-v1` | highway-env 1.12.1 |
| 1.2 | **Custom `RampMerge-v0`** | Subclass `MergeEnv` so the **ego spawns on the on-ramp** `("j","k",0)` and must merge into main-road traffic. Adds `info["is_success"]` (reached a main lane, x > 370 m, no crash). This makes "merging success rate" from our abstract an actual measurable thing. Li et al. (2025) built a similar ramp scenario on highway-env, so the idea is validated, but we write ours from scratch. | own code |
| 1.3 | **DQN baseline** | SB3 DQN with the official highway hyperparameters (γ=0.8, lr 5e-4, [256,256]) | SB3 2.9.0 |
| 1.4 | **PPO baseline** | SB3 PPO, 8 × `SubprocVecEnv` | SB3 2.9.0 |
| 1.5 | **Our own Double + Dueling DQN** | SB3's DQN is *vanilla* (no Double, Dueling or PER). We subclass it: Double-Q target (≈10 lines in `train()`), Dueling head (custom `QNetwork`), and SB3 ≥2.7's built-in `n_steps=3`. Original code that also beats vanilla. | own code on SB3 |
| 1.6 | **Rule-based baseline** | Ego driven by highway-env's `IDMVehicle` (IDM + MOBIL). Comparison only. Plus random and constant-speed baselines. | highway-env |
| 1.7 | **Metrics harness** | Collision rate, avg speed, merge success, cumulative reward, **plus TTC, min headway, jerk, lane changes/km**. 5 seeds × 100 episodes on fixed eval seeds. | own code |
| 1.8 | **Diverse traffic** | Density {0.5, 1, 1.5, 2, 2.5}, NPC type {IDM, Defensive, Aggressive}, lanes {2, 3, 4} | highway-env config |
| 1.9 | **Statistics** | IQM + stratified-bootstrap 95% CIs, probability of improvement | rliable 1.2.0 |
| 1.10 | **Videos/GIFs** | Before and after training, per scenario, embedded in this README | gymnasium `RecordVideo` |

### Tier 2 — Differentiators (relative-marks winners) · ~10–14 person-days

| # | Feature | Why graders care | Effort | Risk |
|---|---|---|---|---|
| 2.1 | **Safety as a constraint: CMDP + PPO-Lagrangian** with TTC/headway cost (§4.5) | The textbook safe-RL formulation. It replaces hand-tuned penalty weights with an interpretable safety budget *d*. We implement it ourselves (~100 lines: a `VecEnvWrapper` + `BaseCallback` dual update, PID-Lagrangian for stability). | 3–5 d | Low–Med |
| 2.2 | **Ego-attention feature extractor** | Fixes the order-dependence of MLP-on-Kinematics (the highway-env FAQ calls MLP results "reasonable but sub-optimal"). It is permutation-invariant and handles variable vehicle counts. **Attention weights drawn as lines on the video** make a strong visual. | 4–6 d | Med |
| 2.3 | **Domain randomisation + curriculum** | Per-episode random density, NPC aggressiveness and IDM params (`TIME_WANTED`, `POLITENESS`). Curriculum ramps density 0.5→2.5 when success ≥ 90%. | 2–3 d | Low |
| 2.4 | **Generalisation heat-map** | Train on {highway, merge, mixed} × test on {highway, merge, roundabout, ramp-merge, densities 0.5×–3×} | 2 d | Low |
| 2.5 | **Robustness curves** | Gaussian observation noise σ∈{0, .05, .1, .2}, vehicle dropout p∈{0, .1, .3}, 1–3-step observation delay | 1–2 d | Low |
| 2.6 | **Action-masking shield (ablation)** | `MaskablePPO` + mask of road-illegal actions (`get_available_actions()`) and TTC-unsafe lane changes. Report collisions **with and without** the shield and the shield-intervention rate, to show the RL policy does the driving. | 2 d | Low |

### Tier 3 — Bonus "exceptional effort" (pick 2–3) · 2–6 person-days each

| # | Feature | Notes |
|---|---|---|
| 3.1 | **Speed-vs-safety Pareto front** | Sweep 8–12 reward weightings (or the Lagrangian budget *d*), plot avg speed vs collision rate, compute hypervolume. One plot that shows a principled trade-off analysis. |
| 3.2 | **RSS formal safety check** (Mobileye, Shalev-Shwartz 2017) | Compute RSS safe longitudinal distance each step and report the % of steps in RSS violation and "who was to blame" in each crash. Links the project to industry standards (IEEE 2846). Cheap: ~1 day. |
| 3.3 | **Risk-sensitive QR-DQN + CVaR** | `sb3_contrib.QRDQN`, act on CVaR_α (mean of the worst α quantiles). Plot α vs collisions and speed as a tunable "cautiousness knob". |
| 3.4 | **Explainability pack** | Attention maps, Integrated Gradients (Captum) per vehicle slot and feature, action-probability timeline synced to video. |
| 3.5 | **Live demo** | Gradio app: choose scenario, density, policy and noise → returns an MP4 + metrics. Optionally host on Hugging Face Spaces. |
| 3.6 | **Adversarial NPC (RARL-style)** | 2 controlled vehicles; the adversary is rewarded for inducing ego crashes and penalised for crashing itself. Highest novelty, highest risk; do it last or as eval-only. |
| 3.7 | **Multi-agent cooperative merging** | highway-env multi-agent mode with shared-parameter PPO. Measures merge throughput. |

**Recommended bundle for a 3–4 week window:** all of Tier 1, then 2.1, 2.2, 2.3, 2.4, 2.6, then 3.1, 3.2 and 3.5.
Together these tell one story — *"safe, robust, explainable RL driving with formal checks"* — which reads
better in a 2-page report than a pile of unrelated tricks.

---

## 4. RL formulation (S, A, R, P, γ)

Written so it can be pasted into the report.

### 4.1 Environment & dynamics

- **Simulator:** `highway-env` 1.12.1 (Farama Foundation). 2-D kinematic bicycle model. The ego is
  controlled through low-level PID controllers that track a target lane and speed.
- **Traffic:** NPCs follow **IDM** for longitudinal control and **MOBIL** for lane changes:

  $$a = a_{max}\left[1 - \left(\tfrac{v}{v_0}\right)^{\delta} - \left(\tfrac{s^*(v,\Delta v)}{s}\right)^2\right], \quad s^* = s_0 + vT + \tfrac{v\,\Delta v}{2\sqrt{a_{max} b}}$$

  Defaults: $a_{max}=6$, comfort accel 3 / −5 m/s², $s_0=10$ m, $T=1.5$ s, $\delta\in[3.5,4.5]$, MOBIL politeness 0.
- **Timing:** simulation 15 Hz (5 Hz in the `-fast` variant), policy decisions at **1 Hz**. One action = 1 s of driving.
- **Episode:** highway ends on collision or after 40 s (30 s in the fast variant). Merge ends on collision or at x > 370 m.

### 4.2 State space S

Main choice: **Kinematics** observation, matrix $s_t \in \mathbb{R}^{V \times F}$.

| Variant | V (vehicles) | F (features) | Used by |
|---|---|---|---|
| Default | 5 | 5: `presence, x, y, vx, vy` | DQN/PPO MLP baselines |
| Rich | 15 | 7: + `cos_h, sin_h` | ego-attention model |

- Row 0 is the ego. The other rows are the nearest vehicles sorted by distance, given **relative to the ego** (`absolute=False`), normalised to [−1, 1] (x: ±200 m, vx/vy: ±80 m/s), and zero-padded when there are fewer vehicles.
- Partial observability: only the V nearest vehicles are visible (`see_behind=False` by default).
- Alternatives evaluated in an ablation (§5.4): OccupancyGrid (4×11×11), TimeToCollision (3×3×10), Grayscale (4×128×64).

### 4.3 Action space A

**Discrete meta-actions**, $|A| = 5$:

| id | action | effect |
|---|---|---|
| 0 | `LANE_LEFT` | PID steers to the left lane |
| 1 | `IDLE` | keep lane and speed |
| 2 | `LANE_RIGHT` | PID steers to the right lane |
| 3 | `FASTER` | next target speed (default set {20, 25, 30} m/s; we use 9 levels in 15–35 m/s for finer speed control) |
| 4 | `SLOWER` | previous target speed |

This covers **speed control** (3/4), **lane changing** (0/2) and, implicitly, **merging** (a lane change from the ramp).

### 4.4 Reward function R

The default highway-env reward (verified in the 1.12.1 source):

$$r_t = \Big[\, w_c\,\mathbb{1}_{crash} + w_r \tfrac{\ell_t}{L-1} + w_v\,\mathrm{clip}\!\left(\tfrac{v_t\cos\psi_t - 20}{30-20},0,1\right)\Big]_{\to[0,1]} \cdot \mathbb{1}_{on\ road}$$

with $w_c=-1$, $w_r=0.1$ (keep right), $w_v=0.4$, linearly rescaled to [0, 1].
Because crashes **terminate** the episode, the main penalty for a crash is the loss of all future reward,
not the −1 itself.

**Our shaped reward** (each term is ablated):

$$r_t = w_v\,\tilde v_t \;+\; w_r\,\tilde\ell_t \;-\; w_c\,\mathbb{1}_{crash} \;-\; w_{lc}\,\mathbb{1}_{lane\ change} \;-\; w_j\,|j_t| \;+\; w_m\,\mathbb{1}_{merged} \;+\; \underbrace{\gamma\Phi(s_{t+1}) - \Phi(s_t)}_{\text{potential shaping}}$$

- $\tilde v_t$: normalised forward speed. $\tilde\ell_t$: keep-right term. $j_t$: jerk (comfort). $\mathbb{1}_{merged}$: one-time bonus in RampMerge.
- $\Phi(s) = -\max(0,\,2\,\text{s} - h(s))$ is the "2-second rule" headway potential. **Potential-based shaping does not change the optimal policy** (Ng et al., 1999), which is the argument for it in the report.
- Note: highway-env's `lane_change_reward` key is **ignored in `highway-v0`** (it is not among the reward components; see §14), so we add the term ourselves.

### 4.5 Safety as a constraint (CMDP) — Tier 2.1

$$\max_\pi\; J_r(\pi)=\mathbb{E}_\pi\Big[\sum_t \gamma^t r_t\Big] \quad \text{s.t.}\quad J_c(\pi)=\mathbb{E}_\pi\Big[\sum_t c_t\Big] \le d$$

Per-step cost:
$c_t = \mathbb{1}[\mathrm{TTC}_t < 2\,\text{s}] + \mathbb{1}[h_t < 1\,\text{s}] + 10\cdot\mathbb{1}_{crash}$,
where $\mathrm{TTC} = \Delta x / (v_{ego} - v_{lead})$ if closing, else ∞, and $h = \Delta x / v_{ego}$ is the time headway.

Lagrangian relaxation: $\mathcal{L}(\pi,\lambda) = J_r(\pi) - \lambda\,(J_c(\pi) - d)$.
PPO maximises $r_t - \lambda c_t$. After every rollout the dual variable updates as
$\lambda \leftarrow \max\big(0,\ \lambda + \eta_\lambda(\hat J_c - d)\big)$, with PID-Lagrangian damping (Stooke et al., 2020).

### 4.6 Discount γ and horizon

At 1 decision/s, γ = 0.8 gives an effective horizon of $1/(1-\gamma) = 5$ s; this is the official highway example and is myopic but stable.
γ = 0.9–0.95 (10–20 s) is better for anticipating merges. γ is tuned per scenario (§12).

### 4.7 Algorithms

| Algorithm | Role | Why |
|---|---|---|
| DQN (vanilla) | baseline | in the abstract; off-policy, sample-efficient |
| **Double + Dueling DQN, n-step 3** (ours) | improved value-based | less overestimation; IDLE-heavy action values benefit from the dueling split |
| PPO | baseline | in the abstract; stable, parallel |
| **PPO + ego-attention** | main model | permutation-invariant traffic reasoning |
| **PPO-Lagrangian** (ours) | safety model | constraint satisfaction |
| MaskablePPO | shield ablation | quantifies how much safety comes from masking |
| QR-DQN (+CVaR) | bonus | risk-sensitive acting |

---

## 5. Technology choices and their trade-offs

### 5.1 Simulator

| Option | Fidelity | Speed / compute | Fit for us | Verdict |
|---|---|---|---|---|
| **highway-env 1.12.1** | 2-D kinematic, behaviour-level (IDM/MOBIL NPCs) | Pure Python, CPU. Measured: 76 steps/s (`highway-fast-v0`), 7 steps/s (`highway-v0`) per core | pip install, Gymnasium API, has merge / roundabout / intersection / exit, easy to subclass | ✅ **Use** |
| MetaDrive | 3-D physics (Panda3D), procedural maps, real-log replay, built-in safe-RL cost | Claims 1000+ FPS lightweight; CPU OK | Great for generalisation, but the PyPI build pins Python < 3.12 and adds a second API to learn | Optional stretch (zero-shot test only) |
| CARLA 0.9.16 / 0.10 (UE5) | Photorealistic, full sensors | GPU 8–16 GB VRAM, 20–130 GB disk, far too slow for RL in 3 weeks | Overkill (perception isn't our topic) | ❌ |
| SUMO 1.28 + sumo-rl | Realistic microscopic traffic networks | CPU; libsumo ~8× faster than TraCI | sumo-rl is built for **traffic signals**; ego control needs custom TraCI code | ❌ (wrong focus) |
| SMARTS / BARK / Flow / CommonRoad-RL | Varies | — | Stale (last releases 2019–2024), painful installs | ❌ |
| Waymax | Data-driven (Waymo logs), JAX | GPU/TPU | Needs dataset approval; non-commercial licence | ❌ |

**Trade-off we accept:** highway-env is not photorealistic and has perfect perception. We address the sim-to-real gap
in the report with the noise/dropout robustness tests and domain randomisation (§3, 2.3/2.5).

### 5.2 RL library

| Library | Version (Oct 2026) | Pros | Cons | Use for |
|---|---|---|---|---|
| **Stable-Baselines3** | 2.9.0 (needs Py ≥3.10, torch ≥2.8, gymnasium <2.0) | Reliable, well documented, callbacks, VecEnvs, custom extractors, CPU-friendly | DQN is vanilla; no Rainbow or PER | ✅ **Backbone** |
| **sb3-contrib** | 2.9.0 | MaskablePPO, RecurrentPPO, QR-DQN, TRPO, ARS, CrossQ, TQC | MaskablePPO cannot be recurrent | ✅ masks, QR-DQN, LSTM |
| Tianshou | 2.0.1 (Py ≥3.11) | Rainbow, C51, IQN, FQF, PER, discrete SAC built in | v2 broke the v1 API; smaller community | Only if we want full Rainbow without writing it |
| OmniSafe | 0.5.0 | PPO-Lag, CPO, PID-Lag, many safe-RL algos | Needs a custom CMDP wrapper; heavier deps (safety-gymnasium, wandb) | Optional cross-check of our own PPO-Lag |
| RLlib (Ray 2.59) | — | Rainbow options, multi-agent, scale | Heavy, config-heavy, overkill for one agent | Only for Tier 3.7 multi-agent |
| CleanRL | — | Single-file readable algos | Designed to be copied → **plagiarism risk** | Read for understanding only; never copy |
| TorchRL 0.14 | — | Modular, GPU-native | Verbose, steep learning curve | ❌ |

### 5.3 Algorithm trade-offs for this task

| | DQN family | PPO family |
|---|---|---|
| Sample efficiency | **Higher** (replay buffer reuses crash transitions) | Lower (on-policy) |
| Stability | Lower (Q overestimation, sensitive to target-update interval) | **Higher** (clipped objective) |
| Wall-clock | Single env → slower | **Parallel envs → faster**. Measured: 70 steps/s (DQN, 1 env) vs 288 steps/s (PPO, 8 envs) |
| Safety extensions | Mask Q-values to −∞; risk via QR-DQN/CVaR | MaskablePPO; **Lagrangian is natural** (policy gradient on r − λc) |
| Best when | budget ≤ 5e4 steps, single core | multi-core machine, constraints, attention models |

Extensions and what each buys:

| Extension | Gains | Cost |
|---|---|---|
| Double DQN | removes max-operator overestimation | ~0 |
| Dueling head | better when many actions have similar value (IDLE vs SLOWER on an empty road) | ~0 |
| n-step (SB3 `n_steps`) | faster credit propagation for collision consequences | slight off-policy bias |
| PER | replays rare crash transitions more often | slower steps; needs β annealing |
| QR-DQN | return distribution → risk-aware acting | more parameters; re-tune (defaults are Atari) |
| RecurrentPPO (LSTM) | infers other drivers' intent over time | ~2–3× slower; frame-stacking is the cheap alternative |
| SAC / TD3 (continuous) | realistic throttle and steering | must also learn lane-keeping; much harder in our budget → **not used** |

### 5.4 Observation type trade-offs

| Observation | Shape | Pros | Cons | Decision |
|---|---|---|---|---|
| **Kinematics** | 5×5 / 15×7 | Compact, fast, MLP-friendly | Order-dependent for an MLP (solved by attention) | ✅ main |
| OccupancyGrid | 4×11×11 | Order-free, fixed size, encodes road | Coarse 5 m cells | ablation |
| TimeToCollision | 3×3×10 | Directly safety-relevant, tiny | Assumes constant speed | ablation |
| Grayscale (CNN) | 4×128×64 | No feature engineering | Renders every step (slow); ~5× more samples (official CNN example: 1e5 vs 2e4) | ablation if time |

### 5.5 Action type trade-offs

| Action | Pros | Cons | Decision |
|---|---|---|---|
| **DiscreteMetaAction** (5) | Safe low-level control, fast learning, matches "lane change / speed" decisions | Coarse speed (fix: 9 target speeds) | ✅ |
| DiscreteAction (9) | Finer control | Has to learn lane keeping | ❌ |
| ContinuousAction | Most realistic | Much harder, lots of off-road early on | ❌ (mention as future work) |

### 5.6 Network architecture trade-offs

| Architecture | Params | Pros | Cons |
|---|---|---|---|
| MLP [256,256] | ~70k | Simplest; official baseline | Depends on vehicle order; fixed V |
| **Ego-attention** (embed [64,64], 2 heads, d=64) | ~35k | Permutation-invariant, variable V, interpretable weights | Needs tuning (lr ≈ 2e-3 in the official transformer example) |
| CNN (NatureCNN) | ~1.7M | For image/grid observations | Slow, GPU-hungry |
| LSTM (RecurrentPPO) | +256 hidden | Temporal reasoning | Slower; tricky inference API (`state`, `episode_start`) |

Ego-attention, written ourselves as an SB3 `BaseFeaturesExtractor`:
reshape obs → (B, V, F) → shared per-vehicle MLP → `nn.MultiheadAttention(query=ego row, keys/values=all rows, key_padding_mask=presence==0)` → residual + LayerNorm → 64-d features → policy/value heads.

### 5.7 Safety-method trade-offs

| Method | Guarantee | Pure RL? | Cost | Decision |
|---|---|---|---|---|
| Reward penalty (TTC, crash) | none; weights hand-tuned | ✅ | trivial | baseline |
| **PPO-Lagrangian (CMDP)** | constraint satisfied *on average* | ✅ | ~100 LOC | ✅ **main** |
| CPO (OmniSafe) | per-update trust region | ✅ | library wrapper | optional |
| Action masking / shield | blocks unsafe actions at runtime | hybrid (rule + RL) | ~50 LOC | **ablation only**, results shown with and without |
| Control barrier functions / MPC shield | formal (model-based) | hybrid | high | ❌ future work |

### 5.8 Tooling

| Need | Choice | Trade-off |
|---|---|---|
| Env / package manager | **uv** (+ `pyproject.toml`, `uv.lock`) | Very fast, reproducible lock. pip/conda also work. |
| Python | **3.12** | The system Python here is 3.14, but 3.12 has the widest wheel coverage (MetaDrive needs < 3.12; torch CPU wheels are solid on 3.12). |
| Experiment tracking | **TensorBoard** (always) + **W&B** (team dashboards, video logging) | TB is offline and zero-setup but weak for comparing runs. W&B needs an account. MLflow has no native SB3 callback. |
| Config | **Hydra** or plain YAML + argparse | Hydra gives multirun sweeps for free but adds a learning curve; YAML is fine if time is short. |
| HP tuning | **Optuna 5.0** (TPE + MedianPruner, `TrialEvalCallback`) | ~20–50 trials × 2e4 steps fits on one laptop overnight |
| Statistics | **rliable 1.2.0** | IQM + bootstrap CIs. Repo archived Oct 2025 but still installs fine. |
| Explainability | Captum (IG), attention maps; SHAP optional | Attention ≠ explanation (first-layer only); show IG next to it |
| Demo | **Gradio** | Faster to build than Streamlit for "pick options → see video" |
| Quality | pytest, ruff, pre-commit, GitHub Actions (tests + 1k-step smoke train) | ~half a day; shows engineering maturity |
| Compute | **Local CPU** (16 threads) for MLP/attention; RTX 4070 only for CNN/attention big batches | SB3 docs: PPO with MLP runs best on CPU. Colab gives only 2 vCPUs; Kaggle ~30 GPU-h/week. |

---

## 6. Where to use what (decision map)

| Situation | Use | Not |
|---|---|---|
| Training | `highway-fast-v0` (or our fast config) | `highway-v0`: **11× slower** (measured) |
| Final evaluation | `highway-v0` + density/aggressiveness sweeps + `merge-v1` + `RampMerge-v0` + `roundabout-v1` | the training env only |
| Merge experiments | `merge-v1` (ego yields to a merger) **and** our `RampMerge-v0` (ego merges) | `merge-v0`: deprecated in 1.12 |
| Quick baseline in ≤ 5e4 steps, 1 core | DQN / our Double-Dueling DQN | PPO (sample-hungry) |
| Many cores, stable curves | PPO with 8–12 `SubprocVecEnv` | `DummyVecEnv` for expensive envs |
| Dense traffic (density ≥ 1.5) | Ego-attention extractor, V=15 | MLP with V=5 (can't see enough cars) |
| Safety budget required | PPO-Lagrangian | tuning crash weight by hand |
| Showing "the RL agent, not a rule, drives" | report shield-intervention rate; train without the shield | a shield-only result |
| Partial observability / intent | frame-stack 3 first; RecurrentPPO if that helps | LSTM by default |
| Risk-averse behaviour knob | QR-DQN + CVaR_α | changing γ |
| Comparing methods | IQM + 95% CI over ≥5 seeds | mean of 1 seed |
| Logging | TensorBoard always; W&B for team sharing | print statements |
| Image observations / CNN | GPU (`device="cuda"`) | CPU |
| MLP / attention PPO | CPU (`device="cpu"`, `torch.set_num_threads(1)` per worker) | GPU (transfer overhead) |

---

## 7. System architecture & repository layout

```
                ┌────────────────────────── configs/*.yaml (env, algo, reward, seeds) ─────────────────────┐
                ▼                                                                                          │
 ┌───────────────────────┐   obs, info    ┌─────────────────────────────┐   action   ┌──────────────────┐   │
 │  highway-env scenarios │──────────────▶│  Wrappers (ours)            │──────────▶│  Agent (SB3)     │   │
 │  highway / merge-v1 /  │◀──────────────│  • RewardShaping (+Φ)       │◀──────────│  DQN · D3QN(ours)│   │
 │  RampMerge (ours) /    │   action      │  • SafetyCost (TTC, headway)│  obs, r, c │  PPO · PPO-Lag   │   │
 │  roundabout / ...      │               │  • DomainRandomizer         │            │  MaskablePPO     │   │
 └───────────────────────┘               │  • ObsNoise / Dropout       │            │  + EgoAttention  │   │
          ▲                               │  • ActionMask (ablation)    │            └────────┬─────────┘   │
          │ config per episode            └─────────────────────────────┘                     │ checkpoints │
 ┌────────┴──────────┐                                                                ┌────────▼─────────┐ │
 │ Curriculum (cb)   │◀────────────── success rate ───────────────────────────────────│ Callbacks        │ │
 └───────────────────┘                                                                │ Eval · λ-update  │ │
                                                                                      │ metrics → TB/W&B │ │
 ┌────────────────────────────────────────────────────────────────┐                   └──────────────────┘ │
 │ Evaluation harness: seeds × scenarios × episodes → parquet →   │◀───── models/*.zip ─────────────────────┘
 │ rliable IQM/CIs · TTC/headway/jerk/RSS · heat-maps · videos    │
 └────────────────────────────────────────────────────────────────┘
```

```
safelane-rl/
├── README.md
├── pyproject.toml / uv.lock          # pinned deps
├── Makefile                          # make install | train-all | eval | figures | reproduce | test
├── configs/
│   ├── env/{highway_fast,highway_eval,merge,ramp_merge,roundabout,dr}.yaml
│   ├── algo/{dqn,d3qn,ppo,ppo_attn,ppo_lag,maskable_ppo,qrdqn}.yaml
│   └── experiment/{baselines,safety,generalisation,robustness,pareto}.yaml
├── safelane/
│   ├── envs/
│   │   ├── ramp_merge.py             # RampMergeEnv(MergeEnv): ego on ramp + is_success
│   │   └── registration.py           # gymnasium.register("RampMerge-v0", ...)
│   ├── wrappers/
│   │   ├── reward_shaping.py         # comfort, lane-change, potential-based headway
│   │   ├── safety_cost.py            # TTC / headway cost → info["cost"]
│   │   ├── domain_randomization.py   # density, NPC type, IDM params per episode
│   │   ├── obs_noise.py              # noise, dropout, delay
│   │   └── action_mask.py            # legal + TTC-safe actions (ablation)
│   ├── agents/
│   │   ├── d3qn.py                   # Double + Dueling DQN on top of SB3 DQN
│   │   ├── ppo_lagrangian.py         # VecEnvWrapper r−λc + PID λ-update callback
│   │   └── extractors.py             # EgoAttentionExtractor (BaseFeaturesExtractor)
│   ├── baselines/{random_policy,constant_speed,idm_mobil}.py
│   ├── callbacks/{metrics,curriculum}.py
│   ├── eval/{runner,metrics,rss,stats,plots}.py
│   └── xai/{attention_overlay,integrated_gradients}.py
├── scripts/{train.py,evaluate.py,tune.py,record_video.py,make_figures.py}
├── app/demo.py                       # Gradio
├── models/                           # final trained agents (committed)
├── results/                          # parquet + csv tables (committed)
├── figures/                          # report & README figures
├── videos/                           # GIFs for README
├── tests/                            # env shapes, TTC, masks never empty, eval determinism
└── report/                           # Overleaf two-column LaTeX source + PDF
```

---

## 8. Installation (verified)

Tested on 21 Oct 2026: Ubuntu, Python 3.12 via `uv`, highway-env 1.12.1, SB3 2.9.0, gymnasium 1.4.0, torch 2.14.1 (CPU).

```bash
# 1. Python 3.12 environment (system Python 3.14 also works for the core stack; 3.12 is safest)
curl -LsSf https://astral.sh/uv/install.sh | sh        # if uv is missing
uv venv --python 3.12 .venv && source .venv/bin/activate

# 2. PyTorch — CPU wheel is enough (small nets); use the CUDA wheel only for CNN experiments
uv pip install --index-url https://download.pytorch.org/whl/cpu torch

# 3. Core stack
uv pip install "highway-env==1.12.1" "stable-baselines3[extra]==2.9.0" "sb3-contrib==2.9.0" \
               "gymnasium==1.4.0" tensorboard rliable optuna pandas pyarrow seaborn moviepy

# 4. Optional extras
uv pip install wandb captum gradio hydra-core            # tracking, XAI, demo, configs
# uv pip install omnisafe                                 # only to cross-check PPO-Lag

# 5. Headless servers / video
sudo apt-get install -y ffmpeg                            # video encoding
# no display needed: highway-env >=1.12 renders offscreen with render_mode="rgb_array".
# Do NOT set SDL_VIDEODRIVER=dummy - with pygame-ce it produces all-black frames (verified).
```

> ⚠️ Don't install `mo-gymnasium` into the same env without checking: it depends on `pygame`, while
> highway-env ≥1.12 depends on `pygame-ce`. Both provide the `pygame` module and can clash.
> Use a separate venv for the MO-Gymnasium experiment, or implement the weight sweep yourself (easy).

Sanity check:
```bash
python -c "import gymnasium as gym, highway_env; e=gym.make('highway-fast-v0'); print(e.reset(seed=0)[0].shape)"
# -> (5, 5)
```

---

## 9. Usage — planned commands

(Interface we will implement; kept here so the team builds to one spec.)

```bash
# Train
python scripts/train.py algo=dqn          env=highway_fast seed=0 steps=50000
python scripts/train.py algo=d3qn         env=highway_fast seed=0 steps=50000
python scripts/train.py algo=ppo          env=highway_fast seed=0 steps=200000 n_envs=8
python scripts/train.py algo=ppo_attn     env=dr           seed=0 steps=300000 n_envs=8
python scripts/train.py algo=ppo_lag      env=dr           seed=0 cost_limit=0.5
python scripts/train.py algo=ppo          env=ramp_merge   seed=0 curriculum=true

# All seeds of an experiment group (Hydra multirun)
python scripts/train.py -m experiment=baselines seed=0,1,2,3,4

# Tune
python scripts/tune.py algo=ppo env=highway_fast n_trials=40

# Evaluate (100 episodes × fixed eval seeds per scenario) → results/*.parquet
python scripts/evaluate.py model=models/ppo_lag_s0.zip scenarios=all episodes=100

# Figures, tables, videos
python scripts/make_figures.py      # learning curves, IQM/CI, heat-map, Pareto, robustness
python scripts/record_video.py model=models/ppo_attn_s0.zip env=ramp_merge attention_overlay=true

# Everything from scratch
make reproduce

# Demo
python app/demo.py                  # http://localhost:7860

# Monitor
tensorboard --logdir runs/
```

---

## 10. Evaluation protocol & metrics

### 10.1 Protocol
- **Seeds:** 5 training seeds per method (10 if cheap). Eval seeds 1000–1099 are fixed and **never** used in training.
- **Episodes:** 100 per (model × scenario), `deterministic=True`.
- **Aggregation:** IQM with 95% stratified-bootstrap CI (rliable). Also performance profiles and P(method A > method B).
- **Fairness:** identical step budgets per comparison; hyperparameters tuned with the same Optuna budget per method.

### 10.2 Metrics

| Metric | Definition | From |
|---|---|---|
| **Collision rate** | episodes ending with `info["crashed"]` / episodes | info |
| **Average speed** | mean of `info["speed"]` over steps (m/s and km/h) | info |
| **Merge success rate** | RampMerge: `info["is_success"]` (merged into a main lane, passed x > 370 m, no crash). merge-v1: no crash and passed x > 370 m. | our env |
| **Cumulative reward** | Σ r_t per episode (report the default env reward, so it is comparable across methods) | env |
| Time-to-collision | per step to the lead vehicle; report 5th percentile and % steps with TTC < 1.5 s | wrapper |
| Min time headway | $h=\Delta x/v_{ego}$; % steps < 1 s (the "2-second rule" violation rate) | wrapper |
| Near-miss rate | episodes with TTC < 1 s and no crash | wrapper |
| Comfort | RMS jerk $j=\Delta a/\Delta t$, RMS acceleration | wrapper |
| Lane changes / km | lane-index changes per km | wrapper |
| RSS violation % | steps closer than the RSS safe distance | `eval/rss.py` |
| Constraint satisfaction | $\hat J_c$ vs budget *d*, both during and after training | PPO-Lag |
| Shield intervention % | steps where the mask removed the policy's preferred action | ablation |
| Distance travelled / episode | progress | env |

### 10.3 Baselines (all evaluated on the same seeds)
1. **Random** policy (lower bound)
2. **Constant speed** (always `IDLE`)
3. **IDM + MOBIL** rule-based driver (the "classic AV stack" reference)
4. **Vanilla DQN** and **vanilla PPO** (official hyperparameters)

### 10.4 Scenario matrix ("diverse traffic scenarios")

| Axis | Values |
|---|---|
| Road | highway (3/4 lanes), merge-v1, RampMerge, roundabout-v1, (exit-v1, intersection-v2 for OOD) |
| Density | 0.5, 1.0, 1.5, 2.0, 2.5 (eval also 3.0 = out-of-distribution) |
| Driver type | IDM, Defensive, Aggressive, mixed |
| Sensor | clean, σ = 0.05 / 0.1 / 0.2 noise, 10% / 30% dropout |

---

## 11. Measured numbers so far & expected results

### 11.1 Measured on our machine (21 Oct 2026; Ryzen-class 16-thread laptop CPU, RTX 4070 laptop GPU unused)

| What | Result |
|---|---|
| Raw env speed, random policy, 1 core | highway-v0 **6.9** · highway-fast-v0 **76** · merge-v0 **113** · roundabout-v0 **64** · intersection-v0 **37** steps/s |
| SB3 DQN training throughput (1 env, CPU) | ~**70** steps/s → 2e4 steps ≈ 5 min, 1e5 ≈ 25 min |
| SB3 PPO training throughput (8 × SubprocVecEnv) | ~**288** steps/s → 2e5 steps ≈ 12 min |

Baselines on `highway-fast-v0`, 30 eval episodes (seeds 1000–1029), 1 training seed. A quick sanity run, not the final protocol:

| Policy | Collision rate | Avg speed (m/s) | Mean return |
|---|---|---|---|
| Random | **100%** | 23.8 | 8.1 |
| Always `IDLE` (constant speed) | 93% | 24.4 | 12.4 |
| Vanilla DQN, official hyperparameters, 2e4 steps (309 s training) | **33%** | 24.5 | 20.1 |
| Same DQN, zero-shot on `merge-v1` | 53% | 25.7 | 10.3 |

Takeaways: (1) 5 minutes of DQN already beats the trivial baselines, but **1 in 3 episodes still crashes**, so
there is a lot of headroom for D3QN, attention, PPO-Lag and longer training to show gains. (2) Average speed
barely moves (the agent hasn't learned to overtake yet). (3) **Zero-shot transfer to merging is poor**, which
motivates the generalisation experiments (E6) and domain randomisation.

**Compute budget estimate:** 7 methods × 5 seeds × ~2e5 steps ≈ 7e6 steps ≈ 7 hours of PPO-style training on
this laptop (run overnight). Evaluation of 100 episodes × ~10 scenarios × 35 models ≈ 2–3 hours.

### 11.2 Targets (from literature + our runs; for orientation only)

| Method | Collision rate (highway, density 1) | Notes |
|---|---|---|
| Random | 100% | measured (§11.1) |
| Vanilla DQN, 2e4 steps | 33% | measured (§11.1); the official docs show learning curves only, no table |
| Literature: Lagrangian SAC-Discrete + MPC action shield, **ego on-ramp merge built on highway-env** | 99.0–99.5% success, 0.3–0.5% collisions | Li et al. 2025, Table IV (arXiv 2503.02624). Closest prior work to our RampMerge: cite it, and compare against it qualitatively. |
| **Our target (PPO-Lag + attention, ≥ 2e5 steps)** | < 5% collisions at density 1, with avg speed ≥ IDM baseline | stretch: < 2% at density 1, < 10% at density 2.5 |

No standard PPO-vs-DQN benchmark table exists for highway-env, so **our Table 1 is itself a contribution**.

---

## 12. Experiments matrix

| ID | Question | Compare | Figure / table in report |
|---|---|---|---|
| E1 | Does RL beat classic baselines? | Random, IDLE, IDM+MOBIL, DQN, PPO | **Table 1**: 4 core metrics, IQM ± CI |
| E2 | Do our DQN improvements help? | DQN vs Double vs Dueling vs D3QN (+n-step) | learning curves with shaded CI |
| E3 | Does attention help in dense traffic? | MLP (V=5), MLP (V=15), Ego-attention | collision rate vs density (line plot) |
| E4 | Constraint vs penalty | PPO, PPO + TTC penalty, **PPO-Lag** (d ∈ {0.25, 0.5, 1}) | return vs cost curves; λ over training |
| E5 | Merging ability | all methods on merge-v1 and **RampMerge** | merge success bar chart |
| E6 | Generalisation | train {highway, merge, DR} × test {6 scenarios × densities} | **heat-map** |
| E7 | Robustness | noise σ, dropout p, delay | degradation curves |
| E8 | Curriculum & DR | none vs curriculum vs DR | sample-efficiency curves |
| E9 | Shield ablation | MaskablePPO with and without the TTC mask | collisions + intervention % |
| E10 | Speed–safety trade-off | reward-weight / budget sweep | **Pareto front** + hypervolume |
| E11 | Explainability | attention maps, IG | qualitative figure + video frames |
| E12 | Formal safety | RSS violation %, blame analysis | small table |

The 2-page limit means the report shows Table 1 plus 3–4 figures (E3/E4/E6/E10 are the most striking).
Everything else goes in the README with links.

---

## 13. Timeline, team split & commit plan

The deadline is not stated in the PDF; this assumes **4 weeks** from 21 Oct 2026. Compress Weeks 3–4 if needed.

| Week | Milestone (git tag) | Deliverables |
|---|---|---|
| 1 (21–27 Oct) | `v0.1-baselines` | repo skeleton, pinned env, RampMerge env + tests, eval harness + metrics, baselines (random/IDLE/IDM), vanilla DQN & PPO, first videos |
| 2 (28 Oct–3 Nov) | `v0.2-improved` | D3QN, ego-attention extractor, DR + curriculum, Optuna sweep, W&B dashboard |
| 3 (4–10 Nov) | `v0.3-safety` | PPO-Lagrangian, safety cost wrapper, mask ablation, RSS checker, 5-seed runs launched overnight |
| 4 (11–17 Nov) | `v1.0-final` | generalisation + robustness + Pareto experiments, rliable figures, XAI overlays, Gradio demo, final README, 2-page report |

**Team split (adapt to team size):**

| Role | Owns |
|---|---|
| A — Environments | `envs/`, `wrappers/` (RampMerge, DR, noise, safety cost), scenario configs |
| B — Agents | D3QN, ego-attention extractor, PPO-Lag, MaskablePPO, tuning |
| C — Evaluation | eval harness, metrics, RSS, rliable stats, all figures, report writing |
| D (if present) — XAI & Demo | attention overlay, IG, videos, Gradio app, CI |

**Commit hygiene (graders read history):** one feature branch per task → PR → teammate review → merge.
Conventional commits (`feat(env): add RampMerge success flag`, `fix(eval): seed eval envs`). Small, frequent
commits from **every** member. Never squash the whole project into one commit, and never backdate or fake authorship;
forged history is easy to detect and counts as misconduct.

---

## 14. Gotchas we verified

All checked against highway-env 1.12.1 source or by running it.

1. **`import highway_env` is required** before `gym.make(...)`, otherwise you get `NameNotFound`. With `SubprocVecEnv`, every worker must import it (use `gym.register_envs(highway_env)` or import inside the env factory). Guard scripts with `if __name__ == "__main__":`.
2. **`merge-v0`, `roundabout-v0`, `exit-v0`, `intersection-v0/v1` print "out of date — upgrade"** in 1.12 (a fix to neighbour-vehicle detection across connected lanes). Use **`merge-v1`, `roundabout-v1`, `exit-v1`, `intersection-v2`**. `highway-v0` / `highway-fast-v0` are current.
3. **In `merge-v0/v1` the ego is on the main road**; the merging car is an NPC, and the "merging_speed_reward" is an *altruistic* penalty for slowing it. There is **no ego-merge task and no success flag**, hence our `RampMerge-v0`.
4. **Merge envs never truncate** (`_is_truncated` returns False). That is fine with default target speeds ≥ 20 m/s, but if you allow speed 0 (e.g. `target_speeds` starting at 0) the episode can last forever. Wrap with `gymnasium.wrappers.TimeLimit`.
5. **`highway-v0` is ~11× slower than `highway-fast-v0`** (6.9 vs 76 steps/s, measured): train fast, evaluate full.
6. **`lane_change_reward` is ignored in `highway-v0`**: it's in the config but not in `_rewards()`. Add it in a wrapper if you want it.
7. **`normalize_reward=True`** maps rewards to [0, 1], so a crash gives reward 0, not negative. The real crash penalty is losing all future reward through termination. Keep this in mind when designing shaping (and say it in the report).
8. **highway-fast-v0 disables NPC–NPC collision checks**: fine for training, but evaluate on `highway-v0`.
9. **SB3 DQN is vanilla**: no Double, Dueling or PER, contrary to what many blogs assume.
10. **SB3 `net_arch=[dict(pi=..., vf=...)]`** (old list form, still in the official highway PPO script) is deprecated. Use `net_arch=dict(pi=[256,256], vf=[256,256])`.
11. **γ = 0.8** in the official scripts is only a 5 s horizon at 1 Hz; merges benefit from 0.9–0.95.
12. **pygame vs pygame-ce** conflict with `mo-gymnasium` (see §8).
13. **Rendering changed in 1.12:** `render_mode="rgb_array"` is headless by default, and the `OFFSCREEN_RENDERING` env var is ignored. For smooth videos, set `simulation_frequency=15` and call `env.unwrapped.set_record_video_wrapper(video_env)`.
14. **Change config only before `reset()`**: `env.unwrapped.config.update({...}); env.reset()`, or `env.reset(options={"config": {...}})`.
15. **MaskablePPO** needs an `action_masks()` method on the env when using `SubprocVecEnv`, and the maskable `evaluate_policy` / `MaskableEvalCallback`.
16. **RecurrentPPO inference** needs `state` and `episode_start` passed to `predict()`, or results will be silently wrong.

---

## 15. Plagiarism-safe rules

The PDF promises automated similarity checks plus manual inspection, and a zero for copying.

- ✅ **Allowed:** `pip install` libraries (highway-env, SB3, sb3-contrib, rliable, Optuna) and calling their APIs; reading docs and papers; citing them.
- ❌ **Never:** fork or clone highway-env's `scripts/`, rl-agents, CleanRL files, or GitHub "highway-dqn" repos. Don't paste the official `sb3_highway_*.py` examples, even modified; write our own `train.py` around our config system.
- Write every custom part ourselves: RampMerge env, wrappers, D3QN, ego-attention extractor, PPO-Lagrangian, metrics, RSS, plots. These are also exactly what earns the "unique effort" bonus.
- Hyperparameters taken from the official examples are facts, not code; cite their source in the README and report.
- Keep design notes in issues/PRs. Real development history is the best defence in a similarity review.

---

## 16. 2-page report skeleton

Two-column Overleaf (IEEEtran `conference` class). GitHub link + team at the top.

1. **Title, team, GitHub link**
2. **Problem & motivation** (3–4 lines): highway merging and lane changing are a leading source of crashes; mapping them to a CMDP.
3. **RL formulation**: S (§4.2, with dims), A (§4.3), R (§4.4 equation), cost and constraint (§4.5), γ, dynamics (IDM/MOBIL equation).
4. **Environment & algorithms**: scenarios, custom RampMerge, DQN/D3QN/PPO/PPO-Lag/attention. One architecture sketch.
5. **Experimental setup**: seeds, budgets, eval protocol, baselines.
6. **Results**: Table 1 (E1), Fig. 1 learning curves (E2/E4), Fig. 2 density curve or generalisation heat-map (E3/E6), Fig. 3 Pareto front (E10), plus one line each on robustness and RSS.
7. **Discussion & limitations**: sim-to-real gap, perfect perception, IDM traffic realism, shield caveat.
8. **References** (do not count toward the 2 pages).

---

## 17. References

**Simulator & libraries**
- highway-env docs — https://highway-env.farama.org/ · source — https://github.com/Farama-Foundation/HighwayEnv · PyPI — https://pypi.org/project/highway-env/
- highway-env neighbour-vehicle fix (v1 envs) — https://highway-env.farama.org/main/dynamics/road/neighbour_vehicles/
- Stable-Baselines3 — https://stable-baselines3.readthedocs.io/ (changelog, DQN, PPO, custom policy, VecEnv, RL tips)
- sb3-contrib — https://sb3-contrib.readthedocs.io/
- rliable — https://github.com/google-research/rliable · Optuna SB3 example — https://github.com/optuna/optuna-examples/blob/main/rl/sb3_simple.py
- OmniSafe — https://github.com/PKU-Alignment/omnisafe · Safety-Gymnasium — https://arxiv.org/abs/2310.12567
- MO-Gymnasium — https://mo-gymnasium.farama.org/ · MetaDrive — https://github.com/metadriverse/metadrive

**Core RL**
- Mnih et al. 2015, DQN — https://www.nature.com/articles/nature14236
- van Hasselt et al. 2016, Double DQN — https://arxiv.org/abs/1509.06461
- Wang et al. 2016, Dueling — https://arxiv.org/abs/1511.06581
- Schaul et al. 2016, PER — https://arxiv.org/abs/1511.05952
- Hessel et al. 2018, Rainbow — https://arxiv.org/abs/1710.02298
- Dabney et al. 2018, QR-DQN — https://arxiv.org/abs/1710.10044
- Schulman et al. 2017, PPO — https://arxiv.org/abs/1707.06347
- Andrychowicz et al. 2021, What matters in on-policy RL — https://arxiv.org/abs/2006.05990
- Huang & Ontañón 2022, Invalid action masking — https://arxiv.org/abs/2006.14171
- Ng, Harada & Russell 1999, Potential-based reward shaping — ICML 1999
- Agarwal et al. 2021, Statistical precipice — https://arxiv.org/abs/2108.13264

**Safe / robust RL**
- Altman 1999, Constrained MDPs
- Achiam et al. 2017, CPO — https://arxiv.org/abs/1705.10528
- Tessler et al. 2019, RCPO — https://arxiv.org/abs/1805.11074
- Stooke et al. 2020, PID Lagrangian — https://arxiv.org/abs/2007.03964
- Alshiekh et al. 2018, Safe RL via Shielding — https://arxiv.org/abs/1708.08611
- Shalev-Shwartz et al. 2017, RSS — https://arxiv.org/abs/1708.06374
- Pinto et al. 2017, RARL — https://arxiv.org/abs/1703.02702
- Tobin et al. 2017, Domain randomisation — https://arxiv.org/abs/1703.06907
- Narvekar et al. 2020, Curriculum learning survey — https://arxiv.org/abs/2003.04960

**Driving-specific**
- Treiber et al. 2000, IDM · Kesting et al. 2007, MOBIL
- Leurent & Mercat 2019, Social Attention for AD in Dense Traffic — https://arxiv.org/abs/1911.12250
- Li et al. 2025, Human-aligned Safe RL for Highway On-Ramp Merging (highway-env) — https://arxiv.org/abs/2503.02624
- Han et al. 2024, LLM-generated rewards for highway driving (highway-env) — https://arxiv.org/abs/2406.10540
- Abouelazm et al. 2025, Automatic curriculum for driving — https://arxiv.org/abs/2505.08264
- Chen et al. 2021, MARL on-ramp merging with action masking — https://arxiv.org/abs/2105.05701
- Hoel et al. 2020, Tactical decisions with uncertainty estimation — https://arxiv.org/abs/2004.10439
- Lu et al. 2022, Waymo BC-SAC (imitation + RL in hard scenarios) — https://arxiv.org/abs/2212.11419
- Kendall et al. 2018, Learning to Drive in a Day (Wayve) — https://arxiv.org/abs/1807.00412
