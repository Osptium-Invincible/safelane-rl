PY ?= python
WORKERS ?= 8

.PHONY: install test smoke train eval figures videos explain demo reproduce clean-cache check

install:            ## create .venv (Python 3.12) and install everything (CPU torch)
	uv venv --python 3.12 .venv
	. .venv/bin/activate && uv pip install --index-url https://download.pytorch.org/whl/cpu torch && uv pip install -e ".[demo,dev]"

test:               ## unit tests (~10 s)
	$(PY) -m pytest -q

smoke:              ## 1-minute end-to-end check: tiny training run + evaluation
	$(PY) scripts/train.py env=ramp_merge algo=ppo_lag algo.total_timesteps=2048 algo.n_envs=2 run_dir=runs/_smoke/seed0
	$(PY) scripts/record_videos.py --run runs/_smoke/seed0 --scenario ramp_merge --out videos/_smoke.gif

train:              ## all 46 training runs of configs/experiment/main.yaml (resumable)
	$(PY) scripts/run_experiments.py configs/experiment/main.yaml --heavy 1 --light 1

eval:               ## evaluate every finished run + baselines on all scenarios (resumable)
	$(PY) scripts/evaluate.py --workers $(WORKERS)

figures:            ## tables and figures from results/ and runs/
	$(PY) scripts/make_figures.py

videos:             ## showcase GIFs (with attention overlay) into videos/
	$(PY) scripts/record_videos.py

explain:            ## Integrated-Gradients + attention figures for the main models
	$(PY) scripts/explain.py --run runs/ramp_ppo_lag/seed0 --scenario ramp_merge
	$(PY) scripts/explain.py --run runs/hw_ppo_lag/seed0 --scenario highway_fast

demo:               ## Gradio app on http://localhost:7860
	$(PY) app/demo.py

reproduce: train eval figures videos explain   ## everything, from scratch

clean-cache:        ## drop cached evaluation jobs (forces re-evaluation)
	rm -rf results/eval/cache

check:              ## verify every number quoted in README.md and report/main.tex against results/
	$(PY) scripts/check_numbers.py
