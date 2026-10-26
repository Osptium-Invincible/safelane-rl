#!/bin/bash
# Full unattended pipeline: train (main + long budget) -> evaluate -> figures -> videos -> explanations.
cd /home/caesar/ai_project && source .venv/bin/activate
log=runs/pipeline.log
echo "[$(date)] start" >> $log
python scripts/run_experiments.py configs/experiment/main.yaml --heavy 1 --light 1 > runs/experiments.log 2>&1
echo "[$(date)] main training done: $(grep -c ' ok ' runs/experiments.log) ok, $(grep -c FAILED runs/experiments.log) failed" >> $log
python scripts/run_experiments.py configs/experiment/extended.yaml --heavy 1 --light 1 > runs/extended.log 2>&1
echo "[$(date)] extended training done: $(grep -c ' ok ' runs/extended.log) ok, $(grep -c FAILED runs/extended.log) failed" >> $log
python scripts/evaluate.py --workers 12 > runs/evaluate.log 2>&1
echo "[$(date)] evaluation done: $(tail -1 runs/evaluate.log)" >> $log
python scripts/make_figures.py > runs/figures.log 2>&1
python scripts/record_videos.py > runs/videos.log 2>&1
for r in ext_ramp_ppo_lag ext_hw_ppo_lag; do python scripts/explain.py --run runs/$r/seed0 --scenario $( [ $r = ext_hw_ppo_lag ] && echo highway_fast || echo ramp_merge ) >> runs/explain.log 2>&1; done
python scripts/explain.py --run runs/hw_ppo/seed0 --scenario highway_fast >> runs/explain.log 2>&1
echo "[$(date)] PIPELINE FINISHED" >> $log
