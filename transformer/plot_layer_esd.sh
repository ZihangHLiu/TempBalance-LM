#!/bin/bash
#SBATCH -p rise
#SBATCH -p rise # partition (queue)
#SBATCH --nodelist=steropes
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=6 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH -t 1-24:00 # time requested (D-HH:MM)

export PYTHONUNBUFFERED=1
bash ~/.bashrc


cd /home/eecs/yefan0726/zihang/layer-wise-learning-rate-schedule-/transformer
python plot_layer_esd.py

scp -r /data/yefan0726/checkpoints/zihang/figures/training_dynamic yefan0726@watson.millennium.berkeley.edu:/data/yefan0726/checkpoints/zihang/figures/