#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH --nodelist=steropes
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=6 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH -t 0-12:00 # time requested (D-HH:MM)
pwd
hostname
date
echo starting job...


cd /home/eecs/yefan0726/zihang/TempBalance/language_modeling/BTD-Transformer
python slurm_test.py