#!/bin/bash
#SBATCH --job-name=topk0
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-10:00:00
#SBATCH --output=/om2/user/ericjm/structured-sae/experiments/topk_baselines0/slurm-%A.out

python /om2/user/ericjm/structured-sae/experiments/topk_baselines0/train.py
