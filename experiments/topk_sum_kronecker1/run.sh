#!/bin/bash
#SBATCH --job-name=sk1
#SBATCH --ntasks=1
#SBATCH --mem=24GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-24:00:00
#SBATCH --output=/om2/user/ericjm/structured-sae/experiments/topk_sum_kronecker1/logs/slurm-%A_%a.out
#SBATCH --array=0-8

python /om2/user/ericjm/structured-sae/experiments/topk_sum_kronecker1/train.py $SLURM_ARRAY_TASK_ID
