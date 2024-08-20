#!/bin/bash
#SBATCH --job-name=sk0
#SBATCH --ntasks=1
#SBATCH --mem=24GB
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-20:00:00
#SBATCH --output=/om2/user/ericjm/structured-sae/experiments/topk_sum_kronecker0/logs/slurm-%A_%a.out
#SBATCH --array=0-8

python /om2/user/ericjm/structured-sae/experiments/topk_sum_kronecker0/train.py $SLURM_ARRAY_TASK_ID
