#!/bin/bash
#SBATCH --job-name=skron0
#SBATCH --ntasks=1
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00:00
#SBATCH --output=/om2/user/ericjm/structured-sae/experiments/sum_kronecker0/logs/slurm-%A_%a.out
#SBATCH --array=0-111

python /om2/user/ericjm/structured-sae/experiments/sum_kronecker0/train.py $SLURM_ARRAY_TASK_ID
