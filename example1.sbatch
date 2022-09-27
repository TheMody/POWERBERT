#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --job-name=POWERBERT
#SBATCH --output=POWERBERT.log
#SBATCH --time=96:00:00

module load cuda/10.1

prefix=${HOME}/POWERBERT
srun ${prefix}/slurm_example.sh config.json ${prefix}
