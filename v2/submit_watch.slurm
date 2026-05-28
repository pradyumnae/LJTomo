#!/bin/bash
#SBATCH -A als
#SBATCH -J watch_pca
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32

module load gpu
export IMAGE="nersc/pytorch:25.02.01"
export PYTHONUSERBASE=$PSCRATCH/lejepa_tomography/env
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.12/site-packages:$PYTHONPATH

srun shifter --image=$IMAGE python watch_checkpoints.py
