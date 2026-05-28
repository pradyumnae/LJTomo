#!/bin/bash
#SBATCH -A als
#SBATCH -J vitl_dino
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --exclusive

module load gpu
export IMAGE="nersc/pytorch:25.02.01"

export PYTHONUSERBASE=$PSCRATCH/lejepa_tomography/env
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.12/site-packages:$PYTHONPATH

export NCCL_DEBUG=INFO
export NCCL_NET_GDR_LEVEL=PHB
export OMP_NUM_THREADS=4
export FI_CXI_RX_DISABLED=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun shifter --image=$IMAGE bash -c "
    torchrun \
        --nnodes=\$SLURM_JOB_NUM_NODES \
        --nproc_per_node=4 \
        --rdzv_id=\$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \
        main.py pretrained=True bs=8
"
