#!/bin/bash
#SBATCH -N 20
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -J lejepa_mega_recon
#SBATCH -A amsc006_g
#SBATCH --image=dperl/tomocam:latest
#SBATCH --output=logs/mega_%j.out

export NUMEXPR_MAX_THREADS=999

# We have 20 nodes. Each node will run 2 processes (one per GPU)
# Total 40 parallel processes.

for node_idx in {0..19}; do
    for gpu_idx in {0..1}; do
        GLOBAL_GPU_IDX=$((node_idx * 2 + gpu_idx))
        
        # Use srun to launch on specific nodes/gpus
        # --relative ensures we distribute across the 20 nodes allocated
        # --gpus-per-task=1 ensures each process gets its own GPU
        srun --nodes=1 --ntasks=1 --relative=$node_idx --gpus-per-task=1 --exclusive \
            shifter --image=dperl/tomocam:latest python3 batch_reconstruct_mega.py $GLOBAL_GPU_IDX 40 &
    done
done

wait
