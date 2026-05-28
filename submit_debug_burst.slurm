#!/bin/bash
#SBATCH -N 4
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -J lejepa_debug_burst
#SBATCH -A amsc006_g
#SBATCH --image=dperl/tomocam:latest
#SBATCH --output=logs/debug_burst.out

export NUMEXPR_MAX_THREADS=999

# Process a specific chunk of files (e.g., from index 100 to 180)
START_OFFSET=100
FILES_PER_GPU=10

# We have 4 nodes, 4 GPUs per node = 16 GPUs total.
# But we only use 2 GPUs per node to avoid OOM.
# Total 8 GPUs. 8 GPUs * 10 files = 80 files.

for node_gpu in {0..7}; do
    NODE_ID=$((node_gpu / 2))
    GPU_ID=$((node_gpu % 2))
    
    # Use srun to launch on specific nodes/gpus
    srun --nodes=1 --ntasks=1 --relative=$NODE_ID --gpus-per-task=1 \
        shifter --image=dperl/tomocam:latest python3 batch_reconstruct_chunk.py $START_OFFSET $node_gpu $FILES_PER_GPU &
done

wait
