#!/bin/bash
#SBATCH -A amsc006
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -c 128
#SBATCH -G 4
#SBATCH -t 01:00:00
#SBATCH -J lejepa_batch_recon
#SBATCH --image=dperl/tomocam:latest
#SBATCH --output=logs/batch-%A_%a.out
#SBATCH --array=0-1443%40

export NUMEXPR_MAX_THREADS=999

# Each array task processes 2 files (one per GPU) to avoid OOM
START_IDX=$((SLURM_ARRAY_TASK_ID * 2))

for i in {0..1}; do
    FILE_IDX=$((START_IDX + i))
    # Get the file path from the list (1-indexed for sed)
    RAW_FILE=$(sed -n "$((FILE_IDX + 1))p" file_list.txt)
    
    if [ ! -z "$RAW_FILE" ]; then
        echo "GPU $i: Starting reconstruction for $RAW_FILE (Index $FILE_IDX)"
        # Run reconstruction on a specific GPU
        CUDA_VISIBLE_DEVICES=$i shifter --image=dperl/tomocam:latest python3 batch_reconstruct.py "$RAW_FILE" &
    fi
done

# Wait for all 4 background processes to finish
wait

echo "Array task $SLURM_ARRAY_TASK_ID completed."
