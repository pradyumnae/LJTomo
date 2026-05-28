#!/bin/bash
#SBATCH -A amsc006
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -c 64
#SBATCH -G 4
#SBATCH -t 01:00:00
#SBATCH -J lejepa_recon
#SBATCH --image=dperl/tomocam:latest
#SBATCH --output=logs/slurm-%j.out

export NUMEXPR_MAX_THREADS=999

# The raw file to reconstruct
RAW_FILE=$1

if [ -z "$RAW_FILE" ]; then
    echo "Error: No raw file provided."
    echo "Usage: sbatch submit_recon.slurm <path_to_raw_h5>"
    exit 1
fi

echo "Starting reconstruction for $RAW_FILE"

# Run the python script using shifter with the tomocam image
shifter --image=dperl/tomocam:latest python3 batch_reconstruct.py "$RAW_FILE"

echo "Job completed!"
