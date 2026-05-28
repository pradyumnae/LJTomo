#!/bin/bash
#SBATCH -N 4
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -J lejepa_heavy
#SBATCH --output=logs/heavy_%j.out

# FILE_LIST is passed as an env var via sbatch --export
export FILE_LIST=${FILE_LIST:-file_list.txt}

module load shifter
# Match Perlmutter's physical core count and suppress warnings
export NUMEXPR_MAX_THREADS=128

# RUNNING 1 PROCESS PER NODE (256GB RAM per worker)
# We have 4 nodes. Each node will run 1 process on GPU 0.
for node_idx in {0..3}; do
    # Only using 1 GPU per node to maximize CPU RAM per reconstruction
    srun --nodes=1 --ntasks=1 --relative=$node_idx --gpus-per-task=1 --exclusive \
        shifter --image=dperl/tomocam:latest python3 batch_reconstruct_mega.py --list $FILE_LIST &
done

wait
