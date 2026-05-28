import sys
import os
import subprocess

def run_chunk(start_offset, gpu_relative_idx, count):
    # Load the file list
    with open("file_list.txt", "r") as f:
        all_files = [line.strip() for line in f.readlines()]
    
    # Calculate which files this specific GPU should handle
    # Each of the 8 GPUs handles 'count' files starting from start_offset
    my_start = start_offset + (gpu_relative_idx * count)
    my_files = all_files[my_start : my_start + count]
    
    print(f"GPU {gpu_relative_idx} starting burst for {len(my_files)} files (indices {my_start} to {my_start + len(my_files) - 1})")
    
    for f in my_files:
        print(f"--- GPU {gpu_relative_idx} processing {os.path.basename(f)} ---")
        # Note: CUDA_VISIBLE_DEVICES is handled by srun/slurm for this process
        subprocess.run(["python3", "batch_reconstruct.py", f])

if __name__ == "__main__":
    offset = int(sys.argv[1])
    gpu_idx = int(sys.argv[2])
    count = int(sys.argv[3])
    run_chunk(offset, gpu_idx, count)
