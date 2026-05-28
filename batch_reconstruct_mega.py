import sys
import os
import subprocess
import random

def run_mega(file_list_path):
    # Load the file list
    with open(file_list_path, "r") as f:
        all_files = [line.strip() for line in f.readlines() if line.strip()]
    
    # Shuffle to distribute load if multiple workers are on the same list (optional)
    random.shuffle(all_files)
    
    out_dir = "/global/homes/e/elavarpa/pscratch/lejepa_tomography/data/lejepa_v2/"
    
    for f_path in all_files:
        f_name = os.path.basename(f_path)
        out_path = os.path.join(out_dir, "recon_" + f_name)
        
        if os.path.exists(out_path):
            print(f"Skipping {f_name}, already exists.")
            continue
            
        print(f"--- Processing {f_name} ---")
        # batch_reconstruct.py handles the actual reconstruction
        subprocess.run(["python3", "batch_reconstruct.py", f_path])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str, default='file_list.txt', help='Path to the file list to process')
    args = parser.parse_args()
    
    run_mega(args.list)
