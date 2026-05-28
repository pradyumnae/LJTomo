import os
import h5py

folders = [
    "_als-12703_xu",
    "_als-12717_bessire",
    "_als-13051_gilbert",
    "_als-13091_mccormack",
    "_als-13385_manga",
    "_als-13561_mcelrone",
    "ALS-13540_lisabeth",
    "BLS-00577_dyparkinson",
    "_als-13286_drisdell"
]

base_path = "/global/cfs/cdirs/als/data_mover/share/alsdata/"
existing_h5 = "/global/homes/e/elavarpa/pscratch/microct_sr_2d_project/data/processed/serpentinite_train.h5"

# Get existing keys to avoid duplication
with h5py.File(existing_h5, 'r') as f:
    existing_keys = set(f.keys())

files_to_process = []

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} not found.")
        continue
        
    for f in os.listdir(folder_path):
        if f.endswith(".h5"):
            full_path = os.path.join(folder_path, f)
            
            # Skip if already in the main dataset
            # (Note: keys in existing_h5 start with 'rec', so we check against filename)
            is_duplicate = False
            for key in existing_keys:
                if f.replace(".h5", "") in key:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
                
            # Skip empty files
            if os.path.getsize(full_path) < 1024 * 1024: # Less than 1MB
                continue
                
            files_to_process.append(full_path)

print(f"Total files to process: {len(files_to_process)}")

# Write to a text file for Slurm Array Job
with open("file_list.txt", "w") as out:
    for fp in files_to_process:
        out.write(f"{fp}\n")
