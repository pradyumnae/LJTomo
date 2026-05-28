import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

def verify_reconstruction(h5_path, output_dir):
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(h5_path, 'r') as f:
        data = f['reconstruction']
        mid_slice_idx = data.shape[0] // 2
        
        # Load a few slices safely
        slices_to_view = [max(0, mid_slice_idx - 10), mid_slice_idx, min(data.shape[0]-1, mid_slice_idx + 10)]
        
        for idx in set(slices_to_view):
            slice_data = data[idx]
            
            plt.figure(figsize=(10, 10))
            plt.imshow(slice_data, cmap='gray')
            plt.title(f"Reconstruction Slice {idx}")
            plt.colorbar()
            
            save_path = os.path.join(output_dir, f"slice_{idx}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved verification slice to {save_path}")

if __name__ == "__main__":
    data_dir = "/global/homes/e/elavarpa/pscratch/lejepa_tomography/data/lejepa_v2/"
    out_dir = "/global/homes/e/elavarpa/pscratch/lejepa_tomography/verification_images"
    
    for f in os.listdir(data_dir):
        if f.endswith(".h5"):
            print(f"Verifying {f}...")
            verify_reconstruction(os.path.join(data_dir, f), os.path.join(out_dir, f.replace(".h5", "")))
