import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def check_one(h5_path, output_png):
    with h5py.File(h5_path, 'r') as f:
        data = f['reconstruction']
        mid = data.shape[0] // 2
        slice_data = data[mid]
        
        # Apply percentile clipping for contrast
        v_min, v_max = np.percentile(slice_data, [1, 99])
        
        plt.figure(figsize=(10, 10))
        plt.imshow(slice_data, cmap='gray', vmin=v_min, vmax=v_max)
        plt.title(f"Verification: {os.path.basename(h5_path)}")
        plt.colorbar()
        plt.savefig(output_png)
        plt.close()
        print(f"Saved {output_png}")

if __name__ == "__main__":
    check_one(sys.argv[1], sys.argv[2])
