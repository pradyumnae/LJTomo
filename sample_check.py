import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def verify_sample(file_list, data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for f_name in file_list:
        h5_path = os.path.join(data_dir, f_name)
        print(f"Processing {f_name}...")
        try:
            with h5py.File(h5_path, 'r') as f:
                data = f['reconstruction']
                mid_idx = data.shape[0] // 2
                slice_data = data[mid_idx]
                
                plt.figure(figsize=(10, 10))
                # Use 1st/99th percentile for robust display
                vmin, vmax = np.percentile(slice_data, [1, 99])
                plt.imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax)
                plt.title(f"{f_name} - Mid Slice")
                plt.colorbar()
                
                save_path = os.path.join(out_dir, f_name.replace('.h5', '.png'))
                plt.savefig(save_path)
                plt.close()
                print(f"Saved to {save_path}")
        except Exception as e:
            print(f"Error processing {f_name}: {e}")

if __name__ == "__main__":
    files = [
        "recon_20251029_014438_RTV_MB_7A_x00y02.h5",
        "recon_20251028_184837_RTV_MB_5A_x00y03.h5",
        "recon_20230925_154748_testScan1.h5",
        "recon_20251029_180006_RTV_MB_1B_x00y02.h5",
        "recon_20251029_023535_RTV_MB_7A_x00y01.h5"
    ]
    verify_sample(files, "data/lejepa_v2/", "verification_images/sample_check")
