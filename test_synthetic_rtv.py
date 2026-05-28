import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
import h5py
import numpy as np
import tomopy
import dxchange
from skimage import transform

def reconstruct_synthetic(path, out_path):
    print(f"Loading {path}...")
    # Read just a few slices to be fast
    # (angles, slices, rays)
    tomo, flat, dark, angles = dxchange.exchange.read_aps_tomoscan_hdf5(path, proj=slice(0,None), sino=slice(600, 610))
    tomo = tomo.astype('float32')
    flat = flat.astype('float32')
    if dark is not None:
        dark = dark.astype('float32')
    
    # Check if flat is placeholder
    if np.mean(flat) < 2.0:
        print("Placeholder flat detected. Using max(tomo) as synthetic flat.")
        flat = np.full_like(flat, np.max(tomo))
    
    # Preprocess
    tomopy.normalize(tomo, flat, dark, out=tomo)
    tomopy.minus_log(tomo, out=tomo)
    
    # Downsample rays 2x (sino is already small since we read just 10 slices)
    tomo = np.asarray([transform.downscale_local_mean(proj, (1, 2), cval=0).astype(proj.dtype) for proj in tomo])
    
    # COR
    cor = tomopy.find_center_pc(tomo[0], tomo[-1])
    print(f"COR: {cor}")
    
    # Recon
    print("Reconstructing...")
    recon = tomopy.recon(tomo, angles, center=cor, algorithm='astra_fbp', options={'proj_type': 'cuda', 'method': 'FBP'})
    
    # Save a slice
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    vmin, vmax = np.percentile(recon[5], [1, 99])
    plt.imshow(recon[5], cmap='gray', vmin=vmin, vmax=vmax)
    plt.title("Synthetic Flat Recon")
    plt.colorbar()
    plt.savefig("verification_images/synthetic_rtv_test.png")
    print("Saved to verification_images/synthetic_rtv_test.png")

if __name__ == "__main__":
    reconstruct_synthetic('/global/cfs/cdirs/als/data_mover/share/alsdata/_als-12717_bessire/20251029_014438_RTV_MB_7A_x00y02.h5', 'test.h5')
