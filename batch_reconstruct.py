import sys
import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
import h5py
import numpy as np
import tomopy
import dxchange

# Add the recon toolkit path
sys.path.append('/global/cfs/cdirs/als/users/parkinson/als_microct-recon')
import ALS_recon_functions as als

def custom_read_metadata(path, print_flag=True):
    """
    Robust version of Parkinson's read_metadata to handle varying HDF5 structures.
    """
    def safe_get(p, key, idx=0):
        try:
            val = dxchange.read_hdf5(p, key)
            if val is None or len(val) == 0:
                return 0.0
            return val[idx] if len(val) > idx else val[0]
        except Exception:
            return 0.0

    numslices = int(safe_get(path, "/measurement/instrument/detector/dimension_y"))
    numrays = int(safe_get(path, "/measurement/instrument/detector/dimension_x"))
    pxsize = safe_get(path, "/measurement/instrument/detector/pixel_size") / 10.0
    numangles = int(safe_get(path, "/process/acquisition/rotation/num_angles"))
    
    # Try index 1 first as in original, fallback to 0
    propagation_dist = safe_get(path, "/measurement/instrument/camera_motor_stack/setup/camera_distance", idx=1)
    
    # Check for energy
    energy_raw = dxchange.read_hdf5(path, "/measurement/instrument/monochromator/energy")
    if energy_raw is not None and len(energy_raw) > 0 and not np.isinf(energy_raw[0]):
        kev = energy_raw[0] / 1000.0
    else:
        kev = 30.0 # Default fallback for white light
        
    angularrange = safe_get(path, "/process/acquisition/rotation/range")
    
    filename = os.path.split(path)[-1]
    if print_flag:
        print(f'{filename}:')
        print(f'numslices: {numslices}, rays: {numrays}, numangles: {numangles}')
        print(f'angularrange: {angularrange}, pxsize: {pxsize*10000:.2f} um, distance: {propagation_dist:.2f} mm. energy: {kev:.2f} keV')
        
    return {'numslices': numslices,
            'numrays': numrays,
            'pxsize': pxsize,
            'numangles': numangles,
            'propagation_dist': propagation_dist,
            'kev': kev,
            'angularrange': angularrange}

# CRITICAL: Monkeypatch the internal reference in the als module
# so that als.auto_find_cor() uses our robust version!
als.read_metadata = custom_read_metadata

# We do not monkeypatch als.read_data here because we need to intercept the data before normalization.

def reconstruct_file(filepath, out_dir):
    filename = os.path.basename(filepath)
    out_path = os.path.join(out_dir, f"recon_{filename}")
    
    if os.path.exists(out_path):
        print(f"Skipping {filename}, already reconstructed.")
        return
        
    print(f"--- Starting reconstruction for {filename} ---")
    
    # Custom COR detection to handle NaNs
    def custom_auto_find_cor(path):
        metadata = custom_read_metadata(path, print_flag=False)
        # Read first and last projections
        tomo, _, _, _ = dxchange.exchange.read_aps_tomoscan_hdf5(path, proj=slice(0, None, metadata['numangles']-1), dtype=np.float32)
        
        # Sanitize projections for COR detection
        tomo = np.nan_to_num(tomo, nan=0.0, posinf=0.0, neginf=0.0)
        
        if tomo.shape[0] < 2:
            raise IndexError("Not enough projections for COR detection.")
            
        cor = tomopy.find_center_pc(tomo[0], tomo[-1], tol=0.25)
        # The ALS functions expect COR relative to center
        cor = cor - tomo.shape[2]/2
        return cor

    try:
        # 1. Read metadata
        metadata = custom_read_metadata(filepath, print_flag=True)
        
        # 2. Determine COR automatically
        print("Calculating Center of Rotation (COR)...")
        cor = custom_auto_find_cor(filepath)
        print(f"Calculated COR: {cor}")
        
        # 3. Read raw data and apply pre/post log processing (ring removal)
        preprocessing_args = {
            "minimum_transmission": 0.01,
            "snr": 3.0,
            "la_size": 1,
            "sm_size": 11,
            "outlier_diff_1D": 750,
            "outlier_size_1D": 3
        }
        postprocessing_args = {
            "ringSigma": 3.0,
            "ringLevel": 8
        }
        
        print("Reading and preprocessing data (2x downsampling)...")
        # Custom read logic to handle placeholder flats
        tomo, flat, dark, angles = dxchange.exchange.read_aps_tomoscan_hdf5(filepath, dtype=np.float32)
        angles = angles.squeeze()
        
        # Apply synthetic flat if necessary
        if np.mean(flat) < 2.0:
            print("Placeholder flat detected. Using max(tomo) as synthetic flat.")
            flat = np.full_like(flat, np.max(tomo))
            
        # Preprocess
        tomopy.normalize(tomo, flat, dark, out=tomo)
        
        # IMPORTANT: Remove NaNs before further processing
        tomo = np.nan_to_num(tomo, nan=0.0, posinf=1.0, neginf=0.0)
        
        tomo = als.prelog_process_tomo(tomo, preprocessing_args)
        tomopy.minus_log(tomo, out=tomo)
        
        # Downsample
        import skimage.transform as transform
        tomo = np.asarray([transform.downscale_local_mean(proj, (2, 2), cval=0).astype(proj.dtype) for proj in tomo])
        tomo = als.postlog_process_tomo(tomo, postprocessing_args)
        
        # Sanitize again
        tomo = np.nan_to_num(tomo, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 4. Perform ASTRA FBP Recon on GPU
        print("Running ASTRA FBP reconstruction on GPU...")
        # Adjust COR for downsampling
        recon = als.astra_fbp_recon(tomo, angles, COR=cor/2.0, fc=1, gpu=True)
        
        # Optionally mask the reconstruction halo
        recon = als.mask_recon(recon)
        
        print(f"Reconstruction complete. Output shape: {recon.shape}")
        
        # 5. Save to HDF5
        print(f"Saving to {out_path}...")
        os.makedirs(out_dir, exist_ok=True)
        with h5py.File(out_path, 'w') as f:
            # Save as float32. The shape should be [slices, width, height]
            f.create_dataset('reconstruction', data=recon, dtype='float32', compression="gzip", compression_opts=4)
            
        print(f"--- Finished {filename} ---")
        
    except (ZeroDivisionError, IndexError, ValueError) as e:
        print(f"!!! Error reconstructing {filename}: {e} !!!")
        print("Skipping this file.")
    except Exception as e:
        print(f"!!! Unexpected error for {filename}: {e} !!!")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_reconstruct.py <path_to_raw_h5>")
        sys.exit(1)
        
    raw_file = sys.argv[1]
    out_directory = "/global/homes/e/elavarpa/pscratch/lejepa_tomography/data/lejepa_v2"
    reconstruct_file(raw_file, out_directory)
