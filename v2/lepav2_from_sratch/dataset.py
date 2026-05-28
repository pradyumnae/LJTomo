import os
import glob
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from augmentations import get_lejepa_transforms

class TomographyH5Dataset(Dataset):
    """
    Treats a directory of HDF5 files (each with a 'tomo' dataset) as a single large continuous volume.
    """
    def __init__(self, data_dir, dataset_key='reconstruction', V=2, 
                 vmin=0.0, vmax=65535.0, is_train=True):
         self.data_dir = data_dir
         self.dataset_key = dataset_key
         self.V = V
         self.is_train = is_train
         self.vmin = vmin
         self.vmax = vmax
         
         # Discover all h5 files in the directory
         self.files = sorted(glob.glob(os.path.join(self.data_dir, "recon_*.h5")))
         if not self.files:
             raise ValueError(f"No recon_*.h5 files found in {self.data_dir}")

         self.scan_infos = []
         self.total_len = 0
         
         for fpath in self.files:
             try:
                 with h5py.File(fpath, 'r') as f:
                     if self.dataset_key in f:
                         dset = f[self.dataset_key]
                         d, h, w = dset.shape
                         self.scan_infos.append({
                             'path': fpath,
                             'start_idx': self.total_len,
                             'end_idx': self.total_len + d,
                             'shape': (d, h, w)
                         })
                         self.total_len += d
             except Exception as e:
                 print(f"Warning: Could not read {fpath}: {e}")

         self.transform_target = get_lejepa_transforms(vmin, vmax, is_target=True)
         self.transform_context = get_lejepa_transforms(vmin, vmax, is_target=False)
         self.open_files = {}

    def __len__(self):
         return self.total_len

    def __getitem__(self, idx):
         # Find which scan this index belongs to
         target_info = None
         for info in self.scan_infos:
             if info['start_idx'] <= idx < info['end_idx']:
                 target_info = info
                 break
         
         if target_info is None:
             raise IndexError("Global index out of range")

         local_idx = idx - target_info['start_idx']
         path = target_info['path']
         
         try:
             if path not in self.open_files:
                 if len(self.open_files) > 100:
                     # Cache eviction
                     k = next(iter(self.open_files))
                     self.open_files[k].close()
                     del self.open_files[k]
                 self.open_files[path] = h5py.File(path, 'r')
                 
             h5_file = self.open_files[path]
             
             # Read the slice
             slice_data = h5_file[self.dataset_key][local_idx]
             # Convert to float32 tensor [1, H, W]
             img_tensor = torch.from_numpy(slice_data.astype(np.float32)).unsqueeze(0)
         except Exception as e:
             # Fallback to a zero tensor of the expected shape if file read fails 
             # (e.g. file is corrupted or currently being written to by a reconstruction job)
             print(f"Warning: Failed to read slice {local_idx} from {path}: {e}")
             d, h, w = target_info['shape']
             img_tensor = torch.zeros((1, h, w), dtype=torch.float32)
         
         # Apply LeJEPA multi-view augmentations
         if self.is_train:
             views = [self.transform_target(img_tensor)]
             for _ in range(1, self.V):
                 views.append(self.transform_context(img_tensor))
             return torch.stack(views), 0 
         else:
             return self.transform_target(img_tensor), 0
