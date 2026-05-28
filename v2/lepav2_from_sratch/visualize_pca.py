import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F

def generate_pca_viz(model, dataset, num_samples=5, seed=42):
    """
    Generates PCA visualizations from the current model representations.
    Returns a list of matplotlib Figure objects.
    """
    figs = []
    
    # We use a standalone generator for reproducible sampling
    g = torch.Generator()
    g.manual_seed(seed)
    
    # Use a basic dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=g)
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for i, (img, _) in enumerate(loader):
            if i >= num_samples: break
            
            img = img.to(device)
            # Handle DDP wrapped models if needed
            backbone = model.module.backbone if hasattr(model, 'module') else model.backbone
            features = backbone.forward_features(img)
            
            # For patch8, the grid is 64x64 patches (512 / 8 = 64)
            # Let's dynamically compute the grid size based on patch size
            # vit_small_patch8_224 patch size is 8
            # Total tokens: CLS token + patch tokens
            # num_prefix_tokens in patch8 is 1 (just CLS token)
            # Let's extract prefix tokens dynamically
            num_prefix = backbone.num_prefix_tokens if hasattr(backbone, 'num_prefix_tokens') else 1
            patch_tokens = features[:, num_prefix:, :] # [1, H*W/P^2, C]
            patch_tokens = patch_tokens.squeeze(0) # [NumPatches, C]
            
            num_patches = patch_tokens.shape[0]
            grid_size = int(np.sqrt(num_patches)) # Should be 64 for patch8
            
            # Normalize tokens
            patch_tokens = patch_tokens / patch_tokens.norm(dim=-1, keepdim=True)
            
            # PCA using SVD
            u, s, v = torch.pca_lowrank(patch_tokens, q=3)
            
            # Project tokens onto the first 3 PCs
            pcs = patch_tokens @ v[:, :3] # [NumPatches, 3]
            
            # Reshape tokens to [C, H, W] for upsampling
            pcs = pcs.reshape(grid_size, grid_size, 3).permute(2, 0, 1).unsqueeze(0) # [1, 3, grid_size, grid_size]
            
            # Upsample to full image size (512x512)
            pca_img = F.interpolate(pcs, size=(512, 512), mode='bilinear', align_corners=False)
            pca_img = pca_img.squeeze(0).permute(1, 2, 0).cpu().numpy() # [512, 512, 3]
            
            # Robust normalization (clipping extremes) for plotting
            for c in range(3):
                low, high = np.percentile(pca_img[..., c], [1, 99])
                pca_img[..., c] = np.clip((pca_img[..., c] - low) / (high - low), 0, 1)
            
            # Plot
            orig_img = img.squeeze().cpu().numpy()
            fig = plt.figure(figsize=(25, 5))
            
            # Original
            plt.subplot(1, 5, 1)
            plt.imshow(orig_img, cmap='gray')
            plt.title(f"Original Slice (Sample {i})")
            plt.axis('off')
            
            # RGB Composite
            plt.subplot(1, 5, 2)
            plt.imshow(pca_img)
            plt.title("PCA (RGB Composite)")
            plt.axis('off')
            
            # Individual PCs
            for j in range(3):
                plt.subplot(1, 5, 3 + j)
                pc_map = pca_img[..., j]
                plt.imshow(pc_map, cmap='viridis')
                plt.title(f"PC {j+1}")
                plt.axis('off')
                
            plt.tight_layout()
            figs.append(fig)
            
    return figs

if __name__ == "__main__":
    from model import ViTEncoder
    from dataset import TomographyH5Dataset
    
    print("Initializing DINOv3 model...", flush=True)
    model = ViTEncoder(proj_dim=128, img_size=512, in_chans=1).to("cuda")
    model.eval()
    
    print("Loading dataset...", flush=True)
    data_dir = "/global/homes/e/elavarpa/pscratch/lejepa_tomography/data/lejepa_v2"
    ds = TomographyH5Dataset(data_dir=data_dir, dataset_key='reconstruction', vmin=0.0, vmax=65535.0, is_train=False)
    
    print("Generating MORE baseline PCA visualizations...", flush=True)
    os.makedirs("baseline_pca_more", exist_ok=True)
    figs = generate_pca_viz(model, ds, num_samples=10, seed=999)
    
    for i, fig in enumerate(figs):
        out_path = f"baseline_pca_more/baseline_slice_seed999_{i}.png"
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}", flush=True)
