import os
import torch
import matplotlib.pyplot as plt
from model import ViTEncoder
from dataset import TomographyH5Dataset
from visualize_pca import generate_pca_viz

def load_checkpoint(model, ckpt_path):
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "").replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

if __name__ == "__main__":
    print("Initializing DINOv3 model...", flush=True)
    model = ViTEncoder(proj_dim=128, img_size=512, in_chans=1).to("cuda")
    
    # Load 6000th step checkpoint
    ckpt_path = "checkpoints/lejepa_step_6000.pth"
    model = load_checkpoint(model, ckpt_path)
    model.eval()
    
    print("Loading dataset...", flush=True)
    data_dir = "/global/homes/e/elavarpa/pscratch/lejepa_tomography/data/lejepa_v2"
    ds = TomographyH5Dataset(data_dir=data_dir, dataset_key='reconstruction', vmin=0.0, vmax=65535.0, is_train=False)
    
    out_dir = "checkpoint_pca_6000"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Generating PCA visualizations for seed=42 (same slices as baseline)...", flush=True)
    figs_42 = generate_pca_viz(model, ds, num_samples=5, seed=42)
    for i, fig in enumerate(figs_42):
        out_path = f"{out_dir}/slice_seed42_{i}.png"
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}", flush=True)
        
    print("Generating PCA visualizations for seed=999 (same slices as baseline_more)...", flush=True)
    figs_999 = generate_pca_viz(model, ds, num_samples=10, seed=999)
    for i, fig in enumerate(figs_999):
        out_path = f"{out_dir}/slice_seed999_{i}.png"
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}", flush=True)

    print("Done!")
