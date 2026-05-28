import os
import sys
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

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_pca_for_step.py <step>")
        sys.exit(1)
        
    step = int(sys.argv[1])
    ckpt_path = f"checkpoints/lejepa_step_{step}.pth"
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        sys.exit(1)
        
    print(f"Initializing model for step {step}...", flush=True)
    model = ViTEncoder(proj_dim=128, img_size=512, in_chans=1).to("cuda")
    model = load_checkpoint(model, ckpt_path)
    model.eval()
    
    print("Loading dataset...", flush=True)
    data_dir = "/global/homes/e/elavarpa/pscratch/lejepa_tomography/data/lejepa_v2"
    ds = TomographyH5Dataset(data_dir=data_dir, dataset_key='reconstruction', vmin=0.0, vmax=65535.0, is_train=False)
    
    out_dir = f"checkpoint_pca_{step}"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Generating PCA visualizations for seed=42 (5 samples) -> {out_dir}...", flush=True)
    figs_42 = generate_pca_viz(model, ds, num_samples=5, seed=42)
    for i, fig in enumerate(figs_42):
        out_path = f"{out_dir}/slice_seed42_{i}.png"
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}", flush=True)
        
    print(f"Generating PCA visualizations for seed=999 (10 samples) -> {out_dir}...", flush=True)
    figs_999 = generate_pca_viz(model, ds, num_samples=10, seed=999)
    for i, fig in enumerate(figs_999):
        out_path = f"{out_dir}/slice_seed999_{i}.png"
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {out_path}", flush=True)

    print("Done!")

if __name__ == "__main__":
    main()
