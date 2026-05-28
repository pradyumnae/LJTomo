import os
import time
import torch
from model import ViTEncoder
from dataset import TomographyH5Dataset
from visualize_pca import generate_pca_viz
import matplotlib.pyplot as plt

def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cuda")
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("_orig_mod.", "").replace("module.", "")
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

def process_checkpoint(ckpt_path, step):
    out_dir = f"checkpoint_pca_{step}"
    # Safeguard: skip if directory already exists with files in it
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) >= 15:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checkpoint PCA for step {step} already exists, skipping.", flush=True)
        return
        
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Found new 5000-step checkpoint: {ckpt_path}. Generating PCA...", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    
    # Initialize model & load weights
    model = ViTEncoder(proj_dim=128, img_size=512, in_chans=1).to("cuda")
    model = load_checkpoint(model, ckpt_path)
    model.eval()
    
    # Load dataset
    data_dir = "/global/homes/e/elavarpa/pscratch/lejepa_tomography/data/lejepa_v2"
    ds = TomographyH5Dataset(data_dir=data_dir, dataset_key='reconstruction', vmin=0.0, vmax=65535.0, is_train=False)
    
    # Generate seed=42 slices (baseline)
    figs_42 = generate_pca_viz(model, ds, num_samples=5, seed=42)
    for i, fig in enumerate(figs_42):
        fig.savefig(f"{out_dir}/slice_seed42_{i}.png", bbox_inches='tight')
        plt.close(fig)
        
    # Generate seed=999 slices (baseline_more)
    figs_999 = generate_pca_viz(model, ds, num_samples=10, seed=999)
    for i, fig in enumerate(figs_999):
        fig.savefig(f"{out_dir}/slice_seed999_{i}.png", bbox_inches='tight')
        plt.close(fig)
        
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Successfully saved 15 PCA slices to {out_dir}/", flush=True)

if __name__ == "__main__":
    print("Starting checkpoint auto-visualizer hook...", flush=True)
    processed_steps = set()
    
    while True:
        if os.path.exists("checkpoints"):
            ckpt_files = [f for f in os.listdir("checkpoints") if f.endswith(".pth") and "step" in f]
            for f in ckpt_files:
                try:
                    # Extract step number, e.g. "lejepa_step_10000.pth" -> 10000
                    step = int(f.split("_")[-1].replace(".pth", ""))
                    if step % 5000 == 0 and step not in processed_steps:
                        ckpt_path = os.path.join("checkpoints", f)
                        process_checkpoint(ckpt_path, step)
                        processed_steps.add(step)
                except Exception as e:
                    print(f"Error processing {f}: {e}", flush=True)
                    
        # Poll every 60 seconds
        time.sleep(60)
