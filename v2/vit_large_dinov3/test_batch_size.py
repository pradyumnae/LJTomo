import sys
import torch
from model import ViTEncoder, SIGReg

def test_batch_size(bs):
    print(f"\n--- Testing Batch Size: {bs} ---", flush=True)
    device = "cuda"
    
    try:
        # 1. Initialize ViT-Large model and load to GPU
        print("Initializing ViT-Large model on GPU...", flush=True)
        model = ViTEncoder(proj_dim=128, img_size=512, in_chans=1, pretrained=False).to(device)
        sigreg = SIGReg().to(device)
        
        # 2. Setup optimizer to simulate full training memory profile
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats(device)
        
        # 3. Create synthetic multi-view tomography inputs [N, V, C, H, W]
        # V = 2 views (target + context) of size 512x512
        print("Generating synthetic inputs...", flush=True)
        inputs = torch.randn(bs, 2, 1, 512, 512, device=device)
        
        # 4. Simulate training step
        print("Running Forward Pass...", flush=True)
        emb, proj = model(inputs)
        inv_loss = (proj.mean(0) - proj).square().mean()
        sigreg_loss = sigreg(proj)
        loss = sigreg_loss * 0.5 + inv_loss * 0.5
        
        print("Running Backward Pass...", flush=True)
        loss.backward()
        
        print("Optimizer Step...", flush=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
        
        # 5. Measure peak memory usage
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3) # in GB
        print(f"SUCCESS! Batch Size {bs} ran successfully.", flush=True)
        print(f"Peak VRAM Allocated: {peak_mem:.2f} GB / 40.00 GB", flush=True)
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"OOM ERROR: Batch Size {bs} failed with Out of Memory!", flush=True)
            # Clear CUDA cache
            for p in model.parameters():
                p.grad = None
            del model, sigreg, opt, inputs
            torch.cuda.empty_cache()
            return False
        else:
            raise e

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_batch_size.py <batch_size>")
        sys.exit(1)
    bs = int(sys.argv[1])
    test_batch_size(bs)
