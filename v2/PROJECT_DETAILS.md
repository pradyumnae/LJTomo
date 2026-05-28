# LeJEPA Tomography: Self-Supervised Learning for Micro-CT Reconstructions

This document provides a comprehensive description of the **LeJEPA Tomography** project, detailing every stage from dataset reconstruction to self-supervised ViT training, validation, and automated quality-control hooks.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Data Pipeline & Reconstruction](#2-data-pipeline--reconstruction)
3. [Model Architecture](#3-model-architecture)
4. [Training Setup & Robust Checkpointing](#4-training-setup--robust-checkpointing)
5. [Multi-Node High-Speed Scaling](#5-multi-node-high-speed-scaling)
6. [GPU Memory & Batch Size Optimization](#6-gpu-memory--batch-size-optimization)
7. [Representation Evaluation via PCA](#7-representation-evaluation-via-pca)
8. [Out-of-Loop Background Automation](#8-out-of-loop-background-automation)

---

## 1. Project Overview
The objective of this project is to train a self-supervised Vision Transformer (ViT) representation model specifically tailored for **Micro-CT (Tomography) scans**. Micro-CT produces high-resolution, volumetric datasets containing microstructural features. 

By leveraging self-supervised pre-training (utilizing a joint-embedding style setup combined with custom statistical constraints), we learn rich anatomical features from 2D slices without requiring manual annotations.

---

## 2. Data Pipeline & Reconstruction

### 2.1 Reconstruction Jobs (`lejepa_h`)
* **Task**: Process raw Micro-CT scans into normalized, standardized `.h5` datasets suitable for machine learning pipelines.
* **Outcome**: Generated exactly **2,889** fully reconstructed `.h5` volumes in the target data directory `/global/homes/e/elavarpa/pscratch/lejepa_tomography/data/lejepa_v2/`.

### 2.2 Resilient Data Loading (`dataset.py`)
Micro-CT reconstructions can occasionally suffer from corrupted files or bad writes. To prevent training runs from crashing mid-epoch, the dataloader is hardened:
* **Fault Tolerance**: Wrapped in a robust try-except filter catching file read exceptions (e.g., `Unable to synchronously open file` or bad object header versions).
* **Grayscale Normalization**: Re-scales the micro-CT reconstruction density values from $[0, 65535]$ down to normal floating-point ranges.

---

## 3. Model Architecture (`model.py`)

The architecture integrates a state-of-the-art vision backbone with custom projections:

```
[Input Slice (1 x 512 x 512)]
            │
            ▼
┌───────────────────────┐
│     ViT Encoder       │  ◄── timm ViT-Small (patch16 DINOv3) OR ViT-Large (patch14 DINOv2)
└───────────────────────┘
            │
            ▼ [Embeddings: 512-dim for Small / 1024-dim for Large]
┌───────────────────────┐
│    Projection MLP     │  ◄── 3-layer MLP with BatchNorm
└───────────────────────┘
            │
            ▼
 [128-dim Representation]
```

### 3.1 Supported Scales
1. **ViT-Small (`vit_small_patch16_dinov3`)**:
   * Uses 12 blocks, 12 attention heads, and embedding dimension of 384.
   * Extremely fast iteration and highly efficient.
2. **ViT-Large (`vit_large_patch14_dinov2`)**:
   * Meta's premier large representation backbone with 24 blocks, 16 attention heads, and embedding dimension of 1024.
   * Learns significantly more complex structural abstractions.
   * Supports both **Pre-trained DINOv2 initialization** and **From Scratch** randomized training for a rigorous 1-to-1 baseline comparison.

---

## 4. Training Setup & Robust Checkpointing

### 4.1 Multi-GPU Execution (`main.py` & `submit_lejepa.slurm`)
* **Scale**: Runs PyTorch Distributed Data Parallel (DDP) across **4 GPUs** per node via `torchrun`.
* **Resource Management**: Launched using Slurm with custom resource constraints (`#SBATCH -C gpu`, `--gpus-per-node=4`, `--cpus-per-task=128`).
* **Runtime**: Adjusted to **12:00:00 hours** wallclock limit to allow thorough, steady training epochs.

### 4.2 Auto-Save Checkpoint Hook
Following a Slurm timeout incident, the checkpointing architecture was modernized to prevent any progress loss:
* **Autosave Frequency**: Automatically serializes the entire network (`model_state_dict`), optimizer (`optimizer_state_dict`), scheduler (`scheduler_state_dict`), and current epoch/step metadata **every 1,000 steps**.
* **Checkpoints Directory**: All step checkpoints are securely archived as `checkpoints/lejepa_step_X.pth`.
* **Auto-Resume**: On startup, `main.py` dynamically scans the directory, locates the absolute latest `.pth` file, and resumes training from that exact step.

---

## 5. Multi-Node High-Speed Scaling

For the massive **ViT-Large** runs, training is parallelized across **2 nodes (8 NVIDIA A100 GPUs total)** to bypass single-node memory and throughput limits.

### 5.1 Distributed Network Coordination
To coordinate communication across nodes, the Slurm script dynamically queries the allocated nodes to assign the primary coordinator (Master Node) and binds a port:
```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
```

### 5.2 Dynamic c10d Rendezvous Launching
We utilize PyTorch `torchrun`'s dynamic **`c10d` rendezvous backend** (`--rdzv_backend=c10d`) with `rdzv_endpoint` to launch tasks on each node:
```bash
srun shifter --image=$IMAGE bash -c "
    torchrun \
        --nnodes=\$SLURM_JOB_NUM_NODES \
        --nproc_per_node=4 \
        --rdzv_id=\$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \
        main.py pretrained=True bs=8
"
```
This guarantees automatic synchronization of gradients via NCCL across Perlmutter's high-speed Slingshot-11 interconnect, eliminating node synchronization failures and maximizing GPU scaling efficiency.

---

## 6. GPU Memory & Batch Size Optimization

To extract the **absolute maximum training throughput** for ViT-Large on Perlmutter's A100 GPUs, we launched a synthetic GPU memory profiling suite (`test_batch_size.py`) under active interactive node constraints.

### 6.1 Benchmark Results (A100 40GB VRAM)
* **Batch Size 2**: **SUCCESS** — uses **10.08 GB** VRAM.
* **Batch Size 4**: **SUCCESS** — uses **18.75 GB** VRAM.
* **Batch Size 8**: **SUCCESS** — uses **36.27 GB** VRAM (Optimal!).
* **Batch Size 12**: **OOM** (Out of Memory) during Forward Pass.
* **Batch Size 16**: **OOM** (Out of Memory) during Forward Pass.

### 6.2 Chosen Profile
Based on these findings, we set the optimal batch size to **`bs=8`** for both the DINO-initialized and Scratch ViT-Large runs, utilizing a highly optimal **90.6% GPU Memory saturation** to maximize training speed!

---

## 7. Representation Evaluation via PCA

To evaluate what features the self-supervised ViT is learning, we project high-dimensional patch embeddings into visually interpretable RGB space using Principal Component Analysis (PCA).

### 7.1 The PCA Pipeline (`visualize_pca.py`)
1. **Patch Extraction**: Slices are fed into the ViT encoder to retrieve patch tokens (skipping DINO's registers and CLS tokens).
2. **SVD projection**: Fits a low-rank PCA (`torch.pca_lowrank`) over the patch embeddings to capture the top 3 principal components (PCs).
3. **Upsampling**: Reshapes the grid of top PCs back to $[512 \times 512 \times 3]$.
4. **Quantile Scaling**: Applies $1\%$ to $99\%$ percentile clipping to maximize contrast, mapping the top 3 components directly to Red, Green, and Blue composite channels.

This creates beautiful anatomical heatmaps where similar structures map to similar color ranges!

### 7.2 Verification at Step 6,000 (`generate_ckpt_pca.py`)
* Successfully evaluated the `lejepa_step_6000.pth` checkpoint.
* Generated and verified **15 total PCA comparison images** under `checkpoint_pca_6000/` across reproducible seeds:
  * **Seed 42** (Baseline slices 0-4)
  * **Seed 999** (More baseline slices 0-9)

---

## 8. Out-of-Loop Background Automation

To monitor quality control dynamically during long runs, we deployed an automated checkpoint watcher hook.

```
       ┌────────────────────────┐
       │   sbatch main training │ (Saves checkpoints/lejepa_step_X.pth every 1k steps)
       └───────────┬────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  watch_checkpoints.py (Background)   │ ◄── Polling checkpoints directory every 60s
└──────────────────┬───────────────────┘
                   │
                   ▼ [If step % 5000 == 0]
┌──────────────────────────────────────┐
│  Automated GPU PCA Slice Generation  │ ◄── Saves 15 slices to "checkpoint_pca_X"
└──────────────────────────────────────┘
```

### 8.1 Daemon Monitor (`watch_checkpoints.py`)
* **Behavior**: Continually polls `checkpoints/` for newly saved model states.
* **Interval**: Triggers specifically when a step checkpoint divisible by **5,000** (e.g. `10000`, `15000`, `20000`...) is completed.
* **Auto-Execution**: Performs the GPU forward pass, fits the low-rank PCA, and exports the 15 visualization plots directly to `checkpoint_pca_<step>/` without interrupting the main training pipeline.

### 8.2 Slurm Deployment (`submit_watch.slurm`)
* **Job ID**: `53445231`
* **Queue**: Deployed as a background job on the `shared` GPU partition (`-q shared`) for 12 hours. This requires only 1 shared GPU, keeping resources extremely cheap while providing real-time visual progress monitoring!
