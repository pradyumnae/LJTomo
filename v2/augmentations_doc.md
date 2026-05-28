# LeJEPA Tomography Augmentation Pipeline (v2)

In Self-Supervised Learning (SSL), specifically in Joint-Embedding Predictive Architectures (JEPA), the quality of representations is heavily dependent on the augmentations applied to the data. 

For the `LeJEPA_Tomography` project, the augmentation pipeline dynamically splits the views into a **Target** view (what the model predicts) and a **Context** view (what the model sees). 

## Base Pipeline (Applied to All Views)
Every slice that passes through the dataset loader undergoes the following foundational transformations:
1. **Custom Intensity Windowing**: Dynamically calculates the 1st and 99th percentiles of the slice (using fast random subsampling) to clip outliers, and then normalizes the intensity linearly into a `[0, 1]` range.
2. **Resize**: Resizes the arbitrary slice geometry to `512x512` using anti-aliasing.
3. **Geometric Flips**: Applies Random Horizontal and Random Vertical flips (50% probability each) to enforce rotational invariance, since rock/geological cores lack a strict "up" orientation.

## Context Corruptions
The Context view receives domain-specific corruptions to force the neural network to ignore scanner-induced noise and artifacts, and focus entirely on the underlying physical structure of the sample.

1. **Grid Distortion (`v2.ElasticTransform`)**
   - Applies an elastic grid transformation to simulate slight geometric deformations or varied alignment.
2. **Gaussian Sensor Noise (`RandomGaussianNoise`)** *(40% chance)*
   - Adds simulated continuous electronic noise.
3. **Photon Starvation (`RandomPoissonNoise`)** *(40% chance)*
   - Simulates X-ray shot noise. The slice is scaled to a randomized peak photon count, passed through a Poisson distribution, and scaled back. This is physically accurate for low-dose CT scans.
4. **Concentric Ring Artifacts (`RandomRingArtifact`)** *(30% chance)*
   - Simulates defective or miscalibrated detector pixels in the scanning array. Generates up to 3 concentric rings of varying widths and intensities originating from the center.
5. **Linear Streak Artifacts (`RandomStreakArtifact`)** *(30% chance)*
   - Simulates beam hardening or extreme photon starvation caused by high-density materials (like metals) inside the sample. Randomly draws faint, intersecting bright or dark lines across the slice at random angles.

By selectively applying these corruptions *only* to the Context view, the DINOv3 model is forced to predict the pristine Target view *through* the noise, effectively learning to denoise and de-streak the images directly in its latent space.
