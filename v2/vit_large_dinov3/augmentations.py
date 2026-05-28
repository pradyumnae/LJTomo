import torch
import torch.nn as nn
from torchvision.transforms import v2
import numpy as np

class CustomIntensityWindowing(nn.Module):
    def __init__(self, p_low=0.01, p_high=0.99):
        super().__init__()
        self.p_low = p_low
        self.p_high = p_high

    def forward(self, img):
        flat = img.view(-1)
        if flat.numel() > 100000:
            indices = torch.randint(0, flat.numel(), (100000,))
            sample = flat[indices]
            q_low = torch.quantile(sample, self.p_low)
            q_high = torch.quantile(sample, self.p_high)
        else:
            q_low = torch.quantile(flat, self.p_low)
            q_high = torch.quantile(flat, self.p_high)
            
        if q_high == q_low:
            return torch.zeros_like(img)
            
        img_clipped = torch.clamp(img, q_low, q_high)
        img_norm = (img_clipped - q_low) / (q_high - q_low)
        return img_norm

class RandomGaussianNoise(nn.Module):
    def __init__(self, std_range=(0.01, 0.05)):
        super().__init__()
        self.std_range = std_range

    def forward(self, img):
        std = torch.empty(1).uniform_(self.std_range[0], self.std_range[1]).item()
        noise = torch.randn_like(img) * std
        return torch.clamp(img + noise, 0, 1)

class RandomRingArtifact(nn.Module):
    def __init__(self, max_rings=3, max_intensity=0.1):
        super().__init__()
        self.max_rings = max_rings
        self.max_intensity = max_intensity

    def forward(self, img):
        _, h, w = img.shape
        center_y, center_x = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        r = torch.sqrt((y - center_y)**2 + (x - center_x)**2)
        artifact_mask = torch.zeros_like(r)
        
        num_rings = torch.randint(1, self.max_rings + 1, (1,)).item()
        for _ in range(num_rings):
            radius = torch.empty(1).uniform_(10, min(h, w) // 2).item()
            width = torch.empty(1).uniform_(1, 3).item()
            intensity = torch.empty(1).uniform_(0.02, self.max_intensity).item()
            ring = torch.exp(-0.5 * ((r - radius) / width) ** 2)
            artifact_mask += ring * intensity
            
        return torch.clamp(img + artifact_mask.unsqueeze(0), 0, 1)

class RandomPoissonNoise(nn.Module):
    def __init__(self, peak_photons_range=(100, 10000)):
        super().__init__()
        self.peak_photons_range = peak_photons_range

    def forward(self, img):
        peak = torch.empty(1).uniform_(self.peak_photons_range[0], self.peak_photons_range[1]).item()
        photons = img * peak
        photons = torch.clamp(photons, min=0)
        noisy_photons = torch.poisson(photons)
        return torch.clamp(noisy_photons / peak, 0, 1)

class RandomStreakArtifact(nn.Module):
    def __init__(self, max_streaks=5, max_intensity=0.2):
        super().__init__()
        self.max_streaks = max_streaks
        self.max_intensity = max_intensity

    def forward(self, img):
        _, h, w = img.shape
        artifact_mask = torch.zeros_like(img)
        num_streaks = torch.randint(1, self.max_streaks + 1, (1,)).item()
        
        y, x = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device), indexing="ij")
        
        for _ in range(num_streaks):
            angle = torch.empty(1).uniform_(0, np.pi).item()
            center_x = torch.empty(1).uniform_(0, w).item()
            center_y = torch.empty(1).uniform_(0, h).item()
            width = torch.empty(1).uniform_(1, 5).item()
            intensity = torch.empty(1).uniform_(-self.max_intensity, self.max_intensity).item()
            dist = torch.abs(np.cos(angle) * (y - center_y) - np.sin(angle) * (x - center_x))
            streak = torch.exp(-0.5 * (dist / width) ** 2)
            artifact_mask += streak * intensity
            
        return torch.clamp(img + artifact_mask, 0, 1)

def get_lejepa_transforms(vmin=0.0, vmax=65535.0, is_target=False):
    base_transforms = [
        CustomIntensityWindowing(p_low=0.01, p_high=0.99),
    ]
    
    if not is_target:
        aug_transforms = [
            v2.Resize((512, 512), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ElasticTransform(alpha=50.0, sigma=5.0),
            v2.RandomApply([RandomGaussianNoise()], p=0.4),
            v2.RandomApply([RandomPoissonNoise()], p=0.4),
            v2.RandomApply([RandomRingArtifact()], p=0.3),
            v2.RandomApply([RandomStreakArtifact()], p=0.3),
        ]
    else:
        aug_transforms = [
            v2.Resize((512, 512), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ]
        
    return v2.Compose(base_transforms + aug_transforms)
