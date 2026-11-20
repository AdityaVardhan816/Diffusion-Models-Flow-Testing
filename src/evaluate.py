"""Evaluation utilities: sampling latency, throughput, NFE tracking, and sample grids."""

import time
import json
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional
from torchvision.utils import save_image, make_grid


def save_sample_grid(images: torch.Tensor, output_path: str, nrow: int = 4):
    """Normalize images from [-1, 1] to [0, 1] and save as a grid."""
    # Scale from [-1, 1] to [0, 1]
    imgs = (images.clamp(-1.0, 1.0) + 1.0) / 2.0
    save_image(imgs, output_path, nrow=nrow)


def measure_sampling_profile(
    sample_fn,
    shape: Tuple[int, ...],
    device: torch.device,
    warmup: int = 1,
    repeats: int = 3,
) -> Dict[str, Any]:
    """Profile sampling latency, throughput, and NFE after warmup."""
    # Warmup
    for _ in range(warmup):
        _ = sample_fn(shape, device)

    # Sync accelerator if applicable
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    total_nfe = 0
    samples = None

    for _ in range(repeats):
        start = time.perf_counter()
        samples, nfe = sample_fn(shape, device)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total_nfe = nfe

    batch_size = shape[0]
    avg_latency_s = float(np.mean(times))
    std_latency_s = float(np.std(times))
    throughput = batch_size / avg_latency_s

    return {
        "batch_size": batch_size,
        "nfe": total_nfe,
        "latency_sec_mean": avg_latency_s,
        "latency_sec_std": std_latency_s,
        "throughput_samples_per_sec": throughput,
        "samples": samples,
    }


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Fréchet distance between two Gaussians (mu1, sigma1) and (mu2, sigma2)."""
    import scipy.linalg
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

