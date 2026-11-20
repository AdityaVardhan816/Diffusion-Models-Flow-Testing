"""Evaluation and benchmark script comparing Diffusion vs Flow Matching across NFE budgets."""

import os
import json
import argparse
import torch
from pathlib import Path

from src.model import CompactUNet
from src.diffusion import GaussianDiffusion
from src.flow_matching import FlowMatching
from src.evaluate import measure_sampling_profile, save_sample_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion or Flow Matching model across NFE budgets")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--output-dir", type=str, default="results/eval", help="Directory to save evaluation results")
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 10, 20, 50], help="Sampling step budgets to test")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for sampling evaluation")
    parser.add_argument("--device", type=str, default=None, help="Device: 'mps', 'cuda', or 'cpu'")
    return parser.parse_args()


def get_device(requested: str = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint} onto {device}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    method_name = cfg["method"].lower()

    # Shared backbone
    model_cfg = cfg.get("model", {})
    # Determine channel size from dataset
    dataset_name = cfg.get("dataset", "fashion_mnist")
    in_channels = 3 if dataset_name == "cifar10" else 1

    backbone = CompactUNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=model_cfg.get("base_channels", 32),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4])),
        time_dim=model_cfg.get("time_dim", 128),
    ).to(device)

    if method_name == "diffusion":
        diff_cfg = cfg.get("diffusion", {})
        method = GaussianDiffusion(
            model=backbone,
            timesteps=diff_cfg.get("timesteps", 1000),
            beta_start=diff_cfg.get("beta_start", 1e-4),
            beta_end=diff_cfg.get("beta_end", 0.02),
        ).to(device)
    else:
        fm_cfg = cfg.get("flow_matching", {})
        method = FlowMatching(
            model=backbone,
            sigma_min=fm_cfg.get("sigma_min", 1e-4),
        ).to(device)

    method.load_state_dict(ckpt["model_state"])
    method.eval()

    shape = (args.batch_size, in_channels, 32, 32)
    results = []

    print(f"\n--- Benchmarking {method_name.upper()} across NFE budgets {args.steps} ---")

    for s in args.steps:
        print(f"\nEvaluating with step count = {s}...")
        if method_name == "diffusion":
            sample_fn = lambda shp, dev: method.sample_ddim(shp, dev, steps=s)
            solver_name = "DDIM"
        else:
            sample_fn = lambda shp, dev: method.sample_euler(shp, dev, steps=s)
            solver_name = "Euler"

        profile = measure_sampling_profile(sample_fn, shape, device, warmup=1, repeats=3)
        samples = profile.pop("samples")
        profile["solver"] = solver_name
        profile["method"] = method_name
        profile["requested_steps"] = s

        print(f"  NFE: {profile['nfe']} | Latency: {profile['latency_sec_mean']*1000:.1f}ms | Throughput: {profile['throughput_samples_per_sec']:.2f} img/s")

        # Save sample grid for this budget
        grid_path = output_dir / f"{method_name}_{solver_name}_nfe_{profile['nfe']}.png"
        save_sample_grid(samples.cpu(), str(grid_path), nrow=4)
        profile["grid_path"] = str(grid_path)
        results.append(profile)

    # Save benchmark summary
    summary_path = output_dir / f"{method_name}_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

