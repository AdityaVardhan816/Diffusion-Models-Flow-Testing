"""Unified training script for Diffusion and Flow Matching experiments."""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn as nn
from pathlib import Path

from src.data import get_dataloader
from src.model import CompactUNet
from src.diffusion import GaussianDiffusion
from src.flow_matching import FlowMatching
from src.evaluate import save_sample_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion or Flow Matching model")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
    parser.add_argument("--sanity-check", action="store_true", help="Run 100-step overfitting check on 4 samples")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--device", type=str, default=None, help="Device: 'mps', 'cuda', or 'cpu'")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def get_device(requested_device: str = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Output directories
    run_name = cfg["run_name"]
    output_dir = Path("results") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Dataset & DataLoader
    dataset_name = cfg.get("dataset", "fashion_mnist")
    batch_size = 4 if args.sanity_check else cfg.get("batch_size", 64)
    max_samples = 4 if args.sanity_check else cfg.get("max_samples", None)
    
    loader = get_dataloader(
        dataset_name=dataset_name,
        root=cfg.get("data_dir", "./data"),
        batch_size=batch_size,
        num_workers=0,
        train=True,
        max_samples=max_samples,
        seed=seed,
    )

    sample_x, _ = next(iter(loader))
    in_channels = sample_x.shape[1]
    img_size = sample_x.shape[2]

    # Shared backbone
    model_cfg = cfg.get("model", {})
    backbone = CompactUNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=model_cfg.get("base_channels", 32),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4])),
        time_dim=model_cfg.get("time_dim", 128),
    ).to(device)

    param_count = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Backbone parameter count: {param_count:,}")

    # Method
    method_name = cfg["method"].lower()
    if method_name == "diffusion":
        diff_cfg = cfg.get("diffusion", {})
        method = GaussianDiffusion(
            model=backbone,
            timesteps=diff_cfg.get("timesteps", 1000),
            beta_start=diff_cfg.get("beta_start", 1e-4),
            beta_end=diff_cfg.get("beta_end", 0.02),
        ).to(device)
    elif method_name in ("flow_matching", "flow"):
        fm_cfg = cfg.get("flow_matching", {})
        method = FlowMatching(
            model=backbone,
            sigma_min=fm_cfg.get("sigma_min", 1e-4),
        ).to(device)
    else:
        raise ValueError(f"Unknown method '{method_name}'. Supported: 'diffusion', 'flow_matching'")

    # Optimizer & Scheduler
    train_cfg = cfg.get("training", {})
    lr = train_cfg.get("lr", 3e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(method.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Sanity check mode
    if args.sanity_check:
        print("\n--- Running Sanity Check (Overfitting on 4 samples) ---")
        method.train()
        batch_data = sample_x.to(device)
        losses = []

        for step in range(1, 101):
            optimizer.zero_grad()
            loss = method.compute_loss(batch_data)
            assert torch.isfinite(loss), f"Non-finite loss detected at step {step}: {loss.item()}"
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if step % 20 == 0 or step == 1:
                print(f"Sanity Step {step:3d}/100 | Loss: {loss.item():.6f}")

        # Verify gradient flow
        has_grad = any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in method.parameters())
        assert has_grad, "Sanity check failed: no parameter gradients observed!"
        print(f"Sanity Check PASSED! Final loss: {losses[-1]:.6f}")

        # Sample 4 images to verify pipeline
        print("Generating test sample...")
        shape = (4, in_channels, img_size, img_size)
        if method_name == "diffusion":
            samples, nfe = method.sample_ddim(shape, device=device, steps=20)
        else:
            samples, nfe = method.sample_euler(shape, device=device, steps=20)

        out_path = str(output_dir / "sanity_samples.png")
        save_sample_grid(samples.cpu(), out_path, nrow=2)
        print(f"Saved sanity samples to {out_path} (NFE: {nfe})")
        return

    # Regular training loop
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 20)
    print(f"\n--- Starting Training ({method_name.upper()}) for {epochs} epochs ---")

    history = {"step": [], "loss": [], "epoch": [], "epoch_time_s": []}
    step = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        method.train()
        epoch_start = time.time()
        running_loss = 0.0
        batches = 0

        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = method.compute_loss(x)
            if not torch.isfinite(loss):
                print(f"Warning: Non-finite loss at epoch {epoch}, step {step}. Skipping.")
                continue

            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(method.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            batches += 1
            step += 1

            if step % cfg.get("log_interval", 50) == 0:
                history["step"].append(step)
                history["loss"].append(loss.item())
                print(f"Epoch {epoch:2d}/{epochs} | Step {step:5d} | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / max(1, batches)
        epoch_elapsed = time.time() - epoch_start
        scheduler.step()
        history["epoch"].append(epoch)
        history["epoch_time_s"].append(epoch_elapsed)
        print(f"--> Epoch {epoch:2d} finished in {epoch_elapsed:.1f}s | Avg Loss: {epoch_loss:.4f}")

        # Sample generation
        if epoch % cfg.get("sample_interval", 5) == 0 or epoch == epochs:
            print("Generating preview samples...")
            shape = (16, in_channels, img_size, img_size)
            if method_name == "diffusion":
                samples, _ = method.sample_ddim(shape, device=device, steps=50)
            else:
                samples, _ = method.sample_euler(shape, device=device, steps=50)

            sample_path = str(sample_dir / f"samples_epoch_{epoch:03d}.png")
            save_sample_grid(samples.cpu(), sample_path, nrow=4)
            print(f"Saved preview samples to {sample_path}")

        # Save checkpoint
        if epoch % cfg.get("save_interval", 10) == 0 or epoch == epochs:
            ckpt_path = output_dir / "checkpoint_latest.pt"
            torch.save({
                "epoch": epoch,
                "step": step,
                "model_state": method.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
            }, ckpt_path)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s.")

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
