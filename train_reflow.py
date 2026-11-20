"""Train 2-Rectified Flow (Reflow) and compare 1-RF vs 2-RF on 1-4 step generation."""

import os
import sys
import json
import time
import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from src.model import CompactUNet
from src.flow_matching import FlowMatching
from src.reflow import Reflow, ReflowDataset, generate_coupled_pairs
from src.evaluate import save_sample_grid, measure_sampling_profile


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2-Rectified Flow (Reflow)")
    parser.add_argument("--teacher-ckpt", type=str, default="results/flow_matching_baseline/checkpoint_latest.pt")
    parser.add_argument("--num-pairs", type=int, default=4000, help="Number of coupled pairs to generate")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs on coupled data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--force-pairs", action="store_true", help="Force regenerate coupled pairs")
    parser.add_argument("--device", type=str, default=None)
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
    print(f"Using device: {device}")

    output_dir = Path("results/reflow_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path("results/eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load teacher 1-Rectified Flow model
    print(f"\n--- Loading 1-Rectified Flow Teacher: {args.teacher_ckpt} ---")
    teacher_data = torch.load(args.teacher_ckpt, map_location=device)
    cfg = teacher_data["config"]
    model_cfg = cfg.get("model", {})
    in_channels = 1

    teacher_backbone = CompactUNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=model_cfg.get("base_channels", 32),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4])),
        time_dim=model_cfg.get("time_dim", 128),
    ).to(device)

    teacher_fm = FlowMatching(teacher_backbone).to(device)
    teacher_fm.load_state_dict(teacher_data["model_state"])
    teacher_fm.eval()

    # 2. Generate or load coupled (Z0, Z1) dataset
    pairs_path = output_dir / "coupled_pairs.pt"
    if pairs_path.exists() and not args.force_pairs:
        print(f"Loading existing coupled pairs from {pairs_path}...")
        pairs = torch.load(pairs_path)
        z0, z1 = pairs["z0"], pairs["z1"]
    else:
        t0 = time.time()
        z0, z1 = generate_coupled_pairs(
            teacher_fm,
            num_samples=args.num_pairs,
            batch_size=args.batch_size,
            shape=(in_channels, 32, 32),
            ode_steps=50,
            device=device,
        )
        print(f"Coupled pair generation finished in {time.time() - t0:.1f}s.")
        torch.save({"z0": z0, "z1": z1}, pairs_path)
        print(f"Saved coupled pairs to {pairs_path}")

    dataset = ReflowDataset(z0, z1)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 3. Initialize 2-Rectified Flow student model (fine-tuned from teacher for fast convergence)
    print("\n--- Initializing 2-Rectified Flow Student ---")
    student_backbone = CompactUNet(
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=model_cfg.get("base_channels", 32),
        channel_mults=tuple(model_cfg.get("channel_mults", [1, 2, 4])),
        time_dim=model_cfg.get("time_dim", 128),
    ).to(device)
    student_backbone.load_state_dict(teacher_backbone.state_dict())

    student_rf = Reflow(student_backbone).to(device)
    optimizer = torch.optim.AdamW(student_rf.parameters(), lr=args.lr, weight_decay=1e-4)

    # 4. Train 2-Rectified Flow on straight paired trajectories
    print(f"\n--- Training 2-Rectified Flow on Coupled Data ({args.epochs} epochs) ---")
    history = {"step": [], "loss": [], "epoch": []}
    step = 0
    t_train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        student_rf.train()
        running_loss = 0.0
        batches = 0

        for b_z0, b_z1 in loader:
            b_z0, b_z1 = b_z0.to(device), b_z1.to(device)
            optimizer.zero_grad()
            loss = student_rf.compute_loss(b_z0, b_z1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_rf.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            batches += 1
            step += 1

            if step % 25 == 0:
                history["step"].append(step)
                history["loss"].append(loss.item())

        avg_loss = running_loss / max(1, batches)
        print(f"Epoch {epoch:2d}/{args.epochs} | Avg Paired Loss: {avg_loss:.4f}")

    print(f"Reflow training completed in {time.time() - t_train_start:.1f}s.")

    # Save student checkpoint
    ckpt_path = output_dir / "checkpoint_latest.pt"
    torch.save({
        "epoch": args.epochs,
        "step": step,
        "model_state": student_rf.state_dict(),
        "config": cfg,
    }, ckpt_path)
    print(f"Saved Reflow checkpoint to {ckpt_path}")

    # 5. Comparative Evaluation at Few-Step Budgets: 1, 2, 4, 8 steps
    print("\n--- Evaluating 1-RF vs 2-RF across Few-Step Budgets: [1, 2, 4, 8] ---")
    budgets = [1, 2, 4, 8]
    eval_results = {"1_rf": [], "2_rf": []}

    student_rf.eval()
    teacher_fm.eval()

    torch.manual_seed(42)
    test_noise = torch.randn((16, in_channels, 32, 32), device=device)

    fig, axes = plt.subplots(len(budgets), 2, figsize=(8, 14))

    for idx, steps in enumerate(budgets):
        # 1-RF (Teacher)
        dt = 1.0 / steps
        x_1rf = test_noise.clone()
        for i in range(steps):
            t_val = i * dt
            t_b = torch.full((16,), t_val, device=device, dtype=torch.float32)
            v = teacher_fm.model(x_1rf, t_b)
            x_1rf = x_1rf + dt * v
        x_1rf = torch.clamp(x_1rf, -1.0, 1.0)

        # 2-RF (Reflowed Student)
        x_2rf = test_noise.clone()
        for i in range(steps):
            t_val = i * dt
            t_b = torch.full((16,), t_val, device=device, dtype=torch.float32)
            v = student_rf.model(x_2rf, t_b)
            x_2rf = x_2rf + dt * v
        x_2rf = torch.clamp(x_2rf, -1.0, 1.0)

        # Save images
        p_1rf = eval_dir / f"1rf_steps_{steps}.png"
        p_2rf = eval_dir / f"2rf_steps_{steps}.png"
        save_sample_grid(x_1rf.cpu(), str(p_1rf), nrow=4)
        save_sample_grid(x_2rf.cpu(), str(p_2rf), nrow=4)

        # Plot for comparative figure
        img1 = plt.imread(str(p_1rf))
        img2 = plt.imread(str(p_2rf))

        axes[idx, 0].imshow(img1)
        axes[idx, 0].set_title(f"1-Rectified Flow (NFE = {steps})")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(img2)
        axes[idx, 1].set_title(f"2-Rectified Flow (Reflow) (NFE = {steps})")
        axes[idx, 1].axis("off")

        print(f"  Completed comparison for NFE = {steps}")

    comp_fig_path = eval_dir / "reflow_minimal_steps_comparison.png"
    plt.tight_layout()
    plt.savefig(str(comp_fig_path), dpi=150)
    plt.close()
    print(f"\nSaved side-by-side comparison to {comp_fig_path}!")


if __name__ == "__main__":
    main()

