"""Orchestrate training of both Diffusion and Flow Matching, followed by evaluation and comparative report."""

import os
import sys
import json
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path


def run_command(cmd):
    print(f"\n>>> Running: {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        print(f"Command failed with exit code {res.returncode}")
        sys.exit(res.returncode)


def plot_loss_comparison(diff_history_path, fm_history_path, out_path):
    with open(diff_history_path) as f:
        diff_hist = json.load(f)
    with open(fm_history_path) as f:
        fm_hist = json.load(f)

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(diff_hist["step"], diff_hist["loss"], label="Diffusion (Noise MSE)", color="tab:blue")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Diffusion Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(fm_hist["step"], fm_hist["loss"], label="Flow Matching (Velocity MSE)", color="tab:orange")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Flow Matching Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved loss curves comparison to {out_path}")


def main():
    python = ".venv/bin/python3"
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = "/tmp"

    print("==================================================")
    print("STEP 1: Training Diffusion Baseline")
    print("==================================================")
    run_command(f"MPLCONFIGDIR=/tmp {python} train.py --config configs/diffusion.json")

    print("\n==================================================")
    print("STEP 2: Training Flow Matching Baseline")
    print("==================================================")
    run_command(f"MPLCONFIGDIR=/tmp {python} train.py --config configs/flow_matching.json")

    print("\n==================================================")
    print("STEP 3: Comparative Evaluation Across NFE Budgets")
    print("==================================================")
    run_command(f"MPLCONFIGDIR=/tmp {python} evaluate.py --checkpoint results/diffusion_baseline/checkpoint_latest.pt --output-dir results/eval --steps 5 10 20 50")
    run_command(f"MPLCONFIGDIR=/tmp {python} evaluate.py --checkpoint results/flow_matching_baseline/checkpoint_latest.pt --output-dir results/eval --steps 5 10 20 50")

    print("\n==================================================")
    print("STEP 4: Generating Comparative Loss Plot & Summary")
    print("==================================================")
    diff_hist = Path("results/diffusion_baseline/history.json")
    fm_hist = Path("results/flow_matching_baseline/history.json")
    if diff_hist.exists() and fm_hist.exists():
        plot_loss_comparison(diff_hist, fm_hist, "results/eval/loss_curves.png")

    print("\nALL RUNS AND EVALUATIONS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()

