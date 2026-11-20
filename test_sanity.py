"""Self-check and sanity test suite for model, diffusion, and flow matching."""

import torch
from src.model import CompactUNet
from src.diffusion import GaussianDiffusion
from src.flow_matching import FlowMatching
from src.reflow import Reflow, ReflowDataset


def test_compact_unet():
    print("[1/3] Testing CompactUNet...")
    device = torch.device("cpu")
    model = CompactUNet(in_channels=1, out_channels=1, base_channels=16, channel_mults=(1, 2), time_dim=64).to(device)
    x = torch.randn(2, 1, 32, 32, device=device)
    t = torch.tensor([0.2, 0.8], device=device)
    out = model(x, t)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert torch.isfinite(out).all(), "Non-finite output detected in CompactUNet"
    print("  ✓ CompactUNet forward pass verified.")


def test_diffusion():
    print("[2/3] Testing GaussianDiffusion...")
    device = torch.device("cpu")
    backbone = CompactUNet(in_channels=1, out_channels=1, base_channels=16, channel_mults=(1, 2), time_dim=64).to(device)
    diffusion = GaussianDiffusion(backbone, timesteps=100).to(device)
    
    # 4 dummy samples
    x0 = torch.randn(4, 1, 32, 32, device=device).clamp(-1.0, 1.0)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-3)

    # Overfit for 10 steps to verify gradients
    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        loss = diffusion.compute_loss(x0)
        assert torch.isfinite(loss), "Non-finite diffusion loss"
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] or losses[-1] < 1.0, "Loss did not behave expectedly"

    # Test DDIM sampling
    samples, nfe = diffusion.sample_ddim((2, 1, 32, 32), device=device, steps=5)
    assert samples.shape == (2, 1, 32, 32), f"Sample shape mismatch: {samples.shape}"
    assert nfe == 5, f"Expected 5 NFE, got {nfe}"
    assert torch.isfinite(samples).all(), "Non-finite values in DDIM sample"
    print("  ✓ GaussianDiffusion training and DDIM sampling verified.")


def test_flow_matching():
    print("[3/3] Testing FlowMatching...")
    device = torch.device("cpu")
    backbone = CompactUNet(in_channels=1, out_channels=1, base_channels=16, channel_mults=(1, 2), time_dim=64).to(device)
    fm = FlowMatching(backbone).to(device)

    x1 = torch.randn(4, 1, 32, 32, device=device).clamp(-1.0, 1.0)
    optimizer = torch.optim.Adam(fm.parameters(), lr=1e-3)

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        loss = fm.compute_loss(x1)
        assert torch.isfinite(loss), "Non-finite flow matching loss"
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Test Euler and Heun sampling
    samples_euler, nfe_euler = fm.sample_euler((2, 1, 32, 32), device=device, steps=5)
    assert samples_euler.shape == (2, 1, 32, 32)
    assert nfe_euler == 5, f"Expected 5 NFE, got {nfe_euler}"

    samples_heun, nfe_heun = fm.sample_heun((2, 1, 32, 32), device=device, steps=5)
    assert samples_heun.shape == (2, 1, 32, 32)
    assert nfe_heun == 10, f"Expected 10 NFE, got {nfe_heun}"

    assert torch.isfinite(samples_euler).all()
    assert torch.isfinite(samples_heun).all()
    print("  ✓ FlowMatching training and Euler/Heun sampling verified.")


def test_reflow():
    print("[4/4] Testing Reflow (2-Rectified Flow)...")
    device = torch.device("cpu")
    backbone = CompactUNet(in_channels=1, out_channels=1, base_channels=16, channel_mults=(1, 2), time_dim=64).to(device)
    rf = Reflow(backbone).to(device)

    z0 = torch.randn(4, 1, 32, 32, device=device)
    z1 = torch.randn(4, 1, 32, 32, device=device)
    optimizer = torch.optim.Adam(rf.parameters(), lr=1e-3)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = rf.compute_loss(z0, z1)
        assert torch.isfinite(loss), "Non-finite reflow loss"
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Test 1-step Euler generation
    samples_1step, nfe_1step = rf.sample_euler((2, 1, 32, 32), device=device, steps=1)
    assert samples_1step.shape == (2, 1, 32, 32)
    assert nfe_1step == 1, f"Expected 1 NFE, got {nfe_1step}"
    assert torch.isfinite(samples_1step).all()
    print("  ✓ Reflow paired training and 1-step generation verified.")


if __name__ == "__main__":
    test_compact_unet()
    test_diffusion()
    test_flow_matching()
    test_reflow()
    print("\nALL SANITY CHECKS PASSED SUCCESSFULLY! ✓")

