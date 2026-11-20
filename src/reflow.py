"""2-Rectified Flow (Reflow) implementation for trajectory straightening and 1-2 step generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class ReflowDataset(Dataset):
    """Coupled dataset of (Z_0, Z_1) trajectories generated from 1-Rectified Flow."""

    def __init__(self, z0: torch.Tensor, z1: torch.Tensor):
        assert z0.shape == z1.shape
        self.z0 = z0
        self.z1 = z1

    def __len__(self):
        return len(self.z0)

    def __getitem__(self, idx):
        return self.z0[idx], self.z1[idx]


@torch.no_grad()
def generate_coupled_pairs(
    flow_model: nn.Module,
    num_samples: int = 3200,
    batch_size: int = 64,
    shape: Tuple[int, int, int] = (1, 32, 32),
    ode_steps: int = 50,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate coupled (Z_0, Z_1) pairs by integrating the 1-Rectified Flow ODE."""
    flow_model.eval()
    all_z0 = []
    all_z1 = []

    batches = (num_samples + batch_size - 1) // batch_size
    print(f"Generating {num_samples} coupled (Z_0, Z_1) pairs using {ode_steps}-step ODE solver...")

    for b in range(batches):
        cur_bs = min(batch_size, num_samples - b * batch_size)
        z0 = torch.randn((cur_bs, *shape), device=device)
        
        # Integrate from t=0 to t=1 using Euler
        z_t = z0.clone()
        dt = 1.0 / ode_steps
        for i in range(ode_steps):
            t_val = i * dt
            t_batch = torch.full((cur_bs,), t_val, device=device, dtype=torch.float32)
            v = flow_model.model(z_t, t_batch)
            z_t = z_t + dt * v

        z1 = torch.clamp(z_t, -1.0, 1.0)
        all_z0.append(z0.cpu())
        all_z1.append(z1.cpu())

        if (b + 1) % 10 == 0 or (b + 1) == batches:
            print(f"  Generated {(b + 1) * batch_size}/{num_samples} pairs")

    return torch.cat(all_z0, dim=0)[:num_samples], torch.cat(all_z1, dim=0)[:num_samples]


class Reflow(nn.Module):
    """2-Rectified Flow model trained on coupled (Z_0, Z_1) pairs."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def compute_loss(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """Compute regression loss on coupled trajectories."""
        batch_size = z0.shape[0]
        device = z0.device

        # Sample time t ~ U(0, 1)
        t = torch.rand(batch_size, device=device)
        t_expand = t.view(batch_size, 1, 1, 1)

        # Straight interpolation between coupled pairs
        z_t = (1.0 - t_expand) * z0 + t_expand * z1
        target_velocity = z1 - z0

        pred_velocity = self.model(z_t, t)
        return F.mse_loss(pred_velocity, target_velocity)

    @torch.no_grad()
    def sample_euler(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        steps: int = 1,
    ) -> Tuple[torch.Tensor, int]:
        """Integrate ODE from t=0 to t=1 using Euler method with few steps (1-4)."""
        self.eval()
        dt = 1.0 / steps
        x = torch.randn(shape, device=device)
        nfe = 0

        for i in range(steps):
            t_val = i * dt
            t_batch = torch.full((shape[0],), t_val, device=device, dtype=torch.float32)
            v = self.model(x, t_batch)
            nfe += 1
            x = x + dt * v

        return torch.clamp(x, -1.0, 1.0), nfe

