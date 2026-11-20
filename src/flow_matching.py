"""Flow Matching (Rectified Flow / Conditional Flow Matching) implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FlowMatching(nn.Module):
    """Conditional Flow Matching with straight interpolation path (Rectified Flow).
    
    Path:
        t in [0, 1]
        x_0 ~ N(0, I)   (source noise at t=0)
        x_1 ~ p_data    (target clean data at t=1)
        x_t = (1 - t) * x_0 + t * x_1
        Target velocity: v_t = x_1 - x_0
    """

    def __init__(self, model: nn.Module, sigma_min: float = 1e-4):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min

    def compute_loss(self, x_1: torch.Tensor) -> torch.Tensor:
        """Compute flow matching regression loss."""
        batch_size = x_1.shape[0]
        device = x_1.device

        # Sample time t ~ U(0, 1)
        t = torch.rand(batch_size, device=device)
        t_expand = t.view(batch_size, 1, 1, 1)

        # Sample base Gaussian noise
        x_0 = torch.randn_like(x_1)

        # Straight linear interpolation path
        x_t = (1.0 - t_expand) * x_0 + t_expand * x_1
        target_velocity = x_1 - x_0

        # Predict velocity field
        pred_velocity = self.model(x_t, t)
        return F.mse_loss(pred_velocity, target_velocity)

    @torch.no_grad()
    def sample_euler(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        steps: int = 50,
    ) -> Tuple[torch.Tensor, int]:
        """Integrate ODE from t=0 to t=1 using Euler method."""
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

    @torch.no_grad()
    def sample_heun(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        steps: int = 25,
    ) -> Tuple[torch.Tensor, int]:
        """Integrate ODE from t=0 to t=1 using 2nd-order Heun (RK2) method."""
        self.eval()
        dt = 1.0 / steps
        x = torch.randn(shape, device=device)
        nfe = 0

        for i in range(steps):
            t0 = i * dt
            t1 = (i + 1) * dt
            t0_batch = torch.full((shape[0],), t0, device=device, dtype=torch.float32)
            t1_batch = torch.full((shape[0],), t1, device=device, dtype=torch.float32)

            k1 = self.model(x, t0_batch)
            nfe += 1

            x_guess = x + dt * k1
            k2 = self.model(x_guess, t1_batch)
            nfe += 1

            x = x + 0.5 * dt * (k1 + k2)

        return torch.clamp(x, -1.0, 1.0), nfe

