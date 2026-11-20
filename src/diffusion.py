"""Diffusion model implementation with DDPM and DDIM samplers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class GaussianDiffusion(nn.Module):
    """Gaussian diffusion with linear beta schedule, epsilon-prediction, and DDPM/DDIM sampling."""

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        # Linear schedule
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers for device management
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffuse data x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        return x_t, noise

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Compute epsilon-prediction MSE loss."""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=x_0.device, dtype=torch.long)
        x_t, noise = self.q_sample(x_0, t)
        # Normalize t to [0, 1] or pass as float for continuous embedding
        t_normalized = t.float() / self.timesteps
        pred_noise = self.model(x_t, t_normalized)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample_ddim(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        steps: int = 50,
        eta: float = 0.0,
    ) -> Tuple[torch.Tensor, int]:
        """Sample using DDIM with configurable number of steps (NFE)."""
        self.eval()
        batch_size = shape[0]
        # Evenly spaced step indices from timesteps-1 down to 0
        step_indices = torch.linspace(self.timesteps - 1, 0, steps, dtype=torch.long, device=device)
        
        x = torch.randn(shape, device=device)
        nfe = 0

        for i in range(len(step_indices)):
            t = step_indices[i]
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            t_norm = t_batch.float() / self.timesteps

            # 1 model function evaluation
            pred_noise = self.model(x, t_norm)
            nfe += 1

            alpha_cumprod_t = self.alphas_cumprod[t]
            
            if i < len(step_indices) - 1:
                t_prev = step_indices[i + 1]
                alpha_cumprod_prev = self.alphas_cumprod[t_prev]
            else:
                alpha_cumprod_prev = torch.tensor(1.0, device=device)

            # Predict x_0
            pred_x0 = (x - torch.sqrt(1.0 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # DDIM update formula
            sigma = (
                eta
                * torch.sqrt((1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t))
                * torch.sqrt(1.0 - alpha_cumprod_t / alpha_cumprod_prev)
            )
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_cumprod_prev - sigma ** 2, min=0.0)) * pred_noise
            
            noise = torch.randn_like(x) if sigma > 0 else 0.0
            x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt + sigma * noise

        return x, nfe

