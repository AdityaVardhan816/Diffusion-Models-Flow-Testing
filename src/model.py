"""Shared compact U-Net backbone for Diffusion and Flow Matching."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for continuous or discrete timesteps."""

    def __init__(self, dim: int, scale: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: shape (B,) or (B, 1)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        elif t.dim() > 1:
            t = t.view(-1)
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        scaled_t = t * self.scale
        emb = scaled_t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block(nn.Module):
    """Residual convolutional block with time embedding injection."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.act = nn.SiLU()

        self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)

        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(self.proj(x)))
        # Inject time embedding
        time_bias = self.time_proj(t_emb)[:, :, None, None]
        h = h + time_bias
        h = self.act(self.norm2(self.conv(h)))
        return h + self.res_conv(x)


class CompactUNet(nn.Module):
    """Compact shared U-Net for 32x32 image generation.
    
    Can predict either noise epsilon(x_t, t) for Diffusion or velocity v(x_t, t) for Flow Matching.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        channel_mults: tuple = (1, 2, 4),
        time_dim: int = 128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Downsampling path
        self.downs = nn.ModuleList()
        chs = [base_channels]
        curr_ch = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.downs.append(
                nn.ModuleList([
                    Block(curr_ch, out_ch, time_dim),
                    Block(out_ch, out_ch, time_dim),
                    nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1),  # downsample
                ])
            )
            curr_ch = out_ch
            chs.append(curr_ch)

        # Mid block
        self.mid1 = Block(curr_ch, curr_ch, time_dim)
        self.mid2 = Block(curr_ch, curr_ch, time_dim)

        # Upsampling path
        self.ups = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.ups.append(
                nn.ModuleList([
                    nn.ConvTranspose2d(curr_ch, curr_ch, kernel_size=4, stride=2, padding=1),  # upsample
                    Block(curr_ch + out_ch, out_ch, time_dim),
                    Block(out_ch, out_ch, time_dim),
                ])
            )
            curr_ch = out_ch

        # Final projection
        self.out_norm = nn.GroupNorm(min(8, curr_ch), curr_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(curr_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        t: (B,) continuous or discrete timestep values
        """
        t_emb = self.time_mlp(t)
        h = self.init_conv(x)

        skips = [h]
        for b1, b2, down in self.downs:
            h = b1(h, t_emb)
            h = b2(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for up, b1, b2 in self.ups:
            h = up(h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = b1(h, t_emb)
            h = b2(h, t_emb)

        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)

