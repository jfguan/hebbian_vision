"""Hebbian Vision: pyramid architecture with conv early stages + Hebbian memory late stages.

Stage 1-2: SwiGLU + 3x3 depthwise conv2d (local only, high resolution)
Stage 3-4: SwiGLU + 3x3 depthwise conv2d + Hebbian memory K^T V / T (local + global)
Downsampling: 3x3 stride-2 conv between stages, doubling channels.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    n_classes: int = 200
    img_size: int = 64
    in_channels: int = 3
    # stages: (n_layers, channels, use_memory)
    stages: Tuple[Tuple[int, int, bool], ...] = (
        (2, 64, False),    # stage 1: 32×32, conv only
        (2, 128, False),   # stage 2: 16×16, conv only
        (4, 256, True),    # stage 3: 8×8,  conv + memory
        (4, 512, True),    # stage 4: 4×4,  conv + memory
    )
    expand: int = 2
    memory_alpha: float = 1.0
    drop_path: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep
        return x * mask / keep


class ConvBlock(nn.Module):
    """SwiGLU + 2D depthwise conv. Local mixing only."""

    def __init__(self, D, expand, grid_size, drop_path=0.0):
        super().__init__()
        d_inner = expand * D
        self.grid_size = grid_size
        self.drop_path = DropPath(drop_path)
        self.norm = RMSNorm(D)
        self.proj = nn.Linear(D, d_inner, bias=False)
        self.gate = nn.Linear(D, d_inner, bias=False)
        self.conv2d = nn.Conv2d(d_inner, d_inner, 3, padding=1, groups=d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, D, bias=False)

    def forward(self, x):
        h = self.norm(x)
        B, T, D = h.shape
        G = self.grid_size
        val = self.proj(h)
        val = val.transpose(1, 2).reshape(B, -1, G, G)
        val = F.silu(self.conv2d(val))
        val = val.reshape(B, -1, T).transpose(1, 2)
        out = self.out_proj(val * F.silu(self.gate(h)))
        return x + self.drop_path(out)


class HebbianBlock(nn.Module):
    """SwiGLU + 2D depthwise conv + Hebbian memory. Local + global."""

    def __init__(self, D, expand, grid_size, memory_alpha=1.0, drop_path=0.0):
        super().__init__()
        d_inner = expand * D
        self.grid_size = grid_size
        self.drop_path = DropPath(drop_path)
        self.norm = RMSNorm(D)
        self.proj = nn.Linear(D, d_inner, bias=False)
        self.gate = nn.Linear(D, d_inner, bias=False)
        self.conv2d = nn.Conv2d(d_inner, d_inner, 3, padding=1, groups=d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, D, bias=False)
        self.proj_k = nn.Linear(D, D, bias=False)
        self.proj_v = nn.Linear(D, D, bias=False)
        self.log_alpha = nn.Parameter(torch.tensor(memory_alpha).log())

    def forward(self, x):
        h = self.norm(x)
        B, T, D = h.shape
        G = self.grid_size
        val = self.proj(h)
        val = val.transpose(1, 2).reshape(B, -1, G, G)
        val = F.silu(self.conv2d(val))
        val = val.reshape(B, -1, T).transpose(1, 2)
        out = self.out_proj(val * F.silu(self.gate(h)))

        K = self.proj_k(h)
        V = self.proj_v(out)
        W = torch.bmm(K.transpose(1, 2), V) / T
        read = torch.bmm(K, W)
        out = out + self.log_alpha.exp() * read

        return x + self.drop_path(out)


class Downsample(nn.Module):
    """2×2 stride-2 conv to halve spatial dims and change channels."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x, grid_size):
        B, T, D = x.shape
        x = x.transpose(1, 2).reshape(B, D, grid_size, grid_size)
        x = self.conv(x)
        return x.flatten(2).transpose(1, 2)


class HebbianVision(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Stem: 3×3 stride-2 conv to halve input resolution
        first_ch = cfg.stages[0][1]
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, first_ch, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
        )
        grid = cfg.img_size // 2  # after stem

        # Build stages
        total_layers = sum(s[0] for s in cfg.stages)
        dpr = [cfg.drop_path * i / max(total_layers - 1, 1) for i in range(total_layers)]
        layer_idx = 0

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_ch = first_ch

        for i, (n_layers, ch, use_memory) in enumerate(cfg.stages):
            # Downsample between stages (not before first)
            if i > 0:
                self.downsamples.append(Downsample(prev_ch, ch))
                grid = grid // 2
            else:
                self.downsamples.append(nn.Identity())

            stage = nn.ModuleList()
            for _ in range(n_layers):
                dp = dpr[layer_idx]
                if use_memory:
                    stage.append(HebbianBlock(ch, cfg.expand, grid, cfg.memory_alpha, dp))
                else:
                    stage.append(ConvBlock(ch, cfg.expand, grid, dp))
                layer_idx += 1
            self.stages.append(stage)
            prev_ch = ch

        final_ch = cfg.stages[-1][1]
        self.norm = RMSNorm(final_ch)
        self.head = nn.Linear(final_ch, cfg.n_classes)

        # Store grid sizes for downsampling
        self._grid_sizes = []
        g = cfg.img_size // 2
        for i in range(len(cfg.stages)):
            if i > 0:
                g = g // 2
            self._grid_sizes.append(g)

    def forward(self, images, targets=None):
        x = self.stem(images)  # (B, C, H/2, W/2)
        B = x.shape[0]
        x = x.flatten(2).transpose(1, 2)  # (B, T, D)

        for i, (ds, stage) in enumerate(zip(self.downsamples, self.stages)):
            if i > 0:
                x = ds(x, self._grid_sizes[i - 1])
            for block in stage:
                x = block(x)

        x = self.norm(x).mean(dim=1)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss
