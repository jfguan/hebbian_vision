"""Hebbian Vision: SwiGLU + 2D conv + Hebbian associative memory.

Per layer: norm → SwiGLU (nonlinear features) → 3x3 depthwise conv2d (local spatial mixing)
         → Hebbian memory K^T V / T (global binding) → project → residual
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    n_classes: int = 100
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    d_model: int = 128
    expand: int = 2
    n_layers: int = 6
    memory_alpha: float = 1.0
    drop_path: float = 0.0  # [DeiT] stochastic depth rate


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class DropPath(nn.Module):
    """[DeiT] Stochastic depth — randomly drop entire residual block during training."""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        mask = torch.rand(x.shape[0], 1, 1, device=x.device) < keep
        return x * mask / keep


class HebbianLayer(nn.Module):
    """SwiGLU + 2D depthwise conv + global Hebbian memory + residual."""

    def __init__(self, cfg: Config, drop_path: float = 0.0):
        super().__init__()
        D = cfg.d_model
        d_inner = cfg.expand * D
        self.grid_size = cfg.img_size // cfg.patch_size  # 8 for 32/4
        self.drop_path = DropPath(drop_path)  # [DeiT] stochastic depth

        # SwiGLU: project up, activate, gate, project down
        self.norm = RMSNorm(D)
        self.proj = nn.Linear(D, d_inner, bias=False)
        self.gate = nn.Linear(D, d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, D, bias=False)

        # 2D depthwise conv on patch grid
        self.conv2d = nn.Conv2d(
            d_inner, d_inner, kernel_size=3, padding=1,
            groups=d_inner, bias=True,
        )

        # Hebbian memory: W = K^T V / T, read = K @ W
        self.proj_k = nn.Linear(D, D, bias=False)
        self.proj_v = nn.Linear(D, D, bias=False)
        self.proj_read = nn.Linear(D, D, bias=False)
        self.log_alpha = nn.Parameter(torch.tensor(cfg.memory_alpha).log())

    def forward(self, x):
        h = self.norm(x)
        B, T, D = h.shape
        G = self.grid_size

        # SwiGLU + 2D depthwise conv
        val = self.proj(h)                          # (B, T, d_inner)
        val = val.transpose(1, 2).reshape(B, -1, G, G)  # (B, d_inner, G, G)
        val = F.silu(self.conv2d(val))              # local spatial mixing
        val = val.reshape(B, -1, T).transpose(1, 2) # (B, T, d_inner)
        out = self.out_proj(val * F.silu(self.gate(h)))  # gate and project down

        # Global associative memory: K from pre-conv (raw), V from post-conv (spatially mixed)
        K = self.proj_k(h)                          # (B, T, D) — patch identity
        V = self.proj_v(out)                        # (B, T, D) — conv-mixed features
        W = torch.bmm(K.transpose(1, 2), V) / T    # (B, D, D)
        read = torch.bmm(K, W)                     # (B, T, D)
        out = out + self.log_alpha.exp() * self.proj_read(read)

        return x + self.drop_path(out)


class PatchEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.proj = nn.Conv2d(
            cfg.in_channels, cfg.d_model,
            kernel_size=cfg.patch_size, stride=cfg.patch_size,
        )
        n_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, cfg.d_model) * 0.02)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x + self.pos_embed


class HebbianVision(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        # [DeiT] linearly increasing drop path rate per layer
        dpr = [cfg.drop_path * i / max(cfg.n_layers - 1, 1) for i in range(cfg.n_layers)]
        self.layers = nn.ModuleList([HebbianLayer(cfg, drop_path=dpr[i]) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes)

    def forward(self, images, targets=None):
        x = self.patch_embed(images)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x).mean(dim=1)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss
