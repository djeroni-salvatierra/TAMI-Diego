import torch
import torch.nn as nn
import torch.nn.functional as F


class NonLocalBlock(nn.Module):
    """
    Embedded Gaussian Non-Local Block.
    Captures long-range dependencies by computing self-attention across the entire image/patch.

    Optimised for PyTorch 2.x on Ampere GPUs (RTX 3060):
      - Uses F.scaled_dot_product_attention (FlashAttention / memory-efficient kernel)
        with `scale=1.0` so it is mathematically 100% identical to the classic
        Embedded Gaussian formulation (Wang et al., 2018) while avoiding O(N^2) memory.
      - Reshapes using `.flatten(2).transpose(1, 2)` to work seamlessly with both
        contiguous (NCHW) and `channels_last` (NHWC) tensors without extra copies.
    """

    def __init__(self, in_channels: int):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = max(in_channels // 2, 1)

        # 1x1 Convolutions to generate the projections (theta, phi, g)
        # Keeping bias=True preserves exact compatibility with existing checkpoints.
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        # 1x1 Convolution to restore the channel dimension before the residual connection
        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

        # Initialize W to zero so the block initially acts as an identity mapping
        nn.init.zeros_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Project and reshape to (B, 1, N, inter_C) where N = H * W
        q = self.theta(x).flatten(2).transpose(1, 2).unsqueeze(1)
        k = self.phi(x).flatten(2).transpose(1, 2).unsqueeze(1)
        v = self.g(x).flatten(2).transpose(1, 2).unsqueeze(1)

        # Fused FlashAttention / memory-efficient attention with scale=1.0
        # (matches standard Embedded Gaussian softmax(theta^T @ phi) @ g)
        y = F.scaled_dot_product_attention(q, k, v, scale=1.0)

        # Reshape back to spatial dimensions: (B, inter_C, H, W)
        y = y.squeeze(1).transpose(1, 2).reshape(B, self.inter_channels, H, W)

        return self.W(y) + x
