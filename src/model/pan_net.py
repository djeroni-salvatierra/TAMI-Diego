import torch
import torch.nn as nn

from src.model.layers.residual import ResBlock
from src.model.layers.non_local import NonLocalBlock


class PanNet(nn.Module):
    """
    Pan-Sharpening Network based on High-Frequency Residual Learning.
    Fuses upsampled Multi-Spectral (MS) and Panchromatic (PAN) images.
    """

    def __init__(self, ms_channels: int = 4, features: int = 64, blocks: int = 4, kernel_size: int = 3):
        """Initializes PanNet for high-frequency residual pan-sharpening.

        Args:
            ms_channels: Number of multi-spectral bands.
            features: Number of intermediate feature channels.
            blocks: Number of residual blocks.
            kernel_size: Spatial kernel size for 2D convolutions.
        """
        super(PanNet, self).__init__()

        in_channels = ms_channels + 1
        padding = kernel_size // 2

        self.conv_in = nn.Conv2d(in_channels, features, kernel_size=kernel_size, padding=padding)
        self.residual_blocks = nn.ModuleList(
            [ResBlock(features, kernel_size=kernel_size) for _ in range(blocks)]
        )
        self.conv_out = nn.Conv2d(features, ms_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, ms_up: torch.Tensor, pan: torch.Tensor) -> torch.Tensor:
        """Forward pass fusing upsampled MS and PAN images via residual learning.

        Args:
            ms_up: Upsampled multi-spectral tensor of shape [B, ms_channels, H, W].
            pan: Panchromatic tensor of shape [B, 1, H, W].

        Returns:
            Pan-sharpened multi-spectral tensor of shape [B, ms_channels, H, W].
        """
        # [B, ms_channels, H, W] + [B, 1, H, W] -> [B, ms_channels + 1, H, W]
        x = torch.cat([ms_up, pan], dim=1)
        # [B, ms_channels + 1, H, W] -> [B, features, H, W]
        features = self.conv_in(x)
        for block in self.residual_blocks:
            features = block(features)
        # [B, features, H, W] -> [B, ms_channels, H, W]
        residual = self.conv_out(features)
        return ms_up + residual


class PanNetNonLocal(nn.Module):
    """
    Non-Local Pan-Sharpening Network.
    Inserts an Embedded Gaussian Non-Local self-attention block in the middle of the
    residual backbone to capture long-range spatial-spectral dependencies across the scene.
    """

    def __init__(self, ms_channels: int = 4, features: int = 64, blocks: int = 4, kernel_size: int = 3):
        """Initializes PanNetNonLocal with a NonLocalBlock in the middle of the backbone.

        Args:
            ms_channels: Number of multi-spectral bands.
            features: Number of intermediate feature channels.
            blocks: Total number of residual blocks, split around the NonLocalBlock.
            kernel_size: Spatial kernel size for 2D convolutions.
        """
        super(PanNetNonLocal, self).__init__()

        in_channels = ms_channels + 1
        padding = kernel_size // 2
        half_blocks = blocks // 2

        self.conv_in = nn.Conv2d(in_channels, features, kernel_size=kernel_size, padding=padding)
        self.residual_part1 = nn.ModuleList(
            [ResBlock(features, kernel_size=kernel_size) for _ in range(half_blocks)]
        )
        self.non_local = NonLocalBlock(features)
        self.residual_part2 = nn.ModuleList(
            [ResBlock(features, kernel_size=kernel_size) for _ in range(blocks - half_blocks)]
        )
        self.conv_out = nn.Conv2d(features, ms_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, ms_up: torch.Tensor, pan: torch.Tensor) -> torch.Tensor:
        """Forward pass fusing upsampled MS and PAN images with mid-backbone self-attention.

        Args:
            ms_up: Upsampled multi-spectral tensor of shape [B, ms_channels, H, W].
            pan: Panchromatic tensor of shape [B, 1, H, W].

        Returns:
            Pan-sharpened multi-spectral tensor of shape [B, ms_channels, H, W].
        """
        # [B, ms_channels, H, W] + [B, 1, H, W] -> [B, ms_channels + 1, H, W]
        x = torch.cat([ms_up, pan], dim=1)
        # [B, ms_channels + 1, H, W] -> [B, features, H, W]
        features = self.conv_in(x)
        for block in self.residual_part1:
            features = block(features)
        # [B, features, H, W] -> [B, features, H, W]
        features = self.non_local(features)
        for block in self.residual_part2:
            features = block(features)
        # [B, features, H, W] -> [B, ms_channels, H, W]
        residual = self.conv_out(features)
        return ms_up + residual
