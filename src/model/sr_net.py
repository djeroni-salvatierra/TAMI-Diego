import torch.nn as nn
import torch.nn.functional as F

from src.model.layers.residual import ResBlock


class SRNet(nn.Module):
    """
    A Super-Resolution Network using Global Residual Learning.
    Instead of learning the whole high-res image, it learns only the
    missing high-frequency details (residuals) and adds them to a bicubic upscaled base.
    """

    def __init__(
        self,
        sampling: int,
        features: int = 64,
        kernel_size: int = 3,
        blocks: int = 3,
        channels: int = 3,
    ):
        """Initializes SRNet with a learnable transposed-conv upsampler and residual backbone.

        Args:
            sampling: Spatial upscaling factor (2, 4, or 8).
            features: Number of intermediate feature channels.
            kernel_size: Spatial kernel size for 2D convolutions.
            blocks: Number of residual blocks in the backbone.
            channels: Number of input/output image channels.

        Raises:
            ValueError: If ``sampling`` is not 2, 4, or 8.
        """
        super(SRNet, self).__init__()

        if sampling not in (2, 4, 8):
            raise ValueError(f"sampling must be 2, 4 or 8, got {sampling}")

        up_kernel = 2 * sampling + 1
        self.sampling = sampling

        # Attribute names kept identical to existing checkpoints for 100% compatibility
        self.upsample = nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=up_kernel,
            stride=sampling,
            padding=up_kernel // 2,
            output_padding=sampling - 1,
        )
        self.add_features = nn.Conv2d(
            channels, features, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.residual = nn.ModuleList(
            [ResBlock(features, kernel_size=kernel_size) for _ in range(blocks)]
        )
        self.obtain_channels = nn.Conv2d(
            features, channels, kernel_size=kernel_size, padding=kernel_size // 2
        )

    def forward(self, low):
        """Forward pass with global residual learning.

        Args:
            low: Low-resolution input tensor of shape [B, C, H, W].

        Returns:
            Super-resolved output tensor of shape [B, C, H * sampling, W * sampling].
        """
        # [B, C, H, W] -> [B, C, H*s, W*s]
        bic = F.interpolate(
            low, scale_factor=self.sampling, mode="bicubic", align_corners=False
        )
        # [B, C, H, W] -> [B, C, H*s, W*s]
        x_up = self.upsample(low)

        # [B, C, H*s, W*s] -> [B, features, H*s, W*s]
        features = self.add_features(x_up)
        for res_block in self.residual:
            features = res_block(features)

        # [B, features, H*s, W*s] -> [B, C, H*s, W*s]
        res = self.obtain_channels(features)
        return bic + res
