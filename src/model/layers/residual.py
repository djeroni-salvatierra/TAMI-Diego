import torch.nn as nn


class ResBlock(nn.Module):
    """
    A standard Residual Block (ResBlock) for Convolutional Neural Networks (CNNs).
    It maintains the same number of input/output channels and preserves spatial dimensions.
    Note: BatchNorm is deliberately omitted following EDSR (Lim et al., 2017) for
    Super-Resolution and Pan-Sharpening, which preserves radiometric/contrast fidelity,
    saves ~40% activation memory, and remains compatible with existing checkpoints.
    """

    def __init__(self, in_channels: int, kernel_size: int = 3):
        super(ResBlock, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding)
        self.act_out = nn.ReLU(inplace=False)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return self.act_out(res + x)
