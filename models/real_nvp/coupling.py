import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import IntEnum
from models.real_nvp.real_nvp_layer import RealNVPLayer
from util import checkerboard_mask, channel_wise_mask, WNConv2d


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


class Coupling(RealNVPLayer):
    """Coupling layer in RealNVP.

    Args:
        mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
    """
    def __init__(self, in_channels, mid_channels, mask_type, reverse_mask):
        super(Coupling, self).__init__()

        # Save mask info
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask

        # Build neural networks for scale and translate
        self.scale = ScaleTranslateNetwork(in_channels, mid_channels, use_tanh=True,
                                           num_blocks=8, kernel_size=3, padding=1)
        self.translate = ScaleTranslateNetwork(in_channels, mid_channels, use_tanh=False,
                                               num_blocks=8, kernel_size=3, padding=1)

    def forward(self, x, sldj, z):
        # Get mask
        b = self._get_mask(x)

        # Get scale and translate factors
        x_b = x * b
        s = self.scale(x_b, b)
        s_exp = s.exp()
        if torch.isnan(s_exp).any():
            raise RuntimeError('Scale factor has NaN entries')
        t = self.translate(x_b, b)

        # Scale and translate
        y = x_b + (1 - b) * (x * s_exp + t)

        # Add log-determinant of the Jacobian
        sldj += s.view(s.size(0), -1).sum(-1)

        return y, sldj, z

    def backward(self, y, z):
        # Get mask
        b = self._get_mask(y)

        # Get scale and translate factors
        y_b = y * b
        s = self.scale(y_b, b)
        s_exp = s.mul(-1).exp()
        if torch.isnan(s_exp).any():
            raise RuntimeError('Scale factor has NaN entries')
        t = self.translate(y_b, b)

        # Scale and translate
        x = y_b + s_exp * ((1 - b) * y - t)

        return x, z

    def _get_mask(self, x):
        if self.mask_type == MaskType.CHECKERBOARD:
            mask = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
        elif self.mask_type == MaskType.CHANNEL_WISE:
            mask = channel_wise_mask(x.size(1), self.reverse_mask, device=x.device)
        else:
            raise ValueError('Mask type must be Coupling.checkerboard or Coupling.channel_wise')

        return mask


class ScaleTranslateNetwork(nn.Module):
    """Neural network for learning scale and translate factors. Based on ResNet.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        use_tanh (bool): Whether to use tanh output head (True for s, False for t).
        num_blocks (int): Number of residual blocks in the network.
        kernel_size (int): Side length of each filter in convolutional layers.
    """
    def __init__(self, in_channels, mid_channels, use_tanh, num_blocks, kernel_size, padding):
        super(ScaleTranslateNetwork, self).__init__()

        self.in_conv = nn.Sequential(nn.BatchNorm2d(in_channels),
                                     WNConv2d(in_channels, mid_channels, kernel_size, padding),
                                     nn.BatchNorm2d(mid_channels),
                                     nn.ReLU())

        self.blocks = nn.ModuleList([
            nn.Sequential(WNConv2d(mid_channels, mid_channels, kernel_size, padding),
                          nn.BatchNorm2d(mid_channels),
                          nn.ReLU(),
                          WNConv2d(mid_channels, mid_channels, kernel_size, padding),
                          nn.BatchNorm2d(mid_channels))
            for _ in range(num_blocks)])

        self.use_tanh = use_tanh
        self.out_conv = WNConv2d(mid_channels, in_channels, kernel_size=1, padding=0)
        self.out_scale = nn.utils.weight_norm(Scalar()) if use_tanh else None

    def forward(self, x, b):
        x = self.in_conv(x)

        for residual in self.blocks:
            x = x + residual(x)
            x = F.relu(x)

        x = self.out_conv(x)
        x = x * (1 - b)

        if self.use_tanh:
            x = torch.tanh(x)
            x = self.out_scale(x)

        return x


class Scalar(nn.Module):
    """Scalar that can be wrapped with `torch.nn.utils.weight_norm`."""
    def __init__(self):
        super(Scalar, self).__init__()
        self.weight = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = self.weight * x
        return x
