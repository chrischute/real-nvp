import torch
import torch.nn as nn

from enum import IntEnum
from models.resnet import ResNet
from util import checkerboard_mask, channel_wise_mask


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


class Coupling(nn.Module):
    """Coupling layer in RealNVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
    """
    def __init__(self, in_channels, mid_channels, num_blocks, mask_type, reverse_mask):
        super(Coupling, self).__init__()

        # Save mask info
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask

        # Build neural network for scale and translate
        self.st_net = ResNet(in_channels, mid_channels, 2 * in_channels,
                             num_blocks=num_blocks, kernel_size=3, padding=1)

        # Learnable scale for s
        self.scale = nn.utils.weight_norm(Scalar())

    def forward(self, x, sldj=None, reverse=True):
        # Get scale and translate factors
        b = self._get_mask(x)
        x_b = x * b
        st = self.st_net(x_b, b)
        s, t = st.chunk(2, dim=1)
        s = self.scale(torch.tanh(s))
        s = s * (1 - b)
        t = t * (1 - b)

        # Scale and translate
        if reverse:
            inv_exp_s = s.mul(-1).exp()
            if torch.isnan(inv_exp_s).any():
                raise RuntimeError('Scale factor has NaN entries')
            x = x_b + inv_exp_s * ((1 - b) * x - t)
        else:
            exp_s = s.exp()
            if torch.isnan(exp_s).any():
                raise RuntimeError('Scale factor has NaN entries')
            x = x_b + (1 - b) * (x * exp_s + t)

            # Add log-determinant of the Jacobian
            sldj += s.view(s.size(0), -1).sum(-1)

        return x, sldj

    def _get_mask(self, x):
        if self.mask_type == MaskType.CHECKERBOARD:
            mask = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
        elif self.mask_type == MaskType.CHANNEL_WISE:
            mask = channel_wise_mask(x.size(1), self.reverse_mask, device=x.device)
        else:
            raise ValueError('Mask type must be MaskType.CHECKERBOARD or MaskType.CHANNEL_WISE')

        return mask


class Scalar(nn.Module):
    """Scalar that can be wrapped with `torch.nn.utils.weight_norm`."""
    def __init__(self):
        super(Scalar, self).__init__()
        self.weight = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = self.weight * x
        return x
