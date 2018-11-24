import torch
import torch.nn as nn

from enum import IntEnum
from models import ResNet
from models.real_nvp.real_nvp_layer import RealNVPLayer
from util import checkerboard_mask, channel_wise_mask


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

        # Build neural network for scale and translate
        self.st_net = ResNet(in_channels, mid_channels, 2 * in_channels,
                             num_blocks=8, kernel_size=3, padding=1)

        # Learnable scale for s
        self.scale = nn.utils.weight_norm(Scalar())

    def forward(self, x, sldj, z):
        y, sldj = self._flow(x, sldj, forward=True)

        return y, sldj, z

    def backward(self, y, z):
        x, _ = self._flow(y, forward=False)

        return x, z

    def _flow(self, x, sldj=None, forward=True):
        # Get scale and translate factors
        b = self._get_mask(x)
        x_b = x * b
        st = self.st_net(x_b, b)
        s, t = st.chunk(2, dim=1)
        s = self.scale(torch.tanh(s))
        s = s * (1 - b)
        t = t * (1 - b)

        # Scale and translate
        if forward:
            exp_s = s.exp()
            if torch.isnan(exp_s).any():
                raise RuntimeError('Scale factor has NaN entries')
            x = x_b + (1 - b) * (x * exp_s + t)

            # Add log-determinant of the Jacobian
            sldj += s.view(s.size(0), -1).sum(-1)
        else:
            exp_neg_s = s.mul(-1).exp()
            if torch.isnan(exp_neg_s).any():
                raise RuntimeError('Scale factor has NaN entries')
            x = x_b + exp_neg_s * ((1 - b) * x - t)

        return x, sldj

    def _get_mask(self, x):
        if self.mask_type == MaskType.CHECKERBOARD:
            mask = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
        elif self.mask_type == MaskType.CHANNEL_WISE:
            mask = channel_wise_mask(x.size(1), self.reverse_mask, device=x.device)
        else:
            raise ValueError('Mask type must be Coupling.checkerboard or Coupling.channel_wise')

        return mask


class Scalar(nn.Module):
    """Scalar that can be wrapped with `torch.nn.utils.weight_norm`."""
    def __init__(self):
        super(Scalar, self).__init__()
        self.weight = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = self.weight * x
        return x
