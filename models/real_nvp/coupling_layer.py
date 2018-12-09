import torch
import torch.nn as nn

from enum import IntEnum
from models.resnet import ResNet
from util import BatchNormStats2d, checkerboard_mask


class MaskType(IntEnum):
    CHECKERBOARD = 0
    CHANNEL_WISE = 1


class CouplingLayer(nn.Module):
    """Coupling layer in RealNVP.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the `s` and `t` network.
        num_blocks (int): Number of residual blocks in the `s` and `t` network.
        mask_type (MaskType): One of `MaskType.CHECKERBOARD` or `MaskType.CHANNEL_WISE`.
        reverse_mask (bool): Whether to reverse the mask. Useful for alternating masks.
    """
    def __init__(self, in_channels, mid_channels, num_blocks, mask_type, reverse_mask):
        super(CouplingLayer, self).__init__()

        # Save mask info
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask

        # Build scale and translate network
        if self.mask_type == MaskType.CHANNEL_WISE:
            in_channels //= 2
        self.st_net = ResNet(in_channels, mid_channels, 2 * in_channels,
                             num_blocks=num_blocks, kernel_size=3, padding=1)

        # Learnable scale for s
        self.scale = nn.utils.weight_norm(Scalar())

        # Out batch norm
        self.norm = BatchNormStats2d(in_channels)
        self.use_norm = True

    def forward(self, x, sldj=None, reverse=True):
        if self.mask_type == MaskType.CHECKERBOARD:
            # Checkerboard mask
            b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
            x_b = x * b
            st = self.st_net(x_b)
            s, t = st.chunk(2, dim=1)
            s = self.scale(torch.tanh(s))
            s = s * (1 - b)
            t = t * (1 - b)

            # Scale and translate
            if reverse:
                if self.use_norm:
                    m, v = self.norm(x * (1 - b), training=False)
                    log_v = v.log()
                    x = x * (.5 * log_v * (1 - b)).exp() + m * (1 - b)

                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = x * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = (x + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.view(s.size(0), -1).sum(-1)

                if self.use_norm:
                    m, v = self.norm(x * (1 - b), self.training)
                    log_v = v.log()
                    x = (x - m * (1 - b)) * (-.5 * log_v * (1 - b)).exp()
                    sldj -= (.5 * log_v * (1 - b)).view(log_v.size(0), -1).sum(-1)
        else:
            # Channel-wise mask
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)

            st = self.st_net(x_id)
            s, t = st.chunk(2, dim=1)
            s = self.scale(torch.tanh(s))

            # Scale and translate
            if reverse:
                if self.use_norm:
                    m, v = self.norm(x_change, training=False)
                    log_v = v.log()
                    x_change = x_change * (.5 * log_v).exp() + m

                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = x_change * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = (x_change + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.view(s.size(0), -1).sum(-1)

                if self.use_norm:
                    m, v = self.norm(x_change, self.training)
                    log_v = v.log()
                    x_change = (x_change - m) * (-.5 * log_v).exp()
                    sldj -= (.5 * log_v).view(log_v.size(0), -1).sum(-1)

            if self.reverse_mask:
                x = torch.cat((x_id, x_change), dim=1)
            else:
                x = torch.cat((x_change, x_id), dim=1)

        return x, sldj


class Scalar(nn.Module):
    """Scalar that can be wrapped with `torch.nn.utils.weight_norm`."""
    def __init__(self):
        super(Scalar, self).__init__()
        self.weight = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = self.weight * x
        return x
