import torch
import torch.nn as nn
import torch.nn.functional as F

from models.real_nvp.coupling import Coupling, MaskType
from models.real_nvp.splitting import Splitting
from models.real_nvp.squeezing import Squeezing


class RealNVP(nn.Module):
    """RealNVP Model

    Based on the paper:
    "Density estimation using Real NVP"
    by Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio
    (https://arxiv.org/abs/1605.08803).

    Args:
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
        `Coupling` layers.
    """
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super(RealNVP, self).__init__()
        # Register data_constraint to pre-process images, not learnable
        self.register_buffer('data_constraint', torch.tensor([0.9], dtype=torch.float32))

        # Save final number of channels for sampling
        self.z_channels = 4 ** (num_scales - 1) * in_channels

        # Get inner layers
        layers = []
        for scale in range(num_scales):
            layers += [Coupling(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False),
                       Coupling(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True),
                       Coupling(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False)]

            if scale < num_scales - 1:
                in_channels *= 4   # Account for the squeeze
                mid_channels *= 2  # When squeezing, double the number of hidden-layer features in s and t
                layers += [Squeezing(),
                           Coupling(in_channels, mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
                           Coupling(in_channels, mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
                           Coupling(in_channels, mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False)]
            else:
                layers += [Coupling(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True)]

            layers += [Splitting(scale)]
            in_channels //= 2  # Account for the split

        self.layers = nn.ModuleList(layers)

    def forward(self, x, reverse=False):
        if reverse:
            # Reshape z to match dimensions of final latent space
            z = x
            if z.size(2) != z.size(3):
                raise ValueError('Expected z with height = width, got shape {}'.format(z.size()))
            if z.size(1) != self.z_channels:
                side_length = int((z.size(1) * z.size(2) * z.size(3) // self.z_channels) ** 0.5)
                z = z.view(-1, self.z_channels, side_length, side_length)

            # Apply inverse flows
            y = None
            for layer in reversed(self.layers):
                y, z = layer.backward(y, z)

            x = torch.sigmoid(y)

            return x, None
        else:
            # Expect inputs in [0, 1]
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got x with min/max {}/{}'
                                 .format(x.min(), x.max()))

            # Dequantize and convert to logits
            y, sldj = self.pre_process(x)

            # Apply forward flows
            z = None
            for layer in self.layers:
                y, sldj, z = layer.forward(y, sldj, z)

            z = torch.cat((z, y), dim=1)

            return z, sldj

    def pre_process(self, x):
        """Dequantize the input image `x` and convert to logits.

        Args:
            x (torch.Tensor): Input image.

        Returns:
            y (torch.Tensor): Dequantized logits of `x`.

        See Also:
            - Dequantization: https://arxiv.org/abs/1511.01844, Section 3.1
            - Modeling logits: https://arxiv.org/abs/1605.08803, Section 4.1
        """
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.data_constraint
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Initialize sum of log-determinants of Jacobians
        ldj = F.softplus(y) + F.softplus(-y) \
            - F.softplus((1. - self.data_constraint).log() - self.data_constraint.log())
        sldj = ldj.view(ldj.size(0), -1).sum(-1)

        return y, sldj
