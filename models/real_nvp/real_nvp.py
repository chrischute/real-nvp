import torch
import torch.nn as nn

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
    """
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64):
        super(RealNVP, self).__init__()
        self.alpha = 1e-5

        # Save final number of channels for sampling
        self.z_channels = 4 ** (num_scales - 1) * in_channels

        # Get inner layers
        layers = []
        for scale in range(num_scales - 1):
            layers += [Coupling(in_channels, mid_channels, MaskType.CHECKERBOARD, reverse_mask=False),
                       Coupling(in_channels, mid_channels, MaskType.CHECKERBOARD, reverse_mask=True),
                       Coupling(in_channels, mid_channels, MaskType.CHECKERBOARD, reverse_mask=False),
                       Squeezing()]
            in_channels *= 4
            layers += [Coupling(in_channels, mid_channels, MaskType.CHANNEL_WISE, reverse_mask=False),
                       Coupling(in_channels, mid_channels, MaskType.CHANNEL_WISE, reverse_mask=True),
                       Coupling(in_channels, mid_channels, MaskType.CHANNEL_WISE, reverse_mask=False),
                       Splitting(scale)]
            in_channels //= 2

        # Get the last layer
        layers += [Coupling(in_channels, mid_channels, MaskType.CHECKERBOARD, reverse_mask=False),
                   Coupling(in_channels, mid_channels, MaskType.CHECKERBOARD, reverse_mask=True),
                   Coupling(in_channels, mid_channels, MaskType.CHECKERBOARD, reverse_mask=False),
                   Coupling(in_channels, mid_channels, MaskType.CHECKERBOARD, reverse_mask=True),
                   Splitting(num_scales - 1)]

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        if x.min() < 0 or x.max() > 1:
            raise ValueError('Expected 0 < x < 1, got x with min/max {}/{}'.format(x.min(), x.max()))

        # Dequantize the input image
        # See https://arxiv.org/abs/1511.01844, Section 3.1
        y = (x * 255. + torch.rand_like(x)) / 256.

        # Model density of logits, rather than y itself
        # See https://arxiv.org/abs/1605.08803, Section 4.1
        y = self.alpha * 0.5 + (1 - self.alpha) * y
        sldj = -(y.log() + (1 - y).log())
        sldj = sldj.view(sldj.size(0), -1).sum(-1)
        y = y.log() - (1 - y).log()

        # Apply forward flows
        z = None
        for layer in self.layers:
            y, sldj, z = layer.forward(y, sldj, z)

        z = torch.cat((z, y), dim=1)

        return z, sldj

    def backward(self, z):
        # Reshape z to match dimensions of final latent space
        if z.size(2) != z.size(3):
            raise ValueError('Expected z with height = width, got shape {}'.format(z.size()))
        side_length = int((z.size(1) * z.size(2) * z.size(3) // self.z_channels) ** 0.5)
        z = z.view(-1, self.z_channels, side_length, side_length)

        # Apply inverse flows
        y = None
        for layer in reversed(self.layers):
            y, z = layer.backward(y, z)

        # Convert from logits to normalized pixel values in (0, 1)
        x = torch.sigmoid(y)

        return x
