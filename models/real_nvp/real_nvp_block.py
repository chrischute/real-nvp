import torch
import torch.nn as nn

from models.real_nvp.coupling import Coupling, MaskType
from util import squeeze_2x2


class RealNVPBlock(nn.Module):
    """Block (i.e., one scale) in a RealNVP model.

    Called recursively to construct `RealNVP`.

    Args:
        scale_idx (int): Index of current scale.
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
    """
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super(RealNVPBlock, self).__init__()

        self.mid_channels = mid_channels * 2 ** scale_idx
        self.is_last_block = scale_idx == num_scales - 1

        self.in_couplings = nn.ModuleList([
            Coupling(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False),
            Coupling(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True),
            Coupling(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=False)
        ])

        if self.is_last_block:
            self.out_coupling = Coupling(in_channels, mid_channels, num_blocks, MaskType.CHECKERBOARD, reverse_mask=True)
        else:
            self.out_couplings = nn.ModuleList([
                Coupling(in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False),
                Coupling(in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=True),
                Coupling(in_channels, 2 * mid_channels, num_blocks, MaskType.CHANNEL_WISE, reverse_mask=False)
            ])
            self.next_block = RealNVPBlock(scale_idx + 1, num_scales, in_channels, mid_channels, num_blocks)

    def forward(self, x, sldj, reverse=False):

        if reverse:
            for coupling in reversed(self.out_couplings):
                x, sldj = coupling(x, sldj, reverse)

            # TODO: STOPPED HERE
        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj)

            if self.is_last_block:
                x, sldj = self.out_coupling(x, sldj)
            else:
                # Squeeze, 3x channel-wise coupling, unsqueeze
                x = squeeze_2x2(x, reverse=False)
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj)
                x = squeeze_2x2(x, reverse=True)

                # Ordered squeeze, next block, ordered unsqueeze
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = torch.split(x, 2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat(x, x_split, dim=1)

        return x, sldj
