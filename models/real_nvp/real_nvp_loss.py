import numpy as np
import torch.nn as nn


class RealNVPLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self):
        super(RealNVPLoss, self).__init__()

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.view(z.size(0), -1).sum(-1)
        ll = prior_ll + sldj
        loss = -ll.mean()

        return loss
