import numpy as np
import torch.nn as nn


class RealNVPLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803

    Args:
        cardinality: Cardinality of discrete probability distribution per dimension of x.
    """
    def __init__(self, cardinality=256):
        super(RealNVPLoss, self).__init__()
        self.cardinality = cardinality

    def forward(self, z, sldj):
        log_prob = self.log_prob_x(z, sldj, self.cardinality)
        loss = -log_prob.sum()

        return loss

    @staticmethod
    def log_prob_x(z, sldj, cardinality):
        """Get the log-probability of `x` given output `z` and sum of
        log-determinants of Jacobians `sldj`.

        Args:
            z: Latent variable outputs of the RealNVP model.
            sldj: Sum of log-determinants of Jacobians accumulated in forward pass.
            cardinality: Cardinality of discrete probability distribution per dimension of x.

        Returns:
            Log-probability of the input `x` which produced `z`.
        """
        # Get dimension of Gaussian
        k = z.size(1) * z.size(2) * z.size(3)

        # Get log-density assuming z ~ N(0, I)
        log_density_z = -0.5 * ((z ** 2).view(z.size(0), k).sum(-1) + k * np.log(2 * np.pi))
        log_density_x = log_density_z + sldj

        # Convert from density to probability
        log_prob_x = log_density_x - k * np.log(cardinality)

        return log_prob_x
