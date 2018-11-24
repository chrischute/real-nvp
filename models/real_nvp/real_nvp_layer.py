import torch.nn as nn


class RealNVPLayer(nn.Module):
    def __init__(self):
        super(RealNVPLayer, self).__init__()

    def forward(self, x, sldj, z):
        """Forward pass of a RealNVP layer.

        Args:
            x (tensor): Input features.
            sldj (tensor): Sum of log determinants of Jacobians.
            z (tensor): Input latent variables.

        Returns:
            y (tensor): Output features.
            sldj (tensor): Updated sum of log determinants of Jacobians.
            z (tensor): Updated latent variables.
        """
        raise NotImplementedError('Subclass of BaseLayer must implement forward.')

    def backward(self, y, z):
        """Backward pass of a RealNVP layer.

        Args:
            y (tensor): Output features.
            z (tensor): Output latent variables.

        Returns:
            x (tensor): Input features.
            z (tensor): Input latent variables.
        """
        raise NotImplementedError('Subclass of BaseLayer must implement backward.')
