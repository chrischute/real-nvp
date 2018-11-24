import torch

from models.real_nvp.real_nvp_layer import RealNVPLayer


class Splitting(RealNVPLayer):
    """Split out half the features of the input and send them
    directly to the latent space.

    Args:
        scale (int): Index of the scale at which this `Splitting` layer occurs.
    """
    def __init__(self, scale):
        super(Splitting, self).__init__()
        self.scale = scale

    def forward(self, x, sldj, z):
        if x.size(1) % 2 != 0:
            raise ValueError('x must have an even number of features: {}'.format(x.size()))

        # Split in half along channel dimension
        new_z, y = x.chunk(2, dim=1)

        # Concatenate split dimensions, i.e., send directly to the latent space
        if z is None:
            z = new_z
        else:
            z = torch.cat((z, new_z), dim=1)

        return y, sldj, z

    def backward(self, y, z):
        # Get number of features to take back from z
        if y is None:
            num_take = z.size(1) // (2 ** self.scale)
        else:
            num_take = y.size(1)

        # Take back split features
        if num_take == z.size(1):
            new_y = z
            z = None
        else:
            z, new_y = z.split((z.size(1) - num_take, num_take), dim=1)

        # Add split features back to x
        if y is None:
            x = new_y
        else:
            x = torch.cat((new_y, y), dim=1)

        return x, z
