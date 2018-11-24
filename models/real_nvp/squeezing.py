import util

from models.real_nvp.real_nvp_layer import RealNVPLayer


class Squeezing(RealNVPLayer):
    """Trade spatial size for number of channels.

    For each channel, divide the input into sub-squares of shape `2x2xC`,
    then reshape them into sub-squares of shape `1x1x4C`.

    See Also:
        - Figure 3 of RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self):
        super(Squeezing, self).__init__()

    def forward(self, x, sldj, z=None):
        if x.size(2) % 2 != 0 or x.size(3) % 2 != 0:
            raise ValueError('x height/width must be even: {}'.format(x.size()))

        y = util.space_to_depth(x, 2)
        if z is not None:
            z = util.space_to_depth(z, 2)

        return y, sldj, z

    def backward(self, y, z):
        if y.size(1) % 4 != 0:
            raise ValueError('y channels must be divisible by 4: {}'.format(y.size()))

        x = util.depth_to_space(y, 2)
        if z is not None:
            z = util.depth_to_space(z, 2)

        return x, z
