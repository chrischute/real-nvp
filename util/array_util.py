import torch


def depth_to_space(x, block_size):
    """For each spatial position, a sub-volume of shape `1x1x(N^2 * C)`,
    reshape into a sub-volume of shape `NxNxC`, where `N = block_size`.

    Adapted from: https://gist.github.com/jalola/f41278bb27447bed9cd3fb48ec142aec

    See Also:
        - TensorFlow nn.depth_to_space: https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
        - Figure 3 of RealNVP paper: https://arxiv.org/abs/1605.08803

    Args:
        x (tensor): Input tensor of shape (B, C, H, W).
        block_size (int): Side length of each spatial block.
    """
    output = x.permute(0, 2, 3, 1)
    (batch_size, d_height, d_width, d_depth) = output.size()
    s_depth = int(d_depth / block_size ** 2)
    s_width = int(d_width * block_size)
    s_height = int(d_height * block_size)
    t_1 = output.reshape(batch_size, d_height, d_width, block_size ** 2, s_depth)
    spl = t_1.split(block_size, 3)
    stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
    output = torch.stack(stack, 0)
    output = output.transpose(0, 1)
    output = output.permute(0, 2, 1, 3, 4)
    output = output.reshape(batch_size, s_height, s_width, s_depth)
    output = output.permute(0, 3, 1, 2)
    return output


def space_to_depth(x, block_size):
    """For each channel, divide the input into sub-squares of shape `NxNxC`,
    then reshape them into sub-squares of shape `1x1x(N^2 * C)`, where `N = block_size`.

    Adapted from: https://gist.github.com/jalola/f41278bb27447bed9cd3fb48ec142aec

    See Also:
        - TensorFlow nn.space_to_depth: https://www.tensorflow.org/api_docs/python/tf/nn/space_to_depth
        - Figure 3 of RealNVP paper: https://arxiv.org/abs/1605.08803

    Args:
        x (tensor): Input tensor of shape (B, C, H, W).
        block_size (int): Size of each spatial block.
    """
    output = x.permute(0, 2, 3, 1)
    (batch_size, s_height, s_width, s_depth) = output.size()
    d_depth = s_depth * block_size ** 2
    d_height = int(s_height / block_size)
    t_1 = output.split(block_size, 2)
    stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
    output = torch.stack(stack, 1)
    output = output.permute(0, 2, 1, 3)
    output = output.permute(0, 3, 1, 2)
    return output


def checkerboard_mask(height, width, reverse=False, dtype=torch.float32,
                      device=None, requires_grad=False):
    """Get a checkerboard mask, such that no two entries adjacent entries
    have the same value. In non-reversed mask, top-left entry is 0.

    Args:
        height (int): Number of rows in the mask.
        width (int): Number of columns in the mask.
        reverse (bool): If True, reverse the mask (i.e., make top-left entry 1).
            Useful for alternating masks in RealNVP.
        dtype (torch.dtype): Data type of the tensor.
        device (torch.device): Device on which to construct the tensor.
        requires_grad (bool): Whether the tensor requires gradient.


    Returns:
        mask (torch.tensor): Checkerboard mask of shape (1, 1, height, width).
    """
    checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
    mask = torch.tensor(checkerboard, dtype=dtype, device=device, requires_grad=requires_grad)

    if reverse:
        mask = 1 - mask

    # Reshape to (1, 1, height, width) for broadcasting with tensors of shape (B, C, H, W)
    mask = mask.view(1, 1, height, width)

    return mask


def channel_wise_mask(num_channels, reverse=False, dtype=torch.float32,
                      device=None, requires_grad=False):
    """Get a channel-wise mask. In non-reversed mask, first N/2 channels are 0,
    and last N/2 channels are 1.

    Args:
        num_channels (int): Number of channels in the mask.
        reverse (bool): If True, reverse the mask (i.e., make first N/2 channels 1).
            Useful for alternating masks in RealNVP.
        dtype (torch.dtype): Data type of the tensor.
        device (torch.device): Device on which to construct the tensor.
        requires_grad (bool): Whether the tensor requires gradient.

    Returns:
        mask (torch.tensor): channel-wise mask of shape (1, num_channels, 1, 1).
    """
    half_channels = num_channels // 2
    channel_wise = [int(i < half_channels) for i in range(num_channels)]
    mask = torch.tensor(channel_wise, dtype=dtype, device=device, requires_grad=requires_grad)

    if reverse:
        mask = 1 - mask

    # Reshape to (1, num_channels, 1, 1) for broadcasting with tensors of shape (B, C, H, W)
    mask = mask.view(1, num_channels, 1, 1)

    return mask
