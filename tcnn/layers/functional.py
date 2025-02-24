from typing import Tuple

import torch
from torch import Tensor

import torch.nn.functional as F

"""
This file contains the functional implementations of the layers.
"""
"""
prepares the input and weight tensors for 1d matrices
"""


def preprocess1d(input: Tensor, weight: Tensor, stride: int):
    """
    Preprocesses the input and weight tensors for 1D convolution.

    Args:
        input (Tensor): The input tensor of shape (n, c, l_in).
        weight (Tensor): The weight tensor of shape (d, c, k).
        stride (int): The stride value for the convolution.

    Returns:
        Tuple[Tensor, Tensor]: The preprocessed input and weight tensors.
    """
    n, c, l_in = input.shape
    d, c, k = weight.shape
    input_transform = input.unfold(2, k, stride).unsqueeze(1)
    weight_transform = weight.unsqueeze(2).unsqueeze(0)
    return input_transform, weight_transform


def conv1d(input: Tensor, weight: Tensor):
    """
    Compute the 1D convolution between the input tensor and the weight tensor.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.

    Returns:
        Tensor: The result of the 1D convolution.
    """
    temp = input * weight
    # return reduce(temp, "n d c l i-> n d c l", "sum")
    return torch.sum(temp, dim=4)


def plus1d(input: Tensor, weight: Tensor):
    """
    Applies the plus operation to a 1D input tensor.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.

    Returns:
        Tensor: The result tensor after applying the plus operation.
    """
    return input + weight


def max1d(input: Tensor):
    """
    Applies a max pooling operation over the channel dimension of a 1D input tensor.

    Args:
        input (Tensor): The input tensor of shape (batch_size, num_channels, length).

    Returns:
        Tensor: The output tensor after applying max pooling operation over the channel dimension.
    """
    # return reduce(input, "n d c l i -> n d c l", "max")
    return torch.amax(input, dim=4)


def min1d(input: Tensor):
    """
    Compute the minimum value along the channel dimension of a 1D input tensor.

    Args:
        input (Tensor): The input tensor of shape (n, d, c, l).

    Returns:
        Tensor: The output tensor of shape (n, d, l) with the minimum value along the channel dimension.
    """
    # return reduce(input, "n d c l i -> n d c l", "min")
    return torch.amin(input, dim=4)


def max_plus1d(input: Tensor, weight: Tensor):
    """
    Applies the max-plus operation to a 1D input tensor.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.

    Returns:
        Tensor: The result tensor after applying the max-plus operation.
    """
    temp = input + weight
    # return reduce(temp, "n d c l i -> n d c l", "max")
    return torch.amax(temp, dim=4)


def min_plus1d(input: Tensor, weight: Tensor):
    """
    Applies the min-plus operation to a 1D input tensor and weight tensor.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.

    Returns:
        Tensor: The result tensor after applying the min-plus operation.
    """
    temp = input + weight
    # return reduce(temp, "n d c l i -> n d c l", "min")
    return torch.amin(temp, dim=4)


"""
operates for 1d channels
"""


def channels_sum1d(input: Tensor):
    """
    Sums the channels dimension of a 1D tensor.

    Args:
        input (Tensor): The input tensor of shape (n, d, c, l).

    Returns:
        Tensor: The output tensor of shape (n, d, l) where the channels dimension is summed.
    """
    # return reduce(input, "n d c l -> n d l", "sum")
    return input.sum(dim=2)


def channels_max1d(input: Tensor):
    """
    Applies a max pooling operation over the channel dimension of a 1D input tensor.

    Args:
        input (Tensor): The input tensor of shape (batch_size, num_channels, length).

    Returns:
        Tensor: The output tensor after applying max pooling operation over the channel dimension.
    """
    return input.max(dim=2)[0]


def channels_min1d(input: Tensor):
    """
    Compute the minimum value along the channel dimension of a 1D input tensor.

    Args:
        input (Tensor): The input tensor of shape (n, d, c, l).

    Returns:
        Tensor: The output tensor of shape (n, d, l) with the minimum value along the channel dimension.
    """
    return input.min(dim=2)[0]


"""
prepares the input and weight tensors for 2d matrices
"""
def unfold(x, kernel_size, stride=1, padding=0):
    """
    Unfolds the input tensor for 2D convolution.
    """
    B, C, H, W = x.shape
    kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    weight = torch.zeros(C * kH * kW, C, kH, kW, device=x.device)
    for i in range(C):
        for kh in range(kH):
            for kw in range(kW):
                idx = i * kH * kW + kh * kW + kw
                weight[idx, i, kh, kw] = 1.0
    y = F.conv2d(x, weight, stride=stride, padding=padding)
    return y.flatten(2)

def preprocess2d(inputs: Tensor, weight: Tensor, stride: Tuple[int, int]):
    """
    Preprocesses the input and weight tensors for 2D convolution.

    Args:
        input (Tensor): The input tensor of shape (n, c, h_in, w_in).
        weight (Tensor): The weight tensor of shape (d, c, k, j).
        stride (Tuple[int, int]): The stride values for the convolution.

    Returns:
        Tuple[Tensor, Tensor]: The preprocessed input and weight tensors.
    """
    n, c, h_in, w_in = inputs.shape
    d, c, k, j = weight.shape
    h_out = (h_in - k) // stride[0] + 1
    w_out = (w_in - j) // stride[1] + 1

    # equal to the following code
    inputs = inputs.unfold(2, k, stride[0]).unfold(3, j, stride[1]).unsqueeze(1)
    # inputs = unfold(inputs, kernel_size=(k, j), stride=stride).unsqueeze(1)
    # inputs = inputs.view(n, c, k, j, h_out, w_out)
    # inputs = inputs.permute(0, 1, 4, 5, 2, 3).unsqueeze(1)  
    weight = weight.unsqueeze(2).unsqueeze(3).unsqueeze(0)
    return inputs, weight


"""
operates for 2d matrices
"""


def conv2d(input: Tensor, weight: Tensor):
    """
    Compute the 2D convolution between the input tensor and the weight tensor.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.

    Returns:
        Tensor: The result of the 2D convolution.
    """
    temp = input * weight
    # return reduce(temp, "n d c h w i j -> n d c h w", "sum")
    return torch.sum(temp, dim=(5, 6))


def plus2d(input: Tensor, weight: Tensor):
    """
    Applies the plus operation to a 2D input tensor.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.

    Returns:
        Tensor: The result tensor after applying the plus operation.
    """
    return input + weight


def max2d(input: Tensor):
    """
    Applies a max pooling operation over the channel dimension of a 2D input tensor.

    Args:
        input (Tensor): The input tensor of shape (batch_size, num_channels, height, width).

    Returns:
        Tensor: The output tensor after applying max pooling operation over the channel dimension.
    """
    # return reduce(input, "n d c h w i j -> n d c h w", "max")
    return torch.amax(input, dim=(5, 6))


def min2d(input: Tensor):
    """
    Compute the minimum value along the channel dimension of a 2D input tensor.

    Args:
        input (Tensor): The input tensor of shape (n, d, c, h, w).

    Returns:
        Tensor: The output tensor of shape (n, d, h, w) with the minimum value along the channel dimension.
    """
    # return reduce(input, "n d c h w i j -> n d c h w", "min")
    return torch.amin(input, dim=(5, 6))


def max_plus2d(input: Tensor, weight: Tensor):
    """
    Applies the max-plus operation to a 2D input tensor.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.

    Returns:
        Tensor: The result tensor after applying the max-plus operation.
    """
    temp = input + weight
    # return reduce(temp, "n d c h w i j -> n d c h w", "max")
    return torch.amax(temp, dim=(5, 6))


def min_plus2d(input: Tensor, weight: Tensor):
    """
    Applies the min-plus operation to a 2D input tensor and weight tensor.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.

    Returns:
        Tensor: The result tensor after applying the min-plus operation.
    """
    temp = input + weight
    # return reduce(temp, "n d c h w i j -> n d c h w", "min")
    return torch.amin(temp, dim=(5, 6))


"""
operates for 2d channels
"""


def channels_sum2d(input: Tensor):
    """
    Sums the channels dimension of a 2D tensor.

    Args:
        input (Tensor): The input tensor of shape (n, d, c, h, w).

    Returns:
        Tensor: The output tensor of shape (n, d, h, w) where the channels dimension is summed.
    """
    # return reduce(input, "n d c h w -> n d h w", "sum")
    return input.sum(dim=2)


def channels_max2d(input: Tensor):
    """
    Applies a max pooling operation over the channel dimension of a 4D input tensor.

    Args:
        input (Tensor): The input tensor of shape (batch_size, num_channels, height, width).

    Returns:
        Tensor: The output tensor after applying max pooling operation over the channel dimension.
    """
    return input.max(dim=2)[0]


def channels_min2d(input: Tensor):
    """
    Compute the minimum value along the channel dimension of a 2D input tensor.

    Args:
        input (Tensor): The input tensor of shape (n, d, c, h, w).

    Returns:
        Tensor: The output tensor of shape (n, d, h, w) with the minimum value along the channel dimension.
    """
    return input.min(dim=2)[0]


"""
operates for combined 2d matrices
"""


"""
prepares the input and weight tensors for 3d matrices
"""


def preprocess3d(input: Tensor, weight: Tensor, stride: Tuple[int, int, int]):
    """
    Preprocesses the input and weight tensors for 3D convolution.

    Args:
        input (Tensor): The input tensor of shape (n, c, d_in, h_in, w_in).
        weight (Tensor): The weight tensor of shape (d, c, k, j, i).
        stride (Tuple[int, int, int]): The stride values for the convolution.

    Returns:
        Tuple[Tensor, Tensor]: The preprocessed input and weight tensors.
    """
    n, c, d_in, h_in, w_in = input.shape
    d, c, k, j, i = weight.shape
    input = (
        input.unfold(2, k, stride[0])
        .unfold(3, j, stride[1])
        .unfold(4, i, stride[2])
        .unsqueeze(1)
    )
    weight = weight.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(0)
    return input, weight


"""
operates for 3d matrices
"""


def conv3d(input: Tensor, weight: Tensor):
    """
    Compute the 3D convolution between the input tensor and the weight tensor.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.

    Returns:
        Tensor: The result of the 3D convolution.
    """
    temp = input * weight
    # return reduce(temp, "n d c depth h w i j k -> n d c depth h w", "sum")
    return torch.sum(temp, dim=(6, 7, 8))


def plus3d(input: Tensor, weight: Tensor):
    """
    Applies the plus operation to a 3D input tensor.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.

    Returns:
        Tensor: The result tensor after applying the plus operation.
    """
    return input + weight


def max3d(input: Tensor):
    """
    Applies a max pooling operation over the channel dimension of a 3D input tensor.

    Args:
        input (Tensor): The input tensor of shape (batch_size, num_channels, depth, height, width).

    Returns:
        Tensor: The output tensor after applying max pooling operation over the channel dimension.
    """
    # return reduce(input, "n d c depth h w i j k -> n d c depth h w", "max")
    return torch.amax(input, dim=(6, 7, 8))


def min3d(input: Tensor):
    """
    Compute the minimum value along the channel dimension of a 3D input tensor.

    Args:
        input (Tensor): The input tensor of shape (n, d, c, depth, h, w).

    Returns:
        Tensor: The output tensor of shape (n, d, depth, h, w) with the minimum value along the channel dimension.
    """
    # return reduce(input, "n d c depth h w i j k -> n d c depth h w", "min")
    return torch.amin(input, dim=(6, 7, 8))


def max_plus3d(input: Tensor, weight: Tensor):
    """
    Applies the max-plus operation to a 3D input tensor.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.

    Returns:
        Tensor: The result tensor after applying the max-plus operation.
    """
    temp = input + weight
    # return reduce(temp, "n d c depth h w i j k -> n d c depth h w", "max")
    return torch.amax(temp, dim=(6, 7, 8))


def min_plus3d(input: Tensor, weight: Tensor):
    """
    Applies the min-plus operation to a 3D input tensor and weight tensor.

    Args:
        input (Tensor): The input tensor.
        weight (Tensor): The weight tensor.

    Returns:
        Tensor: The result tensor after applying the min-plus operation.
    """
    temp = input + weight
    # return reduce(temp, "n d c depth h w i j k -> n d c depth h w", "min")
    return torch.amin(temp, dim=(6, 7, 8))


"""
operates for 3d channels
"""


def channels_sum3d(input: Tensor):
    """
    Sums the channels dimension of a 3D tensor.

    Args:
        input (Tensor): The input tensor of shape (n, d, c, depth, h, w).

    Returns:
        Tensor: The output tensor of shape (n, d, depth, h, w) where the channels dimension is summed.
    """
    # return reduce(input, "n d c depth h w -> n d depth h w", "sum")
    return input.sum(dim=2)


def channels_max3d(input: Tensor):
    """
    Applies a max pooling operation over the channel dimension of a 3D input tensor.

    Args:
        input (Tensor): The input tensor of shape (batch_size, num_channels, depth, height, width).

    Returns:
        Tensor: The output tensor after applying max pooling operation over the channel dimension.
    """
    return input.max(dim=2)[0]


def channels_min3d(input: Tensor):
    """
    Compute the minimum value along the channel dimension of a 3D input tensor.

    Args:
        input (Tensor): The input tensor of shape (n, d, c, depth, h, w).

    Returns:
        Tensor: The output tensor of shape (n, d, depth, h, w) with the minimum value along the channel dimension.
    """
    return input.min(dim=2)[0]


def combine_sum_3d1p(input1: Tensor, input2: Tensor, alpha: Tensor):
    """
    Computes the combined sum of two 3D tensors with a parameterized weight.

    Args:
        input1 (Tensor): The first input tensor.
        input2 (Tensor): The second input tensor.
        alpha (Tensor): The weight tensor.

    Returns:
        Tensor: The combined sum of input1 and input2, weighted by alpha.

    """
    return (
        alpha[None, :, :, None, None, None] * input1
        + (1 - alpha[None, :, :, None, None, None]) * input2
    )


def combine_sum_3d2p(input1: Tensor, input2: Tensor, alpha: Tensor, beta: Tensor):
    """
    Computes the combined sum of two tensors.

    Args:
        input1 (Tensor): The first input tensor.
        input2 (Tensor): The second input tensor.
        alpha (Tensor): The weight tensor for input1.
        beta (Tensor): The weight tensor for input2.

    Returns:
        Tensor: The combined sum of input1 and input2, weighted by alpha and beta.

    """
    return (
        alpha[None, :, :, None, None, None] * input1
        + beta[None, :, :, None, None, None] * input2
    )


def combine_sum_2d1p(input1: Tensor, input2: Tensor, alpha: Tensor):
    """
    Computes the combined sum of two 2D tensors with a parameterized weight.

    Args:
        input1 (Tensor): The first input tensor.
        input2 (Tensor): The second input tensor.
        alpha (Tensor): The weight tensor.

    Returns:
        Tensor: The combined sum of input1 and input2, weighted by alpha.

    """
    return (
        alpha[None, :, :, None, None] * input1
        + (1 - alpha[None, :, :, None, None]) * input2
    )


def combine_sum_2d2p(input1: Tensor, input2: Tensor, alpha: Tensor, beta: Tensor):
    """
    Computes the combined sum of two tensors.

    Args:
        input1 (Tensor): The first input tensor.
        input2 (Tensor): The second input tensor.
        alpha (Tensor): The weight tensor for input1.
        beta (Tensor): The weight tensor for input2.

    Returns:
        Tensor: The combined sum of input1 and input2, weighted by alpha and beta.

    """
    return (
        alpha[None, :, :, None, None] * input1 + beta[None, :, :, None, None] * input2
    )


"""
operates for combined 1d matrices
"""


def combine_sum_1d1p(input1: Tensor, input2: Tensor, alpha: Tensor):
    """
    Computes the combined sum of two 2D tensors with a parameterized weight.

    Args:
        input1 (Tensor): The first input tensor.
        input2 (Tensor): The second input tensor.
        alpha (Tensor): The weight tensor.

    Returns:
        Tensor: The combined sum of input1 and input2, weighted by alpha.

    """
    return alpha[None, :, :, None] * input1 + (1 - alpha[None, :, :, None]) * input2


def combine_sum_1d2p(input1: Tensor, input2: Tensor, alpha: Tensor, beta: Tensor):
    """
    Computes the compound sum of two tensors.

    Args:
        input1 (Tensor): The first input tensor.
        input2 (Tensor): The second input tensor.
        alpha (Tensor): The weight tensor for input1.
        beta (Tensor): The weight tensor for input2.

    Returns:
        Tensor: The combined sum of input1 and input2, weighted by alpha and beta.

    """
    return alpha[None, :, :, None] * input1 + beta[None, :, :, None] * input2


"""
"""


def unsqueeze(input: Tensor, weight: Tensor) -> Tensor:
    r"""
    Unsqueeze input and weight tensors to have the same number of dimensions.

    Args:
        input: input tensor
        weight: weight tensor

    Returns:
        Tuple of unsqueezed input and weight tensors

    Example:
        >>> input = torch.randn(128, 20)
        >>> weight = torch.randn(30, 20)
        >>> input_expanded, weight_expanded = _unsqueeze(input, weight)
        >>> print(input_expanded.size())
        torch.Size([128, 20, 1])
        >>> print(weight_expanded.size())
        torch.Size([1, 30, 20])

    Raises:
        ValueError: if the second dimension of the input tensor and the first dimension of the weight tensor don't match
    """
    if input.size(1) == weight.size(1):
        return input.unsqueeze(2), weight.t().unsqueeze(0)
    else:
        raise ValueError(
            f"Input size {input.size(1)} and weight size {weight.size(1)} don't match"
        )


def maxplus(input: Tensor, weight: Tensor) -> Tensor:
    r"""
    Applies a maxplus transformation to the incoming data: :math:`y = max(x + A^T) + b`.

    Args:
        input: input tensor
        weight: weight tensor

    Returns:
        Output tensor

    Example:
        >>> input = torch.randn(128, 20, 1)
        >>> weight = torch.randn(1, 30, 20)
        >>> output = maxplus(input, weight)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    result = input + weight
    return result.max(dim=1).values


def minplus(input: Tensor, weight: Tensor) -> Tensor:
    r"""
    Applies a minplus transformation to the incoming data: :math:`y = min(x + A^T) + b`.

    Args:
        input: input tensor
        weight: weight tensor

    Returns:
        Output tensor

    Example:
        >>> input = torch.randn(128, 20, 1)
        >>> weight = torch.randn(1, 30, 20)
        >>> output = minplus(input, weight)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    result = input + weight
    return result.min(dim=1).values


    

