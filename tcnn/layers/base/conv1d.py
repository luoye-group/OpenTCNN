from typing import Optional, Union

from torch import Tensor
from torch.nn import functional as F

from .. import functional as function
from ..common_types import _size_1_t
from ..utils import _single
from ._conv import _Conv


class Conv1d(_Conv):
    """
    Implementation of a one-dimensional convolutional layer.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or Tuple[int]): Size of the convolutional kernel.
        stride (int or Tuple[int], optional): Stride of the convolution. Default is 1.
        padding (int or Tuple[int] or str, optional): Padding size of the input. Default is 0.
        dilation (int or Tuple[int], optional): Dilation size of the convolutional kernel. Default is 1.
        groups (int, optional): Connection pattern between input and output channels. Default is 1.
        bias (bool, optional): Whether to use bias. Default is True.
        padding_mode (str, optional): Padding mode. Default is 'zeros'.
        device (str or torch.device, optional): Device where the tensor is located. Default is None.
        dtype (torch.dtype, optional): Data type of the tensor. Default is None.

    Attributes:
        weight (Tensor): Weight tensor of the convolutional kernel.
        bias (Tensor): Bias tensor.

    Methods:
        forward(input: Tensor) -> Tensor:
            Forward propagation function that computes the output tensor of the convolutional layer.

    Returns:
        torch.Tensor: Output tensor after applying the convolution operation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _single(kernel_size)
        stride_ = _single(stride)
        padding_ = padding if isinstance(padding, str) else _single(padding)
        dilation_ = _single(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )

    def _conv_forward(
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        if self.padding_mode != "zeros":
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
        else:
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode="constant", value=0
            )

        input, weight = function.preprocess1d(input, weight, self.stride[0])
        output = function.conv1d(input, weight)
        output = function.channels_sum1d(output)
        if bias is not None:
            output += bias.view(1, -1, 1)
        return output

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
