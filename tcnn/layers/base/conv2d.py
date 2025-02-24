from typing import Optional, Union

from einops import reduce
from torch import Tensor
from torch.nn import functional as F

from .. import functional as function
from ..common_types import _size_1_t
from ..utils import _pair
from ._conv import _Conv


class Conv2d(_Conv):
    """
    2D convolutional layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (Union[int, Tuple[int, int]]): Size of the convolutional kernel.
        stride (Union[int, Tuple[int, int]], optional): Stride of the convolution. Defaults to 1.
        padding (Union[int, Tuple[int, int]] or str, optional): Padding size or padding mode. Defaults to 0.
        dilation (Union[int, Tuple[int, int]], optional): Dilation rate. Defaults to 1.
        groups (int, optional): Number of groups for grouped convolution. Defaults to 1.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
        padding_mode (str, optional): Padding mode. Defaults to 'zeros'.
        device (str or torch.device, optional): Device for the layer's parameters. Defaults to None.
        dtype (torch.dtype, optional): Data type for the layer's parameters. Defaults to None.

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
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,  # type: ignore
            False,
            _pair(0),
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
        # ---solution---: fix the shape of input to compatible with the onnx
        # shape = input.shape
        # shape = [int (i) for i in shape]
        # input = input.reshape(shape)
        # ---solution---
        input, weight = function.preprocess2d(input, weight, self.stride)  # type: ignore
        output = function.conv2d(input, weight)
        output = function.channels_sum2d(output)

        if bias is not None:
            output += bias.view(1, -1, 1, 1)
        return output

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
