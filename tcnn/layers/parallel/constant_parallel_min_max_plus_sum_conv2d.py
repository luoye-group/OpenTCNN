from typing import Optional, Union

from einops import reduce
from torch import Tensor
from torch.nn import functional as F

from .. import functional as function
from ..common_types import _size_2_t
from ..utils import _pair
from ._parallel import _Parallel


class ConstantParallelMinMaxPlusSumConv2d(_Parallel):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
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
        self,
        input: Tensor,
        weight1: Tensor,
        weight2: Tensor,
        bias: Optional[Tensor],
    ) -> Tensor:
        if self.padding_mode != "zeros":
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
        else:
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode="constant", value=0
            )
        input1, weight1 = function.preprocess2d(input, weight1, self.stride)  # type: ignore
        input2, weight2 = function.preprocess2d(input, weight2, self.stride)  # type: ignore
        output1 = function.max_plus2d(input1, weight1)
        output2 = function.min_plus2d(input2, weight2)
        output = output1 + output2
        output = function.channels_sum2d(output)

        if bias is not None:
            output += bias.view(1, -1, 1, 1)
        return output

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(
            input, self.weight1, self.weight2, self.bias
        )
