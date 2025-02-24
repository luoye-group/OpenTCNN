from typing import Optional, Union

from einops import reduce
from torch import Tensor
from torch.nn import functional as F

from .. import functional as function
from ..common_types import _size_3_t
from ..utils import _triple
from ._parallel import _Parallel


class ConstantParallelMinMaxPlusSumConv3d(_Parallel):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = padding if isinstance(padding, str) else _triple(padding)
        dilation_ = _triple(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _triple(0),
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
        input1, weight1 = function.preprocess3d(input, weight1, self.stride)  # type: ignore
        input2, weight2 = function.preprocess3d(input, weight2, self.stride)  # type: ignore
        output1 = function.max_plus3d(input1, weight1)
        output2 = function.min_plus3d(input2, weight2)
        output = output1 + output2
        output = function.channels_sum3d(output)

        if bias is not None:
            output += bias.view(1, -1, 1, 1, 1)
        return output

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(
            input, self.weight1, self.weight2, self.bias
        )
