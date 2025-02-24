from typing import Optional, Union

from torch import Tensor
from torch.nn import functional as F

from .. import functional as function
from ..common_types import _size_1_t
from ..utils import _single
from ._compound1p import _Compound1p


class CompoundMinMaxPlusSumConv1d1p(_Compound1p):
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
        # we create new variables below to make mypy happy since kernel_size has
        # type Union[int, Tuple[int]] and kernel_size_ has type Tuple[int]
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
        self, input: Tensor, weight: Tensor, alpha: Tensor, bias: Optional[Tensor]
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
        temp = function.plus1d(input, weight)
        output1 = function.max1d(temp)
        output2 = function.min1d(temp)
        output = function.combine_sum_1d1p(output1, output2, alpha)
        output = function.channels_sum1d(output)
        if bias is not None:
            output += bias.view(1, -1, 1)
        return output

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.alpha, self.bias)
