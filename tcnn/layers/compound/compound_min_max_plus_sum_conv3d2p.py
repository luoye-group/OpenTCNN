from typing import Optional, Union

from einops import reduce
from torch import Tensor
from torch.nn import functional as F

from .. import functional as function
from ..common_types import _size_3_t
from ..utils import _triple
from ._compound2p import _Compound2p


class CompoundMinMaxPlusSumConv3d2p(_Compound2p):

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
        weight: Tensor,
        alpha: Tensor,
        beta: Tensor,
        bias: Optional[Tensor],
    ):
        if self.padding_mode != "zeros":
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
        else:
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode="constant", value=0
            )
        input, weight = function.preprocess3d(input, weight, self.stride)  # type: ignore
        temp = function.plus3d(input, weight)
        output1 = function.max3d(temp)
        output2 = function.min3d(temp)
        output = function.combine_sum_3d2p(output1, output2, alpha, beta)
        output = function.channels_sum3d(output)

        if bias is not None:
            output += bias.view(1, -1, 1, 1, 1)
        return output

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.alpha, self.beta, self.bias)
