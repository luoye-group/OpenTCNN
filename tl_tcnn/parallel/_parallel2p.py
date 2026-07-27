import math
from typing import Tuple

import torch
from torch.nn import Module, init
from torch.nn.parameter import Parameter

from ..utils import _reverse_repeat_tuple

"""
The implementation of _Conv is copied from torch.nn.modules.conv._ConvNd
url: https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/conv.py#L46
"""


class _Parallel2p(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        transposed: bool,
        output_padding: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:
        # factory_kwargs = {"device": device, "dtype": dtype}
        factory_kwargs = {}
        if device is not None and not isinstance(device, type):
            factory_kwargs["device"] = device
        if dtype is not None and not isinstance(dtype, type):
            factory_kwargs["dtype"] = dtype
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}"
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )

        if transposed:
            self.weight1 = Parameter(
                torch.empty(
                    (in_channels, out_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    (in_channels, out_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        else:
            self.weight1 = Parameter(
                torch.empty(
                    (out_channels, in_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    (out_channels, in_channels // groups, *kernel_size),
                    **factory_kwargs,
                )
            )
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        # add our settings
        in_channels_per_group = in_channels // groups
        self.alpha = Parameter(
            torch.empty((out_channels, in_channels_per_group), **factory_kwargs)
        )
        self.beta = Parameter(
            torch.empty((out_channels, in_channels_per_group), **factory_kwargs)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
        # inti our parameters
        init.normal_(self.alpha, mean=0.0, std=1.0)
        init.normal_(self.beta, mean=0.0, std=1.0)
        # init.normal_(self.alpha, mean=0.0, std=0.1)
        # init.normal_(self.beta, mean=0.0, std=0.1)
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight1)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def reset_parameters_1(self):
        fan_in = self.in_channels // self.groups * self.kernel_size[0] * self.kernel_size[1]
        fan_out = self.out_channels // self.groups * self.kernel_size[0] * self.kernel_size[1]
        
        # For MaxPlus2d: Use Gumbel distribution (max-stable)
        # Gumbel(loc=0, scale=beta) where beta = sqrt(2/pi) * sigma for variance matching
        # Or simpler: scale based on Xavier/Kaiming logic
        scale_max = 1.0 / math.sqrt(fan_in)  # Kaiming-like scale
        
        # For MinPlus2d: Use negative Gumbel or Gumbel with negative mean
        # Or use reversed Gumbel: -Gumbel() 
        
        # Gumbel parameters: loc=0, scale=beta
        # Sampling: loc - scale * log(-log(Uniform(0,1)))
        
        # Implementation for MaxPlus weights
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.weight1) + 1e-8))
        self.weight1.data = scale_max * gumbel_noise
        
        # For MinPlus: use negative Gumbel or Logistic
        gumbel_noise2 = -torch.log(-torch.log(torch.rand_like(self.weight2) + 1e-8))
        self.weight2.data = -scale_max * gumbel_noise2  # Negative Gumbel for min
        
        # Alternative: Laplace distribution (also max-stable with different tails)
        # laplace_noise = torch.distributions.Laplace(0, scale_max).sample(self.weight1.shape)
        # self.weight1.data = laplace_noise
        
        # Alpha/Beta: Kaiming Normal as you have is fine
        # But consider: alpha and beta should be positive if you want convex combination
        # Otherwise, they can learn any mixing
        
        alpha_std = 1.0 / math.sqrt(fan_in)
        init.normal_(self.alpha, mean=0, std=alpha_std) 
        init.normal_(self.beta, mean=0, std=alpha_std)
        
        # # Initialise alpha and beta to 1.0
        # init.ones_(self.alpha)
        # init.ones_(self.beta)
        
        # Or use positive-constrained initialization
        # init.uniform_(self.alpha, 0.0, 1.0)
        # init.uniform_(self.beta, 0.0, 1.0)
        
        if self.bias is not None:
            init.zeros_(self.bias)
        

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"
