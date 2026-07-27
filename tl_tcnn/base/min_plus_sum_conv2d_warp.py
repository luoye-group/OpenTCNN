"""Warp for MinPlusSumConv2d.

Forward uses `min_plus_sum_conv2d_kernel` (NCHW input, OIHW weight, NCHW output).
Backward uses split kernels: dweight + dinput.
"""

from typing import Optional, Union
from functools import lru_cache

import torch
from torch import Tensor, autograd, zeros, zeros_like
from torch.nn import functional as F

from ..common_types import _size_2_t
from ..utils import _pair, _kernel_cache_key
from ._conv import _Conv
from .min_plus_sum_conv2d import (
    min_plus_sum_conv2d_kernel,
    min_plus_sum_conv2d_kernel_backward_dweight,
    min_plus_sum_conv2d_kernel_backward_dinput,
)
from tilelang.autotuner import set_autotune_inputs


@lru_cache(maxsize=1)
def _compile_forward_kernel_cached(*key):
    return min_plus_sum_conv2d_kernel(*key)


@lru_cache(maxsize=1)
def _compile_backward_kernel_dweight_cached(*key, maxmin_gradient="multiple"):
    return min_plus_sum_conv2d_kernel_backward_dweight(*key, maxmin_gradient=maxmin_gradient)


@lru_cache(maxsize=1)
def _compile_backward_kernel_dinput_cached(*key, maxmin_gradient="multiple"):
    return min_plus_sum_conv2d_kernel_backward_dinput(*key, maxmin_gradient=maxmin_gradient)


def clear_min_plus_sum_conv2d_kernel_cache():
    """Clear Python-side compilation caches for MinPlusSumConv2d kernels."""
    _compile_forward_kernel_cached.cache_clear()
    _compile_backward_kernel_dweight_cached.cache_clear()
    _compile_backward_kernel_dinput_cached.cache_clear()


class _MinPlusSumConv2d(autograd.Function):
    """Autograd function for the MinPlusSumConv2d operator."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        conv_module,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        kernel_height,
        kernel_width,
        out_height,
        out_width,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        dtype="float32",
        int_dtype_str="int32",
        autotune=False,
        maxmin_gradient="multiple",
    ):
        input_nchw = input.contiguous()
        weight_oihw = weight.contiguous()

        output = torch.zeros(
            (batch_size, out_channels, out_height, out_width),
            device=input.device,
            dtype=input.dtype,
        )

        conv_module._run_forward_kernel(input_nchw, weight_oihw, output, out_height, out_width)

        ctx.save_for_backward(input_nchw, weight_oihw)
        ctx.meta = (
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
            kernel_height,
            kernel_width,
            out_height,
            out_width,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            dtype,
            int_dtype_str,
            autotune,
            maxmin_gradient,
        )
        ctx.conv_module = conv_module
        return output.contiguous()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input_nchw, weight_oihw = ctx.saved_tensors
        (
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
            kernel_height,
            kernel_width,
            out_height,
            out_width,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            dtype,
            int_dtype_str,
            autotune,
            maxmin_gradient,
        ) = ctx.meta

        conv_module = ctx.conv_module
        grad_output_nchw = grad_output.contiguous()
        grad_input_nchw = zeros_like(input_nchw)
        grad_weight_oihw = zeros_like(weight_oihw)

        conv_module._run_backward_kernel(
            input_nchw,
            weight_oihw,
            grad_output_nchw,
            grad_input_nchw,
            grad_weight_oihw,
            out_height,
            out_width,
            maxmin_gradient,
        )

        return (
            grad_input_nchw,
            grad_weight_oihw,
        ) + (None,) * (1 + len(ctx.meta))


class MinPlusSumConv2d(_Conv):
    """Min-Plus-Sum 2D Convolutional Layer with 2 Parameters."""

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
        int_dtype_str: str = "int32",
        autotune: bool = False,
        maxmin_gradient: str = "multiple",
    ) -> None:
        """Initialize the Convolutional class."""
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

        self.int_dtype_str = int_dtype_str
        self.autotune = autotune
        self.maxmin_gradient = maxmin_gradient

        self.forward_kernels = {}
        self.backward_kernels_dweight = {}
        self.backward_kernels_dinput = {}

    def _get_dtype_string(self, dtype):
        """Map torch dtype to the TileLang string identifier."""
        dtype_map = {
            torch.float32: "float32",
            torch.float64: "float64",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
        }
        return dtype_map.get(dtype, "float32")

    def _run_forward_kernel(self, input_nchw, weight_oihw, output, out_height, out_width):
        key = _kernel_cache_key(
            input_nchw.shape[0],
            self.in_channels,
            self.out_channels,
            input_nchw.shape[2],
            input_nchw.shape[3],
            self.kernel_size[0],
            self.kernel_size[1],
            out_height,
            out_width,
            self.stride[0],
            self.stride[1],
            self.dilation[0],
            self.dilation[1],
            self._get_dtype_string(input_nchw.dtype),
            self.int_dtype_str,
            self.autotune,
        )
        if key not in self.forward_kernels:
            with set_autotune_inputs(input_nchw, weight_oihw, output):
                self.forward_kernels[key] = _compile_forward_kernel_cached(*key)
        self.forward_kernels[key](input_nchw, weight_oihw, output)

    def _run_backward_kernel(
        self,
        input_nchw,
        weight_oihw,
        grad_output_nchw,
        grad_input_nchw,
        grad_weight_oihw,
        out_height,
        out_width,
        maxmin_gradient,
    ):
        key = _kernel_cache_key(
            input_nchw.shape[0],
            self.in_channels,
            self.out_channels,
            input_nchw.shape[2],
            input_nchw.shape[3],
            self.kernel_size[0],
            self.kernel_size[1],
            out_height,
            out_width,
            self.stride[0],
            self.stride[1],
            self.dilation[0],
            self.dilation[1],
            self._get_dtype_string(input_nchw.dtype),
            self.int_dtype_str,
            self.autotune,
            maxmin_gradient,
        )

        if key not in self.backward_kernels_dweight:
            with set_autotune_inputs(input_nchw, weight_oihw, grad_output_nchw, grad_weight_oihw):
                self.backward_kernels_dweight[key] = _compile_backward_kernel_dweight_cached(
                    *key[:-1],
                    maxmin_gradient=maxmin_gradient,
                )
        self.backward_kernels_dweight[key](
            input_nchw,
            weight_oihw,
            grad_output_nchw,
            grad_weight_oihw,
        )

        if key not in self.backward_kernels_dinput:
            with set_autotune_inputs(input_nchw, weight_oihw, grad_output_nchw, grad_input_nchw):
                self.backward_kernels_dinput[key] = _compile_backward_kernel_dinput_cached(
                    *key[:-1],
                    maxmin_gradient=maxmin_gradient,
                )
        self.backward_kernels_dinput[key](
            input_nchw,
            weight_oihw,
            grad_output_nchw,
            grad_input_nchw,
        )

    def _conv_forward(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
    ) -> Tensor:
        """A forward warp from kernel."""
        if self.padding_mode != "zeros":
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
        else:
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode="constant", value=0
            )

        padded_height, padded_width = (
            input.shape[2],
            input.shape[3],
        )  # use padded spatial dims
        out_height = (
            padded_height - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        out_width = (
            padded_width - self.dilation[1] * (self.kernel_size[1] - 1) - 1
        ) // self.stride[1] + 1
        if self.groups != 1:
            raise NotImplementedError("groups != 1 is not supported in base warp yet")

        output = _MinPlusSumConv2d.apply(
            input,
            weight,
            self,
            input.shape[0],
            self.in_channels,
            self.out_channels,
            input.shape[2],
            input.shape[3],
            self.kernel_size[0],
            self.kernel_size[1],
            out_height,
            out_width,
            self.stride[0],
            self.stride[1],
            self.dilation[0],
            self.dilation[1],
            self._get_dtype_string(input.dtype),
            self.int_dtype_str,
            self.autotune,
            self.maxmin_gradient,
        )

        if bias is not None:
            output += bias.view(1, -1, 1, 1)
        return output

    def forward(self, input: Tensor) -> Tensor:
        """Forward function for the MinPlusSumConv2d layer."""
        return self._conv_forward(input, self.weight, self.bias)
