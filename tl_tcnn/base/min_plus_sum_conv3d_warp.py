"""Warp for MinPlusSumConv3d.

Forward uses `min_plus_sum_conv3d_kernel` (NCDHW input, OIDHW weight, NCDHW output).
Backward uses split kernels: dweight + dinput.
"""

from typing import Optional, Union
from functools import lru_cache

import torch
from torch import Tensor, autograd, zeros_like
from torch.nn import functional as F

from ..common_types import _size_3_t
from ..utils import _triple, _kernel_cache_key
from ._conv import _Conv
from .min_plus_sum_conv3d import (
    min_plus_sum_conv3d_kernel,
    min_plus_sum_conv3d_kernel_backward_dweight,
    min_plus_sum_conv3d_kernel_backward_dinput,
)
from tilelang.autotuner import set_autotune_inputs


@lru_cache(maxsize=1)
def _compile_forward_kernel_cached(*key):
    return min_plus_sum_conv3d_kernel(*key)


@lru_cache(maxsize=1)
def _compile_backward_kernel_dweight_cached(*key, maxmin_gradient="multiple"):
    return min_plus_sum_conv3d_kernel_backward_dweight(*key, maxmin_gradient=maxmin_gradient)


@lru_cache(maxsize=1)
def _compile_backward_kernel_dinput_cached(*key, maxmin_gradient="multiple"):
    return min_plus_sum_conv3d_kernel_backward_dinput(*key, maxmin_gradient=maxmin_gradient)


def clear_min_plus_sum_conv3d_kernel_cache():
    """Clear Python-side compilation caches for MinPlusSumConv3d kernels."""
    _compile_forward_kernel_cached.cache_clear()
    _compile_backward_kernel_dweight_cached.cache_clear()
    _compile_backward_kernel_dinput_cached.cache_clear()


class _MinPlusSumConv3d(autograd.Function):
    """Autograd function for the MinPlusSumConv3d operator."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        conv_module,
        batch_size,
        in_channels,
        out_channels,
        in_depth,
        in_height,
        in_width,
        kernel_depth,
        kernel_height,
        kernel_width,
        out_depth,
        out_height,
        out_width,
        stride_d,
        stride_h,
        stride_w,
        dilation_d,
        dilation_h,
        dilation_w,
        dtype="float32",
        int_dtype_str="int32",
        autotune=False,
        maxmin_gradient="multiple",
    ):
        input_ncdhw = input.contiguous()
        weight_oidhw = weight.contiguous()

        output = torch.zeros(
            (batch_size, out_channels, out_depth, out_height, out_width),
            device=input.device,
            dtype=input.dtype,
        )

        conv_module._run_forward_kernel(
            input_ncdhw,
            weight_oidhw,
            output,
            out_depth,
            out_height,
            out_width,
        )

        ctx.save_for_backward(input_ncdhw, weight_oidhw)
        ctx.meta = (
            batch_size,
            in_channels,
            out_channels,
            in_depth,
            in_height,
            in_width,
            kernel_depth,
            kernel_height,
            kernel_width,
            out_depth,
            out_height,
            out_width,
            stride_d,
            stride_h,
            stride_w,
            dilation_d,
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
        input_ncdhw, weight_oidhw = ctx.saved_tensors
        (
            batch_size,
            in_channels,
            out_channels,
            in_depth,
            in_height,
            in_width,
            kernel_depth,
            kernel_height,
            kernel_width,
            out_depth,
            out_height,
            out_width,
            stride_d,
            stride_h,
            stride_w,
            dilation_d,
            dilation_h,
            dilation_w,
            dtype,
            int_dtype_str,
            autotune,
            maxmin_gradient,
        ) = ctx.meta

        conv_module = ctx.conv_module
        grad_output_ncdhw = grad_output.contiguous()
        grad_input_ncdhw = zeros_like(input_ncdhw)
        grad_weight_oidhw = zeros_like(weight_oidhw)

        conv_module._run_backward_kernel(
            input_ncdhw,
            weight_oidhw,
            grad_output_ncdhw,
            grad_input_ncdhw,
            grad_weight_oidhw,
            out_depth,
            out_height,
            out_width,
            maxmin_gradient,
        )

        return (
            grad_input_ncdhw,
            grad_weight_oidhw,
        ) + (None,) * (1 + len(ctx.meta))


class MinPlusSumConv3d(_Conv):
    """Min-Plus-Sum 3D Convolutional Layer."""

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
        int_dtype_str: str = "int32",
        autotune: bool = False,
        maxmin_gradient: str = "multiple",
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
            **factory_kwargs,
        )

        self.int_dtype_str = int_dtype_str
        self.autotune = autotune
        self.maxmin_gradient = maxmin_gradient

        self.forward_kernels = {}
        self.backward_kernels_dweight = {}
        self.backward_kernels_dinput = {}

    def _get_dtype_string(self, dtype):
        dtype_map = {
            torch.float32: "float32",
            torch.float64: "float64",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
        }
        return dtype_map.get(dtype, "float32")

    def _run_forward_kernel(self, input_ncdhw, weight_oidhw, output, out_depth, out_height, out_width):
        key = _kernel_cache_key(
            input_ncdhw.shape[0],
            self.in_channels,
            self.out_channels,
            input_ncdhw.shape[2],
            input_ncdhw.shape[3],
            input_ncdhw.shape[4],
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
            out_depth,
            out_height,
            out_width,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.dilation[0],
            self.dilation[1],
            self.dilation[2],
            self._get_dtype_string(input_ncdhw.dtype),
            self.int_dtype_str,
            self.autotune,
        )
        if key not in self.forward_kernels:
            with set_autotune_inputs(input_ncdhw, weight_oidhw, output):
                self.forward_kernels[key] = _compile_forward_kernel_cached(*key)
        self.forward_kernels[key](input_ncdhw, weight_oidhw, output)

    def _run_backward_kernel(
        self,
        input_ncdhw,
        weight_oidhw,
        grad_output_ncdhw,
        grad_input_ncdhw,
        grad_weight_oidhw,
        out_depth,
        out_height,
        out_width,
        maxmin_gradient,
    ):
        key = _kernel_cache_key(
            input_ncdhw.shape[0],
            self.in_channels,
            self.out_channels,
            input_ncdhw.shape[2],
            input_ncdhw.shape[3],
            input_ncdhw.shape[4],
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
            out_depth,
            out_height,
            out_width,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.dilation[0],
            self.dilation[1],
            self.dilation[2],
            self._get_dtype_string(input_ncdhw.dtype),
            self.int_dtype_str,
            self.autotune,
            maxmin_gradient,
        )

        if key not in self.backward_kernels_dweight:
            with set_autotune_inputs(input_ncdhw, weight_oidhw, grad_output_ncdhw, grad_weight_oidhw):
                self.backward_kernels_dweight[key] = _compile_backward_kernel_dweight_cached(
                    *key[:-1],
                    maxmin_gradient=maxmin_gradient,
                )
        self.backward_kernels_dweight[key](
            input_ncdhw,
            weight_oidhw,
            grad_output_ncdhw,
            grad_weight_oidhw,
        )

        if key not in self.backward_kernels_dinput:
            with set_autotune_inputs(input_ncdhw, weight_oidhw, grad_output_ncdhw, grad_input_ncdhw):
                self.backward_kernels_dinput[key] = _compile_backward_kernel_dinput_cached(
                    *key[:-1],
                    maxmin_gradient=maxmin_gradient,
                )
        self.backward_kernels_dinput[key](
            input_ncdhw,
            weight_oidhw,
            grad_output_ncdhw,
            grad_input_ncdhw,
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        if self.padding_mode != "zeros":
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
        else:
            input = F.pad(input, self._reversed_padding_repeated_twice, mode="constant", value=0)

        padded_depth, padded_height, padded_width = (
            input.shape[2],
            input.shape[3],
            input.shape[4],
        )

        out_depth = (
            padded_depth - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        out_height = (
            padded_height - self.dilation[1] * (self.kernel_size[1] - 1) - 1
        ) // self.stride[1] + 1
        out_width = (
            padded_width - self.dilation[2] * (self.kernel_size[2] - 1) - 1
        ) // self.stride[2] + 1

        if self.groups != 1:
            raise NotImplementedError("groups != 1 is not supported in base warp yet")

        output = _MinPlusSumConv3d.apply(
            input,
            weight,
            self,
            input.shape[0],
            self.in_channels,
            self.out_channels,
            input.shape[2],
            input.shape[3],
            input.shape[4],
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
            out_depth,
            out_height,
            out_width,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.dilation[0],
            self.dilation[1],
            self.dilation[2],
            self._get_dtype_string(input.dtype),
            self.int_dtype_str,
            self.autotune,
            self.maxmin_gradient,
        )

        if bias is not None:
            output += bias.view(1, -1, 1, 1, 1)
        return output

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
