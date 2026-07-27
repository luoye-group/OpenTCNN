"""Warp for MaxPlusSumConv1d.

Forward uses `max_plus_sum_conv1d_kernel` (NCL input, OIK weight, NCL output).
Backward uses split kernels: dweight + dinput.
"""

from typing import Optional, Union
from functools import lru_cache

import torch
from torch import Tensor, autograd, zeros_like
from torch.nn import functional as F

from ..common_types import _size_1_t
from ..utils import _single, _kernel_cache_key
from ._conv import _Conv
from .max_plus_sum_conv1d import (
    max_plus_sum_conv1d_kernel,
    max_plus_sum_conv1d_kernel_backward_dweight,
    max_plus_sum_conv1d_kernel_backward_dinput,
)
from tilelang.autotuner import set_autotune_inputs


@lru_cache(maxsize=1)
def _compile_forward_kernel_cached(*key):
    return max_plus_sum_conv1d_kernel(*key)


@lru_cache(maxsize=1)
def _compile_backward_kernel_dweight_cached(*key, maxmin_gradient="multiple"):
    return max_plus_sum_conv1d_kernel_backward_dweight(*key, maxmin_gradient=maxmin_gradient)


@lru_cache(maxsize=1)
def _compile_backward_kernel_dinput_cached(*key, maxmin_gradient="multiple"):
    return max_plus_sum_conv1d_kernel_backward_dinput(*key, maxmin_gradient=maxmin_gradient)


def clear_max_plus_sum_conv1d_kernel_cache():
    """Clear Python-side compilation caches for MaxPlusSumConv1d kernels."""

    _compile_forward_kernel_cached.cache_clear()
    _compile_backward_kernel_dweight_cached.cache_clear()
    _compile_backward_kernel_dinput_cached.cache_clear()


class _MaxPlusSumConv1d(autograd.Function):
    """Autograd function for the MaxPlusSumConv1d operator."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        conv_module,
        batch_size,
        in_channels,
        out_channels,
        in_length,
        kernel_length,
        out_length,
        stride_l,
        dilation_l,
        dtype="float32",
        int_dtype_str="int32",
        autotune=False,
        maxmin_gradient="multiple",
    ):
        input_ncl = input.contiguous()
        weight_oik = weight.contiguous()

        output = torch.zeros(
            (batch_size, out_channels, out_length),
            device=input.device,
            dtype=input.dtype,
        )

        conv_module._run_forward_kernel(input_ncl, weight_oik, output, out_length)

        ctx.save_for_backward(input_ncl, weight_oik)
        ctx.meta = (
            batch_size,
            in_channels,
            out_channels,
            in_length,
            kernel_length,
            out_length,
            stride_l,
            dilation_l,
            dtype,
            int_dtype_str,
            autotune,
            maxmin_gradient,
        )
        ctx.conv_module = conv_module
        return output.contiguous()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input_ncl, weight_oik = ctx.saved_tensors
        (
            batch_size,
            in_channels,
            out_channels,
            in_length,
            kernel_length,
            out_length,
            stride_l,
            dilation_l,
            dtype,
            int_dtype_str,
            autotune,
            maxmin_gradient,
        ) = ctx.meta

        conv_module = ctx.conv_module
        grad_output_ncl = grad_output.contiguous()
        grad_input_ncl = zeros_like(input_ncl)
        grad_weight_oik = zeros_like(weight_oik)

        conv_module._run_backward_kernel(
            input_ncl,
            weight_oik,
            grad_output_ncl,
            grad_input_ncl,
            grad_weight_oik,
            out_length,
            maxmin_gradient,
        )

        return (
            grad_input_ncl,
            grad_weight_oik,
        ) + (None,) * (1 + len(ctx.meta))


class MaxPlusSumConv1d(_Conv):
    """Max-Plus-Sum 1D Convolutional Layer."""

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
        int_dtype_str: str = "int32",
        autotune: bool = False,
        maxmin_gradient: str = "single",
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

    def _run_forward_kernel(self, input_ncl, weight_oik, output, out_length):
        key = _kernel_cache_key(
            input_ncl.shape[0],
            self.in_channels,
            self.out_channels,
            input_ncl.shape[2],
            self.kernel_size[0],
            out_length,
            self.stride[0],
            self.dilation[0],
            self._get_dtype_string(input_ncl.dtype),
            self.int_dtype_str,
            self.autotune,
        )
        if key not in self.forward_kernels:
            with set_autotune_inputs(input_ncl, weight_oik, output):
                self.forward_kernels[key] = _compile_forward_kernel_cached(*key)
        self.forward_kernels[key](input_ncl, weight_oik, output)

    def _run_backward_kernel(
        self,
        input_ncl,
        weight_oik,
        grad_output_ncl,
        grad_input_ncl,
        grad_weight_oik,
        out_length,
        maxmin_gradient,
    ):
        key = _kernel_cache_key(
            input_ncl.shape[0],
            self.in_channels,
            self.out_channels,
            input_ncl.shape[2],
            self.kernel_size[0],
            out_length,
            self.stride[0],
            self.dilation[0],
            self._get_dtype_string(input_ncl.dtype),
            self.int_dtype_str,
            self.autotune,
            maxmin_gradient,
        )

        if key not in self.backward_kernels_dweight:
            with set_autotune_inputs(input_ncl, weight_oik, grad_output_ncl, grad_weight_oik):
                self.backward_kernels_dweight[key] = _compile_backward_kernel_dweight_cached(
                    *key[:-1],
                    maxmin_gradient=maxmin_gradient,
                )
        self.backward_kernels_dweight[key](
            input_ncl,
            weight_oik,
            grad_output_ncl,
            grad_weight_oik,
        )

        if key not in self.backward_kernels_dinput:
            with set_autotune_inputs(input_ncl, weight_oik, grad_output_ncl, grad_input_ncl):
                self.backward_kernels_dinput[key] = _compile_backward_kernel_dinput_cached(
                    *key[:-1],
                    maxmin_gradient=maxmin_gradient,
                )
        self.backward_kernels_dinput[key](
            input_ncl,
            weight_oik,
            grad_output_ncl,
            grad_input_ncl,
        )

    def _conv_forward(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
    ) -> Tensor:
        if self.padding_mode != "zeros":
            input = F.pad(
                input, self._reversed_padding_repeated_twice, mode=self.padding_mode
            )
        else:
            if any(self.padding):
                input = F.pad(input, self._reversed_padding_repeated_twice)

        padded_length = input.shape[2]
        out_length = (
            padded_length - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        if self.groups != 1:
            raise NotImplementedError("groups != 1 is not supported in base warp yet")

        output = _MaxPlusSumConv1d.apply(
            input,
            weight,
            self,
            input.shape[0],
            self.in_channels,
            self.out_channels,
            padded_length,
            self.kernel_size[0],
            out_length,
            self.stride[0],
            self.dilation[0],
            self._get_dtype_string(input.dtype),
            self.int_dtype_str,
            self.autotune,
            self.maxmin_gradient,
        )

        if bias is not None:
            output = output + bias.view(1, -1, 1)
        return output

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)
