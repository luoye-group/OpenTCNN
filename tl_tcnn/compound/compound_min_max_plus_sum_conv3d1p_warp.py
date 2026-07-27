"""This is the warp file for kernels used for CompoundMinMaxPlusSumConv3d operation."""

from typing import Optional, Union
from functools import lru_cache

import torch
from torch import Tensor, autograd, zeros, zeros_like
from torch.nn import functional as F
from ..common_types import _size_3_t
from ..utils import _triple, DataType
from ._compound1p import _Compound1p
from .compound_min_max_plus_sum_conv3d1p import (
    compound_min_max_plus_sum_conv3d1p_kernel,
    compound_min_max_plus_sum_conv3d1p_backward_da,
    compound_min_max_plus_sum_conv3d1p_backward_dweight,
    compound_min_max_plus_sum_conv3d1p_backward_dinput,
)
from tilelang.autotuner import set_autotune_inputs
from ..utils import _kernel_cache_key


@lru_cache(maxsize=1)
def _compile_forward_kernel_cached(*key):
    return compound_min_max_plus_sum_conv3d1p_kernel(
        *key,
    )


@lru_cache(maxsize=1)
def _compile_backward_kernel_da_cached(*key):
    return compound_min_max_plus_sum_conv3d1p_backward_da(
        *key,
    )


@lru_cache(maxsize=1)
def _compile_backward_kernel_dweight_cached(*key, maxmin_gradient="multiple"):
    return compound_min_max_plus_sum_conv3d1p_backward_dweight(
        *key,
        maxmin_gradient=maxmin_gradient,
    )


@lru_cache(maxsize=1)
def _compile_backward_kernel_dinput_cached(*key, maxmin_gradient="multiple"):
    return compound_min_max_plus_sum_conv3d1p_backward_dinput(
        *key,
        maxmin_gradient=maxmin_gradient,
    )


class _CompoundMinMaxPlusSumConv3d1p(autograd.Function):
    """Autograd function for the CompoundMinMaxPlusSumConv3d1p operator."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        alpha: Tensor,
        conv_module,
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        in_depth,
        kernel_height,
        kernel_width,
        kernel_depth,
        out_height,
        out_width,
        out_depth,
        stride_h,
        stride_w,
        stride_d,
        dilation_h,
        dilation_w,
        dilation_d,
        dtype="float32",
        int_dtype_str="int32",
        autotune=False,
        maxmin_gradient="multiple",
    ):
        input_nchw = input.contiguous()
        weight_oikk = weight.contiguous()  # (K,K,IC,OC)
        alpha_oi = alpha.contiguous()

        output = torch.zeros(
            # (batch, out_height, out_width, out_depth, out_channels),
            (batch_size, out_channels, out_height, out_width, out_depth),
            device=input.device,
            dtype=input.dtype,
        )

        conv_module._run_forward_kernel(input_nchw, weight_oikk, alpha_oi, output, out_height, out_width, out_depth)
        ctx.save_for_backward(input_nchw, weight_oikk, alpha_oi)
        ctx.meta = (
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
            in_depth,
            kernel_height,
            kernel_width,
            kernel_depth,
            out_height,
            out_width,
            out_depth,
            stride_h,
            stride_w,
            stride_d,
            dilation_h,
            dilation_w,
            dilation_d,
            dtype,
            int_dtype_str,
            autotune,
            maxmin_gradient,
        )
        ctx.conv_module = conv_module
        return output.contiguous()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        input_nchw, weight_oikk, alpha_oi = ctx.saved_tensors
        (
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
            in_depth,
            kernel_height,
            kernel_width,
            kernel_depth,
            out_height,
            out_width,
            out_depth,
            stride_h,
            stride_w,
            stride_d,
            dilation_h,
            dilation_w,
            dilation_d,
            dtype,
            int_dtype_str,
            autotune,
            maxmin_gradient,
        ) = ctx.meta
        conv_module = ctx.conv_module
                
        grad_input_nchw = zeros_like(input_nchw)
        grad_weight_oikk = zeros_like(weight_oikk)
        grad_alpha_oi = zeros_like(alpha_oi)
        grad_output_nchw = grad_output.contiguous()

        conv_module._run_backward_kernel(
            input_nchw, weight_oikk, alpha_oi, 
            grad_output_nchw, 
            grad_input_nchw, grad_weight_oikk, grad_alpha_oi,
            out_height,
            out_width,
            out_depth,
            maxmin_gradient,
        )

        return (grad_input_nchw, grad_weight_oikk, grad_alpha_oi) + (None,) * (1 + len(ctx.meta))


class CompoundMinMaxPlusSumConv3d1p(_Compound1p):
    """Compound Min-Max-Plus-Sum 3D Convolutional Layer with 2 Parameters."""

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
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
        int_dtype_str="int32",
        autotune=False,
        maxmin_gradient: str = "multiple",
    ) -> None:
        """Initialize the Convolutional class."""
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
            dilation_,  # type: ignore
            False,
            _triple(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )
        self.int_dtype_str = int_dtype_str
        self.autotune = autotune
        self.maxmin_gradient = maxmin_gradient
        self.forward_kernels = {}
        self.backward_kernels_da = {}
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

    def _run_forward_kernel(
        self, input_nchw, weight_oikk, alpha_oi, output,
        out_height,
        out_width,
        out_depth,
    ):
        key = _kernel_cache_key(
            input_nchw.shape[0],
            self.in_channels,
            self.out_channels,
            input_nchw.shape[2],
            input_nchw.shape[3],
            input_nchw.shape[4],
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
            out_height,
            out_width,
            out_depth,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.dilation[0],
            self.dilation[1],
            self.dilation[2],
            self._get_dtype_string(input_nchw.dtype),
            self.int_dtype_str,
            self.autotune,
        )
        if key not in self.forward_kernels:
            with set_autotune_inputs(input_nchw, weight_oikk, alpha_oi, output):
                self.forward_kernels[key] = _compile_forward_kernel_cached(*key)
                
        self.forward_kernels[key](input_nchw, weight_oikk, alpha_oi, output)

    def _run_backward_kernel(
        self, input_nchw, weight_oikk, alpha_oi, grad_output_nchw, 
        grad_input_nchw, grad_weight_oikk, grad_alpha_oi,
        out_height,
        out_width,
        out_depth,
        maxmin_gradient,
    ):
        key = _kernel_cache_key(
            input_nchw.shape[0],
            self.in_channels,
            self.out_channels,
            input_nchw.shape[2],
            input_nchw.shape[3],
            input_nchw.shape[4],
            self.kernel_size[0],
            self.kernel_size[1],
            self.kernel_size[2],
            out_height,
            out_width,
            out_depth,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            self.dilation[0],
            self.dilation[1],
            self.dilation[2],
            self._get_dtype_string(input_nchw.dtype),
            self.int_dtype_str,
            self.autotune,
        )
        if key not in self.backward_kernels_da:
            with set_autotune_inputs(
                input_nchw, weight_oikk, 
                grad_output_nchw, grad_alpha_oi
            ):
                self.backward_kernels_da[key] = _compile_backward_kernel_da_cached(*key)
        self.backward_kernels_da[key](
            input_nchw,
            weight_oikk,
            grad_output_nchw,
            grad_alpha_oi,
        )

        dweight_key = (*key, maxmin_gradient)
        if dweight_key not in self.backward_kernels_dweight:
            with set_autotune_inputs(
                input_nchw, weight_oikk, alpha_oi, 
                grad_output_nchw, grad_weight_oikk
            ):
                self.backward_kernels_dweight[dweight_key] = _compile_backward_kernel_dweight_cached(
                    *key,
                    maxmin_gradient=maxmin_gradient,
                )
        self.backward_kernels_dweight[dweight_key](
            input_nchw,
            weight_oikk,
            alpha_oi,
            grad_output_nchw,
            grad_weight_oikk,
        )
        dinput_key = (*key, maxmin_gradient)
        if dinput_key not in self.backward_kernels_dinput:
            with set_autotune_inputs(
                input_nchw, weight_oikk, alpha_oi, 
                grad_output_nchw, grad_input_nchw
            ):
                self.backward_kernels_dinput[dinput_key] = _compile_backward_kernel_dinput_cached(
                    *key,
                    maxmin_gradient=maxmin_gradient,
                )
        self.backward_kernels_dinput[dinput_key](
            input_nchw,
            weight_oikk,
            alpha_oi,
            grad_output_nchw,
            grad_input_nchw,
        )

    def _conv_forward(
        self,
        input: Tensor,
        weight: Tensor,
        alpha: Tensor,
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

        padded_height, padded_width, padded_depth = (
            input.shape[2],
            input.shape[3],
            input.shape[4],
        )  # use padded spatial dims
        out_height = (
            padded_height - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        out_width = (
            padded_width - self.dilation[1] * (self.kernel_size[1] - 1) - 1
        ) // self.stride[1] + 1
        out_depth = (
            padded_depth - self.dilation[2] * (self.kernel_size[2] - 1) - 1
        ) // self.stride[2] + 1
        
        if self.groups == 1:
            output = _CompoundMinMaxPlusSumConv3d1p.apply(
                input,
                weight,
                alpha,
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
                out_height,
                out_width,
                out_depth,
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
        else:
            in_per_group = self.in_channels // self.groups
            out_per_group = self.out_channels // self.groups
            outputs = []
            for g in range(self.groups):
                in_start = g * in_per_group
                in_end = in_start + in_per_group
                out_start = g * out_per_group
                out_end = out_start + out_per_group

                group_input = input[:, in_start:in_end, :, :, :]
                group_weight = weight[out_start:out_end, :, :, :, :]
                group_alpha = alpha[out_start:out_end, in_start:in_end]

                group_output = _CompoundMinMaxPlusSumConv3d1p.apply(
                    group_input,
                    group_weight,
                    group_alpha,
                    self,
                    input.shape[0],
                    in_per_group,
                    out_per_group,
                    input.shape[2],
                    input.shape[3],
                    input.shape[4],
                    self.kernel_size[0],
                    self.kernel_size[1],
                    self.kernel_size[2],
                    out_height,
                    out_width,
                    out_depth,
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
                outputs.append(group_output)

            output = torch.cat(outputs, dim=1)

        if bias is not None:
            output += bias.view(1, -1, 1, 1, 1)
        return output

    def forward(self, input: Tensor) -> Tensor:
        """Forward function for the CompoundMinMaxPlusSumConv3d1p layer."""
        return self._conv_forward(input, self.weight, self.alpha, self.bias)
