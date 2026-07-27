"""This is the warp file for kernels used for ParallelMinMaxPlusSumConv1d operation."""

from typing import Optional, Union
from functools import lru_cache

import torch
from torch import Tensor, autograd, zeros, zeros_like
from torch.nn import functional as F
from ..common_types import _size_1_t
from ..utils import _single, DataType
from ._parallel2p import _Parallel2p
from .parallel_min_max_plus_sum_conv1d2p import (
    parallel_min_max_plus_sum_conv1d2p_kernel,
    parallel_min_max_plus_sum_conv1d2p_backward_dab,
    parallel_min_max_plus_sum_conv1d2p_backward_dweight,
    parallel_min_max_plus_sum_conv1d2p_backward_dinput,
)
from tilelang.autotuner import set_autotune_inputs
from ..utils import _kernel_cache_key


@lru_cache(maxsize=1)
def _compile_forward_kernel_cached(*key):
    return parallel_min_max_plus_sum_conv1d2p_kernel(
        *key,
    )


@lru_cache(maxsize=1)
def _compile_backward_kernel_dab_cached(*key):
    return parallel_min_max_plus_sum_conv1d2p_backward_dab(
        *key,
    )


@lru_cache(maxsize=1)
def _compile_backward_kernel_dweight_cached(*key, maxmin_gradient="multiple"):
    return parallel_min_max_plus_sum_conv1d2p_backward_dweight(
        *key,
        maxmin_gradient=maxmin_gradient,
    )


@lru_cache(maxsize=1)
def _compile_backward_kernel_dinput_cached(*key, maxmin_gradient="multiple"):
    return parallel_min_max_plus_sum_conv1d2p_backward_dinput(
        *key,
        maxmin_gradient=maxmin_gradient,
    )


class _ParallelMinMaxPlusSumConv1d2p(autograd.Function):
    """Autograd function for the ParallelMinMaxPlusSumConv1d2p operator."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight1: Tensor,
        weight2: Tensor,
        alpha: Tensor,
        beta: Tensor,
        conv_module,
        batch_size,
        in_channels,
        out_channels,
        in_width,
        kernel_width,
        out_width,
        stride_w,
        dilation_w,
        dtype="float32",
        int_dtype_str="int32",
        autotune=False,
        maxmin_gradient="multiple",
    ):
        input_nchw = input.contiguous()
        weight1_oikk = weight1.contiguous()  # (K,K,IC,OC)
        weight2_oikk = weight2.contiguous()  # (K,K,IC,OC)
        alpha_oi = alpha.contiguous()
        beta_oi = beta.contiguous()

        output = torch.zeros(
            # (batch, out_width, out_channels),
            (batch_size, out_channels, out_width),
            device=input.device,
            dtype=input.dtype,
        )

        conv_module._run_forward_kernel(input_nchw, weight1_oikk, weight2_oikk, alpha_oi, beta_oi, output, out_width)
        ctx.save_for_backward(input_nchw, weight1_oikk, weight2_oikk, alpha_oi, beta_oi)
        ctx.meta = (
            batch_size,
            in_channels,
            out_channels,
            in_width,
            kernel_width,
            out_width,
            stride_w,
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
        input_nchw, weight1_oikk, weight2_oikk, alpha_oi, beta_oi = ctx.saved_tensors
        (
            batch_size,
            in_channels,
            out_channels,
            in_width,
            kernel_width,
            out_width,
            stride_w,
            dilation_w,
            dtype,
            int_dtype_str,
            autotune,
            maxmin_gradient,
        ) = ctx.meta
        conv_module = ctx.conv_module
                
        grad_input_nchw = zeros_like(input_nchw)
        grad_weight1_oikk = zeros_like(weight1_oikk)
        grad_weight2_oikk = zeros_like(weight2_oikk)
        grad_alpha_oi = zeros_like(alpha_oi)
        grad_beta_oi = zeros_like(beta_oi)
        grad_output_nchw = grad_output.contiguous()

        conv_module._run_backward_kernel(
            input_nchw, weight1_oikk, weight2_oikk, alpha_oi, beta_oi, 
            grad_output_nchw, 
            grad_input_nchw, grad_weight1_oikk, grad_weight2_oikk, grad_alpha_oi, grad_beta_oi,
            out_width,
            maxmin_gradient,
        )

        return (
            grad_input_nchw,
            grad_weight1_oikk,
            grad_weight2_oikk,
            grad_alpha_oi,
            grad_beta_oi,
        ) + (None,) * (1 + len(ctx.meta))


class ParallelMinMaxPlusSumConv1d2p(_Parallel2p):
    """Parallel Min-Max-Plus-Sum 1D Convolutional Layer with 2 Parameters."""

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
        int_dtype_str="int32",
        autotune=False,
        maxmin_gradient: str = "multiple",
    ) -> None:
        """Initialize the Convolutional class."""
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
            dilation_,  # type: ignore
            False,
            _single(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )
        self.int_dtype_str = int_dtype_str
        self.autotune = autotune
        self.maxmin_gradient = maxmin_gradient
        self.forward_kernels = {}
        self.backward_kernels_dab = {}
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
        self, input_nchw, weight1_oikk, weight2_oikk, alpha_oi, beta_oi, output,
        out_width,
    ):
        key = _kernel_cache_key(
            input_nchw.shape[0],
            self.in_channels,
            self.out_channels,
            input_nchw.shape[2],
            self.kernel_size[0],
            out_width,
            self.stride[0],
            self.dilation[0],
            self._get_dtype_string(input_nchw.dtype),
            self.int_dtype_str,
            self.autotune,
        )
        if key not in self.forward_kernels:
            with set_autotune_inputs(input_nchw, weight1_oikk, weight2_oikk, alpha_oi, beta_oi, output):
                self.forward_kernels[key] = _compile_forward_kernel_cached(*key)
                
        self.forward_kernels[key](input_nchw, weight1_oikk, weight2_oikk, alpha_oi, beta_oi, output)

    def _run_backward_kernel(
        self, input_nchw, weight1_oikk, weight2_oikk, alpha_oi, beta_oi, grad_output_nchw, 
        grad_input_nchw, grad_weight1_oikk, grad_weight2_oikk, grad_alpha_oi, grad_beta_oi,
        out_width,
        maxmin_gradient,
    ):
        key = _kernel_cache_key(
            input_nchw.shape[0],
            self.in_channels,
            self.out_channels,
            input_nchw.shape[2],
            self.kernel_size[0],
            out_width,
            self.stride[0],
            self.dilation[0],
            self._get_dtype_string(input_nchw.dtype),
            self.int_dtype_str,
            self.autotune,
        )
        if key not in self.backward_kernels_dab:
            with set_autotune_inputs(
                input_nchw, weight1_oikk, weight2_oikk, 
                grad_output_nchw, grad_alpha_oi, grad_beta_oi
            ):
                self.backward_kernels_dab[key] = _compile_backward_kernel_dab_cached(*key)
        self.backward_kernels_dab[key](
            input_nchw,
            weight1_oikk, weight2_oikk,
            grad_output_nchw,
            grad_alpha_oi,
            grad_beta_oi,
        )

        dweight_key = (*key, maxmin_gradient)
        if dweight_key not in self.backward_kernels_dweight:
            with set_autotune_inputs(
                input_nchw, weight1_oikk, weight2_oikk, alpha_oi, beta_oi, 
                grad_output_nchw, grad_weight1_oikk, grad_weight2_oikk
            ):
                self.backward_kernels_dweight[dweight_key] = _compile_backward_kernel_dweight_cached(
                    *key,
                    maxmin_gradient=maxmin_gradient,
                )
        self.backward_kernels_dweight[dweight_key](
            input_nchw,
            weight1_oikk, weight2_oikk,
            alpha_oi,
            beta_oi,
            grad_output_nchw,
            grad_weight1_oikk, grad_weight2_oikk,
        )
        dinput_key = (*key, maxmin_gradient)
        if dinput_key not in self.backward_kernels_dinput:
            with set_autotune_inputs(
                input_nchw, weight1_oikk, weight2_oikk, alpha_oi, beta_oi, 
                grad_output_nchw, grad_input_nchw
            ):
                self.backward_kernels_dinput[dinput_key] = _compile_backward_kernel_dinput_cached(
                    *key,
                    maxmin_gradient=maxmin_gradient,
                )
        self.backward_kernels_dinput[dinput_key](
            input_nchw,
            weight1_oikk, weight2_oikk,
            alpha_oi,
            beta_oi,
            grad_output_nchw,
            grad_input_nchw,
        )

    def _conv_forward(
        self,
        input: Tensor,
        weight1: Tensor,
        weight2: Tensor,
        alpha: Tensor,
        beta: Tensor,
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

        padded_width = input.shape[2]  # use padded spatial dims
        out_width = (
            padded_width - self.dilation[0] * (self.kernel_size[0] - 1) - 1
        ) // self.stride[0] + 1
        if self.groups == 1:
            output = _ParallelMinMaxPlusSumConv1d2p.apply(
                input,
                weight1,
                weight2,
                alpha,
                beta,
                self,
                input.shape[0],
                self.in_channels,
                self.out_channels,
                input.shape[2],
                self.kernel_size[0],
                out_width,
                self.stride[0],
                self.dilation[0],
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

                group_input = input[:, in_start:in_end, :]
                group_weight1 = weight1[out_start:out_end, :, :]
                group_weight2 = weight2[out_start:out_end, :, :]
                group_alpha = alpha[out_start:out_end, :]
                group_beta = beta[out_start:out_end, :]

                group_output = _ParallelMinMaxPlusSumConv1d2p.apply(
                    group_input,
                    group_weight1,
                    group_weight2,
                    group_alpha,
                    group_beta,
                    self,
                    input.shape[0],
                    in_per_group,
                    out_per_group,
                    input.shape[2],
                    self.kernel_size[0],
                    out_width,
                    self.stride[0],
                    self.dilation[0],
                    self._get_dtype_string(input.dtype),
                    self.int_dtype_str,
                    self.autotune,
                    self.maxmin_gradient,
                )
                outputs.append(group_output)

            output = torch.cat(outputs, dim=1)

        if bias is not None:
            output += bias.view(1, -1, 1)
        return output

    def forward(self, input: Tensor) -> Tensor:
        """Forward function for the ParallelMinMaxPlusSumConv1d2p layer."""
        return self._conv_forward(input, self.weight1, self.weight2, self.alpha, self.beta, self.bias)
