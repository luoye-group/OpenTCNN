"""This is the warp file for kernels used for CompoundMinMaxPlusSumConv2d operation."""

from typing import Optional, Union
from functools import lru_cache

import torch
from torch import Tensor, autograd, zeros, zeros_like
from torch.nn import functional as F
from ..common_types import _size_2_t
from ..utils import _pair, DataType
from ._compound2p import _Compound2p
from .compound_min_max_plus_sum_conv2d2p_sharedOpt import (
    compound_min_max_plus_sum_conv2d2p_kernel,
    # compound_min_max_plus_sum_conv2d2p_kernel_tiled as compound_min_max_plus_sum_conv2d2p_kernel,
    compound_min_max_plus_sum_conv2d2p_backward_dab,
    compound_min_max_plus_sum_conv2d2p_backward_dweight,
    compound_min_max_plus_sum_conv2d2p_backward_dinput,
)
from tilelang.autotuner import set_autotune_inputs
from ..utils import _kernel_cache_key


@lru_cache(maxsize=1)
def _compile_forward_kernel_cached(*key):
    return compound_min_max_plus_sum_conv2d2p_kernel(
        *key,
        # autotune=True,
    )


def clear_compound2p_kernel_cache():
    """Clear Python-side compilation caches for compound2p kernels.

    Useful when toggling `autotune` at runtime or after editing kernel code,
    so the next forward/backward triggers a fresh compile/autotune.
    """
    _compile_forward_kernel_cached.cache_clear()
    _compile_backward_kernel_dab_cached.cache_clear()
    _compile_backward_kernel_dweight_cached.cache_clear()
    _compile_backward_kernel_dinput_cached.cache_clear()


@lru_cache(maxsize=1)
def _compile_backward_kernel_dab_cached(*key):
    return compound_min_max_plus_sum_conv2d2p_backward_dab(
        *key,
        # autotune=True,
    )


@lru_cache(maxsize=1)
def _compile_backward_kernel_dweight_cached(*key, maxmin_gradient="multiple"):
    return compound_min_max_plus_sum_conv2d2p_backward_dweight(
        *key,
        maxmin_gradient=maxmin_gradient,
        # autotune=True,
    )


@lru_cache(maxsize=1)
def _compile_backward_kernel_dinput_cached(*key, maxmin_gradient="multiple"):
    return compound_min_max_plus_sum_conv2d2p_backward_dinput(
        *key,
        maxmin_gradient=maxmin_gradient,
        # autotune=True,
    )


class _CompoundMinMaxPlusSumConv2d2p(autograd.Function):
    """Autograd function for the CompoundMinMaxPlusSumConv2d2p operator."""

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        alpha: Tensor,
        beta: Tensor,
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
        weight_oikk = weight.contiguous()  # (K,K,IC,OC)
        alpha_oi = alpha.contiguous()
        beta_oi = beta.contiguous()

        output = torch.zeros(
            # (batch, out_height, out_width, out_channels),
            (batch_size, out_channels, out_height, out_width),
            device=input.device,
            dtype=input.dtype,
        )

        conv_module._run_forward_kernel(input_nchw, weight_oikk, alpha_oi, beta_oi, output, out_height, out_width)
        ctx.save_for_backward(input_nchw, weight_oikk, alpha_oi, beta_oi)
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
        input_nchw, weight_oikk, alpha_oi, beta_oi = ctx.saved_tensors
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
                
        grad_input_nchw = zeros_like(input_nchw)
        grad_weight_oikk = zeros_like(weight_oikk)
        grad_alpha_oi = zeros_like(alpha_oi)
        grad_beta_oi = zeros_like(beta_oi)
        grad_output_nchw = grad_output.contiguous()

        conv_module._run_backward_kernel(
            input_nchw, weight_oikk, alpha_oi, beta_oi, 
            grad_output_nchw, 
            grad_input_nchw, grad_weight_oikk, grad_alpha_oi, grad_beta_oi,
            out_height,
            out_width,
            maxmin_gradient,
        )

        return (grad_input_nchw,
            grad_weight_oikk,
            grad_alpha_oi,
            grad_beta_oi) + (None,) * (1 + len(ctx.meta))
            
class CompoundMinMaxPlusSumConv2d2p(_Compound2p):
    """Compound Min-Max-Plus-Sum 2D Convolutional Layer with 2 Parameters."""

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
        int_dtype_str="int32",
        autotune=False,
        maxmin_gradient: str = "multiple",
    ) -> None:
        """Initialize the Convolutional class."""
        clear_compound2p_kernel_cache() 
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
        self, input_nchw, weight_oikk, alpha_oi, beta_oi, output,
        out_height,
        out_width,
    ):
        in_channels = input_nchw.shape[1]
        out_channels = weight_oikk.shape[0]
        key = _kernel_cache_key(
            input_nchw.shape[0],
            in_channels,
            out_channels,
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
            self.autotune
        )
        if key not in self.forward_kernels:
            with set_autotune_inputs(input_nchw, weight_oikk, alpha_oi, beta_oi, output):
                self.forward_kernels[key] = _compile_forward_kernel_cached(*key)
                
        self.forward_kernels[key](input_nchw, weight_oikk, alpha_oi, beta_oi, output)

    def _run_backward_kernel(
        self, input_nchw, weight_oikk, alpha_oi, beta_oi, grad_output_nchw, 
        grad_input_nchw, grad_weight_oikk, grad_alpha_oi, grad_beta_oi,
        out_height,
        out_width,
        maxmin_gradient,
    ):
        in_channels = input_nchw.shape[1]
        out_channels = weight_oikk.shape[0]
        key = _kernel_cache_key(
            input_nchw.shape[0],
            in_channels,
            out_channels,
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
            self.autotune
        )
        if key not in self.backward_kernels_dab:
            with set_autotune_inputs(
                input_nchw, weight_oikk, 
                grad_output_nchw, grad_alpha_oi, grad_beta_oi
            ):
                self.backward_kernels_dab[key] = _compile_backward_kernel_dab_cached(*key)
        self.backward_kernels_dab[key](
            input_nchw,
            weight_oikk,
            grad_output_nchw,
            grad_alpha_oi,
            grad_beta_oi,
        )

        dweight_key = (*key, maxmin_gradient)
        if dweight_key not in self.backward_kernels_dweight:
            with set_autotune_inputs(
                input_nchw, weight_oikk, alpha_oi, beta_oi, 
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
            beta_oi,
            grad_output_nchw,
            grad_weight_oikk,
        )
        dinput_key = (*key, maxmin_gradient)
        if dinput_key not in self.backward_kernels_dinput:
            with set_autotune_inputs(
                input_nchw, weight_oikk, alpha_oi, beta_oi, 
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
            beta_oi,
            grad_output_nchw,
            grad_input_nchw,
        )

    def _conv_forward(
        self,
        input: Tensor,
        weight: Tensor,
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
        if self.groups == 1:
            output = _CompoundMinMaxPlusSumConv2d2p.apply(
                input,
                weight,
                alpha,
                beta,
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
        else:
            in_per_group = self.in_channels // self.groups
            out_per_group = self.out_channels // self.groups
            outputs = []
            for g in range(self.groups):
                in_start = g * in_per_group
                in_end = in_start + in_per_group
                out_start = g * out_per_group
                out_end = out_start + out_per_group

                group_input = input[:, in_start:in_end, :, :]
                group_weight = weight[out_start:out_end, :, :, :]
                group_alpha = alpha[out_start:out_end, :]
                group_beta = beta[out_start:out_end, :]

                group_output = _CompoundMinMaxPlusSumConv2d2p.apply(
                    group_input,
                    group_weight,
                    group_alpha,
                    group_beta,
                    self,
                    input.shape[0],
                    in_per_group,
                    out_per_group,
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
                outputs.append(group_output)

            output = torch.cat(outputs, dim=1)

        if bias is not None:
            output += bias.view(1, -1, 1, 1)
        return output

    def forward(self, input: Tensor) -> Tensor:
        """Forward function for the CompoundMinMaxPlusSumConv2d2p layer."""
        return self._conv_forward(input, self.weight, self.alpha, self.beta, self.bias)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("forward_kernels", None)
        state.pop("backward_kernels_dab", None)
        state.pop("backward_kernels_dweight", None)
        state.pop("backward_kernels_dinput", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.forward_kernels = {}
        self.backward_kernels_dab = {}
        self.backward_kernels_dweight = {}
        self.backward_kernels_dinput = {}
