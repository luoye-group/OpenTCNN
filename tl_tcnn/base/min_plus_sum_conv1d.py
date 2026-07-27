"""Kernels for MinPlusSumConv1d.

Definition:

    out[b, oc, ol] = \sum_{ic} \min_{kl}( img[b, ic, ol*sl + kl*dl] + weight[oc, ic, kl] )

Layouts (PyTorch default):
- img:    (B, IC, L)
- weight: (OC, IC, KL)
- out:    (B, OC, OL)

Backward is split into dweight and dinput kernels.
"""

import itertools

import tilelang
import tilelang.language as T

from tilelang.autotuner import AutoTuner

from ..utils import DataType, maybe_autotune
from ..autotune_env import _get_autotune_env


def min_plus_sum_conv1d_kernel(
    batch_size,
    in_channels,
    out_channels,
    in_length,
    kernel_length,
    out_length,
    stride_l,
    dilation_l,
    dtype="float32",
    int_dtype="int32",
    autotune=False,
):
    """Forward kernel for MinPlusSumConv1d (NCL input, OIK weight, NOL output)."""

    batches = batch_size
    patches = out_channels * out_length
    fast, warmup, rep, max_configs = _get_autotune_env()

    def get_configs(*_kernel_params, **_kwargs):
        max_grid_y = 65535
        if autotune:
            BLOCK_B = [1, 2, 4, 8, 16, 32]
            BLOCK_P = [32, 64, 128, 256]
            reduce_threads = [4, 8, 16, 32, 64]
        else:
            BLOCK_B = [1]
            _min_block_p = (patches + max_grid_y - 1) // max_grid_y
            if _min_block_p <= 64:
                BLOCK_P = [64]
            elif _min_block_p <= 128:
                BLOCK_P = [128]
            else:
                BLOCK_P = [256]
            reduce_threads = [8]

        _configs = list(itertools.product(BLOCK_B, BLOCK_P, reduce_threads))
        configs = [
            {
                "BLOCK_B": c[0],
                "BLOCK_P": c[1],
                "reduce_threads": c[2],
                "batch_size": batch_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "in_length": in_length,
                "kernel_length": kernel_length,
                "out_length": out_length,
            }
            for c in _configs
            if (c[0] * c[1] * c[2] <= 1024)
            and ((patches + c[1] - 1) // c[1] <= max_grid_y)
        ]
        if max_configs > 0 and len(configs) > max_configs:
            configs = configs[:max_configs]
        return configs

    @maybe_autotune(configs=get_configs(), warmup=warmup, rep=rep, enabled=autotune)
    @tilelang.jit(target="auto")
    def _kernel(
        BLOCK_B=None,
        BLOCK_P=None,
        reduce_threads=None,
        batch_size=None,
        in_channels=None,
        out_channels=None,
        in_length=None,
        kernel_length=None,
        out_length=None,
    ):
        MAX_TRANSACTION_SIZE_IN_BITS = 128
        TILE_IC = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads

        @T.prim_func
        def main(
            img_buffer: T.Buffer((batch_size, in_channels, in_length), dtype),
            weight_buffer: T.Buffer((out_channels, in_channels, kernel_length), dtype),
            output_buffer: T.Buffer((batch_size, out_channels, out_length), dtype),
        ):
            with T.Kernel(
                T.ceildiv(batches, BLOCK_B),
                T.ceildiv(patches, BLOCK_P),
                threads=(BLOCK_B, BLOCK_P, reduce_threads),
            ) as (bid, pid):
                tb = T.get_thread_binding(0)
                tp = T.get_thread_binding(1)
                tic = T.get_thread_binding(2)

                b = bid * BLOCK_B + tb
                patch_idx = pid * BLOCK_P + tp
                
                valid_b = b < batch_size
                valid_p = patch_idx < patches

                # if valid_b and valid_p:
                ol = patch_idx % out_length
                oc = patch_idx // out_length
                ol_stride = ol * stride_l

                local_acc = T.alloc_local((1,), dtype)
                T.clear(local_acc)

                sum_val = T.alloc_shared((BLOCK_B, BLOCK_P), dtype)
                if tic == 0:
                    sum_val[tb, tp] = T.cast(0, dtype)
                T.sync_threads()
                if valid_b and valid_p:
                    for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                        for ic in T.serial(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                local_min = T.alloc_local((1,), dtype)
                                local_min[0] = T.infinity(dtype)
                                for kl in T.serial(kernel_length):
                                    in_l = ol_stride + kl * dilation_l
                                    if (in_l >= 0) & (in_l < in_length):
                                        val = img_buffer[b, ic_global, in_l] + weight_buffer[oc, ic_global, kl]
                                        if val < local_min[0]:
                                            local_min[0] = val
                                local_acc[0] += local_min[0]

                    T.atomic_add(sum_val[tb, tp], local_acc[0])
                T.sync_threads()
                if valid_b and valid_p:
                    if tic == 0:
                        output_buffer[b, oc, ol] = sum_val[tb, tp]

        return main

    return _kernel()


def min_plus_sum_conv1d_kernel_backward_dweight(
    batch_size,
    in_channels,
    out_channels,
    in_length,
    kernel_length,
    out_length,
    stride_l,
    dilation_l,
    dtype="float32",
    int_dtype="int32",
    autotune=False,
    maxmin_gradient="multiple",
):
    """Compute gradient w.r.t. weight for MinPlusSumConv1d."""

    BOL = batch_size * out_length
    fast, warmup, rep, max_configs = _get_autotune_env()

    def get_configs(*_kernel_params, **_kwargs):
        if autotune:
            if fast:
                BLOCK_OC = [1, 2, 4, 8, 16]
                BLOCK_IC = [1, 2, 4, 8]
                reduce_threads = [8, 16, 32]
            else:
                BLOCK_OC = [1, 2, 4, 8, 16, 32]
                BLOCK_IC = [1, 2, 4, 8, 16]
                reduce_threads = [4, 8, 16, 32, 64]
        else:
            BLOCK_OC = [4]
            BLOCK_IC = [1]
            reduce_threads = [32]

        _configs = list(itertools.product(BLOCK_OC, BLOCK_IC, reduce_threads))
        configs = [
            {
                "BLOCK_OC": c[0],
                "BLOCK_IC": c[1],
                "reduce_threads": c[2],
                "batch_size": batch_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "in_length": in_length,
                "kernel_length": kernel_length,
                "out_length": out_length,
            }
            for c in _configs
            if c[0] * c[1] * c[2] <= 1024
        ]
        if max_configs > 0 and len(configs) > max_configs:
            configs = configs[:max_configs]
        return configs

    @maybe_autotune(configs=get_configs(), warmup=warmup, rep=rep, enabled=autotune)
    @tilelang.jit(target="auto")
    def _kernel(
        BLOCK_OC=None,
        BLOCK_IC=None,
        reduce_threads=None,
        batch_size=None,
        in_channels=None,
        out_channels=None,
        in_length=None,
        kernel_length=None,
        out_length=None,
    ):
        MAX_TRANSACTION_SIZE_IN_BITS = 128
        TILE_BOL = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_BOL = TILE_BOL * reduce_threads

        @T.prim_func
        def compute_dweight(
            img_buffer: T.Buffer((batch_size, in_channels, in_length), dtype),
            weight_buffer: T.Buffer((out_channels, in_channels, kernel_length), dtype),
            dout_buffer: T.Buffer((batch_size, out_channels, out_length), dtype),
            dweight_buffer: T.Buffer((out_channels, in_channels, kernel_length), dtype),
        ):
            with T.Kernel(
                T.ceildiv(out_channels, BLOCK_OC),
                T.ceildiv(in_channels, BLOCK_IC),
                threads=(BLOCK_OC, BLOCK_IC, reduce_threads),
            ) as (oc_block, ic_block):
                toc = T.get_thread_binding(0)
                tic = T.get_thread_binding(1)
                tbol = T.get_thread_binding(2)

                oc = oc_block * BLOCK_OC + toc
                ic = ic_block * BLOCK_IC + tic
                valid_oc = oc < out_channels
                valid_ic = ic < in_channels

                shared = T.alloc_shared((BLOCK_OC, BLOCK_IC, kernel_length, reduce_threads), dtype)
                if (tbol == 0) & (toc == 0) & (tic == 0):
                    T.clear(shared)
                T.sync_threads()

                local_acc = T.alloc_local((kernel_length,), dtype)
                T.clear(local_acc)

                if valid_oc & valid_ic:
                    for bol_block in T.serial(T.ceildiv(BOL, BLOCK_BOL)):
                        bol_start = bol_block * BLOCK_BOL + tbol * TILE_BOL

                        for bol_offset in T.serial(TILE_BOL):
                            bol = bol_start + bol_offset
                            if bol < BOL:
                                ol = bol % out_length
                                b = bol // out_length
                                dout_val = dout_buffer[b, oc, ol]
                                ol_stride = ol * stride_l

                                best = T.alloc_local((1,), dtype)
                                best[0] = T.infinity(dtype)
                                for kl in T.serial(kernel_length):
                                    in_l = ol_stride + kl * dilation_l
                                    if (in_l >= 0) & (in_l < in_length):
                                        val = img_buffer[b, ic, in_l] + weight_buffer[oc, ic, kl]
                                        if val < best[0]:
                                            best[0] = val

                                if maxmin_gradient == "single":
                                    sel = T.alloc_local((1,), int_dtype)
                                    sel[0] = T.cast(0, int_dtype)
                                    found = T.alloc_local((1,), "int32")
                                    found[0] = 0
                                    for kl in T.serial(kernel_length):
                                        in_l = ol_stride + kl * dilation_l
                                        if (in_l >= 0) & (in_l < in_length):
                                            val = img_buffer[b, ic, in_l] + weight_buffer[oc, ic, kl]
                                            if (val == best[0]) & (found[0] == 0):
                                                sel[0] = T.cast(kl, int_dtype)
                                                found[0] = 1
                                    local_acc[T.cast(sel[0], "int32")] += dout_val
                                else:
                                    cnt = T.alloc_local((1,), "int32")
                                    cnt[0] = 0
                                    for kl in T.serial(kernel_length):
                                        in_l = ol_stride + kl * dilation_l
                                        if (in_l >= 0) & (in_l < in_length):
                                            val = img_buffer[b, ic, in_l] + weight_buffer[oc, ic, kl]
                                            if val == best[0]:
                                                cnt[0] += 1
                                    for kl in T.serial(kernel_length):
                                        in_l = ol_stride + kl * dilation_l
                                        if (in_l >= 0) & (in_l < in_length):
                                            val = img_buffer[b, ic, in_l] + weight_buffer[oc, ic, kl]
                                            if val == best[0]:
                                                local_acc[kl] += dout_val / T.cast(cnt[0], dtype)

                    for kl in T.serial(kernel_length):
                        shared[toc, tic, kl, tbol] = local_acc[kl]

                T.sync_threads()

                if valid_oc & valid_ic:
                    if tbol == 0:
                        for kl in T.serial(kernel_length):
                            total = T.alloc_local((1,), dtype)
                            total[0] = T.cast(0, dtype)
                            for r in T.serial(reduce_threads):
                                total[0] = total[0] + shared[toc, tic, kl, r]
                            dweight_buffer[oc, ic, kl] = total[0]

        return compute_dweight

    return _kernel()


def min_plus_sum_conv1d_kernel_backward_dinput(
    batch_size,
    in_channels,
    out_channels,
    in_length,
    kernel_length,
    out_length,
    stride_l,
    dilation_l,
    dtype="float32",
    int_dtype="int32",
    autotune=False,
    maxmin_gradient="multiple",
):
    """Compute gradient w.r.t. input for MinPlusSumConv1d."""

    batches = batch_size
    patches = out_channels * out_length
    fast, warmup, rep, max_configs = _get_autotune_env()

    def get_configs(*_kernel_params, **_kwargs):
        max_grid_y = 65535
        if autotune:
            if fast:
                BLOCK_B = [1, 2, 4]
                BLOCK_P = [64, 128, 256]
                reduce_threads = [8, 16, 32]
            else:
                BLOCK_B = [1, 2, 4, 8]
                BLOCK_P = [32, 64, 128, 256]
                reduce_threads = [4, 8, 16, 32, 64]
        else:
            BLOCK_B = [1]
            _min_block_p = (patches + max_grid_y - 1) // max_grid_y
            if _min_block_p <= 64:
                BLOCK_P = [64]
            elif _min_block_p <= 128:
                BLOCK_P = [128]
            else:
                BLOCK_P = [256]
            reduce_threads = [8]

        _configs = list(itertools.product(BLOCK_B, BLOCK_P, reduce_threads))
        configs = [
            {
                "BLOCK_B": c[0],
                "BLOCK_P": c[1],
                "reduce_threads": c[2],
                "batch_size": batch_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "in_length": in_length,
                "kernel_length": kernel_length,
                "out_length": out_length,
            }
            for c in _configs
            if (c[0] * c[1] * c[2] <= 1024)
            and ((patches + c[1] - 1) // c[1] <= max_grid_y)
        ]
        if max_configs > 0 and len(configs) > max_configs:
            configs = configs[:max_configs]
        return configs

    @maybe_autotune(configs=get_configs(), warmup=warmup, rep=rep, enabled=autotune)
    @tilelang.jit(target="auto")
    def _kernel(
        BLOCK_B=None,
        BLOCK_P=None,
        reduce_threads=None,
        batch_size=None,
        in_channels=None,
        out_channels=None,
        in_length=None,
        kernel_length=None,
        out_length=None,
    ):
        MAX_TRANSACTION_SIZE_IN_BITS = 128
        TILE_IC = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads

        @T.prim_func
        def compute_dinput(
            img_buffer: T.Buffer((batch_size, in_channels, in_length), dtype),
            weight_buffer: T.Buffer((out_channels, in_channels, kernel_length), dtype),
            dout_buffer: T.Buffer((batch_size, out_channels, out_length), dtype),
            dimg_buffer: T.Buffer((batch_size, in_channels, in_length), dtype),
        ):
            with T.Kernel(
                T.ceildiv(batches, BLOCK_B),
                T.ceildiv(patches, BLOCK_P),
                threads=(BLOCK_B, BLOCK_P, reduce_threads),
            ) as (bid, pid):
                tb = T.get_thread_binding(0)
                tp = T.get_thread_binding(1)
                tic = T.get_thread_binding(2)

                b = bid * BLOCK_B + tb
                patch_idx = pid * BLOCK_P + tp

                if b < batch_size and patch_idx < patches:
                    ol = patch_idx % out_length
                    oc = patch_idx // out_length
                    dout_val = dout_buffer[b, oc, ol]
                    ol_stride = ol * stride_l

                    for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                        for ic in T.serial(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                best = T.alloc_local((1,), dtype)
                                best[0] = T.infinity(dtype)
                                for kl in T.serial(kernel_length):
                                    in_l = ol_stride + kl * dilation_l
                                    if (in_l >= 0) & (in_l < in_length):
                                        val = img_buffer[b, ic_global, in_l] + weight_buffer[oc, ic_global, kl]
                                        if val < best[0]:
                                            best[0] = val

                                if maxmin_gradient == "single":
                                    sel = T.alloc_local((1,), int_dtype)
                                    sel[0] = T.cast(0, int_dtype)
                                    found = T.alloc_local((1,), "int32")
                                    found[0] = 0
                                    for kl in T.serial(kernel_length):
                                        in_l = ol_stride + kl * dilation_l
                                        if (in_l >= 0) & (in_l < in_length):
                                            val = img_buffer[b, ic_global, in_l] + weight_buffer[oc, ic_global, kl]
                                            if (val == best[0]) & (found[0] == 0):
                                                sel[0] = T.cast(kl, int_dtype)
                                                found[0] = 1
                                    kl_sel = T.cast(sel[0], "int32")
                                    in_l = ol_stride + kl_sel * dilation_l
                                    if (in_l >= 0) & (in_l < in_length):
                                        T.atomic_add(dimg_buffer[b, ic_global, in_l], dout_val)
                                else:
                                    cnt = T.alloc_local((1,), "int32")
                                    cnt[0] = 0
                                    for kl in T.serial(kernel_length):
                                        in_l = ol_stride + kl * dilation_l
                                        if (in_l >= 0) & (in_l < in_length):
                                            val = img_buffer[b, ic_global, in_l] + weight_buffer[oc, ic_global, kl]
                                            if val == best[0]:
                                                cnt[0] += 1
                                    for kl in T.serial(kernel_length):
                                        in_l = ol_stride + kl * dilation_l
                                        if (in_l >= 0) & (in_l < in_length):
                                            val = img_buffer[b, ic_global, in_l] + weight_buffer[oc, ic_global, kl]
                                            if val == best[0]:
                                                T.atomic_add(dimg_buffer[b, ic_global, in_l], dout_val / T.cast(cnt[0], dtype))

        return compute_dinput

    return _kernel()

    @T.prim_func
    def bwd_all(
        img: T.handle,
        weight: T.handle,
        dout: T.handle,
        dimg: T.handle,
        dweight: T.handle,
    ):
        img_buffer = T.match_buffer(img, (batch_size, in_length, in_channels), dtype)
        weight_buffer = T.match_buffer(
            weight, (kernel_length, in_channels, out_channels), dtype
        )
        dout_buffer = T.match_buffer(
            dout, (batch_size, out_length, out_channels), dtype
        )
        dimg_buffer = T.match_buffer(dimg, (batch_size, in_length, in_channels), dtype)
        dweight_buffer = T.match_buffer(
            dweight, (kernel_length, in_channels, out_channels), dtype
        )

        with T.Kernel(grid_size, threads=block_size) as bid:
            tid = T.get_thread_binding(0)
            patch_idx = bid

            ol = patch_idx % out_length
            b = patch_idx // out_length

            for it_oc in T.serial(T.ceildiv(out_channels, block_size)):
                oc = tid + it_oc * block_size
                if oc < out_channels:
                    dout_val = dout_buffer[b, ol, oc]
                    for ic in T.serial(in_channels):
                        local_min = T.alloc_local((1,), dtype)
                        local_arg = T.alloc_local((1,), int_dtype)
                        local_min[0] = T.infinity(dtype)
                        local_arg[0] = T.cast(0, int_dtype)

                        for kl in T.serial(kernel_length):
                            k_flat = kl

                            val = (
                                img_buffer[
                                    b,
                                    ol * stride_l + kl * dilation_l,
                                    ic,
                                ]
                                + weight_buffer[kl, ic, oc]
                            )
                            if val < local_min[0]:
                                local_min[0] = val
                                local_arg[0] = T.cast(k_flat, int_dtype)

                        flat = local_arg[0]
                        kl_sel = flat
                        # dW accumulation
                        T.atomic_add(dweight_buffer[kl_sel, ic, oc], dout_val)
                        # dImg accumulation
                        in_l = ol * stride_l + kl_sel * dilation_l
                        if (in_l >= 0) and (in_l < in_length):
                            T.atomic_add(dimg_buffer[b, in_l, ic], dout_val)

    return bwd_all
