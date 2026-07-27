"""Kernels for MaxPlusSumConv3d.

Definition:

    out[b, oc, od, oh, ow] = \sum_{ic} \max_{kd,kh,kw}(
        img[b, ic, od*sd + kd*dd, oh*sh + kh*dh, ow*sw + kw*dw] + weight[oc, ic, kd, kh, kw]
    )

Layouts (PyTorch default):
- img:    (B, IC, D, H, W)
- weight: (OC, IC, KD, KH, KW)
- out:    (B, OC, OD, OH, OW)

Backward is split into dweight and dinput kernels.

Tie handling for the inner max is controlled by `maxmin_gradient`:
- "single": pick the first max position
- "multiple": distribute gradient uniformly across all max positions
"""

import itertools

import tilelang
import tilelang.language as T

from tilelang.autotuner import AutoTuner

from ..utils import DataType, maybe_autotune
from ..autotune_env import _get_autotune_env


def max_plus_sum_conv3d_kernel(
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
    int_dtype="int32",
    autotune=False,
):
    """Forward kernel for MaxPlusSumConv3d (NCDHW input, OIDHW weight, NCDHW output)."""

    batches = batch_size
    patches = out_channels * out_depth * out_height * out_width
    fast, warmup, rep, max_configs = _get_autotune_env()

    def get_configs(*_kernel_params, **_kwargs):
        max_grid_y = 65535
        if autotune:
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
                "in_depth": in_depth,
                "in_height": in_height,
                "in_width": in_width,
                "kernel_depth": kernel_depth,
                "kernel_height": kernel_height,
                "kernel_width": kernel_width,
                "out_depth": out_depth,
                "out_height": out_height,
                "out_width": out_width,
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
        in_depth=None,
        in_height=None,
        in_width=None,
        kernel_depth=None,
        kernel_height=None,
        kernel_width=None,
        out_depth=None,
        out_height=None,
        out_width=None,
    ):
        MAX_TRANSACTION_SIZE_IN_BITS = 128
        TILE_IC = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads

        @T.prim_func
        def main(
            img_buffer: T.Buffer((batch_size, in_channels, in_depth, in_height, in_width), dtype),
            weight_buffer: T.Buffer((out_channels, in_channels, kernel_depth, kernel_height, kernel_width), dtype),
            output_buffer: T.Buffer((batch_size, out_channels, out_depth, out_height, out_width), dtype),
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
                ow = patch_idx % out_width
                tmp0 = patch_idx // out_width
                oh = tmp0 % out_height
                tmp = tmp0 // out_height
                od = tmp % out_depth
                oc = tmp // out_depth

                od_stride = od * stride_d
                oh_stride = oh * stride_h
                ow_stride = ow * stride_w

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
                                local_max = T.alloc_local((1,), dtype)
                                local_max[0] = -T.infinity(dtype)
                                for kd in T.serial(kernel_depth):
                                    in_d = od_stride + kd * dilation_d
                                    for kh in T.serial(kernel_height):
                                        in_h = oh_stride + kh * dilation_h
                                        for kw in T.serial(kernel_width):
                                            in_w = ow_stride + kw * dilation_w
                                            if (
                                                (in_d >= 0)
                                                & (in_d < in_depth)
                                                & (in_h >= 0)
                                                & (in_h < in_height)
                                                & (in_w >= 0)
                                                & (in_w < in_width)
                                            ):
                                                val = (
                                                    img_buffer[b, ic_global, in_d, in_h, in_w]
                                                    + weight_buffer[oc, ic_global, kd, kh, kw]
                                                )
                                                if val > local_max[0]:
                                                    local_max[0] = val
                                local_acc[0] += local_max[0]

                    T.atomic_add(sum_val[tb, tp], local_acc[0])
                T.sync_threads()
                if valid_b and valid_p:
                    if tic == 0:
                        output_buffer[b, oc, od, oh, ow] = sum_val[tb, tp]

        return main

    return _kernel()


def max_plus_sum_conv3d_kernel_backward_dweight(
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
    int_dtype="int32",
    autotune=False,
    maxmin_gradient="multiple",
):
    """Compute gradient w.r.t. weight for MaxPlusSumConv3d."""

    BODHW = batch_size * out_depth * out_height * out_width
    KVOL = kernel_depth * kernel_height * kernel_width
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
                "in_depth": in_depth,
                "in_height": in_height,
                "in_width": in_width,
                "kernel_depth": kernel_depth,
                "kernel_height": kernel_height,
                "kernel_width": kernel_width,
                "out_depth": out_depth,
                "out_height": out_height,
                "out_width": out_width,
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
        in_depth=None,
        in_height=None,
        in_width=None,
        kernel_depth=None,
        kernel_height=None,
        kernel_width=None,
        out_depth=None,
        out_height=None,
        out_width=None,
    ):
        MAX_TRANSACTION_SIZE_IN_BITS = 128
        TILE_BODHW = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_BODHW = TILE_BODHW * reduce_threads

        @T.prim_func
        def compute_dweight(
            img_buffer: T.Buffer((batch_size, in_channels, in_depth, in_height, in_width), dtype),
            weight_buffer: T.Buffer((out_channels, in_channels, kernel_depth, kernel_height, kernel_width), dtype),
            dout_buffer: T.Buffer((batch_size, out_channels, out_depth, out_height, out_width), dtype),
            dweight_buffer: T.Buffer((out_channels, in_channels, kernel_depth, kernel_height, kernel_width), dtype),
        ):
            with T.Kernel(
                T.ceildiv(out_channels, BLOCK_OC),
                T.ceildiv(in_channels, BLOCK_IC),
                threads=(BLOCK_OC, BLOCK_IC, reduce_threads),
            ) as (oc_block, ic_block):
                toc = T.get_thread_binding(0)
                tic = T.get_thread_binding(1)
                tbd = T.get_thread_binding(2)

                oc = oc_block * BLOCK_OC + toc
                ic = ic_block * BLOCK_IC + tic
                valid_oc = oc < out_channels
                valid_ic = ic < in_channels

                shared = T.alloc_shared((BLOCK_OC, BLOCK_IC, KVOL, reduce_threads), dtype)
                if (tbd == 0) & (toc == 0) & (tic == 0):
                    T.clear(shared)
                T.sync_threads()

                local_acc = T.alloc_local((KVOL,), dtype)
                T.clear(local_acc)

                if valid_oc & valid_ic:
                    for bodhw_block in T.serial(T.ceildiv(BODHW, BLOCK_BODHW)):
                        start = bodhw_block * BLOCK_BODHW + tbd * TILE_BODHW

                        local_dout = T.alloc_local((TILE_BODHW,), dtype)
                        local_best = T.alloc_local((TILE_BODHW,), dtype)
                        local_img = T.alloc_local((TILE_BODHW, KVOL), dtype)

                        for off in T.vectorized(TILE_BODHW):
                            idx = start + off
                            if idx < BODHW:
                                ow = idx % out_width
                                tmp0 = idx // out_width
                                oh = tmp0 % out_height
                                tmp = tmp0 // out_height
                                od = tmp % out_depth
                                b = tmp // out_depth

                                local_dout[off] = dout_buffer[b, oc, od, oh, ow]
                                local_best[off] = -T.infinity(dtype)

                                od_stride = od * stride_d
                                oh_stride = oh * stride_h
                                ow_stride = ow * stride_w

                                for kd in T.serial(kernel_depth):
                                    in_d = od_stride + kd * dilation_d
                                    for kh in T.serial(kernel_height):
                                        in_h = oh_stride + kh * dilation_h
                                        for kw in T.serial(kernel_width):
                                            in_w = ow_stride + kw * dilation_w
                                            kidx = (kd * kernel_height + kh) * kernel_width + kw
                                            if (
                                                (in_d >= 0)
                                                & (in_d < in_depth)
                                                & (in_h >= 0)
                                                & (in_h < in_height)
                                                & (in_w >= 0)
                                                & (in_w < in_width)
                                            ):
                                                v = (
                                                    img_buffer[b, ic, in_d, in_h, in_w]
                                                    + weight_buffer[oc, ic, kd, kh, kw]
                                                )
                                                local_img[off, kidx] = v
                                                if v > local_best[off]:
                                                    local_best[off] = v
                                            else:
                                                local_img[off, kidx] = -T.infinity(dtype)
                            else:
                                local_dout[off] = T.cast(0, dtype)
                                local_best[off] = -T.infinity(dtype)
                                for kidx in T.serial(KVOL):
                                    local_img[off, kidx] = -T.infinity(dtype)

                        for off in T.serial(TILE_BODHW):
                            idx = start + off
                            if idx < BODHW:
                                best = local_best[off]
                                dout_val = local_dout[off]

                                if maxmin_gradient == "single":
                                    sel = T.alloc_local((1,), int_dtype)
                                    sel[0] = T.cast(0, int_dtype)
                                    found = T.alloc_local((1,), "int32")
                                    found[0] = 0
                                    for kidx in T.serial(KVOL):
                                        if (local_img[off, kidx] == best) & (found[0] == 0):
                                            sel[0] = T.cast(kidx, int_dtype)
                                            found[0] = 1
                                    local_acc[T.cast(sel[0], "int32")] += dout_val
                                else:
                                    cnt = T.alloc_local((1,), "int32")
                                    cnt[0] = 0
                                    for kidx in T.serial(KVOL):
                                        if local_img[off, kidx] == best:
                                            cnt[0] += 1
                                    for kidx in T.serial(KVOL):
                                        if local_img[off, kidx] == best:
                                            local_acc[kidx] += dout_val / T.cast(cnt[0], dtype)

                    for kidx in T.serial(KVOL):
                        shared[toc, tic, kidx, tbd] = local_acc[kidx]

                T.sync_threads()

                if valid_oc & valid_ic:
                    if tbd == 0:
                        for kidx in T.serial(KVOL):
                            total = T.alloc_local((1,), dtype)
                            total[0] = T.cast(0, dtype)
                            for r in T.serial(reduce_threads):
                                total[0] = total[0] + shared[toc, tic, kidx, r]

                            kd = kidx // (kernel_height * kernel_width)
                            rem = kidx % (kernel_height * kernel_width)
                            kh = rem // kernel_width
                            kw = rem % kernel_width
                            dweight_buffer[oc, ic, kd, kh, kw] = total[0]

        return compute_dweight

    return _kernel()


def max_plus_sum_conv3d_kernel_backward_dinput(
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
    int_dtype="int32",
    autotune=False,
    maxmin_gradient="multiple",
):
    """Compute gradient w.r.t. input for MaxPlusSumConv3d."""

    batches = batch_size
    patches = out_channels * out_depth * out_height * out_width
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
                "in_depth": in_depth,
                "in_height": in_height,
                "in_width": in_width,
                "kernel_depth": kernel_depth,
                "kernel_height": kernel_height,
                "kernel_width": kernel_width,
                "out_depth": out_depth,
                "out_height": out_height,
                "out_width": out_width,
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
        in_depth=None,
        in_height=None,
        in_width=None,
        kernel_depth=None,
        kernel_height=None,
        kernel_width=None,
        out_depth=None,
        out_height=None,
        out_width=None,
    ):
        MAX_TRANSACTION_SIZE_IN_BITS = 128
        TILE_IC = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads

        @T.prim_func
        def compute_dinput(
            img_buffer: T.Buffer((batch_size, in_channels, in_depth, in_height, in_width), dtype),
            weight_buffer: T.Buffer((out_channels, in_channels, kernel_depth, kernel_height, kernel_width), dtype),
            dout_buffer: T.Buffer((batch_size, out_channels, out_depth, out_height, out_width), dtype),
            dimg_buffer: T.Buffer((batch_size, in_channels, in_depth, in_height, in_width), dtype),
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
                    ow = patch_idx % out_width
                    tmp0 = patch_idx // out_width
                    oh = tmp0 % out_height
                    tmp = tmp0 // out_height
                    od = tmp % out_depth
                    oc = tmp // out_depth

                    dout_val = dout_buffer[b, oc, od, oh, ow]

                    od_stride = od * stride_d
                    oh_stride = oh * stride_h
                    ow_stride = ow * stride_w

                    for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                        for ic in T.vectorized(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                best = T.alloc_local((1,), dtype)
                                best[0] = -T.infinity(dtype)

                                for kd in T.serial(kernel_depth):
                                    in_d = od_stride + kd * dilation_d
                                    for kh in T.serial(kernel_height):
                                        in_h = oh_stride + kh * dilation_h
                                        for kw in T.serial(kernel_width):
                                            in_w = ow_stride + kw * dilation_w
                                            if (
                                                (in_d >= 0)
                                                & (in_d < in_depth)
                                                & (in_h >= 0)
                                                & (in_h < in_height)
                                                & (in_w >= 0)
                                                & (in_w < in_width)
                                            ):
                                                val = (
                                                    img_buffer[b, ic_global, in_d, in_h, in_w]
                                                    + weight_buffer[oc, ic_global, kd, kh, kw]
                                                )
                                                if val > best[0]:
                                                    best[0] = val

                                if maxmin_gradient == "single":
                                    sel_kd = T.alloc_local((1,), int_dtype)
                                    sel_kh = T.alloc_local((1,), int_dtype)
                                    sel_kw = T.alloc_local((1,), int_dtype)
                                    sel_kd[0] = T.cast(0, int_dtype)
                                    sel_kh[0] = T.cast(0, int_dtype)
                                    sel_kw[0] = T.cast(0, int_dtype)
                                    found = T.alloc_local((1,), "int32")
                                    found[0] = 0
                                    for kd in T.serial(kernel_depth):
                                        in_d = od_stride + kd * dilation_d
                                        for kh in T.serial(kernel_height):
                                            in_h = oh_stride + kh * dilation_h
                                            for kw in T.serial(kernel_width):
                                                in_w = ow_stride + kw * dilation_w
                                                if (
                                                    (in_d >= 0)
                                                    & (in_d < in_depth)
                                                    & (in_h >= 0)
                                                    & (in_h < in_height)
                                                    & (in_w >= 0)
                                                    & (in_w < in_width)
                                                ):
                                                    val = (
                                                        img_buffer[b, ic_global, in_d, in_h, in_w]
                                                        + weight_buffer[oc, ic_global, kd, kh, kw]
                                                    )
                                                    if (val == best[0]) & (found[0] == 0):
                                                        sel_kd[0] = T.cast(kd, int_dtype)
                                                        sel_kh[0] = T.cast(kh, int_dtype)
                                                        sel_kw[0] = T.cast(kw, int_dtype)
                                                        found[0] = 1

                                    in_d = od_stride + T.cast(sel_kd[0], "int32") * dilation_d
                                    in_h = oh_stride + T.cast(sel_kh[0], "int32") * dilation_h
                                    in_w = ow_stride + T.cast(sel_kw[0], "int32") * dilation_w
                                    if (
                                        (in_d >= 0)
                                        & (in_d < in_depth)
                                        & (in_h >= 0)
                                        & (in_h < in_height)
                                        & (in_w >= 0)
                                        & (in_w < in_width)
                                    ):
                                        T.atomic_add(dimg_buffer[b, ic_global, in_d, in_h, in_w], dout_val)
                                else:
                                    cnt = T.alloc_local((1,), "int32")
                                    cnt[0] = 0
                                    for kd in T.serial(kernel_depth):
                                        in_d = od_stride + kd * dilation_d
                                        for kh in T.serial(kernel_height):
                                            in_h = oh_stride + kh * dilation_h
                                            for kw in T.serial(kernel_width):
                                                in_w = ow_stride + kw * dilation_w
                                                if (
                                                    (in_d >= 0)
                                                    & (in_d < in_depth)
                                                    & (in_h >= 0)
                                                    & (in_h < in_height)
                                                    & (in_w >= 0)
                                                    & (in_w < in_width)
                                                ):
                                                    val = (
                                                        img_buffer[b, ic_global, in_d, in_h, in_w]
                                                        + weight_buffer[oc, ic_global, kd, kh, kw]
                                                    )
                                                    if val == best[0]:
                                                        cnt[0] += 1
                                    for kd in T.serial(kernel_depth):
                                        in_d = od_stride + kd * dilation_d
                                        for kh in T.serial(kernel_height):
                                            in_h = oh_stride + kh * dilation_h
                                            for kw in T.serial(kernel_width):
                                                in_w = ow_stride + kw * dilation_w
                                                if (
                                                    (in_d >= 0)
                                                    & (in_d < in_depth)
                                                    & (in_h >= 0)
                                                    & (in_h < in_height)
                                                    & (in_w >= 0)
                                                    & (in_w < in_width)
                                                ):
                                                    val = (
                                                        img_buffer[b, ic_global, in_d, in_h, in_w]
                                                        + weight_buffer[oc, ic_global, kd, kh, kw]
                                                    )
                                                    if val == best[0]:
                                                        T.atomic_add(
                                                            dimg_buffer[b, ic_global, in_d, in_h, in_w],
                                                            dout_val / T.cast(cnt[0], dtype),
                                                        )

        return compute_dinput

    return _kernel()
