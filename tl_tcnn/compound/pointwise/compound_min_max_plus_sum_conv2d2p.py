# pyright: reportInvalidTypeForm=false

"""Pointwise (k=1) compound min/max + plus conv2d2p TileLang kernels.

Implements (no sum over IC):
    Y[n, oc, oh, ow] = a[oc] * max_ic(X[n, ic, ih, iw] + W[oc, ic, 0, 0])
                     + b[oc] * min_ic(X[n, ic, ih, iw] + W[oc, ic, 0, 0])
where ih = oh * stride_h, iw = ow * stride_w on the *padded* input.

Backward tie policy (maxmin_gradient):
- 'single': pick the first (smallest ic) argmax/argmin
- 'multiple': split gradient equally across all ties

Note: This file is intentionally minimal and correctness-first.
"""

import itertools

import tilelang
import tilelang.language as T

from ...autotune_env import _get_autotune_env
from ...utils import DataType, maybe_autotune


def _require_pointwise(kernel_height: int, kernel_width: int) -> None:
    if kernel_height != 1 or kernel_width != 1:
        raise ValueError("This pointwise implementation only supports kernel_size=(1, 1).")


def compound_min_max_plus_sum_conv2d2p_kernel(
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
    int_dtype="int32",
    autotune=False,
):
    _require_pointwise(kernel_height, kernel_width)
    _ = (dilation_h, dilation_w)  # unused for 1x1

    bo_total = batch_size * out_channels
    fast, warmup, rep, max_configs = _get_autotune_env()

    def get_configs(*_kernel_params, **_kwargs):
        max_grid_y = 65535
        max_threads_per_block = 1024

        tile_ic = 128 // DataType(dtype).bits
        if autotune:
            if fast:
                BLOCK_OH = [1, 2, 4, 8]
                BLOCK_OW = [8, 16, 32]
                reduce_threads = [4, 8, 16]
            else:
                BLOCK_OH = [1, 2, 4, 8, 16]
                BLOCK_OW = [1, 2, 4, 8, 16, 32]
                reduce_threads = [1, 2, 4, 8, 16, 32]
        else:
            BLOCK_OH = [1]
            BLOCK_OW = [32]
            reduce_threads = [8]

        configs = []
        for block_oh, block_ow, r_threads in itertools.product(BLOCK_OH, BLOCK_OW, reduce_threads):
            block_p = block_oh * block_ow
            if block_p * r_threads > max_threads_per_block:
                continue
            oh_tiles = (out_height + block_oh - 1) // block_oh
            ow_tiles = (out_width + block_ow - 1) // block_ow
            if oh_tiles * ow_tiles > max_grid_y:
                continue
            # Keep block_ic <= in_channels-ish; not required but avoids silly configs.
            block_ic = tile_ic * r_threads
            if block_ic <= 0:
                continue

            configs.append(
                {
                    "BLOCK_OH": block_oh,
                    "BLOCK_OW": block_ow,
                    "reduce_threads": r_threads,
                    "batch_size": batch_size,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "in_height": in_height,
                    "in_width": in_width,
                    "kernel_height": kernel_height,
                    "kernel_width": kernel_width,
                    "out_height": out_height,
                    "out_width": out_width,
                }
            )

        if max_configs > 0 and len(configs) > max_configs:
            configs = configs[:max_configs]
        if not configs:
            configs = [
                {
                    "BLOCK_OH": 1,
                    "BLOCK_OW": 1,
                    "reduce_threads": 1,
                    "batch_size": batch_size,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "in_height": in_height,
                    "in_width": in_width,
                    "kernel_height": kernel_height,
                    "kernel_width": kernel_width,
                    "out_height": out_height,
                    "out_width": out_width,
                }
            ]
        return configs

    @maybe_autotune(configs=get_configs(), warmup=warmup, rep=rep, enabled=autotune)
    @tilelang.jit(target="auto")
    def _kernel(
        BLOCK_OH=None,
        BLOCK_OW=None,
        reduce_threads=None,
        batch_size=None,
        in_channels=None,
        out_channels=None,
        in_height=None,
        in_width=None,
        kernel_height=None,
        kernel_width=None,
        out_height=None,
        out_width=None,
    ):
        TILE_IC = 128 // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads
        BLOCK_P = BLOCK_OH * BLOCK_OW
        OW_TILES = T.ceildiv(out_width, BLOCK_OW)
        SPATIAL_TILES = T.ceildiv(out_height, BLOCK_OH) * OW_TILES

        @T.prim_func
        def main(
            img: T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight: T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
            alpha: T.Buffer((out_channels,), dtype),
            beta: T.Buffer((out_channels,), dtype),
            out: T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
        ):
            with T.Kernel(bo_total, SPATIAL_TILES, threads=(BLOCK_P, reduce_threads)) as (bo, tile_id):
                tp = T.get_thread_binding(0)
                tic = T.get_thread_binding(1)

                n = bo // out_channels
                oc = bo % out_channels

                tile_oh = tile_id // OW_TILES
                tile_ow = tile_id % OW_TILES
                oh_base = tile_oh * BLOCK_OH
                ow_base = tile_ow * BLOCK_OW

                toh = tp // BLOCK_OW
                tow = tp % BLOCK_OW
                oh = oh_base + toh
                ow = ow_base + tow

                valid = (bo < bo_total) and (oh < out_height) and (ow < out_width)
                ih = oh * stride_h
                iw = ow * stride_w

                local_max = T.alloc_local((1,), dtype)
                local_min = T.alloc_local((1,), dtype)
                local_max[0] = -T.infinity(dtype)
                local_min[0] = T.infinity(dtype)

                for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                    for ii in T.serial(TILE_IC):
                        ic = bic * BLOCK_IC + tic * TILE_IC + ii
                        if valid and ic < in_channels:
                            v = img[n, ic, ih, iw] + weight[oc, ic, 0, 0]
                            if v > local_max[0]:
                                local_max[0] = v
                            if v < local_min[0]:
                                local_min[0] = v

                sh_partial_max = T.alloc_shared((BLOCK_P, reduce_threads), dtype)
                sh_partial_min = T.alloc_shared((BLOCK_P, reduce_threads), dtype)
                sh_partial_max[tp, tic] = local_max[0]
                sh_partial_min[tp, tic] = local_min[0]
                T.sync_threads()

                if tic == 0 and valid:
                    vmax = T.alloc_local((1,), dtype)
                    vmin = T.alloc_local((1,), dtype)
                    vmax[0] = -T.infinity(dtype)
                    vmin[0] = T.infinity(dtype)
                    for r in T.serial(reduce_threads):
                        pmx = sh_partial_max[tp, r]
                        pmn = sh_partial_min[tp, r]
                        if pmx > vmax[0]:
                            vmax[0] = pmx
                        if pmn < vmin[0]:
                            vmin[0] = pmn
                    out[n, oc, oh, ow] = alpha[oc] * vmax[0] + beta[oc] * vmin[0]

        return main

    return _kernel()


def compound_min_max_plus_sum_conv2d2p_kernel_tiled(*args, **kwargs):
    return compound_min_max_plus_sum_conv2d2p_kernel(*args, **kwargs)


def compound_min_max_plus_sum_conv2d2p_backward_dab(
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
    int_dtype="int32",
    autotune=False,
):
    _require_pointwise(kernel_height, kernel_width)
    _ = (int_dtype, dilation_h, dilation_w)

    bo_total = batch_size * out_channels
    fast, warmup, rep, max_configs = _get_autotune_env()

    def get_configs(*_kernel_params, **_kwargs):
        max_grid_y = 65535
        max_threads_per_block = 1024
        tile_ic = 128 // DataType(dtype).bits

        if autotune:
            if fast:
                BLOCK_OH = [1, 2, 4, 8]
                BLOCK_OW = [8, 16, 32]
                reduce_threads = [4, 8, 16]
            else:
                BLOCK_OH = [1, 2, 4, 8, 16]
                BLOCK_OW = [1, 2, 4, 8, 16, 32]
                reduce_threads = [1, 2, 4, 8, 16, 32]
        else:
            BLOCK_OH = [1]
            BLOCK_OW = [32]
            reduce_threads = [8]

        configs = []
        for block_oh, block_ow, r_threads in itertools.product(BLOCK_OH, BLOCK_OW, reduce_threads):
            block_p = block_oh * block_ow
            if block_p * r_threads > max_threads_per_block:
                continue
            oh_tiles = (out_height + block_oh - 1) // block_oh
            ow_tiles = (out_width + block_ow - 1) // block_ow
            if oh_tiles * ow_tiles > max_grid_y:
                continue
            block_ic = tile_ic * r_threads
            if block_ic <= 0:
                continue
            configs.append(
                {
                    "BLOCK_OH": block_oh,
                    "BLOCK_OW": block_ow,
                    "reduce_threads": r_threads,
                    "batch_size": batch_size,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "in_height": in_height,
                    "in_width": in_width,
                    "kernel_height": kernel_height,
                    "kernel_width": kernel_width,
                    "out_height": out_height,
                    "out_width": out_width,
                }
            )

        if max_configs > 0 and len(configs) > max_configs:
            configs = configs[:max_configs]
        if not configs:
            configs = [
                {
                    "BLOCK_OH": 1,
                    "BLOCK_OW": 1,
                    "reduce_threads": 1,
                    "batch_size": batch_size,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "in_height": in_height,
                    "in_width": in_width,
                    "kernel_height": kernel_height,
                    "kernel_width": kernel_width,
                    "out_height": out_height,
                    "out_width": out_width,
                }
            ]
        return configs

    @maybe_autotune(configs=get_configs(), warmup=warmup, rep=rep, enabled=autotune)
    @tilelang.jit(target="auto")
    def _kernel(
        BLOCK_OH=None,
        BLOCK_OW=None,
        reduce_threads=None,
        batch_size=None,
        in_channels=None,
        out_channels=None,
        in_height=None,
        in_width=None,
        kernel_height=None,
        kernel_width=None,
        out_height=None,
        out_width=None,
    ):
        TILE_IC = 128 // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads
        BLOCK_P = BLOCK_OH * BLOCK_OW
        OW_TILES = T.ceildiv(out_width, BLOCK_OW)
        SPATIAL_TILES = T.ceildiv(out_height, BLOCK_OH) * OW_TILES

        @T.prim_func
        def main(
            img: T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight: T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
            dout: T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
            dalpha: T.Buffer((out_channels,), dtype),
            dbeta: T.Buffer((out_channels,), dtype),
        ):
            with T.Kernel(bo_total, SPATIAL_TILES, threads=(BLOCK_P, reduce_threads)) as (bo, tile_id):
                tp = T.get_thread_binding(0)
                tic = T.get_thread_binding(1)

                n = bo // out_channels
                oc = bo % out_channels

                tile_oh = tile_id // OW_TILES
                tile_ow = tile_id % OW_TILES
                oh_base = tile_oh * BLOCK_OH
                ow_base = tile_ow * BLOCK_OW

                toh = tp // BLOCK_OW
                tow = tp % BLOCK_OW
                oh = oh_base + toh
                ow = ow_base + tow

                valid = (bo < bo_total) and (oh < out_height) and (ow < out_width)
                ih = oh * stride_h
                iw = ow * stride_w

                local_max = T.alloc_local((1,), dtype)
                local_min = T.alloc_local((1,), dtype)
                local_max[0] = -T.infinity(dtype)
                local_min[0] = T.infinity(dtype)

                for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                    for ii in T.serial(TILE_IC):
                        ic = bic * BLOCK_IC + tic * TILE_IC + ii
                        if valid and ic < in_channels:
                            v = img[n, ic, ih, iw] + weight[oc, ic, 0, 0]
                            if v > local_max[0]:
                                local_max[0] = v
                            if v < local_min[0]:
                                local_min[0] = v

                sh_partial_max = T.alloc_shared((BLOCK_P, reduce_threads), dtype)
                sh_partial_min = T.alloc_shared((BLOCK_P, reduce_threads), dtype)
                sh_partial_max[tp, tic] = local_max[0]
                sh_partial_min[tp, tic] = local_min[0]
                T.sync_threads()

                if tic == 0 and valid:
                    vmax = T.alloc_local((1,), dtype)
                    vmin = T.alloc_local((1,), dtype)
                    vmax[0] = -T.infinity(dtype)
                    vmin[0] = T.infinity(dtype)
                    for r in T.serial(reduce_threads):
                        pmx = sh_partial_max[tp, r]
                        pmn = sh_partial_min[tp, r]
                        if pmx > vmax[0]:
                            vmax[0] = pmx
                        if pmn < vmin[0]:
                            vmin[0] = pmn
                    g = dout[n, oc, oh, ow]
                    T.atomic_add(dalpha[oc], g * vmax[0])
                    T.atomic_add(dbeta[oc], g * vmin[0])

        return main

    return _kernel()


def compound_min_max_plus_sum_conv2d2p_backward_dweight(
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
    int_dtype="int32",
    autotune=False,
    maxmin_gradient="multiple",
):
    _require_pointwise(kernel_height, kernel_width)
    split_ties = maxmin_gradient != "single"
    _ = (dilation_h, dilation_w)

    bo_total = batch_size * out_channels
    fast, warmup, rep, max_configs = _get_autotune_env()

    def get_configs(*_kernel_params, **_kwargs):
        max_grid_y = 65535
        max_threads_per_block = 1024
        tile_ic = 128 // DataType(dtype).bits

        if autotune:
            if fast:
                BLOCK_OH = [1, 2, 4, 8]
                BLOCK_OW = [8, 16, 32]
                reduce_threads = [4, 8, 16]
            else:
                BLOCK_OH = [1, 2, 4, 8, 16]
                BLOCK_OW = [1, 2, 4, 8, 16, 32]
                reduce_threads = [1, 2, 4, 8, 16, 32]
        else:
            BLOCK_OH = [1]
            BLOCK_OW = [32]
            reduce_threads = [8]

        configs = []
        for block_oh, block_ow, r_threads in itertools.product(BLOCK_OH, BLOCK_OW, reduce_threads):
            block_p = block_oh * block_ow
            if block_p * r_threads > max_threads_per_block:
                continue
            oh_tiles = (out_height + block_oh - 1) // block_oh
            ow_tiles = (out_width + block_ow - 1) // block_ow
            if oh_tiles * ow_tiles > max_grid_y:
                continue
            if (tile_ic * r_threads) <= 0:
                continue
            configs.append(
                {
                    "BLOCK_OH": block_oh,
                    "BLOCK_OW": block_ow,
                    "reduce_threads": r_threads,
                    "batch_size": batch_size,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "in_height": in_height,
                    "in_width": in_width,
                    "kernel_height": kernel_height,
                    "kernel_width": kernel_width,
                    "out_height": out_height,
                    "out_width": out_width,
                }
            )

        if max_configs > 0 and len(configs) > max_configs:
            configs = configs[:max_configs]
        if not configs:
            configs = [
                {
                    "BLOCK_OH": 1,
                    "BLOCK_OW": 1,
                    "reduce_threads": 1,
                    "batch_size": batch_size,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "in_height": in_height,
                    "in_width": in_width,
                    "kernel_height": kernel_height,
                    "kernel_width": kernel_width,
                    "out_height": out_height,
                    "out_width": out_width,
                }
            ]
        return configs

    @maybe_autotune(configs=get_configs(), warmup=warmup, rep=rep, enabled=autotune)
    @tilelang.jit(target="auto")
    def _kernel(
        BLOCK_OH=None,
        BLOCK_OW=None,
        reduce_threads=None,
        batch_size=None,
        in_channels=None,
        out_channels=None,
        in_height=None,
        in_width=None,
        kernel_height=None,
        kernel_width=None,
        out_height=None,
        out_width=None,
        split_ties=split_ties,
    ):
        TILE_IC = 128 // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads
        BLOCK_P = BLOCK_OH * BLOCK_OW
        OW_TILES = T.ceildiv(out_width, BLOCK_OW)
        SPATIAL_TILES = T.ceildiv(out_height, BLOCK_OH) * OW_TILES

        @T.prim_func
        def main(
            img: T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight: T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
            alpha: T.Buffer((out_channels,), dtype),
            beta: T.Buffer((out_channels,), dtype),
            dout: T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
            dweight: T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
        ):
            with T.Kernel(bo_total, SPATIAL_TILES, threads=(BLOCK_P, reduce_threads)) as (bo, tile_id):
                tp = T.get_thread_binding(0)
                tic = T.get_thread_binding(1)

                n = bo // out_channels
                oc = bo % out_channels

                tile_oh = tile_id // OW_TILES
                tile_ow = tile_id % OW_TILES
                oh_base = tile_oh * BLOCK_OH
                ow_base = tile_ow * BLOCK_OW

                toh = tp // BLOCK_OW
                tow = tp % BLOCK_OW
                oh = oh_base + toh
                ow = ow_base + tow

                valid = (bo < bo_total) and (oh < out_height) and (ow < out_width)
                ih = oh * stride_h
                iw = ow * stride_w

                local_max = T.alloc_local((1,), dtype)
                local_min = T.alloc_local((1,), dtype)
                local_max[0] = -T.infinity(dtype)
                local_min[0] = T.infinity(dtype)

                for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                    for ii in T.serial(TILE_IC):
                        ic = bic * BLOCK_IC + tic * TILE_IC + ii
                        if valid and ic < in_channels:
                            v = img[n, ic, ih, iw] + weight[oc, ic, 0, 0]
                            if v > local_max[0]:
                                local_max[0] = v
                            if v < local_min[0]:
                                local_min[0] = v

                sh_partial_max = T.alloc_shared((BLOCK_P, reduce_threads), dtype)
                sh_partial_min = T.alloc_shared((BLOCK_P, reduce_threads), dtype)
                sh_partial_max[tp, tic] = local_max[0]
                sh_partial_min[tp, tic] = local_min[0]
                T.sync_threads()

                vmax = T.alloc_local((1,), dtype)
                vmin = T.alloc_local((1,), dtype)
                vmax[0] = -T.infinity(dtype)
                vmin[0] = T.infinity(dtype)

                sh_kmax = T.alloc_shared((BLOCK_P,), int_dtype)
                sh_kmin = T.alloc_shared((BLOCK_P,), int_dtype)
                sh_idx_max = T.alloc_shared((BLOCK_P,), int_dtype)
                sh_idx_min = T.alloc_shared((BLOCK_P,), int_dtype)
                sh_vmax = T.alloc_shared((BLOCK_P,), dtype)
                sh_vmin = T.alloc_shared((BLOCK_P,), dtype)

                if tic == 0 and valid:
                    for r in T.serial(reduce_threads):
                        pmx = sh_partial_max[tp, r]
                        pmn = sh_partial_min[tp, r]
                        if pmx > vmax[0]:
                            vmax[0] = pmx
                        if pmn < vmin[0]:
                            vmin[0] = pmn
                    sh_kmax[tp] = 0
                    sh_kmin[tp] = 0
                    sh_idx_max[tp] = -1
                    sh_idx_min[tp] = -1
                    sh_vmax[tp] = vmax[0]
                    sh_vmin[tp] = vmin[0]
                T.sync_threads()

                g = T.alloc_local((1,), dtype)
                if valid:
                    g[0] = dout[n, oc, oh, ow]
                else:
                    g[0] = T.cast(0, dtype)

                a = alpha[oc]
                b = beta[oc]

                if split_ties:
                    cnt_max = T.alloc_local((1,), int_dtype)
                    cnt_min = T.alloc_local((1,), int_dtype)
                    cnt_max[0] = 0
                    cnt_min[0] = 0
                    for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                        for ii in T.serial(TILE_IC):
                            ic = bic * BLOCK_IC + tic * TILE_IC + ii
                            if valid and ic < in_channels:
                                v = img[n, ic, ih, iw] + weight[oc, ic, 0, 0]
                                if v == vmax[0]:
                                    cnt_max[0] += 1
                                if v == vmin[0]:
                                    cnt_min[0] += 1

                    sh_cnt_max = T.alloc_shared((BLOCK_P, reduce_threads), int_dtype)
                    sh_cnt_min = T.alloc_shared((BLOCK_P, reduce_threads), int_dtype)
                    sh_cnt_max[tp, tic] = cnt_max[0]
                    sh_cnt_min[tp, tic] = cnt_min[0]
                    T.sync_threads()

                    if tic == 0 and valid:
                        tot_mx = T.alloc_local((1,), int_dtype)
                        tot_mn = T.alloc_local((1,), int_dtype)
                        tot_mx[0] = 0
                        tot_mn[0] = 0
                        for r in T.serial(reduce_threads):
                            tot_mx[0] += sh_cnt_max[tp, r]
                            tot_mn[0] += sh_cnt_min[tp, r]
                        sh_kmax[tp] = tot_mx[0]
                        sh_kmin[tp] = tot_mn[0]
                    T.sync_threads()

                    kmax = sh_kmax[tp]
                    kmin = sh_kmin[tp]
                    for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                        for ii in T.serial(TILE_IC):
                            ic = bic * BLOCK_IC + tic * TILE_IC + ii
                            if valid and ic < in_channels:
                                v = img[n, ic, ih, iw] + weight[oc, ic, 0, 0]
                                if (kmax > 0) and (v == sh_vmax[tp]):
                                    T.atomic_add(dweight[oc, ic, 0, 0], g[0] * a / T.cast(kmax, dtype))
                                if (kmin > 0) and (v == sh_vmin[tp]):
                                    T.atomic_add(dweight[oc, ic, 0, 0], g[0] * b / T.cast(kmin, dtype))
                else:
                    # 'single' tie policy: pick first occurrence by strict >/< scan.
                    if tic == 0 and valid:
                        best_max = T.alloc_local((1,), dtype)
                        best_min = T.alloc_local((1,), dtype)
                        idx_max = T.alloc_local((1,), int_dtype)
                        idx_min = T.alloc_local((1,), int_dtype)
                        best_max[0] = -T.infinity(dtype)
                        best_min[0] = T.infinity(dtype)
                        idx_max[0] = 0
                        idx_min[0] = 0
                        for ic in T.serial(in_channels):
                            v = img[n, ic, ih, iw] + weight[oc, ic, 0, 0]
                            if v > best_max[0]:
                                best_max[0] = v
                                idx_max[0] = ic
                            if v < best_min[0]:
                                best_min[0] = v
                                idx_min[0] = ic
                        T.atomic_add(dweight[oc, idx_max[0], 0, 0], g[0] * a)
                        T.atomic_add(dweight[oc, idx_min[0], 0, 0], g[0] * b)

        return main

    return _kernel()


def compound_min_max_plus_sum_conv2d2p_backward_dinput(
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
    int_dtype="int32",
    autotune=False,
    maxmin_gradient="multiple",
):
    _require_pointwise(kernel_height, kernel_width)
    split_ties = maxmin_gradient != "single"
    _ = (dilation_h, dilation_w)

    bo_total = batch_size * out_channels
    fast, warmup, rep, max_configs = _get_autotune_env()

    def get_configs(*_kernel_params, **_kwargs):
        max_grid_y = 65535
        max_threads_per_block = 1024
        tile_ic = 128 // DataType(dtype).bits

        if autotune:
            if fast:
                BLOCK_OH = [1, 2, 4, 8]
                BLOCK_OW = [8, 16, 32]
                reduce_threads = [4, 8, 16]
            else:
                BLOCK_OH = [1, 2, 4, 8, 16]
                BLOCK_OW = [1, 2, 4, 8, 16, 32]
                reduce_threads = [1, 2, 4, 8, 16, 32]
        else:
            BLOCK_OH = [1]
            BLOCK_OW = [32]
            reduce_threads = [8]

        configs = []
        for block_oh, block_ow, r_threads in itertools.product(BLOCK_OH, BLOCK_OW, reduce_threads):
            block_p = block_oh * block_ow
            if block_p * r_threads > max_threads_per_block:
                continue
            oh_tiles = (out_height + block_oh - 1) // block_oh
            ow_tiles = (out_width + block_ow - 1) // block_ow
            if oh_tiles * ow_tiles > max_grid_y:
                continue
            if (tile_ic * r_threads) <= 0:
                continue
            configs.append(
                {
                    "BLOCK_OH": block_oh,
                    "BLOCK_OW": block_ow,
                    "reduce_threads": r_threads,
                    "batch_size": batch_size,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "in_height": in_height,
                    "in_width": in_width,
                    "kernel_height": kernel_height,
                    "kernel_width": kernel_width,
                    "out_height": out_height,
                    "out_width": out_width,
                }
            )

        if max_configs > 0 and len(configs) > max_configs:
            configs = configs[:max_configs]
        if not configs:
            configs = [
                {
                    "BLOCK_OH": 1,
                    "BLOCK_OW": 1,
                    "reduce_threads": 1,
                    "batch_size": batch_size,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "in_height": in_height,
                    "in_width": in_width,
                    "kernel_height": kernel_height,
                    "kernel_width": kernel_width,
                    "out_height": out_height,
                    "out_width": out_width,
                }
            ]
        return configs

    @maybe_autotune(configs=get_configs(), warmup=warmup, rep=rep, enabled=autotune)
    @tilelang.jit(target="auto")
    def _kernel(
        BLOCK_OH=None,
        BLOCK_OW=None,
        reduce_threads=None,
        batch_size=None,
        in_channels=None,
        out_channels=None,
        in_height=None,
        in_width=None,
        kernel_height=None,
        kernel_width=None,
        out_height=None,
        out_width=None,
        split_ties=split_ties,
    ):
        TILE_IC = 128 // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads
        BLOCK_P = BLOCK_OH * BLOCK_OW
        OW_TILES = T.ceildiv(out_width, BLOCK_OW)
        SPATIAL_TILES = T.ceildiv(out_height, BLOCK_OH) * OW_TILES

        @T.prim_func
        def main(
            img: T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight: T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
            alpha: T.Buffer((out_channels,), dtype),
            beta: T.Buffer((out_channels,), dtype),
            dout: T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
            dimg: T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
        ):
            with T.Kernel(bo_total, SPATIAL_TILES, threads=(BLOCK_P, reduce_threads)) as (bo, tile_id):
                tp = T.get_thread_binding(0)
                tic = T.get_thread_binding(1)

                n = bo // out_channels
                oc = bo % out_channels

                tile_oh = tile_id // OW_TILES
                tile_ow = tile_id % OW_TILES
                oh_base = tile_oh * BLOCK_OH
                ow_base = tile_ow * BLOCK_OW

                toh = tp // BLOCK_OW
                tow = tp % BLOCK_OW
                oh = oh_base + toh
                ow = ow_base + tow

                valid = (bo < bo_total) and (oh < out_height) and (ow < out_width)
                ih = oh * stride_h
                iw = ow * stride_w

                local_max = T.alloc_local((1,), dtype)
                local_min = T.alloc_local((1,), dtype)
                local_max[0] = -T.infinity(dtype)
                local_min[0] = T.infinity(dtype)

                for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                    for ii in T.serial(TILE_IC):
                        ic = bic * BLOCK_IC + tic * TILE_IC + ii
                        if valid and ic < in_channels:
                            v = img[n, ic, ih, iw] + weight[oc, ic, 0, 0]
                            if v > local_max[0]:
                                local_max[0] = v
                            if v < local_min[0]:
                                local_min[0] = v

                sh_partial_max = T.alloc_shared((BLOCK_P, reduce_threads), dtype)
                sh_partial_min = T.alloc_shared((BLOCK_P, reduce_threads), dtype)
                sh_partial_max[tp, tic] = local_max[0]
                sh_partial_min[tp, tic] = local_min[0]
                T.sync_threads()

                vmax = T.alloc_local((1,), dtype)
                vmin = T.alloc_local((1,), dtype)
                vmax[0] = -T.infinity(dtype)
                vmin[0] = T.infinity(dtype)

                sh_kmax = T.alloc_shared((BLOCK_P,), int_dtype)
                sh_kmin = T.alloc_shared((BLOCK_P,), int_dtype)
                sh_idx_max = T.alloc_shared((BLOCK_P,), int_dtype)
                sh_idx_min = T.alloc_shared((BLOCK_P,), int_dtype)
                sh_vmax = T.alloc_shared((BLOCK_P,), dtype)
                sh_vmin = T.alloc_shared((BLOCK_P,), dtype)

                if tic == 0 and valid:
                    for r in T.serial(reduce_threads):
                        pmx = sh_partial_max[tp, r]
                        pmn = sh_partial_min[tp, r]
                        if pmx > vmax[0]:
                            vmax[0] = pmx
                        if pmn < vmin[0]:
                            vmin[0] = pmn
                    sh_kmax[tp] = 0
                    sh_kmin[tp] = 0
                    sh_idx_max[tp] = -1
                    sh_idx_min[tp] = -1
                    sh_vmax[tp] = vmax[0]
                    sh_vmin[tp] = vmin[0]
                T.sync_threads()

                g = T.alloc_local((1,), dtype)
                if valid:
                    g[0] = dout[n, oc, oh, ow]
                else:
                    g[0] = T.cast(0, dtype)

                a = alpha[oc]
                b = beta[oc]

                if split_ties:
                    cnt_max = T.alloc_local((1,), int_dtype)
                    cnt_min = T.alloc_local((1,), int_dtype)
                    cnt_max[0] = 0
                    cnt_min[0] = 0
                    for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                        for ii in T.serial(TILE_IC):
                            ic = bic * BLOCK_IC + tic * TILE_IC + ii
                            if valid and ic < in_channels:
                                v = img[n, ic, ih, iw] + weight[oc, ic, 0, 0]
                                if v == vmax[0]:
                                    cnt_max[0] += 1
                                if v == vmin[0]:
                                    cnt_min[0] += 1

                    sh_cnt_max = T.alloc_shared((BLOCK_P, reduce_threads), int_dtype)
                    sh_cnt_min = T.alloc_shared((BLOCK_P, reduce_threads), int_dtype)
                    sh_cnt_max[tp, tic] = cnt_max[0]
                    sh_cnt_min[tp, tic] = cnt_min[0]
                    T.sync_threads()

                    if tic == 0 and valid:
                        tot_mx = T.alloc_local((1,), int_dtype)
                        tot_mn = T.alloc_local((1,), int_dtype)
                        tot_mx[0] = 0
                        tot_mn[0] = 0
                        for r in T.serial(reduce_threads):
                            tot_mx[0] += sh_cnt_max[tp, r]
                            tot_mn[0] += sh_cnt_min[tp, r]
                        sh_kmax[tp] = tot_mx[0]
                        sh_kmin[tp] = tot_mn[0]
                    T.sync_threads()

                    kmax = sh_kmax[tp]
                    kmin = sh_kmin[tp]
                    for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                        for ii in T.serial(TILE_IC):
                            ic = bic * BLOCK_IC + tic * TILE_IC + ii
                            if valid and ic < in_channels:
                                v = img[n, ic, ih, iw] + weight[oc, ic, 0, 0]
                                if (kmax > 0) and (v == sh_vmax[tp]):
                                    T.atomic_add(dimg[n, ic, ih, iw], g[0] * a / T.cast(kmax, dtype))
                                if (kmin > 0) and (v == sh_vmin[tp]):
                                    T.atomic_add(dimg[n, ic, ih, iw], g[0] * b / T.cast(kmin, dtype))
                else:
                    # 'single' tie policy: pick first occurrence by strict >/< scan.
                    if tic == 0 and valid:
                        best_max = T.alloc_local((1,), dtype)
                        best_min = T.alloc_local((1,), dtype)
                        idx_max = T.alloc_local((1,), int_dtype)
                        idx_min = T.alloc_local((1,), int_dtype)
                        best_max[0] = -T.infinity(dtype)
                        best_min[0] = T.infinity(dtype)
                        idx_max[0] = 0
                        idx_min[0] = 0
                        for ic in T.serial(in_channels):
                            v = img[n, ic, ih, iw] + weight[oc, ic, 0, 0]
                            if v > best_max[0]:
                                best_max[0] = v
                                idx_max[0] = ic
                            if v < best_min[0]:
                                best_min[0] = v
                                idx_min[0] = ic
                        T.atomic_add(dimg[n, idx_max[0], ih, iw], g[0] * a)
                        T.atomic_add(dimg[n, idx_min[0], ih, iw], g[0] * b)

        return main

    return _kernel()


def compound_min_max_plus_sum_conv2d2p_backward_dinput_tiled(*args, **kwargs):
    return compound_min_max_plus_sum_conv2d2p_backward_dinput(*args, **kwargs)
