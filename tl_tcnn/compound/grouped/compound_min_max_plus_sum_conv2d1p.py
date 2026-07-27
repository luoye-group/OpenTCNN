"""This is the file containing kernels used for CompoundMinMaxPlusSumConv2d operation."""

import os
import tilelang
import tilelang.language as T
import itertools
from tilelang.autotuner import AutoTuner
from ...utils import DataType, maybe_autotune
from ...autotune_env import _get_autotune_env

# depthwise (oc = ic = groups), 1 oc 1 ic, do not let different oc share different ic. For example, RGB as IC, the OC should be RGB too, without mixing them, meaning Red OC only connects to Red IC, and so on. This is the depthwise convolution definition.

def compound_min_max_plus_sum_conv2d1p_kernel(
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
    groups=1,
):
    """Fused compound-min-max-plus-sum convolution kernel variant.

    Compared with v2: retained shared memory patch loading, directly used OIHW weight layout
    (out_channels, in_channels, kernel_height, kernel_width), and removed additional kernel
    flattening to avoid parameter mismatch compilation errors.
    """
    bo_total = batch_size * out_channels
    in_per_group = in_channels // groups
    out_per_group = out_channels // groups
    fast, warmup, rep, max_configs = _get_autotune_env()

    def get_configs(*_kernel_params, **_kwargs):
        max_grid_y = 65535
        max_threads_per_block = 1024
        max_shared_bytes_per_block = 48 * 1024
        tile_ic = 128 // DataType(dtype).bits

        if autotune:
            if fast:
                BLOCK_OH = [1, 2, 4, 8, 16,]
                BLOCK_OW = [8, 16, 32]
                reduce_threads = [4, 8, 16]
            else:
                BLOCK_OH = [1, 2, 4, 8, 16, 32]
                BLOCK_OW = [1, 2, 4, 8, 16, 32]
                reduce_threads = [4, 8, 16, 32]
        else:
            BLOCK_OH = [1]
            BLOCK_OW = [32]
            reduce_threads = [8]

        _configs = list(itertools.product(BLOCK_OH, BLOCK_OW, reduce_threads))
        configs = []
        for c in _configs:
            block_oh, block_ow, r_threads = c
            if r_threads > in_per_group:
                continue
            block_p = block_oh * block_ow
            threads_per_block = block_p * r_threads
            if threads_per_block > max_threads_per_block:
                continue

            oh_tiles = (out_height + block_oh - 1) // block_oh
            ow_tiles = (out_width + block_ow - 1) // block_ow
            grid_y = oh_tiles * ow_tiles
            if grid_y > max_grid_y:
                continue

            block_ic = tile_ic * r_threads
            tile_ih = (block_oh - 1) * stride_h + (kernel_height - 1) * dilation_h + 1
            tile_iw = (block_ow - 1) * stride_w + (kernel_width - 1) * dilation_w + 1

            shared_elems = (
                block_ic * tile_ih * tile_iw
                + block_ic * kernel_height * kernel_width
                + block_ic
                + block_ic
                + block_p * r_threads
            )
            shared_bytes = shared_elems * DataType(dtype).bits // 8
            if shared_bytes > max_shared_bytes_per_block:
                continue

            configs.append({
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
            "groups": groups,
        })
        
        if max_configs > 0 and len(configs) > max_configs:
            configs = configs[:max_configs]
        if not configs:
            configs = [{
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
                "groups": groups,
            }]
        
        return configs
    
    @maybe_autotune(
        configs=get_configs(),
        warmup=warmup,
        rep=rep,
        enabled=autotune,
    )
    @tilelang.jit(
        target="auto",
    )
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
        groups=None,
    ):
        MAX_TRANSACTION_SIZE_IN_BITS = 128

        TILE_IC = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads
        BLOCK_P = BLOCK_OH * BLOCK_OW
        TILE_IH = (BLOCK_OH - 1) * stride_h + (kernel_height - 1) * dilation_h + 1
        TILE_IW = (BLOCK_OW - 1) * stride_w + (kernel_width - 1) * dilation_w + 1
        OW_TILES = T.ceildiv(out_width, BLOCK_OW)
        SPATIAL_TILES = T.ceildiv(out_height, BLOCK_OH) * OW_TILES
        TILE_IHW = TILE_IH * TILE_IW
        IMG_ELEMS = BLOCK_IC * TILE_IHW
        KERNEL_HW = kernel_height * kernel_width
        WEIGHT_ELEMS = BLOCK_IC * KERNEL_HW
        
        @T.prim_func
        def main(
            img_buffer : T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight_buffer : T.Buffer((out_channels, in_per_group, kernel_height, kernel_width), dtype),
            alpha_buffer : T.Buffer((out_channels, in_per_group), dtype),
            output_buffer : T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
        ):

            with T.Kernel(bo_total, SPATIAL_TILES, threads=(BLOCK_P, reduce_threads)) as (bo, tile_id):
                tp = T.get_thread_binding(0)
                tic = T.get_thread_binding(1)

                b = bo // out_channels
                oc = bo % out_channels
                group_id = oc // out_per_group
                group_in_base = group_id * in_per_group

                tile_oh = tile_id // OW_TILES
                tile_ow = tile_id % OW_TILES
                oh_base = tile_oh * BLOCK_OH
                ow_base = tile_ow * BLOCK_OW

                toh = tp // BLOCK_OW
                tow = tp % BLOCK_OW
                oh = oh_base + toh
                ow = ow_base + tow

                valid_bo = bo < bo_total
                valid_oh = oh < out_height
                valid_ow = ow < out_width
                valid_out = valid_bo and valid_oh and valid_ow

                # Shared-memory staged tiles.
                sh_img = T.alloc_shared((BLOCK_IC, TILE_IH, TILE_IW), dtype)
                sh_weight = T.alloc_shared((BLOCK_IC, kernel_height, kernel_width), dtype)
                sh_alpha = T.alloc_shared((BLOCK_IC,), dtype)
                sh_sum = T.alloc_shared((BLOCK_P, reduce_threads), dtype)

                local_max = T.alloc_local((1,), dtype)
                local_min = T.alloc_local((1,), dtype)
                local_neg_max = T.alloc_local((1,), dtype)
                local_accum = T.alloc_local((1,), dtype)
                local_accum[0] = T.cast(0, dtype)

                linear_tid = tic * BLOCK_P + tp
                total_threads = BLOCK_P * reduce_threads

                for bic in T.serial(T.ceildiv(in_per_group, BLOCK_IC)):
                    for idx in T.serial(T.ceildiv(IMG_ELEMS, total_threads)):
                        linear = idx * total_threads + linear_tid
                        if linear < IMG_ELEMS:
                            ic_local = linear // (TILE_IHW)
                            rem0 = linear % (TILE_IHW)
                            tih = rem0 // TILE_IW
                            tiw = rem0 % TILE_IW

                            ic_global = group_in_base + bic * BLOCK_IC + ic_local
                            in_h = oh_base * stride_h + tih
                            in_w = ow_base * stride_w + tiw
                            if valid_bo and ic_global < in_channels and in_h < in_height and in_w < in_width:
                                sh_img[ic_local, tih, tiw] = img_buffer[b, ic_global, in_h, in_w]
                            else:
                                sh_img[ic_local, tih, tiw] = T.cast(0, dtype)

                    for idx in T.serial(T.ceildiv(WEIGHT_ELEMS, total_threads)):
                        linear = idx * total_threads + linear_tid
                        if linear < WEIGHT_ELEMS:
                            ic_local = linear // (KERNEL_HW)
                            rem1 = linear % (KERNEL_HW)
                            kh = rem1 // kernel_width
                            kw = rem1 % kernel_width

                            ic_global = bic * BLOCK_IC + ic_local
                            if valid_bo and ic_global < in_per_group:
                                sh_weight[ic_local, kh, kw] = weight_buffer[oc, ic_global, kh, kw]
                            else:
                                sh_weight[ic_local, kh, kw] = T.cast(0, dtype)

                    for idx in T.serial(T.ceildiv(BLOCK_IC, total_threads)):
                        ic_local = idx * total_threads + linear_tid
                        if ic_local < BLOCK_IC:
                            ic_global = bic * BLOCK_IC + ic_local
                            if valid_bo and ic_global < in_per_group:
                                sh_alpha[ic_local] = alpha_buffer[oc, ic_global]
                            else:
                                sh_alpha[ic_local] = T.cast(0, dtype)
                    T.sync_threads()
                    if valid_out:
                        oh_off = toh * stride_h
                        ow_off = tow * stride_w
                        for ic in T.serial(TILE_IC):
                            ic_local = tic * TILE_IC + ic
                            ic_global = bic * BLOCK_IC + ic_local
                            if ic_global < in_channels:
                                local_max[0] = -T.infinity(dtype)
                                local_neg_max[0] = -T.infinity(dtype)

                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_tih = oh_off + kh * dilation_h
                                        in_tiw = ow_off + kw * dilation_w
                                        if in_tih < TILE_IH and in_tiw < TILE_IW:
                                            val = sh_img[ic_local, in_tih, in_tiw] + sh_weight[ic_local, kh, kw]
                                            if val > local_max[0]:
                                                local_max[0] = val
                                            if -val > local_neg_max[0]:
                                                local_neg_max[0] = -val

                                local_min[0] = -local_neg_max[0]
                                if local_max[0] != -T.infinity(dtype):
                                    local_accum[0] += sh_alpha[ic_local] * local_max[0]
                                if local_min[0] != T.infinity(dtype):
                                    local_accum[0] += sh_alpha[ic_local] * local_min[0]
                    T.sync_threads()

                sh_sum[tp, tic] = local_accum[0]
                T.sync_threads()

                if tic == 0 and valid_out:
                    total = T.alloc_local((1,), dtype)
                    total[0] = T.cast(0, dtype)
                    for r in T.serial(reduce_threads):
                        total[0] = total[0] + sh_sum[tp, r]
                    output_buffer[b, oc, oh, ow] = total[0]

        return main
    
    return _kernel()


def compound_min_max_plus_sum_conv2d1p_kernel_tiled(
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
    """Alias to the shared-optimized tiled forward kernel."""
    return compound_min_max_plus_sum_conv2d1p_kernel(
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
        dtype=dtype,
        int_dtype=int_dtype,
        autotune=autotune,
    )




# @tilelang.jit
def compound_min_max_plus_sum_conv2d1p_backward_da(
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
    groups=1,
):
    """Shared-memory optimized gradient kernel for `dalpha`."""
    BHW = batch_size * out_height * out_width
    in_per_group = in_channels // groups
    out_per_group = out_channels // groups
    fast, warmup, rep, max_configs = _get_autotune_env()

    def get_configs(*_kernel_params, **_kwargs):
        if autotune:
            if fast:
                BLOCK_OC = [1]
                BLOCK_IC = [2, 4, 8]
                reduce_threads = [8, 16, 32]
            else:
                BLOCK_OC = [1]
                BLOCK_IC = [1, 2, 4, 8, 16, 32]
                reduce_threads = [1, 2, 4, 8, 16, 32, 64]
        else:
            BLOCK_OC = [1]
            BLOCK_IC = [2]
            reduce_threads = [16]

        cfgs = []
        for bo, bi, rt in itertools.product(BLOCK_OC, BLOCK_IC, reduce_threads):
            if bo * bi * rt > 1024:
                continue
            cfgs.append({
                "BLOCK_OC": bo,
                "BLOCK_IC": bi,
                "reduce_threads": rt,
                "batch_size": batch_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "in_height": in_height,
                "in_width": in_width,
                "kernel_height": kernel_height,
                "kernel_width": kernel_width,
                "out_height": out_height,
                "out_width": out_width,
                "groups": groups,
            })
        if max_configs > 0 and len(cfgs) > max_configs:
            return cfgs[:max_configs]
        return cfgs

    @maybe_autotune(configs=get_configs(), warmup=warmup, rep=rep, enabled=autotune)
    @tilelang.jit(target="auto")
    def _da_kernel(
        BLOCK_OC=None,
        BLOCK_IC=None,
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
        groups=None,
    ):
        @T.prim_func
        def compute_dalpha(
            img_buffer: T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight_buffer: T.Buffer((out_channels, in_per_group, kernel_height, kernel_width), dtype),
            dout_buffer: T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
            dalpha_buffer: T.Buffer((out_channels, in_per_group), dtype),
        ):
            with T.Kernel(
                T.ceildiv(out_channels, BLOCK_OC),
                T.ceildiv(in_per_group, BLOCK_IC),
                threads=(BLOCK_OC, BLOCK_IC, reduce_threads),
            ) as (oc_blk, ic_blk):
                toc = T.get_thread_binding(0)
                tic = T.get_thread_binding(1)
                tr = T.get_thread_binding(2)

                oc = oc_blk * BLOCK_OC + toc
                ic = ic_blk * BLOCK_IC + tic
                group_id = oc // out_per_group
                group_in_base = group_id * in_per_group
                valid = (oc < out_channels) and (ic < in_per_group)

                sh_weight = T.alloc_shared((BLOCK_OC, BLOCK_IC, kernel_height, kernel_width), dtype)
                sh_img = T.alloc_shared((reduce_threads, BLOCK_IC, kernel_height, kernel_width), dtype)
                sh_dout = T.alloc_shared((reduce_threads, BLOCK_OC), dtype)
                sh_dalpha = T.alloc_shared((BLOCK_OC, BLOCK_IC, reduce_threads), dtype)

                local_dalpha = T.alloc_local((1,), dtype)
                local_dalpha[0] = T.cast(0, dtype)

                if valid:
                    for kh in T.serial(kernel_height):
                        for kw in T.serial(kernel_width):
                            sh_weight[toc, tic, kh, kw] = weight_buffer[oc, ic, kh, kw]

                T.sync_threads()

                for bhw_it in T.serial(T.ceildiv(BHW, reduce_threads)):
                    bhw0 = bhw_it * reduce_threads + tr
                    if bhw0 < BHW:
                        tmp1 = bhw0 // out_width
                        ow = bhw0 % out_width
                        b = tmp1 // out_height
                        oh = tmp1 % out_height
                        h0 = oh * stride_h
                        w0 = ow * stride_w

                        if toc == 0 and tic < BLOCK_IC:
                            icg = group_in_base + ic_blk * BLOCK_IC + tic
                            if icg < in_channels:
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        ih = h0 + kh * dilation_h
                                        iw = w0 + kw * dilation_w
                                        if (ih >= 0) and (ih < in_height) and (iw >= 0) and (iw < in_width):
                                            sh_img[tr, tic, kh, kw] = img_buffer[b, icg, ih, iw]
                                        else:
                                            sh_img[tr, tic, kh, kw] = T.cast(0, dtype)
                            else:
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        sh_img[tr, tic, kh, kw] = T.cast(0, dtype)

                        if tic == 0 and toc < BLOCK_OC:
                            ocg = oc_blk * BLOCK_OC + toc
                            if ocg < out_channels:
                                sh_dout[tr, toc] = dout_buffer[b, ocg, oh, ow]
                            else:
                                sh_dout[tr, toc] = T.cast(0, dtype)
                    T.sync_threads()
                    if valid and bhw0 < BHW:
                        vmax = T.alloc_local((1,), dtype)
                        vnegmax = T.alloc_local((1,), dtype)
                        vmax[0] = -T.infinity(dtype)
                        vnegmax[0] = -T.infinity(dtype)
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                v = sh_img[tr, tic, kh, kw] + sh_weight[toc, tic, kh, kw]
                                if v > vmax[0]:
                                    vmax[0] = v
                                if -v > vnegmax[0]:
                                    vnegmax[0] = -v
                        vmin = T.alloc_local((1,), dtype)
                        vmin[0] = -vnegmax[0]
                        if vmax[0] != -T.infinity(dtype):
                            local_dalpha[0] = local_dalpha[0] + sh_dout[tr, toc] * vmax[0]
                        if vmin[0] != T.infinity(dtype):
                            local_dalpha[0] = local_dalpha[0] + sh_dout[tr, toc] * vmin[0]
                    T.sync_threads()

                T.sync_threads()

                if valid:
                    sh_dalpha[toc, tic, tr] = local_dalpha[0]

                T.sync_threads()

                if valid and tr == 0:
                    out_a = T.alloc_local((1,), dtype)
                    out_a[0] = T.cast(0, dtype)
                    for r in T.serial(reduce_threads):
                        out_a[0] = out_a[0] + sh_dalpha[toc, tic, r]
                    dalpha_buffer[oc, ic] = out_a[0]

                T.sync_threads()

        return compute_dalpha

    return _da_kernel()


# @tilelang.jit
def compound_min_max_plus_sum_conv2d1p_backward_dweight(
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
    groups=1,
    maxmin_gradient="multiple" # multiple 1/k for more than one max/min position, single picks one of them (e.i. the first one)
):
    """Shared-memory optimized `dweight` kernel."""
    BHW = batch_size * out_height * out_width
    in_per_group = in_channels // groups
    out_per_group = out_channels // groups
    fast, warmup, rep, max_configs = _get_autotune_env()

    def get_configs(*_kernel_params, **_kwargs):
        if autotune:
            if fast:
                BLOCK_OC = [1]
                BLOCK_IC = [1, 2, 4]
                reduce_threads = [8, 16, 32]
            else:
                BLOCK_OC = [1]
                BLOCK_IC = [1, 2, 4, 8, 16, 32]
                reduce_threads = [1, 2, 4, 8, 16, 32, 64]
        else:
            BLOCK_OC = [1]
            BLOCK_IC = [1]
            reduce_threads = [16]

        cfgs = []
        for bo, bi, rt in itertools.product(BLOCK_OC, BLOCK_IC, reduce_threads):
            if bo * bi * rt > 1024:
                continue
            cfgs.append({
                "BLOCK_OC": bo,
                "BLOCK_IC": bi,
                "reduce_threads": rt,
                "batch_size": batch_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "in_height": in_height,
                "in_width": in_width,
                "kernel_height": kernel_height,
                "kernel_width": kernel_width,
                "out_height": out_height,
                "out_width": out_width,
                "groups": groups,
            })
        if max_configs > 0 and len(cfgs) > max_configs:
            return cfgs[:max_configs]
        return cfgs

    split_ties = (maxmin_gradient != "single")

    @maybe_autotune(configs=get_configs(), warmup=warmup, rep=rep, enabled=autotune)
    @tilelang.jit(target="auto")
    def _kernel(
        BLOCK_OC=None,
        BLOCK_IC=None,
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
        groups=None,
        split_ties=split_ties,
    ):
        @T.prim_func
        def compute_dweight(
            img_buffer: T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight_buffer: T.Buffer((out_channels, in_per_group, kernel_height, kernel_width), dtype),
            alpha_buffer: T.Buffer((out_channels, in_per_group), dtype),
            dout_buffer: T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
            dweight_buffer: T.Buffer((out_channels, in_per_group, kernel_height, kernel_width), dtype),
        ):
            with T.Kernel(
                T.ceildiv(out_channels, BLOCK_OC),
                T.ceildiv(in_per_group, BLOCK_IC),
                threads=(BLOCK_OC, BLOCK_IC, reduce_threads),
            ) as (oc_blk, ic_blk):
                toc = T.get_thread_binding(0)
                tic = T.get_thread_binding(1)
                tr = T.get_thread_binding(2)

                oc = oc_blk * BLOCK_OC + toc
                ic = ic_blk * BLOCK_IC + tic
                group_id = oc // out_per_group
                group_in_base = group_id * in_per_group
                valid = (oc < out_channels) and (ic < in_per_group)

                sh_weight = T.alloc_shared((BLOCK_OC, BLOCK_IC, kernel_height, kernel_width), dtype)
                sh_alpha = T.alloc_shared((BLOCK_OC, BLOCK_IC), dtype)
                sh_img = T.alloc_shared((reduce_threads, BLOCK_IC, kernel_height, kernel_width), dtype)
                sh_dout = T.alloc_shared((reduce_threads, BLOCK_OC), dtype)
                sh_dw = T.alloc_shared((BLOCK_OC, BLOCK_IC, kernel_height, kernel_width, reduce_threads), dtype)

                local_dw = T.alloc_local((kernel_height, kernel_width), dtype)
                T.clear(local_dw)

                if valid:
                    for kh in T.serial(kernel_height):
                        for kw in T.serial(kernel_width):
                            sh_weight[toc, tic, kh, kw] = weight_buffer[oc, ic, kh, kw]
                    sh_alpha[toc, tic] = alpha_buffer[oc, ic]

                T.sync_threads()

                for bhw_it in T.serial(T.ceildiv(BHW, reduce_threads)):
                    bhw0 = bhw_it * reduce_threads + tr
                    if bhw0 < BHW:
                        tmp1 = bhw0 // out_width
                        ow = bhw0 % out_width
                        b = tmp1 // out_height
                        oh = tmp1 % out_height
                        h0 = oh * stride_h
                        w0 = ow * stride_w

                        if toc == 0 and tic < BLOCK_IC:
                            icg = group_in_base + ic_blk * BLOCK_IC + tic
                            if icg < in_channels:
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        ih = h0 + kh * dilation_h
                                        iw = w0 + kw * dilation_w
                                        if (ih >= 0) and (ih < in_height) and (iw >= 0) and (iw < in_width):
                                            sh_img[tr, tic, kh, kw] = img_buffer[b, icg, ih, iw]
                                        else:
                                            sh_img[tr, tic, kh, kw] = T.cast(0, dtype)
                            else:
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        sh_img[tr, tic, kh, kw] = T.cast(0, dtype)

                        if tic == 0 and toc < BLOCK_OC:
                            ocg = oc_blk * BLOCK_OC + toc
                            if ocg < out_channels:
                                sh_dout[tr, toc] = dout_buffer[b, ocg, oh, ow]
                            else:
                                sh_dout[tr, toc] = T.cast(0, dtype)
                    T.sync_threads()

                    if valid and bhw0 < BHW:
                        vmax = T.alloc_local((1,), dtype)
                        vnegmax = T.alloc_local((1,), dtype)
                        vmax[0] = -T.infinity(dtype)
                        vnegmax[0] = -T.infinity(dtype)
                        vals = T.alloc_local((kernel_height, kernel_width), dtype)
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                vals[kh, kw] = sh_img[tr, tic, kh, kw] + sh_weight[toc, tic, kh, kw]
                                if vals[kh, kw] > vmax[0]:
                                    vmax[0] = vals[kh, kw]
                                if -vals[kh, kw] > vnegmax[0]:
                                    vnegmax[0] = -vals[kh, kw]

                        if split_ties:
                            kmax = T.alloc_local((1,), int_dtype)
                            kmin = T.alloc_local((1,), int_dtype)
                            kmax[0] = 0
                            kmin[0] = 0
                            for kh in T.serial(kernel_height):
                                for kw in T.serial(kernel_width):
                                    if (vmax[0] != -T.infinity(dtype)) and (vals[kh, kw] == vmax[0]):
                                        kmax[0] += 1
                                    if (vnegmax[0] != -T.infinity(dtype)) and (-vals[kh, kw] == vnegmax[0]):
                                        kmin[0] += 1
                            for kh in T.serial(kernel_height):
                                for kw in T.serial(kernel_width):
                                    if (kmax[0] > 0) and (vals[kh, kw] == vmax[0]):
                                        local_dw[kh, kw] = local_dw[kh, kw] + (
                                            sh_alpha[toc, tic] * sh_dout[tr, toc] / T.cast(kmax[0], dtype)
                                        )
                                    if (kmin[0] > 0) and (-vals[kh, kw] == vnegmax[0]):
                                        local_dw[kh, kw] = local_dw[kh, kw] + (
                                            sh_alpha[toc, tic] * sh_dout[tr, toc] / T.cast(kmin[0], dtype)
                                        )
                        else:
                            max_h = T.alloc_local((1,), int_dtype)
                            max_w = T.alloc_local((1,), int_dtype)
                            min_h = T.alloc_local((1,), int_dtype)
                            min_w = T.alloc_local((1,), int_dtype)
                            max_h[0] = -1
                            max_w[0] = -1
                            min_h[0] = -1
                            min_w[0] = -1
                            for kh in T.serial(kernel_height):
                                for kw in T.serial(kernel_width):
                                    if (vmax[0] != -T.infinity(dtype)) and (vals[kh, kw] == vmax[0]) and (max_h[0] == -1):
                                        max_h[0] = kh
                                        max_w[0] = kw
                                    if (vnegmax[0] != -T.infinity(dtype)) and (-vals[kh, kw] == vnegmax[0]) and (min_h[0] == -1):
                                        min_h[0] = kh
                                        min_w[0] = kw
                            if max_h[0] >= 0:
                                local_dw[max_h[0], max_w[0]] = local_dw[max_h[0], max_w[0]] + sh_alpha[toc, tic] * sh_dout[tr, toc]
                            if min_h[0] >= 0:
                                local_dw[min_h[0], min_w[0]] = local_dw[min_h[0], min_w[0]] + sh_alpha[toc, tic] * sh_dout[tr, toc]
                    T.sync_threads()

                T.sync_threads()

                if valid:
                    for kh in T.serial(kernel_height):
                        for kw in T.serial(kernel_width):
                            sh_dw[toc, tic, kh, kw, tr] = local_dw[kh, kw]

                T.sync_threads()

                if valid and tr == 0:
                    for kh in T.serial(kernel_height):
                        for kw in T.serial(kernel_width):
                            outv = T.alloc_local((1,), dtype)
                            outv[0] = T.cast(0, dtype)
                            for r in T.serial(reduce_threads):
                                outv[0] = outv[0] + sh_dw[toc, tic, kh, kw, r]
                            dweight_buffer[oc, ic, kh, kw] = outv[0]

                T.sync_threads()

        return compute_dweight

    return _kernel()



# @tilelang.jit
def compound_min_max_plus_sum_conv2d1p_backward_dinput(
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
    groups=1,
    maxmin_gradient="multiple" # multiple 1/k for more than one max/min position, single picks one of them (e.i. the first one)
):
    """Shared-memory optimized `dinput` kernel (single/multiple max-min ties)."""
    bo_total = batch_size * out_channels
    in_per_group = in_channels // groups
    out_per_group = out_channels // groups
    fast, warmup, rep, max_configs = _get_autotune_env()

    def get_configs(*_kernel_params, **_kwargs):
        max_grid_y = 65535
        max_threads_per_block = 1024
        max_shared_bytes_per_block = 48 * 1024
        tile_ic = 128 // DataType(dtype).bits

        if autotune:
            if fast:
                BLOCK_OH = [1, 2, 4, 8]
                BLOCK_OW = [8, 16, 32]
                reduce_threads = [4, 8, 16]
            else:
                BLOCK_OH = [1, 2, 4, 8, 16, 32]
                BLOCK_OW = [1, 2, 4, 8, 16, 32]
                reduce_threads = [1, 2, 4, 8, 16, 32]
        else:
            BLOCK_OH = [1]
            BLOCK_OW = [16]
            reduce_threads = [8]

        cfgs = []
        for boh, bow, rt in itertools.product(BLOCK_OH, BLOCK_OW, reduce_threads):
            block_p = boh * bow
            if block_p * rt > max_threads_per_block:
                continue
            oh_tiles = (out_height + boh - 1) // boh
            ow_tiles = (out_width + bow - 1) // bow
            if oh_tiles * ow_tiles > max_grid_y:
                continue

            block_ic = tile_ic * rt
            shared_elems = (
                block_ic * ((boh - 1) * stride_h + (kernel_height - 1) * dilation_h + 1) * ((bow - 1) * stride_w + (kernel_width - 1) * dilation_w + 1)
                + block_ic * kernel_height * kernel_width
                + block_ic
                + block_ic
            )
            if shared_elems * DataType(dtype).bits // 8 > max_shared_bytes_per_block:
                continue

            cfgs.append({
                "BLOCK_OH": boh,
                "BLOCK_OW": bow,
                "reduce_threads": rt,
                "batch_size": batch_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "in_height": in_height,
                "in_width": in_width,
                "kernel_height": kernel_height,
                "kernel_width": kernel_width,
                "out_height": out_height,
                "out_width": out_width,
                "groups": groups,
            })

        if max_configs > 0 and len(cfgs) > max_configs:
            return cfgs[:max_configs]
        return cfgs

    split_ties = (maxmin_gradient != "single")

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
        groups=None,
        split_ties=split_ties,
    ):
        TILE_IC = 128 // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads
        BLOCK_P = BLOCK_OH * BLOCK_OW
        TILE_IH = (BLOCK_OH - 1) * stride_h + (kernel_height - 1) * dilation_h + 1
        TILE_IW = (BLOCK_OW - 1) * stride_w + (kernel_width - 1) * dilation_w + 1
        OW_TILES = T.ceildiv(out_width, BLOCK_OW)
        SPATIAL_TILES = T.ceildiv(out_height, BLOCK_OH) * OW_TILES
        TILE_IHW = TILE_IH * TILE_IW
        IMG_ELEMS = BLOCK_IC * TILE_IHW
        KERNEL_HW = kernel_height * kernel_width
        WEIGHT_ELEMS = BLOCK_IC * KERNEL_HW
        
        @T.prim_func
        def compute_dinput(
            img_buffer: T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight_buffer: T.Buffer((out_channels, in_per_group, kernel_height, kernel_width), dtype),
            alpha_buffer: T.Buffer((out_channels, in_per_group), dtype),
            dout_buffer: T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
            dimg_buffer: T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
        ):
            with T.Kernel(bo_total, SPATIAL_TILES, threads=(BLOCK_P, reduce_threads)) as (bo, tile_id):
                tp = T.get_thread_binding(0)
                tic = T.get_thread_binding(1)

                b = bo // out_channels
                oc = bo % out_channels
                group_id = oc // out_per_group
                group_in_base = group_id * in_per_group

                tile_oh = tile_id // OW_TILES
                tile_ow = tile_id % OW_TILES
                oh_base = tile_oh * BLOCK_OH
                ow_base = tile_ow * BLOCK_OW

                toh = tp // BLOCK_OW
                tow = tp % BLOCK_OW
                oh = oh_base + toh
                ow = ow_base + tow

                valid = (bo < bo_total) and (oh < out_height) and (ow < out_width)

                sh_img = T.alloc_shared((BLOCK_IC, TILE_IH, TILE_IW), dtype)
                sh_weight = T.alloc_shared((BLOCK_IC, kernel_height, kernel_width), dtype)
                sh_alpha = T.alloc_shared((BLOCK_IC,), dtype)

                linear_tid = tic * BLOCK_P + tp
                total_threads = BLOCK_P * reduce_threads
                local_dout = T.alloc_local((1,), dtype)
                if valid:
                    local_dout[0] = dout_buffer[b, oc, oh, ow]
                else:
                    local_dout[0] = T.cast(0, dtype)

                for bic in T.serial(T.ceildiv(in_per_group, BLOCK_IC)):
                    for idx in T.serial(T.ceildiv(IMG_ELEMS, total_threads)):
                        linear = idx * total_threads + linear_tid
                        if linear < IMG_ELEMS:
                            icl = linear // (TILE_IHW)
                            rem = linear % (TILE_IHW)
                            tih = rem // TILE_IW
                            tiw = rem % TILE_IW
                            icg = group_in_base + bic * BLOCK_IC + icl
                            ih = oh_base * stride_h + tih
                            iw = ow_base * stride_w + tiw
                            if (bo < bo_total) and (icg < in_channels) and (ih < in_height) and (iw < in_width):
                                sh_img[icl, tih, tiw] = img_buffer[b, icg, ih, iw]
                            else:
                                sh_img[icl, tih, tiw] = T.cast(0, dtype)

                    for idx in T.serial(T.ceildiv(WEIGHT_ELEMS, total_threads)):
                        linear = idx * total_threads + linear_tid
                        if linear < WEIGHT_ELEMS:
                            icl = linear // (KERNEL_HW)
                            rem = linear % (KERNEL_HW)
                            kh = rem // kernel_width
                            kw = rem % kernel_width
                            icg = bic * BLOCK_IC + icl
                            if (bo < bo_total) and (icg < in_per_group):
                                sh_weight[icl, kh, kw] = weight_buffer[oc, icg, kh, kw]
                            else:
                                sh_weight[icl, kh, kw] = T.cast(0, dtype)

                    for idx in T.serial(T.ceildiv(BLOCK_IC, total_threads)):
                        icl = idx * total_threads + linear_tid
                        if icl < BLOCK_IC:
                            icg = bic * BLOCK_IC + icl
                            if (bo < bo_total) and (icg < in_per_group):
                                sh_alpha[icl] = alpha_buffer[oc, icg]
                            else:
                                sh_alpha[icl] = T.cast(0, dtype)

                    T.sync_threads()

                    if valid:
                    # for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                        oh_off = toh * stride_h
                        ow_off = tow * stride_w
                        for ii in T.serial(TILE_IC):
                            icl = tic * TILE_IC + ii
                            ic_local = bic * BLOCK_IC + icl
                            icg = group_in_base + ic_local
                            if ic_local < in_per_group:
                                vmax = T.alloc_local((1,), dtype)
                                vnegmax = T.alloc_local((1,), dtype)
                                vmax[0] = -T.infinity(dtype)
                                vnegmax[0] = -T.infinity(dtype)
                                vals = T.alloc_local((kernel_height, kernel_width), dtype)
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        tih = oh_off + kh * dilation_h
                                        tiw = ow_off + kw * dilation_w
                                        if tih < TILE_IH and tiw < TILE_IW:
                                            vals[kh, kw] = sh_img[icl, tih, tiw] + sh_weight[icl, kh, kw]
                                            if vals[kh, kw] > vmax[0]:
                                                vmax[0] = vals[kh, kw]
                                            if -vals[kh, kw] > vnegmax[0]:
                                                vnegmax[0] = -vals[kh, kw]

                                if split_ties:
                                    kmax = T.alloc_local((1,), int_dtype)
                                    kmin = T.alloc_local((1,), int_dtype)
                                    kmax[0] = 0
                                    kmin[0] = 0
                                    for kh in T.serial(kernel_height):
                                        for kw in T.serial(kernel_width):
                                            if (vmax[0] != -T.infinity(dtype)) and (vals[kh, kw] == vmax[0]):
                                                kmax[0] += 1
                                            if (vnegmax[0] != -T.infinity(dtype)) and (-vals[kh, kw] == vnegmax[0]):
                                                kmin[0] += 1
                                    for kh in T.serial(kernel_height):
                                        for kw in T.serial(kernel_width):
                                            ih = oh * stride_h + kh * dilation_h
                                            iw = ow * stride_w + kw * dilation_w
                                            if (ih >= 0) and (ih < in_height) and (iw >= 0) and (iw < in_width):
                                                if (kmax[0] > 0) and (vals[kh, kw] == vmax[0]):
                                                    T.atomic_add(
                                                        dimg_buffer[b, icg, ih, iw],
                                                        sh_alpha[icl] * local_dout[0] / T.cast(kmax[0], dtype),
                                                    )
                                                if (kmin[0] > 0) and (-vals[kh, kw] == vnegmax[0]):
                                                    T.atomic_add(
                                                        dimg_buffer[b, icg, ih, iw],
                                                        sh_alpha[icl] * local_dout[0] / T.cast(kmin[0], dtype),
                                                    )
                                else:
                                    max_h = T.alloc_local((1,), int_dtype)
                                    max_w = T.alloc_local((1,), int_dtype)
                                    min_h = T.alloc_local((1,), int_dtype)
                                    min_w = T.alloc_local((1,), int_dtype)
                                    max_h[0] = -1
                                    max_w[0] = -1
                                    min_h[0] = -1
                                    min_w[0] = -1
                                    for kh in T.serial(kernel_height):
                                        for kw in T.serial(kernel_width):
                                            if (vmax[0] != -T.infinity(dtype)) and (vals[kh, kw] == vmax[0]) and (max_h[0] == -1):
                                                max_h[0] = kh
                                                max_w[0] = kw
                                            if (vnegmax[0] != -T.infinity(dtype)) and (-vals[kh, kw] == vnegmax[0]) and (min_h[0] == -1):
                                                min_h[0] = kh
                                                min_w[0] = kw
                                    if max_h[0] >= 0:
                                        ih1 = oh * stride_h + max_h[0] * dilation_h
                                        iw1 = ow * stride_w + max_w[0] * dilation_w
                                        if (ih1 >= 0) and (ih1 < in_height) and (iw1 >= 0) and (iw1 < in_width):
                                            T.atomic_add(dimg_buffer[b, icg, ih1, iw1], sh_alpha[icl] * local_dout[0])
                                    if min_h[0] >= 0:
                                        ih2 = oh * stride_h + min_h[0] * dilation_h
                                        iw2 = ow * stride_w + min_w[0] * dilation_w
                                        if (ih2 >= 0) and (ih2 < in_height) and (iw2 >= 0) and (iw2 < in_width):
                                            T.atomic_add(dimg_buffer[b, icg, ih2, iw2], sh_alpha[icl] * local_dout[0])

                    T.sync_threads()

        return compute_dinput

    return _kernel()


# @tilelang.jit
def compound_min_max_plus_sum_conv2d1p_backward_dinput_tiled(
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
    maxmin_gradient="multiple" # multiple 1/k for more than one max/min position, single picks one of them (e.i. the first one)
):
    """Alias to the shared-memory optimized dInput kernel."""
    return compound_min_max_plus_sum_conv2d1p_backward_dinput(
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
        dtype=dtype,
        int_dtype=int_dtype,
        autotune=autotune,
        maxmin_gradient=maxmin_gradient,
    )

