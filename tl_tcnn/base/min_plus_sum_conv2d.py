"""This is the file containing kernels used for MinPlusSumConv2d operation."""

import tilelang
import tilelang.language as T

import os
import tilelang
import tilelang.language as T
import itertools
from tilelang.autotuner import AutoTuner
from ..utils import DataType, maybe_autotune
from ..autotune_env import _get_autotune_env

def min_plus_sum_conv2d_kernel(
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
    """Fused compound-min-max-plus-sum convolution kernel variant.

    Compared with v2: retained shared memory patch loading, directly used OIHW weight layout
    (out_channels, in_channels, kernel_height, kernel_width), and removed additional kernel
    flattening to avoid parameter mismatch compilation errors.
    """
    batches = batch_size
    patches = out_channels * out_height * out_width
    fast, warmup, rep, max_configs = _get_autotune_env()

    def get_configs(*_kernel_params, **_kwargs):
        max_grid_y = 65535
        if autotune:
            BLOCK_B = [1, 2, 4, 8, 16, 32, 64, 128] 
            BLOCK_P = [16, 32, 64, 128, 256]
            # Let one thread compute multiple output pixels (patches).
            P_TILES = [1, 2, 4]
            # CUDA blockDim.z is typically limited to 64.
            reduce_threads = [4, 8, 16, 32, 64]
        else:
            BLOCK_B = [1]
            # Avoid gridDim.y overflow (<= 65535) for large OH/OW.
            # patches = OC * OH * OW, grid_y = ceildiv(patches, BLOCK_P)
            _min_block_p = (patches + max_grid_y - 1) // max_grid_y
            if _min_block_p <= 32:
                BLOCK_P = [32]
            elif _min_block_p <= 64:
                BLOCK_P = [64]
            elif _min_block_p <= 128:
                BLOCK_P = [128]
            else:
                BLOCK_P = [256]
            P_TILES = [1]
            reduce_threads = [4]
        _configs = list(itertools.product(
            BLOCK_B,
            BLOCK_P,
            P_TILES,
            reduce_threads,
        ))
        configs = [{
            "BLOCK_B": c[0],
            "BLOCK_P": c[1],
            "P_TILES": c[2],
            "reduce_threads": c[3],
            # Sliding-window strip width (meta):
            # strip_w = (P_TILES - 1) * stride_w + (KW - 1) * dilation_w + 1
            "STRIP_W": (c[2] - 1) * stride_w + (kernel_width - 1) * dilation_w + 1,
            "batch_size": batch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "in_height": in_height,
            "in_width": in_width,
            "kernel_height": kernel_height,
            "kernel_width": kernel_width,
            "out_height": out_height,
            "out_width": out_width,
        } for c in _configs
            if (c[0] * c[1] * c[3] <= 1024)
            and ((patches + (c[1]) - 1) // (c[1]) <= max_grid_y)
        ]
        
        if max_configs > 0 and len(configs) > max_configs:
            configs = configs[:max_configs]
        
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
        BLOCK_B=None,
        BLOCK_P=None,
        P_TILES=None,
        reduce_threads=None,
        STRIP_W=None,
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
        MAX_TRANSACTION_SIZE_IN_BITS = 128

        TILE_IC = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads
        
        @T.prim_func
        def main(
            img_buffer : T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight_buffer : T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
            output_buffer : T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
        ):

            with T.Kernel(T.ceildiv(batches, BLOCK_B), T.ceildiv(patches, BLOCK_P), 
                        threads=(BLOCK_B, BLOCK_P, reduce_threads)) as (bid, pid):
                tb = T.get_thread_binding(0)
                tp = T.get_thread_binding(1)    # each thread handling each patch
                tic = T.get_thread_binding(2)   # each thread handling each ic?
                
                batch_idx = bid * BLOCK_B + tb
                patch_idx = pid * BLOCK_P + tp
                
                valid_b = batch_idx < batch_size
                valid_p = patch_idx < patches

                # if valid_b and valid_p:
                # Decode patch index
                ow = patch_idx % out_width
                tmp = patch_idx // out_width
                oh = tmp % out_height
                oc = tmp // out_height
                
                oh_stride = oh * stride_h
                ow_stride = ow * stride_w
                
                # local memory
                local_neg_max = T.alloc_local((1,), dtype)

                local_accum = T.alloc_local((1,), dtype)
                
                local_img = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                local_weight = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                
                # shared memory
                sum_val = T.alloc_shared((BLOCK_B, BLOCK_P,), dtype)
                
                if tic == 0:
                    sum_val[tb, tp] = T.cast(0, dtype)
                T.clear(local_accum)
                
                T.sync_threads()
                if valid_b and valid_p:
                    # Initialize local max and min arrays
                    for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                        for ic in T.vectorized(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                # load local img
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        
                                        if in_h < in_height and in_w < in_width:
                                            local_img[ic, kh, kw] = img_buffer[
                                                batch_idx,
                                                ic_global,
                                                in_h,
                                                in_w,
                                            ]
                                        local_weight[ic, kh, kw] = weight_buffer[
                                            oc,
                                            ic_global,
                                            kh,
                                            kw
                                        ]

                        for ic in T.serial(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                local_neg_max[0] = -T.infinity(dtype)
                                # Loop kh, kw over the kernel window for this local tile
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        if in_h < in_height and in_w < in_width:
                                            val = (local_img[ic, kh, kw] + local_weight[ic, kh, kw])
                                            if -val > local_neg_max[0]:
                                                local_neg_max[0] = -val
                                local_accum[0] += -local_neg_max[0] 

                    T.atomic_add(sum_val[tb, tp], local_accum[0])
                T.sync_threads()
                if valid_b and valid_p:
                    if tic == 0:
                        output_buffer[batch_idx, oc, oh, ow] = sum_val[tb, tp]
                    # T.atomic_add(output_buffer[batch_idx, oc, oh, ow], local_accum[0])

        return main
    
    return _kernel()



def min_plus_sum_conv2d_kernel_tiled(
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
    """Fused compound-min-max-plus-sum convolution kernel variant.

    Compared with v2: retained shared memory patch loading, directly used OIHW weight layout
    (out_channels, in_channels, kernel_height, kernel_width), and removed additional kernel
    flattening to avoid parameter mismatch compilation errors.
    """
    batches = batch_size
    # patches = out_channels * out_height * out_width
    
    
    IC = in_channels
    fast, warmup, rep, max_configs = _get_autotune_env()

    def get_configs(*_kernel_params, **_kwargs):
        max_grid_y = 65535
        max_threads_per_block = 1024

        if autotune:
            if fast:
                BLOCK_B = [1, 2, 4, 8]
                BLOCK_P = [64, 128, 256]
                # Keep only square tiles; smaller set is enough for most shapes.
                BLOCK_OH = [4, 8, 16, 32]
                BLOCK_OW = [4, 8, 16, 32]
                reduce_threads = [8, 16, 32]
            else:
                BLOCK_B = [1, 2, 4, 8, 16, 32, 64, 128]
                BLOCK_P = [32, 64, 128, 256]
                # Treat BLOCK_OH/BLOCK_OW as a coarse partition of (oh, ow) into tiles.
                BLOCK_OH = [1, 2, 4, 8, 16, 32, 64, 128]
                BLOCK_OW = [1, 2, 4, 8, 16, 32, 64, 128]
                reduce_threads = [4, 8, 16, 32, 64]
        else:
            BLOCK_B = [1]
            BLOCK_OH = [16]
            BLOCK_OW = [16]
            # Avoid gridDim.y overflow (<= 65535).
            _min_block_p = (out_channels * BLOCK_OH[0] * BLOCK_OW[0] + max_grid_y - 1) // max_grid_y
            if _min_block_p <= 32:
                BLOCK_P = [32]
            elif _min_block_p <= 64:
                BLOCK_P = [64]
            elif _min_block_p <= 128:
                BLOCK_P = [128]
            else:
                BLOCK_P = [256]
            # P_TILES = [1]

            reduce_threads = [4]

        _configs = list(itertools.product(
            BLOCK_B,
            BLOCK_P,
            # P_TILES,
            BLOCK_OH,
            BLOCK_OW,
            reduce_threads,
        ))
        configs = []
        for c in _configs:
            b_b, b_p, b_oh, b_ow, r_t = c
            total_threads = b_b * b_p * r_t
            if b_oh != b_ow:
                continue
            if total_threads > max_threads_per_block:
                continue
            if r_t > 64:
                continue
            if b_oh < 1 or b_ow < 1:
                continue
            # Kernel uses patches = out_channels * BLOCK_OH * BLOCK_OW
            patches_cfg = out_channels * b_oh * b_ow
            if (patches_cfg + b_p - 1) // b_p > max_grid_y:
                continue

            configs.append({
                "BLOCK_B": b_b,
                "BLOCK_P": b_p,
                "BLOCK_OH": b_oh,
                "BLOCK_OW": b_ow,
                "reduce_threads": r_t,
                # Sliding-window strip width (meta):
                # strip_w = (P_TILES - 1) * stride_w + (KW - 1) * dilation_w + 1
                # "STRIP_W": (c[2] - 1) * stride_w + (kernel_width - 1) * dilation_w + 1,
                "batch_size": batch_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "in_height": in_height,
                "in_width": in_width,
                "kernel_height": kernel_height,
                "kernel_width": kernel_width,
                "out_height": out_height,
                "out_width": out_width,
            })

        if max_configs > 0 and len(configs) > max_configs:
            return configs[:max_configs]
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
        BLOCK_B=None,
        BLOCK_P=None,
        # P_TILES=None,
        BLOCK_OH=None,
        BLOCK_OW=None,
        reduce_threads=None,
        # STRIP_W=None,
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
        MAX_TRANSACTION_SIZE_IN_BITS = 128

        TILE_IC = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads
        
        patches = out_channels * BLOCK_OH * BLOCK_OW

        # Each (boh, bow) corresponds to one output tile.
        TILE_OH = T.ceildiv(out_height, BLOCK_OH)
        TILE_OW = T.ceildiv(out_width, BLOCK_OW)

        # Input tile needed to cover the output tile with stride/dilation.
        TILE_IH = (TILE_OH - 1) * stride_h + (kernel_height - 1) * dilation_h + 1
        TILE_IW = (TILE_OW - 1) * stride_w + (kernel_width - 1) * dilation_w + 1
        
        @T.prim_func
        def main(
            img_buffer : T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight_buffer : T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
            output_buffer : T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
        ):

            with T.Kernel(T.ceildiv(batches, BLOCK_B), T.ceildiv(patches, BLOCK_P), 
                        threads=(BLOCK_B, BLOCK_P, reduce_threads)) as (bid, pid):
                tb = T.get_thread_binding(0)
                tp = T.get_thread_binding(1)    # each thread handling each patch
                tic = T.get_thread_binding(2)   # each thread handling each ic?
                
                batch_idx = bid * BLOCK_B + tb
                patch_idx = pid * BLOCK_P + tp
                
                # Decode patch index
                bow = patch_idx % BLOCK_OW
                tmp = patch_idx // BLOCK_OW
                boh = tmp % BLOCK_OH
                oc = tmp // BLOCK_OH
                
                base_ow = bow * TILE_OW
                base_oh = boh * TILE_OH
                
                base_h = base_oh * stride_h
                base_w = base_ow * stride_w
                
                valid_bidx = batch_idx < batch_size
                valid_pidx = patch_idx < patches

                local_neg_max = T.alloc_local((1,), dtype)

                local_accum = T.alloc_local((TILE_OH, TILE_OW,), dtype)
                
                # local_img = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                local_img = T.alloc_local((TILE_IC, TILE_IH, TILE_IW), dtype)
                local_weight = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                
                # shared memory
                sum_val = T.alloc_shared((BLOCK_B, BLOCK_P, TILE_OH, TILE_OW), dtype)

                T.clear(local_accum)


                if valid_bidx and valid_pidx:
                    # local memory
                    # local_max = T.alloc_local((TILE_OH, TILE_OW), dtype)
                    # local_min = T.alloc_local((TILE_OH, TILE_OW,), dtype)
                    
                    # Init shared accumulator (once per output element) then sync.
                    if tic == 0:
                        for toh in T.serial(TILE_OH):
                            for tow in T.serial(TILE_OW):
                                sum_val[tb, tp, toh, tow] = T.cast(0, dtype)
                    
                T.sync_threads()

                if valid_bidx and valid_pidx:
                    # Initialize local max and min arrays
                    for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                        for ic in T.vectorized(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                # load local img
                                for tih in T.vectorized(TILE_IH):
                                    for tiw in T.vectorized(TILE_IW):
                                        in_h = base_h + tih
                                        in_w = base_w + tiw
                                        
                                        if in_h < in_height and in_w < in_width:
                                            local_img[ic, tih, tiw] = img_buffer[
                                                batch_idx,
                                                ic_global,
                                                in_h,
                                                in_w,
                                            ]
                                        else:
                                            local_img[ic, tih, tiw] = T.cast(0, dtype)
                                
                                # load local weight
                                for kh in T.vectorized(kernel_height):
                                    for kw in T.vectorized(kernel_width):
                                        local_weight[ic, kh, kw] = weight_buffer[
                                            oc,
                                            ic_global,
                                            kh,
                                            kw
                                        ]

                        for toh in T.serial(TILE_OH):
                            for tow in T.serial(TILE_OW):
                                oh_out = base_oh + toh
                                ow_out = base_ow + tow
                                if oh_out < out_height and ow_out < out_width:
                                    for ic in T.serial(TILE_IC):
                                        ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                                        if ic_global < in_channels:
                                            
                                            local_neg_max[0] = T.infinity(dtype)

                                            base_tih = toh * stride_h
                                            base_tiw = tow * stride_w
                                            
                                            
                                            for kh in T.serial(kernel_height):
                                                tih = base_tih + kh * dilation_h
                                                for kw in T.serial(kernel_width):
                                                    tiw = base_tiw + kw * dilation_w
                                                    val = (local_img[ic, tih, tiw] + local_weight[ic, kh, kw])
                                                    if -val > local_neg_max[0]:
                                                        local_neg_max[0] = -val
                                            local_accum[toh, tow] += -local_neg_max[0] 
                                                    
                    for toh in T.serial(TILE_OH):
                        for tow in T.serial(TILE_OW):
                            T.atomic_add(sum_val[tb, tp, toh, tow], local_accum[toh, tow])

                T.sync_threads()
                if valid_bidx and valid_pidx:
                    if tic == 0:
                        for toh in T.serial(TILE_OH):
                            for tow in T.serial(TILE_OW):
                                oh_out = base_oh + toh
                                ow_out = base_ow + tow
                                if oh_out < out_height and ow_out < out_width:
                                    output_buffer[batch_idx, oc, oh_out, ow_out] = sum_val[tb, tp, toh, tow]
                T.sync_threads()

        return main
    
    return _kernel()


@tilelang.jit
def min_plus_sum_conv2d_kernel_depr(
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
):
    """Fused min-plus-sum convolution kernel variant.

    Compared with v2: retained shared memory patch loading, directly used HWIO weight layout, and removed additional kernel flattening to avoid parameter mismatch compilation errors.
    Layouts:
    img: (B,H,W,C) NHWC
    weight: (KH,KW,IC,OC) HWIO
    output: (B,OH,OW,OC)
    Computation: output[b,oh,ow,oc] = Σ_ic min_{kh,kw}( img[b,in_h,in_w,ic]   weight[kh,kw,ic,oc] )
    """
    total_patches = batch_size * out_height * out_width

    # Block threads cover output channels (or a power-of-two >= oc) capped at 1024
    block_size = 1
    while block_size < out_channels and block_size < 1024:
        block_size <<= 1
    if block_size > 1024:
        block_size = 1024

    grid_size = total_patches

    @T.prim_func
    def fused_kernel_v25(img: T.handle, weight: T.handle, output: T.handle):
        img_buffer = T.match_buffer(
            img, (batch_size, in_height, in_width, in_channels), dtype
        )
        weight_buffer = T.match_buffer(
            weight, (kernel_height, kernel_width, in_channels, out_channels), dtype
        )
        output_buffer = T.match_buffer(
            output, (batch_size, out_height, out_width, out_channels), dtype
        )

        with T.Kernel(grid_size, threads=block_size) as bid:
            tid = T.get_thread_binding(0)
            patch_idx = bid

            # Decode patch index
            ow = patch_idx % out_width
            tmp = patch_idx // out_width
            oh = tmp % out_height
            b = tmp // out_height

            # Each thread handles one output channel
            for it_oc in T.serial(T.ceildiv(out_channels, block_size)):
                oc = tid + it_oc * block_size
                if oc < out_channels:
                    sum_val = T.alloc_local((1,), dtype)
                    sum_val[0] = T.cast(0, dtype)

                    for ic in T.serial(in_channels):
                        local_min = T.alloc_local((1,), dtype)
                        local_min[0] = T.infinity(dtype)

                        # Loop kh, kw, ic; access weight_buffer directly (HWIO)
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                val = (
                                    img_buffer[
                                        b,
                                        oh * stride_h + kh * dilation_h,
                                        ow * stride_w + kw * dilation_w,
                                        ic,
                                    ]
                                    + weight_buffer[kh, kw, ic, oc]
                                )
                                if val < local_min[0]:
                                    local_min[0] = val

                        sum_val[0] = sum_val[0] + local_min[0]
                    output_buffer[b, oh, ow, oc] = sum_val[0]

    return fused_kernel_v25


@tilelang.jit
def min_plus_sum_conv2d_backward_all(
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
):
    """Backward (Type1): recompute argmin on-the-fly (no mask), compute dimg and dweight in one kernel.

    Layouts:
    img: (B,H,W,C) NHWC
    weight: (KH,KW,IC,OC) HWIO
    dout: (B,OH,OW,OC) NHWC
    dimg: (B,H,W,IC) NHWC
    dweight: (KH,KW,IC,OC) HWIO
    """
    total_patches = batch_size * out_height * out_width

    block_size = 1
    while block_size < out_channels and block_size < 1024:
        block_size <<= 1
    if block_size > 1024:
        block_size = 1024

    grid_size = total_patches

    @T.prim_func
    def bwd_all(
        img: T.handle,
        weight: T.handle,
        dout: T.handle,
        dimg: T.handle,
        dweight: T.handle,
    ):
        img_buffer = T.match_buffer(
            img, (batch_size, in_height, in_width, in_channels), dtype
        )
        weight_buffer = T.match_buffer(
            weight, (kernel_height, kernel_width, in_channels, out_channels), dtype
        )
        dout_buffer = T.match_buffer(
            dout, (batch_size, out_height, out_width, out_channels), dtype
        )
        dimg_buffer = T.match_buffer(
            dimg, (batch_size, in_height, in_width, in_channels), dtype
        )
        dweight_buffer = T.match_buffer(
            dweight, (kernel_height, kernel_width, in_channels, out_channels), dtype
        )

        with T.Kernel(grid_size, threads=block_size) as bid:
            tid = T.get_thread_binding(0)
            patch_idx = bid

            ow = patch_idx % out_width
            tmp = patch_idx // out_width
            oh = tmp % out_height
            b = tmp // out_height

            for it_oc in T.serial(T.ceildiv(out_channels, block_size)):
                oc = tid + it_oc * block_size
                if oc < out_channels:
                    dout_val = dout_buffer[b, oh, ow, oc]
                    for ic in T.serial(in_channels):
                        local_min = T.alloc_local((1,), dtype)
                        local_arg = T.alloc_local((1,), int_dtype)
                        local_min[0] = T.infinity(dtype)
                        local_arg[0] = T.cast(0, int_dtype)

                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                k_flat = kh * kernel_width + kw
                                val = (
                                    img_buffer[
                                        b,
                                        oh * stride_h + kh * dilation_h,
                                        ow * stride_w + kw * dilation_w,
                                        ic,
                                    ]
                                    + weight_buffer[kh, kw, ic, oc]
                                )
                                if val < local_min[0]:
                                    local_min[0] = val
                                    local_arg[0] = T.cast(k_flat, int_dtype)

                        flat = local_arg[0]
                        kh_sel = flat // kernel_width
                        kw_sel = flat % kernel_width
                        # dW accumulation
                        T.atomic_add(dweight_buffer[kh_sel, kw_sel, ic, oc], dout_val)
                        # dImg accumulation
                        in_h = oh * stride_h + kh_sel * dilation_h
                        in_w = ow * stride_w + kw_sel * dilation_w
                        if (
                            (in_h >= 0)
                            and (in_h < in_height)
                            and (in_w >= 0)
                            and (in_w < in_width)
                        ):
                            T.atomic_add(dimg_buffer[b, in_h, in_w, ic], dout_val)

    return bwd_all



# @tilelang.jit
def min_plus_sum_conv2d_kernel_backward_dweight(
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
    """Compute dweight gradient.
    
    dweight[kh,kw,ic,oc] = Σ_{b,oh,ow} (
        dout[b,oh,ow,oc] * δ_min(ic,oc,b,oh,ow,kh,kw)
    )
    
    Parallelize over (kh, kw, ic, oc) with tiling over batch*spatial.
    """

    BHW = batch_size * out_height * out_width

    fast, warmup, rep, max_configs = _get_autotune_env()
    def get_configs(*_kernel_params, **_kwargs):
        if autotune:
            BLOCK_OC = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            BLOCK_IC = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            reduce_threads = [4, 8, 16, 32, 64, 128, 256]
        else:
            BLOCK_OC = [4]
            BLOCK_IC = [1]
            reduce_threads = [64]
        _configs = list(itertools.product(
            BLOCK_OC,
            BLOCK_IC,
            reduce_threads,
        ))
        configs = [{
            "BLOCK_OC": c[0],
            "BLOCK_IC": c[1],
            "reduce_threads": c[2],
            "batch_size": batch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "in_height": in_height,
            "in_width": in_width,
            "kernel_height": kernel_height,
            "kernel_width": kernel_width,
            "out_height": out_height,
            "out_width": out_width,
        } for c in _configs if c[0] * c[1] * c[2] <= 1024]
        
        if max_configs > 0 and len(configs) > max_configs:
            return configs[:max_configs]
        
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
    def _kernel_dweight_single(
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
    ):

        MAX_TRANSACTION_SIZE_IN_BITS = 128
        K = kernel_height * kernel_width

        TILE_BHW = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_BHW = TILE_BHW * reduce_threads
        
        @T.prim_func
        def compute_dweight_single(
            img_buffer: T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight_buffer: T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
            dout_buffer: T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
            dweight_buffer: T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
        ):
            # Buffers
            
            # Use 3D grid (kh, kw, oc) and 3D threads (kh, kw, ic). Tile over in_channels serially if needed.
            with T.Kernel(
                T.ceildiv(out_channels, BLOCK_OC),
                T.ceildiv(in_channels, BLOCK_IC),
                threads=(BLOCK_OC, BLOCK_IC, reduce_threads)
            ) as (oc_block, ic_block):
                toc = T.get_thread_binding(0)
                tic = T.get_thread_binding(1)
                tbhw = T.get_thread_binding(2)
                
                # Global indices
                oc = oc_block * BLOCK_OC + toc  # one oc per grid z; expand within loop if BLOCK_OC>1
                ic = ic_block * BLOCK_IC + tic 
                
                # Bounds check for kh/kw/oc
                valid_oc = oc < out_channels
                valid_ic = ic < in_channels
                
                # shared memory (reduce over tbhw without atomics)
                shared_dweight = T.alloc_shared((BLOCK_OC, BLOCK_IC, kernel_height, kernel_width, reduce_threads), dtype)
                
                # local memory
                local_dweight_acc = T.alloc_local((kernel_height, kernel_width,), dtype)
                local_img = T.alloc_local((TILE_BHW, kernel_height, kernel_width), dtype)
                local_weight = T.alloc_local((kernel_height, kernel_width), dtype)
                local_dout = T.alloc_local((TILE_BHW), dtype)
                
                T.sync_threads()

                if tbhw == 0 and toc == 0 and tic == 0:
                    T.clear(shared_dweight)
                T.clear(local_dweight_acc)

                T.sync_threads()
                    
                if valid_ic & valid_oc:
                    for kh in T.serial(kernel_height):
                        for kw in T.serial(kernel_width):
                            local_weight[kh, kw] = weight_buffer[oc, ic, kh, kw]
                    # Process batches and spatial positions in tiles
                    for bhw_block in T.serial(T.ceildiv(BHW, BLOCK_BHW)):
                        bhw_start = bhw_block * BLOCK_BHW + tbhw * TILE_BHW
                        
                        for bhw_offset in T.vectorized(TILE_BHW):
                            bhw = bhw_start + bhw_offset
                            
                            # Check if this spatial position is valid
                            if bhw < BHW:
                                # Decode b, oh, ow
                                tmp1 = bhw // out_width
                                ow = bhw % out_width
                                tmp2 = tmp1 // out_height
                                oh = tmp1 % out_height
                                b = tmp2
                                
                                oh_stride = oh * stride_h
                                ow_stride = ow * stride_w
                                
                                local_dout[bhw_offset] = dout_buffer[b, oc, oh, ow]
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width):
                                            local_img[bhw_offset, kh, kw] = img_buffer[b, ic, in_h, in_w]

                        for bhw_offset in T.serial(TILE_BHW):
                            bhw = bhw_start + bhw_offset
                            
                            # Check if this spatial position is valid
                            if bhw < BHW:
                                # Decode b, oh, ow
                                tmp1 = bhw // out_width
                                ow = bhw % out_width
                                tmp2 = tmp1 // out_height
                                oh = tmp1 % out_height
                                b = tmp2
                                
                                oh_stride = oh * stride_h
                                ow_stride = ow * stride_w
                                
                                # Find max and min over kernel positions
                                neg_max_val = T.alloc_local((1,), dtype)
                                min_argh = T.alloc_local((1,), int_dtype)
                                min_argw = T.alloc_local((1,), int_dtype)
                                neg_max_val[0] = -T.infinity(dtype)
                                min_argh[0] = -1
                                min_argw[0] = -1
                                
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        
                                        # Check bounds
                                        if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width):
                                            # Use the locally cached per-position image value
                                            # to ensure identical behavior to buffer-based computation.
                                            val = (local_img[bhw_offset, kh, kw] + local_weight[kh, kw])

                                            if -val > neg_max_val[0]:
                                                neg_max_val[0] = -val
                                                min_argh[0] = kh
                                                min_argw[0] = kw
                                
                                if min_argh[0] != -1:
                                    local_dweight_acc[min_argh[0], min_argw[0]] += (local_dout[bhw_offset])

                    for kh in T.serial(kernel_height):
                        for kw in T.serial(kernel_width):
                            shared_dweight[toc, tic, kh, kw, tbhw] = local_dweight_acc[kh, kw]
                            # T.atomic_add(shared_dweight[toc, tic, kh, kw, tbhw], local_dweight_acc[kh, kw])

                T.sync_threads()

                if valid_ic & valid_oc:
                    if tbhw == 0:
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                total = T.alloc_local((1,), dtype)
                                total[0] = T.cast(0, dtype)
                                for r in T.serial(reduce_threads):
                                    total[0] = total[0] + shared_dweight[toc, tic, kh, kw, r]
                                dweight_buffer[oc, ic, kh, kw] = total[0] 
                                # T.atomic_add(dweight_buffer[oc, ic, kh, kw], total[0])             
        return compute_dweight_single
    
    
    @maybe_autotune(
        configs=get_configs(),
        warmup=warmup,
        rep=rep,
        enabled=autotune,
    )
    @tilelang.jit(
        target="auto",
    )
    def _kernel_dweight_multi(
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
    ):

        MAX_TRANSACTION_SIZE_IN_BITS = 128
        K = kernel_height * kernel_width

        TILE_BHW = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_BHW = TILE_BHW * reduce_threads
        
        @T.prim_func
        def compute_dweight_multi(
            img_buffer: T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight_buffer: T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
            dout_buffer: T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
            dweight_buffer: T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
        ):
            # Buffers
            
            # Use 3D grid (kh, kw, oc) and 3D threads (kh, kw, ic). Tile over in_channels serially if needed.
            with T.Kernel(
                T.ceildiv(out_channels, BLOCK_OC),
                T.ceildiv(in_channels, BLOCK_IC),
                threads=(BLOCK_OC, BLOCK_IC, reduce_threads)
            ) as (oc_block, ic_block):
                toc = T.get_thread_binding(0)
                tic = T.get_thread_binding(1)
                tbhw = T.get_thread_binding(2)
                
                # Global indices
                oc = oc_block * BLOCK_OC + toc  # one oc per grid z; expand within loop if BLOCK_OC>1
                ic = ic_block * BLOCK_IC + tic 
                
                # Bounds check for kh/kw/oc
                valid_oc = oc < out_channels
                valid_ic = ic < in_channels
                
                # shared memory (reduce over tbhw without atomics)
                shared_dweight = T.alloc_shared((BLOCK_OC, BLOCK_IC, kernel_height, kernel_width, reduce_threads), dtype)
                
                # local memory
                local_dweight_acc = T.alloc_local((kernel_height, kernel_width,), dtype)
                local_img = T.alloc_local((TILE_BHW, kernel_height, kernel_width), dtype)
                local_weight = T.alloc_local((kernel_height, kernel_width), dtype)
                local_dout = T.alloc_local((TILE_BHW), dtype)
                
                local_val = T.alloc_local((kernel_height, kernel_width), dtype)
                
                T.sync_threads()

                if tbhw == 0 and toc == 0 and tic == 0:
                    T.clear(shared_dweight)
                T.clear(local_dweight_acc)

                T.sync_threads()
                    
                if valid_ic & valid_oc:
                    for kh in T.serial(kernel_height):
                        for kw in T.serial(kernel_width):
                            local_weight[kh, kw] = weight_buffer[oc, ic, kh, kw]
                    # Process batches and spatial positions in tiles
                    for bhw_block in T.serial(T.ceildiv(BHW, BLOCK_BHW)):
                        bhw_start = bhw_block * BLOCK_BHW + tbhw * TILE_BHW
                        
                        for bhw_offset in T.vectorized(TILE_BHW):
                            bhw = bhw_start + bhw_offset
                            
                            # Check if this spatial position is valid
                            if bhw < BHW:
                                # Decode b, oh, ow
                                tmp1 = bhw // out_width
                                ow = bhw % out_width
                                tmp2 = tmp1 // out_height
                                oh = tmp1 % out_height
                                b = tmp2
                                
                                oh_stride = oh * stride_h
                                ow_stride = ow * stride_w
                                
                                local_dout[bhw_offset] = dout_buffer[b, oc, oh, ow]
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width):
                                            local_img[bhw_offset, kh, kw] = img_buffer[b, ic, in_h, in_w]

                        for bhw_offset in T.serial(TILE_BHW):
                            bhw = bhw_start + bhw_offset
                            
                            # Check if this spatial position is valid
                            if bhw < BHW:
                                # Decode b, oh, ow
                                tmp1 = bhw // out_width
                                ow = bhw % out_width
                                tmp2 = tmp1 // out_height
                                oh = tmp1 % out_height
                                b = tmp2
                                
                                oh_stride = oh * stride_h
                                ow_stride = ow * stride_w
                                
                                # Find max and min over kernel positions
                                neg_max = T.alloc_local((1,), dtype)
                                max_k = T.alloc_local((1,), int_dtype)
                                min_k = T.alloc_local((1,), int_dtype)

                                min_k[0] = 0

                                neg_max[0] = -T.infinity(dtype) # min_val = -neg_max

                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        
                                        # Check bounds
                                        if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width):
                                            # Use the locally cached per-position image value
                                            # to ensure identical behavior to buffer-based computation.
                                            local_val[kh, kw] = (local_img[bhw_offset, kh, kw] + local_weight[kh, kw])
                                            
                                            if -local_val[kh, kw] > neg_max[0]:
                                                neg_max[0] = -local_val[kh, kw]
                                
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        
                                        # Check bounds
                                        if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width):
                                            if -local_val[kh, kw] == neg_max[0]:
                                                min_k[0] += 1
                                
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        
                                        # Check bounds
                                        if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width):
                                            if (min_k[0] > 0) & (-local_val[kh, kw] == neg_max[0]):
                                                local_dweight_acc[kh, kw] += (
                                                    local_dout[bhw_offset]
                                                    / T.cast(min_k[0], dtype)
                                                )

                    for kh in T.serial(kernel_height):
                        for kw in T.serial(kernel_width):
                            shared_dweight[toc, tic, kh, kw, tbhw] = local_dweight_acc[kh, kw]
                            # T.atomic_add(shared_dweight[toc, tic, kh, kw, tbhw], local_dweight_acc[kh, kw])

                T.sync_threads()

                if valid_ic & valid_oc:
                    if tbhw == 0:
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                total = T.alloc_local((1,), dtype)
                                total[0] = T.cast(0, dtype)
                                for r in T.serial(reduce_threads):
                                    total[0] = total[0] + shared_dweight[toc, tic, kh, kw, r]
                                dweight_buffer[oc, ic, kh, kw] = total[0] 
                                # T.atomic_add(dweight_buffer[oc, ic, kh, kw], total[0])
        return compute_dweight_multi
        
    if maxmin_gradient == "single":
        _kernel = _kernel_dweight_single
    else:
        _kernel = _kernel_dweight_multi

    return _kernel()



# @tilelang.jit
def min_plus_sum_conv2d_kernel_backward_dinput(
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
    """Compute gradient w.r.t. input for compound-min-max-plus-sum 2D convolution with
    two mixing parameters.

    Layouts:
    img     : (B, C, H, W) NCHW
    weight  : (OC, IC, KH, KW) OIHW
    dout    : (B, OC, OH, OW)
    dimg    : (B, C, H, W)
    Gradient: for each (b, oc, oh, ow) and input channel ic, contributions are routed to
    the argmax/argmin locations of img + weight over (kh, kw)
    respectively.
    """    
    batches = batch_size
    patches = out_channels * out_height * out_width
    IC = in_channels
    fast, warmup, rep, max_configs = _get_autotune_env()


    def get_configs(*_kernel_params, **_kwargs):
        max_grid_y = 65535
        if autotune:
            if fast:
                BLOCK_B = [1, 2, 4, 8]
                BLOCK_P = [64, 128, 256]
                reduce_threads = [8, 16, 32]
            else:
                BLOCK_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
                BLOCK_P = [16, 32, 64, 128, 256]
                reduce_threads = [4, 8, 16, 32, 64]
        else:
            BLOCK_B = [1]
            _min_block_p = (patches + max_grid_y - 1) // max_grid_y
            if _min_block_p <= 32:
                BLOCK_P = [32]
            elif _min_block_p <= 64:
                BLOCK_P = [64]
            elif _min_block_p <= 128:
                BLOCK_P = [128]
            else:
                BLOCK_P = [256]
            reduce_threads = [4]
        _configs = list(itertools.product(
            BLOCK_B,
            BLOCK_P,
            reduce_threads,
        ))
        configs = [{
            "BLOCK_B": c[0],
            "BLOCK_P": c[1],
            "reduce_threads": c[2],
            "batch_size": batch_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "in_height": in_height,
            "in_width": in_width,
            "kernel_height": kernel_height,
            "kernel_width": kernel_width,
            "out_height": out_height,
            "out_width": out_width,
        } for c in _configs
            if (c[0] * c[1] * c[2] <= 1024)
            and ((patches + c[1] - 1) // c[1] <= max_grid_y)
        ]
        if max_configs > 0 and len(configs) > max_configs:
            return configs[:max_configs]
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
    def _kernel_dinput_single(
        BLOCK_B=None,
        BLOCK_P=None,
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

        MAX_TRANSACTION_SIZE_IN_BITS = 128

        TILE_IC = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads
        
        @T.prim_func
        def compute_dinput_single(
            img_buffer : T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight_buffer : T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
            dout_buffer : T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
            dimg_buffer : T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
        ):

            with T.Kernel(T.ceildiv(batches, BLOCK_B), T.ceildiv(patches, BLOCK_P),
                        threads=(BLOCK_B, BLOCK_P, reduce_threads)) as (bid, pid):
                tb = T.get_thread_binding(0)
                tp = T.get_thread_binding(1)    # each thread handling each patch
                tic = T.get_thread_binding(2)   # each thread handling each ic?
                
                batch_idx = bid * BLOCK_B + tb
                patch_idx = pid * BLOCK_P + tp

                if batch_idx < batch_size and patch_idx < patches:
                    # Decode patch index
                    ow = patch_idx % out_width
                    tmp = patch_idx // out_width
                    oh = tmp % out_height
                    oc = tmp // out_height
                    oh_stride = oh * stride_h
                    ow_stride = ow * stride_w
                    # local memory
                    local_neg_max = T.alloc_local((1,), dtype)
                    local_min_argh = T.alloc_local((1,), int_dtype)
                    local_min_argw = T.alloc_local((1,), int_dtype)

                    local_img = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                    local_weight = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                    local_dout = T.alloc_local((1,), dtype)

                    local_dout[0] = dout_buffer[batch_idx, oc, oh, ow]
                    # Initialize local max and min arrays
                    for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                        for ic in T.vectorized(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                # load local img
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride+ kw * dilation_w
                                        
                                        if in_h < in_height and in_w < in_width:
                                            local_img[ic, kh, kw] = img_buffer[
                                                batch_idx,
                                                ic_global,
                                                in_h,
                                                in_w,
                                            ]
                                        local_weight[ic, kh, kw] = weight_buffer[
                                            oc,
                                            ic_global,
                                            kh,
                                            kw
                                        ]                     
                                
                        for ic in T.serial(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                local_neg_max[0] = -T.infinity(dtype)
                                local_min_argh[0] = -1
                                local_min_argw[0] = -1
                                # Loop kh, kw over the kernel window for this local tile
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        
                                        if in_h < in_height and in_w < in_width:
                                            val = (local_img[ic, kh, kw] + local_weight[ic, kh, kw])
                                            if -val > local_neg_max[0]:
                                                local_neg_max[0] = -val
                                                local_min_argh[0] = T.cast(kh, int_dtype)
                                                local_min_argw[0] = T.cast(kw, int_dtype)
                            
                                kh_sel2 = local_min_argh[0]
                                kw_sel2 = local_min_argw[0]
                                in_h2 = oh_stride + kh_sel2 * dilation_h
                                in_w2 = ow_stride + kw_sel2 * dilation_w
                                if (
                                    (in_h2 >= 0)
                                    and (in_h2 < in_height)
                                    and (in_w2 >= 0)
                                    and (in_w2 < in_width)
                                ):
                                    T.atomic_add(
                                        dimg_buffer[batch_idx, ic_global, in_h2, in_w2],
                                        local_dout[0],
                                    )
        return compute_dinput_single
    
    
    @maybe_autotune(
        configs=get_configs(),
        warmup=warmup,
        rep=rep,
        enabled=autotune,
    )
    @tilelang.jit(
        target="auto",
    )
    def _kernel_dinput_multi(
        BLOCK_B=None,
        BLOCK_P=None,
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

        MAX_TRANSACTION_SIZE_IN_BITS = 128

        TILE_IC = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads
        
        @T.prim_func
        def compute_dinput_multi(
            img_buffer : T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight_buffer : T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
            dout_buffer : T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
            dimg_buffer : T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
        ):

            with T.Kernel(T.ceildiv(batches, BLOCK_B), T.ceildiv(patches, BLOCK_P),
                        threads=(BLOCK_B, BLOCK_P, reduce_threads)) as (bid, pid):
                tb = T.get_thread_binding(0)
                tp = T.get_thread_binding(1)    # each thread handling each patch
                tic = T.get_thread_binding(2)   # each thread handling each ic?
                
                batch_idx = bid * BLOCK_B + tb
                patch_idx = pid * BLOCK_P + tp

                if batch_idx < batch_size and patch_idx < patches:
                    # Decode patch index
                    ow = patch_idx % out_width
                    tmp = patch_idx // out_width
                    oh = tmp % out_height
                    oc = tmp // out_height
                    oh_stride = oh * stride_h
                    ow_stride = ow * stride_w
                    # local memory
                    local_neg_max = T.alloc_local((1,), dtype)
                    min_k = T.alloc_local((1,), int_dtype)

                    local_img = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                    local_weight = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                    local_dout = T.alloc_local((1,), dtype)

                    local_val = T.alloc_local((kernel_height, kernel_width), dtype)

                    local_dout[0] = dout_buffer[batch_idx, oc, oh, ow]
                    # Initialize local max and min arrays
                    for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                        for ic in T.vectorized(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                # load local img
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride+ kw * dilation_w
                                        
                                        if in_h < in_height and in_w < in_width:
                                            local_img[ic, kh, kw] = img_buffer[
                                                batch_idx,
                                                ic_global,
                                                in_h,
                                                in_w,
                                            ]
                                        local_weight[ic, kh, kw] = weight_buffer[
                                            oc,
                                            ic_global,
                                            kh,
                                            kw
                                        ]

                        for ic in T.serial(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                local_neg_max[0] = -T.infinity(dtype)
                                min_k[0] = 0

                                # Loop kh, kw over the kernel window for this local tile
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        
                                        if in_h < in_height and in_w < in_width:
                                            local_val[kh, kw] = (
                                                local_img[ic, kh, kw] + local_weight[ic, kh, kw]
                                            )
                                            if -local_val[kh, kw] > local_neg_max[0]:
                                                local_neg_max[0] = -local_val[kh, kw]
                                                
                                # local_min[0] = -local_neg_max[0]
                                
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        
                                        if in_h < in_height and in_w < in_width:
                                            if -local_val[kh, kw] == local_neg_max[0]:
                                                min_k[0] += 1
                                
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        
                                        if in_h < in_height and in_w < in_width:
                                            if (min_k[0] > 0) and (-local_val[kh, kw] == local_neg_max[0]):
                                                T.atomic_add(
                                                    dimg_buffer[batch_idx, ic_global, in_h, in_w],
                                                    local_dout[0] / T.cast(min_k[0], dtype),
                                                )
        return compute_dinput_multi

    if maxmin_gradient == "single":
        _kernel = _kernel_dinput_single
    else:
        _kernel = _kernel_dinput_multi

    return _kernel()

