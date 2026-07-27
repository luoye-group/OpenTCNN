"""This is the file containing kernels used for CompoundMinMaxPlusSumConv3d operation."""

import os
import tilelang
import tilelang.language as T
import itertools
from tilelang.autotuner import AutoTuner
from ..utils import DataType, maybe_autotune
from ..autotune_env import _get_autotune_env

def compound_min_max_plus_sum_conv3d1p_kernel(
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
    int_dtype="int32",
    autotune=False,
):
    """Fused compound-min-max-plus-sum convolution kernel variant.

    Compared with v2: retained shared memory patch loading, directly used OIHW weight layout
    (out_channels, in_channels, kernel_height, kernel_width, kernel_depth), and removed additional kernel
    flattening to avoid parameter mismatch compilation errors.
    """
    batches = batch_size
    patches = out_channels * out_height * out_width * out_depth
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
            "in_depth": in_depth,
            "kernel_height": kernel_height,
            "kernel_width": kernel_width,
            "kernel_depth": kernel_depth,
            "out_height": out_height,
            "out_width": out_width,
            "out_depth": out_depth,
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
        in_depth=None,
        kernel_height=None,
        kernel_width=None,
        kernel_depth=None,
        out_height=None,
        out_width=None,
        out_depth=None,
    ):
        MAX_TRANSACTION_SIZE_IN_BITS = 128

        TILE_IC = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads
        
        @T.prim_func
        def main(
            img_buffer : T.Buffer((batch_size, in_channels, in_height, in_width, in_depth), dtype),
            weight_buffer : T.Buffer((out_channels, in_channels, kernel_height, kernel_width, kernel_depth), dtype),
            alpha_buffer : T.Buffer((out_channels, in_channels), dtype),
            output_buffer : T.Buffer((batch_size, out_channels, out_height, out_width, out_depth), dtype),
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
                od = patch_idx % out_depth
                tmp0 = patch_idx // out_depth
                ow = tmp0 % out_width
                tmp = tmp0 // out_width
                oh = tmp % out_height
                oc = tmp // out_height
                
                oh_stride = oh * stride_h
                ow_stride = ow * stride_w
                od_stride = od * stride_d
                
                # local memory
                local_max = T.alloc_local((1,), dtype)
                local_min = T.alloc_local((1,), dtype)
                local_neg_max = T.alloc_local((1,), dtype)

                local_accum = T.alloc_local((1,), dtype)
                
                local_img = T.alloc_local((TILE_IC, kernel_height, kernel_width, kernel_depth), dtype)
                local_weight = T.alloc_local((TILE_IC, kernel_height, kernel_width, kernel_depth), dtype)
                local_alpha = T.alloc_local((TILE_IC,), dtype)
                
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
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            
                                            if in_h < in_height and in_w < in_width and in_d < in_depth:
                                                local_img[ic, kh, kw, kd] = img_buffer[
                                                    batch_idx,
                                                    ic_global,
                                                    in_h,
                                                    in_w,
                                                    in_d,
                                                ]
                                            local_weight[ic, kh, kw, kd] = weight_buffer[
                                                oc,
                                                ic_global,
                                                kh,
                                                kw,
                                                kd,
                                            ]
                                local_alpha[ic] = alpha_buffer[oc, ic_global]

                        for ic in T.serial(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                local_max[0] = -T.infinity(dtype)
                                local_neg_max[0] = -T.infinity(dtype)
                                # Loop kh, kw over the kernel window for this local tile
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            if in_h < in_height and in_w < in_width and in_d < in_depth:
                                                val = (local_img[ic, kh, kw, kd] + local_weight[ic, kh, kw, kd])
                                                if val > local_max[0]:
                                                    local_max[0] = val
                                                if -val > local_neg_max[0]:
                                                    local_neg_max[0] = -val
                                
                                local_min[0] = -local_neg_max[0]
                                local_accum[0] += local_alpha[ic] * local_max[0] + (1 - local_alpha[ic]) * local_min[0]

                    T.atomic_add(sum_val[tb, tp], local_accum[0])
                T.sync_threads()
                if valid_b and valid_p:    
                    if tic == 0:
                        output_buffer[batch_idx, oc, oh, ow, od] = sum_val[tb, tp]
                    # T.atomic_add(output_buffer[batch_idx, oc, oh, ow], local_accum[0])

        return main
    
    return _kernel()



def compound_min_max_plus_sum_conv3d1p_kernel_tiled(
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
    int_dtype="int32",
    autotune=False,
):
    """Fused compound-min-max-plus-sum convolution kernel variant.

    Compared with v2: retained shared memory patch loading, directly used OIHW weight layout
    (out_channels, in_channels, kernel_height, kernel_width, kernel_depth), and removed additional kernel
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
                BLOCK_OD = [4, 8, 16, 32]
                reduce_threads = [8, 16, 32]
            else:
                BLOCK_B = [1, 2, 4, 8, 16, 32, 64, 128]
                BLOCK_P = [32, 64, 128, 256]
                # Treat BLOCK_OH/BLOCK_OW as a coarse partition of (oh, ow) into tiles.
                BLOCK_OH = [1, 2, 4, 8, 16, 32, 64, 128]
                BLOCK_OW = [1, 2, 4, 8, 16, 32, 64, 128]
                BLOCK_OD = [1, 2, 4, 8, 16, 32, 64, 128]
                reduce_threads = [4, 8, 16, 32, 64]
        else:
            BLOCK_B = [1]
            BLOCK_OH = [16]
            BLOCK_OW = [16]
            BLOCK_OD = [16]
            # Avoid gridDim.y overflow (<= 65535).
            _min_block_p = (out_channels * BLOCK_OH[0] * BLOCK_OW[0] * BLOCK_OD[0] + max_grid_y - 1) // max_grid_y
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
            BLOCK_OD,
            reduce_threads,
        ))
        configs = []
        for c in _configs:
            b_b, b_p, b_oh, b_ow, b_od, r_t = c
            total_threads = b_b * b_p * r_t
            if b_oh != b_ow or b_oh != b_od or b_ow != b_od:
                continue
            if total_threads > max_threads_per_block:
                continue
            if r_t > 64:
                continue
            if b_oh < 1 or b_ow < 1 or b_od < 1:
                continue
            # Kernel uses patches = out_channels * BLOCK_OH * BLOCK_OW * BLOCK_OD
            patches_cfg = out_channels * b_oh * b_ow * b_od
            if (patches_cfg + b_p - 1) // b_p > max_grid_y:
                continue

            configs.append({
                "BLOCK_B": b_b,
                "BLOCK_P": b_p,
                "BLOCK_OH": b_oh,
                "BLOCK_OW": b_ow,
                "BLOCK_OD": b_od,
                "reduce_threads": r_t,
                # Sliding-window strip width (meta):
                # strip_w = (P_TILES - 1) * stride_w + (KW - 1) * dilation_w + 1
                # "STRIP_W": (c[2] - 1) * stride_w + (kernel_width - 1) * dilation_w + 1,
                "batch_size": batch_size,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "in_height": in_height,
                "in_width": in_width,
                "in_depth": in_depth,
                "kernel_height": kernel_height,
                "kernel_width": kernel_width,
                "kernel_depth": kernel_depth,
                "out_height": out_height,
                "out_width": out_width,
                "out_depth": out_depth,
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
        BLOCK_OD=None,
        reduce_threads=None,
        # STRIP_W=None,
        batch_size=None,
        in_channels=None,
        out_channels=None,
        in_height=None,
        in_width=None,
        in_depth=None,
        kernel_height=None,
        kernel_width=None,
        kernel_depth=None,
        out_height=None,
        out_width=None,
        out_depth=None,
    ):
        MAX_TRANSACTION_SIZE_IN_BITS = 128

        TILE_IC = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads
        
        patches = out_channels * BLOCK_OH * BLOCK_OW * BLOCK_OD

        # Each (boh, bow) corresponds to one output tile.
        TILE_OH = T.ceildiv(out_height, BLOCK_OH)
        TILE_OW = T.ceildiv(out_width, BLOCK_OW)
        TILE_OD = T.ceildiv(out_depth, BLOCK_OD)

        # Input tile needed to cover the output tile with stride/dilation.
        TILE_IH = (TILE_OH - 1) * stride_h + (kernel_height - 1) * dilation_h + 1
        TILE_IW = (TILE_OW - 1) * stride_w + (kernel_width - 1) * dilation_w + 1
        TILE_ID = (TILE_OD - 1) * stride_d + (kernel_depth - 1) * dilation_d + 1
        
        @T.prim_func
        def main(
            img_buffer : T.Buffer((batch_size, in_channels, in_height, in_width, in_depth), dtype),
            weight_buffer : T.Buffer((out_channels, in_channels, kernel_height, kernel_width, kernel_depth), dtype),
            alpha_buffer : T.Buffer((out_channels, in_channels), dtype),
            output_buffer : T.Buffer((batch_size, out_channels, out_height, out_width, out_depth), dtype),
        ):

            with T.Kernel(T.ceildiv(batches, BLOCK_B), T.ceildiv(patches, BLOCK_P), 
                        threads=(BLOCK_B, BLOCK_P, reduce_threads)) as (bid, pid):
                tb = T.get_thread_binding(0)
                tp = T.get_thread_binding(1)    # each thread handling each patch
                tic = T.get_thread_binding(2)   # each thread handling each ic?
                
                batch_idx = bid * BLOCK_B + tb
                patch_idx = pid * BLOCK_P + tp
                
                # Decode patch index
                bod = patch_idx % BLOCK_OD
                tmp0 = patch_idx // BLOCK_OD
                bow = tmp0 % BLOCK_OW
                tmp = tmp0 // BLOCK_OW
                boh = tmp % BLOCK_OH
                oc = tmp // BLOCK_OH
                
                base_od = bod * TILE_OD 
                base_ow = bow * TILE_OW
                base_oh = boh * TILE_OH
                
                base_h = base_oh * stride_h
                base_w = base_ow * stride_w
                base_d = base_od * stride_d
                
                valid_bidx = batch_idx < batch_size
                valid_pidx = patch_idx < patches

                local_max = T.alloc_local((1,), dtype)
                local_min = T.alloc_local((1,), dtype)
                local_neg_max = T.alloc_local((1,), dtype)

                local_accum = T.alloc_local((TILE_OH, TILE_OW, TILE_OD), dtype)
                
                # local_img = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                local_img = T.alloc_local((TILE_IC, TILE_IH, TILE_IW, TILE_ID), dtype)
                local_weight = T.alloc_local((TILE_IC, kernel_height, kernel_width, kernel_depth), dtype)
                local_alpha = T.alloc_local((TILE_IC,), dtype)
                
                # shared memory
                sum_val = T.alloc_shared((BLOCK_B, BLOCK_P, TILE_OH, TILE_OW, TILE_OD), dtype)

                T.clear(local_accum)


                if valid_bidx and valid_pidx:
                    # local memory
                    # local_max = T.alloc_local((TILE_OH, TILE_OW), dtype)
                    # local_min = T.alloc_local((TILE_OH, TILE_OW,), dtype)
                    
                    # Init shared accumulator (once per output element) then sync.
                    if tic == 0:
                        for toh in T.serial(TILE_OH):
                            for tow in T.serial(TILE_OW):
                                for tod in T.serial(TILE_OD):
                                    sum_val[tb, tp, toh, tow, tod] = T.cast(0, dtype)
                    
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
                                        for tid in T.vectorized(TILE_ID):
                                            in_h = base_h + tih
                                            in_w = base_w + tiw
                                            in_d = base_d + tid
                                            
                                            if in_h < in_height and in_w < in_width and in_d < in_depth:
                                                local_img[ic, tih, tiw, tid] = img_buffer[
                                                    batch_idx,
                                                    ic_global,
                                                    in_h,
                                                    in_w,
                                                    in_d
                                                ]
                                            else:
                                                local_img[ic, tih, tiw, tid] = T.cast(0, dtype)
                                
                                # load local weight
                                for kh in T.vectorized(kernel_height):
                                    for kw in T.vectorized(kernel_width):
                                        for kd in T.vectorized(kernel_depth):
                                            local_weight[ic, kh, kw, kd] = weight_buffer[
                                                oc,
                                                ic_global,
                                                kh,
                                                kw,
                                                kd
                                            ]
                                local_alpha[ic] = alpha_buffer[oc, ic_global]

                        for toh in T.serial(TILE_OH):
                            for tow in T.serial(TILE_OW):
                                for tod in T.serial(TILE_OD):
                                    oh_out = base_oh + toh
                                    ow_out = base_ow + tow
                                    od_out = base_od + tod
                                    
                                    if oh_out < out_height and ow_out < out_width and od_out < out_depth:
                                        for ic in T.serial(TILE_IC):
                                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                                            if ic_global < in_channels:
                                                
                                                local_max[0] = -T.infinity(dtype)
                                                local_neg_max[0] = -T.infinity(dtype)
                                                
                                                base_tih = toh * stride_h
                                                base_tiw = tow * stride_w
                                                base_tid = tod * stride_d
                                                
                                                for kh in T.serial(kernel_height):
                                                    tih = base_tih + kh * dilation_h
                                                    for kw in T.serial(kernel_width):
                                                        tiw = base_tiw + kw * dilation_w
                                                        for kd in T.serial(kernel_depth):
                                                            tid = base_tid + kd * dilation_d
                                                            val = (local_img[ic, tih, tiw, tid] + local_weight[ic, kh, kw, kd])
                                                            if val > local_max[0]:
                                                                local_max[0] = val
                                                            if -val > local_neg_max[0]:
                                                                local_neg_max[0] = -val
                                                
                                                local_min[0] = -local_neg_max[0]
                                                local_accum[toh, tow, tod] += local_alpha[ic] * local_max[0] + (1 - local_alpha[ic]) * local_min[0]
                    for toh in T.serial(TILE_OH):
                        for tow in T.serial(TILE_OW):
                            for tod in T.serial(TILE_OD):
                                T.atomic_add(sum_val[tb, tp, toh, tow, tod], local_accum[toh, tow, tod])

                T.sync_threads()
                if valid_bidx and valid_pidx:
                    if tic == 0:
                        for toh in T.serial(TILE_OH):
                            for tow in T.serial(TILE_OW):
                                for tod in T.serial(TILE_OD):
                                    oh_out = base_oh + toh
                                    ow_out = base_ow + tow
                                    od_out = base_od + tod
                                    if oh_out < out_height and ow_out < out_width and od_out < out_depth:
                                        output_buffer[batch_idx, oc, oh_out, ow_out, od_out] = sum_val[tb, tp, toh, tow, tod]
                T.sync_threads()

        return main
    
    return _kernel()


# @tilelang.jit
def compound_min_max_plus_sum_conv3d1p_backward_da(
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
    int_dtype="int32",
    autotune=False,
):
    """Compute dalpha gradients.
    
    dalpha[oc,ic] = Σ_{b,oh,ow,od} dout[b,oc,oh,ow,od] * (max_val[oc,ic,b,oh,ow,od] + min_val[oc,ic,b,oh,ow,od])
    
    Parallelize over (ic, oc) with tiling over spatial dimensions.
    """

    BHWD = batch_size * out_height * out_width * out_depth
    fast, warmup, rep, max_configs = _get_autotune_env()
    
    def get_configs(*_kernel_params, **_kwargs):
        if autotune:
            BLOCK_OC = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            BLOCK_IC = [1, 2, 4, 8, 16, 32, 64, 128, 256]
            reduce_threads = [4, 8, 16, 32, 64, 128, 256]
        else:
            BLOCK_OC = [8]
            BLOCK_IC = [2]
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
            "in_depth": in_depth,
            "kernel_height": kernel_height,
            "kernel_width": kernel_width,
            "kernel_depth": kernel_depth,
            "out_height": out_height,
            "out_width": out_width,
            "out_depth": out_depth,
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

    def _dab_kernel(
        BLOCK_OC=None,
        BLOCK_IC=None,
        reduce_threads=None,
        batch_size=None,
        in_channels=None,
        out_channels=None,
        in_height=None,
        in_width=None,
        in_depth=None,
        kernel_height=None,
        kernel_width=None,
        kernel_depth=None,
        out_height=None,
        out_width=None,
        out_depth=None,
    ):
        MAX_TRANSACTION_SIZE_IN_BITS = 128
        TILE_BHWD = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_BHWD = TILE_BHWD * reduce_threads

        @T.prim_func
        def compute_dalpha(
            img_buffer: T.Buffer((batch_size, in_channels, in_height, in_width, in_depth), dtype),
            weight_buffer: T.Buffer((out_channels, in_channels, kernel_height, kernel_width, kernel_depth), dtype),
            dout_buffer: T.Buffer((batch_size, out_channels, out_height, out_width, out_depth), dtype),
            dalpha_buffer: T.Buffer((out_channels, in_channels), dtype),
        ):
            # Shared memory for accumulation
            # shared_dalpha = T.alloc_shared((BLOCK_OC, BLOCK_IC), dtype)
            
            shared_dalpha = T.alloc_shared((BLOCK_OC, BLOCK_IC, BLOCK_BHWD), dtype)
            
            with T.Kernel(
                T.ceildiv(out_channels, BLOCK_OC),
                T.ceildiv(in_channels, BLOCK_IC),
                threads=(BLOCK_OC, BLOCK_IC, reduce_threads)
            ) as (oc_block, ic_block):
                toc = T.get_thread_binding(0)
                tic = T.get_thread_binding(1)
                tbhwd = T.get_thread_binding(2)

                # Global indices
                oc = oc_block * BLOCK_OC + toc
                ic = ic_block * BLOCK_IC + tic
                
                # Only process valid channels
                valid_oc = oc < out_channels
                valid_ic = ic < in_channels
                
                # Local accumulation
                local_alpha_acc = T.alloc_local((1,), dtype)
                local_alpha_acc[0] = T.cast(0, dtype)
                # Cache per-bhwd and per-kernel-position image values locally
                # so that local computations match buffer-based correctness.
                local_img = T.alloc_local((TILE_BHWD, kernel_height, kernel_width, kernel_depth), dtype)
                local_weight = T.alloc_local((kernel_height, kernel_width, kernel_depth), dtype)
                local_dout = T.alloc_local((TILE_BHWD), dtype)
                
                T.sync_threads()
                
                if tbhwd == 0 and toc == 0 and tic == 0:
                    # shared_dalpha[toc, tic] = 0
                    # shared_dalpha[toc, tic, tbhwd] = 0
                    T.clear(shared_dalpha)
                
                T.sync_threads()
                
                if valid_oc and valid_ic:
                    # assign local_weight
                    for kh in T.serial(kernel_height):
                        for kw in T.serial(kernel_width):
                            for kd in T.serial(kernel_depth):
                                local_weight[kh, kw, kd] = weight_buffer[oc, ic, kh, kw, kd]
                    
                    # Process batches and spatial positions in tiles
                    for bhwd_block in T.serial(T.ceildiv(BHWD, BLOCK_BHWD)):
                        bhwd_start = bhwd_block * BLOCK_BHWD + tbhwd * TILE_BHWD
                        
                        for bhwd_offset in T.vectorized(TILE_BHWD):
                            bhwd = bhwd_start + bhwd_offset
                            
                            # Check if this spatial position is valid
                            if bhwd < BHWD:
                                # Decode b, oh, ow
                                od = bhwd % out_depth
                                tmp1 = bhwd // out_depth
                                ow = tmp1 % out_width
                                tmp2 = tmp1 // out_width
                                oh = tmp2 % out_height
                                b = tmp2 // out_height
                                
                                oh_stride = oh * stride_h
                                ow_stride = ow * stride_w
                                od_stride = od * stride_d
                                
                                local_dout[bhwd_offset] = dout_buffer[b, oc, oh, ow, od]
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width) & (in_d >= 0) & (in_d < in_depth):
                                                local_img[bhwd_offset, kh, kw, kd] = img_buffer[b, ic, in_h, in_w, in_d]

                        for bhwd_offset in T.serial(TILE_BHWD):
                            bhwd = bhwd_start + bhwd_offset
                            
                            # Check if this spatial position is valid
                            if bhwd < BHWD:
                                # Decode b, oh, ow
                                od = bhwd % out_depth
                                tmp1 = bhwd // out_depth
                                ow = tmp1 % out_width
                                tmp2 = tmp1 // out_width
                                oh = tmp2 % out_height
                                b = tmp2 // out_height
                                
                                oh_stride = oh * stride_h
                                ow_stride = ow * stride_w
                                od_stride = od * stride_d
                                # Find max and min over kernel positions
                                max_val = T.alloc_local((1,), dtype)
                                min_val = T.alloc_local((1,), dtype)
                                max_val[0] = -T.infinity(dtype)
                                neg_max = T.alloc_local((1,), dtype)
                                neg_max[0] = -T.infinity(dtype)
                                
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            
                                            # Check bounds
                                            if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width) & (in_d >= 0) & (in_d < in_depth):
                                                # Use the locally cached per-position image value
                                                # to ensure identical behavior to buffer-based computation.
                                                val = (local_img[bhwd_offset, kh, kw, kd] + local_weight[kh, kw, kd])

                                                if val > max_val[0]:
                                                    max_val[0] = val
                                                if -val > neg_max[0]:
                                                    neg_max[0] = -val

                                min_val[0] = -neg_max[0]
                                local_alpha_acc[0] = local_alpha_acc[0] + local_dout[bhwd_offset] * (max_val[0] - min_val[0])
                    
                T.sync_threads()

                if valid_oc and valid_ic:
                    # accumulate per thread
                    shared_dalpha[toc, tic, tbhwd] = local_alpha_acc[0]
                    
                T.sync_threads()
                    
                if valid_oc and valid_ic:
                    if tbhwd == 0:
                        final_val = T.alloc_local((1,), dtype)
                        T.clear(final_val)
                        
                        for r in T.serial(reduce_threads):
                            final_val[0] += shared_dalpha[toc, tic, r]
                        
                        dalpha_buffer[oc, ic] = final_val[0]
                    
                    # or it can be done with atomic add without the need for final reduction thread, but may have more contention:
                    # T.atomic_add(shared_dalpha[toc, tic], local_alpha_acc[0])
                    # T.sync_threads()
                    # dalpha_buffer[oc, ic] = shared_dalpha[toc, tic]
            
                T.sync_threads()
                    
        return compute_dalpha


    return _dab_kernel()


# @tilelang.jit
def compound_min_max_plus_sum_conv3d1p_backward_dweight(
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
    int_dtype="int32",
    autotune=False,
    maxmin_gradient="multiple" # multiple 1/k for more than one max/min position, single picks one of them (e.i. the first one)
):
    """Compute dweight gradient.
    
    dweight[kh,kw,kd,ic,oc] = Σ_{b,oh,ow,od} (
        alpha[ic,oc] * dout[b,oh,ow,od,oc] * (δ_max(ic,oc,b,oh,ow,od,kh,kw,kd) + δ_min(ic,oc,b,oh,ow,od,kh,kw,kd))
    )
    
    Parallelize over (kh, kw, kd ic, oc) with tiling over batch*spatial.
    """

    BHWD = batch_size * out_height * out_width * out_depth

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
            "in_depth": in_depth,
            "kernel_height": kernel_height,
            "kernel_width": kernel_width,
            "kernel_depth": kernel_depth,
            "out_height": out_height,
            "out_width": out_width,
            "out_depth": out_depth,
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
        in_depth=None,
        kernel_height=None,
        kernel_width=None,
        kernel_depth=None,
        out_height=None,
        out_width=None,
        out_depth=None,
    ):

        MAX_TRANSACTION_SIZE_IN_BITS = 128
        K = kernel_height * kernel_width * kernel_depth
        TILE_BHWD = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_BHWD = TILE_BHWD * reduce_threads
        
        @T.prim_func
        def compute_dweight_single(
            img_buffer: T.Buffer((batch_size, in_channels, in_height, in_width, in_depth), dtype),
            weight_buffer: T.Buffer((out_channels, in_channels, kernel_height, kernel_width, kernel_depth), dtype),
            alpha_buffer: T.Buffer((out_channels, in_channels), dtype),
            dout_buffer: T.Buffer((batch_size, out_channels, out_height, out_width, out_depth), dtype),
            dweight_buffer: T.Buffer((out_channels, in_channels, kernel_height, kernel_width, kernel_depth), dtype),
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
                tbhwd = T.get_thread_binding(2)
                
                # Global indices
                oc = oc_block * BLOCK_OC + toc  # one oc per grid z; expand within loop if BLOCK_OC>1
                ic = ic_block * BLOCK_IC + tic 
                
                # Bounds check for kh/kw/oc
                valid_oc = oc < out_channels
                valid_ic = ic < in_channels
                
                # shared memory (reduce over tbhwd without atomics)
                shared_dweight = T.alloc_shared((BLOCK_OC, BLOCK_IC, kernel_height, kernel_width, kernel_depth, reduce_threads), dtype)
                
                # local memory
                local_dweight_acc = T.alloc_local((kernel_height, kernel_width, kernel_depth), dtype)
                local_alpha = T.alloc_local((1,), dtype)
                local_alpha[0] = alpha_buffer[oc, ic]
                local_img = T.alloc_local((TILE_BHWD, kernel_height, kernel_width, kernel_depth), dtype)
                local_weight = T.alloc_local((kernel_height, kernel_width, kernel_depth), dtype)
                local_dout = T.alloc_local((TILE_BHWD), dtype)
                
                T.sync_threads()

                if tbhwd == 0 and toc == 0 and tic == 0:
                    T.clear(shared_dweight)
                T.clear(local_dweight_acc)

                T.sync_threads()
                    
                if valid_ic & valid_oc:
                    for kh in T.serial(kernel_height):
                        for kw in T.serial(kernel_width):
                            for kd in T.serial(kernel_depth):
                                local_weight[kh, kw, kd] = weight_buffer[oc, ic, kh, kw, kd]
                    # Process batches and spatial positions in tiles
                    for bhwd_block in T.serial(T.ceildiv(BHWD, BLOCK_BHWD)):
                        bhwd_start = bhwd_block * BLOCK_BHWD + tbhwd * TILE_BHWD
                        
                        for bhwd_offset in T.vectorized(TILE_BHWD):
                            bhwd = bhwd_start + bhwd_offset
                            
                            # Check if this spatial position is valid
                            if bhwd < BHWD:
                                # Decode b, oh, ow
                                od = bhwd % out_depth
                                tmp1 = bhwd // out_depth
                                ow = tmp1 % out_width
                                tmp2 = tmp1 // out_width
                                oh = tmp2 % out_height
                                b = tmp2 // out_height
                                
                                oh_stride = oh * stride_h
                                ow_stride = ow * stride_w
                                od_stride = od * stride_d
                                
                                local_dout[bhwd_offset] = dout_buffer[b, oc, oh, ow, od]
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width) & (in_d >= 0) & (in_d < in_depth):
                                                local_img[bhwd_offset, kh, kw, kd] = img_buffer[b, ic, in_h, in_w, in_d]

                        for bhwd_offset in T.serial(TILE_BHWD):
                            bhwd = bhwd_start + bhwd_offset
                            
                            # Check if this spatial position is valid
                            if bhwd < BHWD:
                                # Decode b, oh, ow
                                od = bhwd % out_depth
                                tmp1 = bhwd // out_depth
                                ow = tmp1 % out_width
                                tmp2 = tmp1 // out_width
                                oh = tmp2 % out_height
                                b = tmp2 // out_height
                                
                                oh_stride = oh * stride_h
                                ow_stride = ow * stride_w
                                od_stride = od * stride_d
                                
                                # Find max and min over kernel positions
                                max_val = T.alloc_local((1,), dtype)
                                neg_max_val = T.alloc_local((1,), dtype)
                                max_argh = T.alloc_local((1,), int_dtype)
                                max_argw = T.alloc_local((1,), int_dtype)
                                max_argd = T.alloc_local((1,), int_dtype)
                                min_argh = T.alloc_local((1,), int_dtype)
                                min_argw = T.alloc_local((1,), int_dtype)
                                min_argd = T.alloc_local((1,), int_dtype)
                                max_val[0] = -T.infinity(dtype)
                                neg_max_val[0] = -T.infinity(dtype)
                                max_argh[0] = -1
                                max_argw[0] = -1
                                max_argd[0] = -1
                                min_argh[0] = -1
                                min_argw[0] = -1
                                min_argd[0] = -1
                                
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                        
                                            # Check bounds
                                            if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width) & (in_d >= 0) & (in_d < in_depth):
                                                # Use the locally cached per-position image value
                                                # to ensure identical behavior to buffer-based computation.
                                                val = (local_img[bhwd_offset, kh, kw, kd] + local_weight[kh, kw, kd])
                                                if val > max_val[0]:
                                                    max_val[0] = val
                                                    max_argh[0] = kh
                                                    max_argw[0] = kw
                                                    max_argd[0] = kd
                                                if -val > neg_max_val[0]:
                                                    neg_max_val[0] = -val
                                                    min_argh[0] = kh
                                                    min_argw[0] = kw
                                                    min_argd[0] = kd
                                
                                if max_argh[0] != -1:
                                    local_dweight_acc[max_argh[0], max_argw[0], max_argd[0]] += (local_alpha[0] * local_dout[bhwd_offset])
                                if min_argh[0] != -1:
                                    local_dweight_acc[min_argh[0], min_argw[0], min_argd[0]] += ((1 - local_alpha[0]) * local_dout[bhwd_offset])

                    for kh in T.serial(kernel_height):
                        for kw in T.serial(kernel_width):
                            for kd in T.serial(kernel_depth):
                                shared_dweight[toc, tic, kh, kw, kd, tbhwd] = local_dweight_acc[kh, kw, kd]
                            # T.atomic_add(shared_dweight[toc, tic, kh, kw, tbhwd], local_dweight_acc[kh, kw])

                T.sync_threads()

                if valid_ic & valid_oc:
                    if tbhwd == 0:
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                for kd in T.serial(kernel_depth):
                                    total = T.alloc_local((1,), dtype)
                                    total[0] = T.cast(0, dtype)
                                    for r in T.serial(reduce_threads):
                                        total[0] = total[0] + shared_dweight[toc, tic, kh, kw, kd, r]
                                    dweight_buffer[oc, ic, kh, kw, kd] = total[0] 
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
        in_depth=None,
        kernel_height=None,
        kernel_width=None,
        kernel_depth=None,
        out_height=None,
        out_width=None,
        out_depth=None,
    ):

        MAX_TRANSACTION_SIZE_IN_BITS = 128
        K = kernel_height * kernel_width * kernel_depth
        TILE_BHWD = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_BHWD = TILE_BHWD * reduce_threads
        
        @T.prim_func
        def compute_dweight_multi(
            img_buffer: T.Buffer((batch_size, in_channels, in_height, in_width, in_depth), dtype),
            weight_buffer: T.Buffer((out_channels, in_channels, kernel_height, kernel_width, kernel_depth), dtype),
            alpha_buffer: T.Buffer((out_channels, in_channels), dtype),
            dout_buffer: T.Buffer((batch_size, out_channels, out_height, out_width, out_depth), dtype),
            dweight_buffer: T.Buffer((out_channels, in_channels, kernel_height, kernel_width, kernel_depth), dtype),
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
                tbhwd = T.get_thread_binding(2)
                
                # Global indices
                oc = oc_block * BLOCK_OC + toc  # one oc per grid z; expand within loop if BLOCK_OC>1
                ic = ic_block * BLOCK_IC + tic 
                
                # Bounds check for kh/kw/oc
                valid_oc = oc < out_channels
                valid_ic = ic < in_channels
                
                # shared memory (reduce over tbhwd without atomics)
                shared_dweight = T.alloc_shared((BLOCK_OC, BLOCK_IC, kernel_height, kernel_width, kernel_depth, reduce_threads), dtype)
                
                # local memory
                local_dweight_acc = T.alloc_local((kernel_height, kernel_width, kernel_depth), dtype)
                local_alpha = T.alloc_local((1,), dtype)
                local_alpha[0] = alpha_buffer[oc, ic]
                local_img = T.alloc_local((TILE_BHWD, kernel_height, kernel_width, kernel_depth), dtype)
                local_weight = T.alloc_local((kernel_height, kernel_width, kernel_depth), dtype)
                local_dout = T.alloc_local((TILE_BHWD), dtype)
                
                local_val = T.alloc_local((kernel_height, kernel_width, kernel_depth), dtype)
                
                T.sync_threads()

                if tbhwd == 0 and toc == 0 and tic == 0:
                    T.clear(shared_dweight)
                T.clear(local_dweight_acc)

                T.sync_threads()
                    
                if valid_ic & valid_oc:
                    for kh in T.serial(kernel_height):
                        for kw in T.serial(kernel_width):
                            for kd in T.serial(kernel_depth):
                                local_weight[kh, kw, kd] = weight_buffer[oc, ic, kh, kw, kd]
                    # Process batches and spatial positions in tiles
                    for bhwd_block in T.serial(T.ceildiv(BHWD, BLOCK_BHWD)):
                        bhwd_start = bhwd_block * BLOCK_BHWD + tbhwd * TILE_BHWD
                        
                        for bhwd_offset in T.vectorized(TILE_BHWD):
                            bhwd = bhwd_start + bhwd_offset
                            
                            # Check if this spatial position is valid
                            if bhwd < BHWD:
                                # Decode b, oh, ow
                                od = bhwd % out_depth
                                tmp1 = bhwd // out_depth
                                ow = tmp1 % out_width
                                tmp2 = tmp1 // out_width
                                oh = tmp2 % out_height
                                b = tmp2 // out_height
                                
                                oh_stride = oh * stride_h
                                ow_stride = ow * stride_w
                                od_stride = od * stride_d
                                
                                local_dout[bhwd_offset] = dout_buffer[b, oc, oh, ow, od]
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width) & (in_d >= 0) & (in_d < in_depth):
                                                local_img[bhwd_offset, kh, kw, kd] = img_buffer[b, ic, in_h, in_w, in_d]

                        for bhwd_offset in T.serial(TILE_BHWD):
                            bhwd = bhwd_start + bhwd_offset
                            
                            # Check if this spatial position is valid
                            if bhwd < BHWD:
                                # Decode b, oh, ow
                                od = bhwd % out_depth
                                tmp1 = bhwd // out_depth
                                ow = tmp1 % out_width
                                tmp2 = tmp1 // out_width
                                oh = tmp2 % out_height
                                b = tmp2 // out_height
                                
                                oh_stride = oh * stride_h
                                ow_stride = ow * stride_w
                                od_stride = od * stride_d
                                
                                # Find max and min over kernel positions
                                max_val = T.alloc_local((1,), dtype)
                                neg_max = T.alloc_local((1,), dtype)
                                max_k = T.alloc_local((1,), int_dtype)
                                min_k = T.alloc_local((1,), int_dtype)

                                max_k[0] = 0
                                min_k[0] = 0

                                max_val[0] = -T.infinity(dtype)
                                neg_max[0] = -T.infinity(dtype) # min_val = -neg_max

                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            
                                            # Check bounds
                                            if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width) & (in_d >= 0) & (in_d < in_depth):
                                                # Use the locally cached per-position image value
                                                # to ensure identical behavior to buffer-based computation.
                                                local_val[kh, kw, kd] = (local_img[bhwd_offset, kh, kw, kd] + local_weight[kh, kw, kd])
                                                
                                                if local_val[kh, kw, kd] > max_val[0]:
                                                    max_val[0] = local_val[kh, kw, kd]
                                                if -local_val[kh, kw, kd] > neg_max[0]:
                                                    neg_max[0] = -local_val[kh, kw, kd]
                                
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            
                                            # Check bounds
                                            if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width) & (in_d >= 0) & (in_d < in_depth):
                                                if local_val[kh, kw, kd] == max_val[0]:
                                                    max_k[0] += 1
                                                if -local_val[kh, kw, kd] == neg_max[0]:
                                                    min_k[0] += 1
                                
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            
                                            # Check bounds
                                            if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width) & (in_d >= 0) & (in_d < in_depth):
                                                if (max_k[0] > 0) & (local_val[kh, kw, kd] == max_val[0]):
                                                    local_dweight_acc[kh, kw, kd] += (
                                                        local_alpha[0]
                                                        * local_dout[bhwd_offset]
                                                        / T.cast(max_k[0], dtype)
                                                    )
                                                if (min_k[0] > 0) & (-local_val[kh, kw, kd] == neg_max[0]):
                                                    local_dweight_acc[kh, kw, kd] += (
                                                        (1 - local_alpha[0])
                                                        * local_dout[bhwd_offset]
                                                        / T.cast(min_k[0], dtype)
                                                    )

                    for kh in T.serial(kernel_height):
                        for kw in T.serial(kernel_width):
                            for kd in T.serial(kernel_depth):
                                shared_dweight[toc, tic, kh, kw, kd, tbhwd] = local_dweight_acc[kh, kw, kd]
                                        # T.atomic_add(shared_dweight[toc, tic, kh, kw, tbhwd], local_dweight_acc[kh, kw])

                T.sync_threads()

                if valid_ic & valid_oc:
                    if tbhwd == 0:
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                for kd in T.serial(kernel_depth):
                                    total = T.alloc_local((1,), dtype)
                                    total[0] = T.cast(0, dtype)
                                    for r in T.serial(reduce_threads):
                                        total[0] = total[0] + shared_dweight[toc, tic, kh, kw, kd, r]
                                    dweight_buffer[oc, ic, kh, kw, kd] = total[0] 
                                # T.atomic_add(dweight_buffer[oc, ic, kh, kw], total[0])
        return compute_dweight_multi
        
    if maxmin_gradient == "single":
        _kernel = _kernel_dweight_single
    else:
        _kernel = _kernel_dweight_multi

    return _kernel()



# @tilelang.jit
def compound_min_max_plus_sum_conv3d1p_backward_dinput(
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
    int_dtype="int32",
    autotune=False,
    maxmin_gradient="multiple" # multiple 1/k for more than one max/min position, single picks one of them (e.i. the first one)
):
    """Compute gradient w.r.t. input for compound-min-max-plus-sum 2D convolution with
    two mixing parameters.

    Layouts:
    img     : (B, C, H, W, D) NCHWD
    weight  : (OC, IC, KH, KW, KD) OIHKD
    alpha   : (OC, IC)
    dout    : (B, OC, OH, OW, OD)
    dimg    : (B, C, H, W, D)
    Gradient: for each (b, oc, oh, ow, od) and input channel ic, contributions are routed to
    the argmax/argmin locations of img + weight over (kh, kw, kd), scaled by alpha
    respectively.
    """    
    batches = batch_size
    patches = out_channels * out_height * out_width * out_depth
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
            "in_depth": in_depth,
            "kernel_height": kernel_height,
            "kernel_width": kernel_width,
            "kernel_depth": kernel_depth,
            "out_height": out_height,
            "out_width": out_width,
            "out_depth": out_depth,
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
        in_depth=None,
        kernel_height=None,
        kernel_width=None,
        kernel_depth=None,
        out_height=None,
        out_width=None,
        out_depth=None,
    ):

        MAX_TRANSACTION_SIZE_IN_BITS = 128

        TILE_IC = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads
        
        @T.prim_func
        def compute_dinput_single(
            img_buffer : T.Buffer((batch_size, in_channels, in_height, in_width, in_depth), dtype),
            weight_buffer : T.Buffer((out_channels, in_channels, kernel_height, kernel_width, kernel_depth), dtype),
            alpha_buffer : T.Buffer((out_channels, in_channels), dtype),
            dout_buffer : T.Buffer((batch_size, out_channels, out_height, out_width, out_depth), dtype),
            dimg_buffer : T.Buffer((batch_size, in_channels, in_height, in_width, in_depth), dtype),
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
                    od = patch_idx % out_depth
                    tmp0 = patch_idx // out_depth
                    ow = tmp0 % out_width
                    tmp = tmp0 // out_width
                    oh = tmp % out_height
                    oc = tmp // out_height
                    oh_stride = oh * stride_h
                    ow_stride = ow * stride_w
                    od_stride = od * stride_d
                    # local memory
                    local_max = T.alloc_local((1,), dtype)
                    local_neg_max = T.alloc_local((1,), dtype)
                    local_max_argh = T.alloc_local((1,), int_dtype)
                    local_max_argw = T.alloc_local((1,), int_dtype)
                    local_max_argd = T.alloc_local((1,), int_dtype)
                    local_min_argh = T.alloc_local((1,), int_dtype)
                    local_min_argw = T.alloc_local((1,), int_dtype)
                    local_min_argd = T.alloc_local((1,), int_dtype)

                    local_img = T.alloc_local((TILE_IC, kernel_height, kernel_width, kernel_depth), dtype)
                    local_weight = T.alloc_local((TILE_IC, kernel_height, kernel_width, kernel_depth), dtype)
                    local_alpha = T.alloc_local((TILE_IC,), dtype)
                    local_dout = T.alloc_local((1,), dtype)

                    local_dout[0] = dout_buffer[batch_idx, oc, oh, ow, od]
                    # Initialize local max and min arrays
                    for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                        for ic in T.vectorized(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                # load local img
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride+ kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            
                                            if in_h < in_height and in_w < in_width and in_d < in_depth:
                                                local_img[ic, kh, kw, kd] = img_buffer[
                                                    batch_idx,
                                                    ic_global,
                                                    in_h,
                                                    in_w,
                                                    in_d
                                                ]
                                            local_weight[ic, kh, kw, kd] = weight_buffer[
                                                oc,
                                                ic_global,
                                                kh,
                                                kw,
                                                kd
                                            ]
                                local_alpha[ic] = alpha_buffer[oc, ic_global]                                
                                
                        for ic in T.serial(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                local_max[0] = -T.infinity(dtype)
                                local_neg_max[0] = -T.infinity(dtype)
                                local_max_argh[0] = -1
                                local_max_argw[0] = -1
                                local_max_argd[0] = -1
                                local_min_argh[0] = -1
                                local_min_argw[0] = -1
                                local_min_argd[0] = -1
                                # Loop kh, kw over the kernel window for this local tile
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            
                                            if in_h < in_height and in_w < in_width and in_d < in_depth:
                                                val = (local_img[ic, kh, kw, kd] + local_weight[ic, kh, kw, kd])
                                                if val > local_max[0]:
                                                    local_max[0] = val
                                                    local_max_argh[0] = T.cast(kh, int_dtype)
                                                    local_max_argw[0] = T.cast(kw, int_dtype)
                                                    local_max_argd[0] = T.cast(kd, int_dtype)
                                                if -val > local_neg_max[0]:
                                                    local_neg_max[0] = -val
                                                    local_min_argh[0] = T.cast(kh, int_dtype)
                                                    local_min_argw[0] = T.cast(kw, int_dtype)
                                                    local_min_argd[0] = T.cast(kd, int_dtype)
                            
                                kh_sel1 = local_max_argh[0]
                                kw_sel1 = local_max_argw[0]
                                kd_sel1 = local_max_argd[0]
                                kh_sel2 = local_min_argh[0]
                                kw_sel2 = local_min_argw[0]
                                kd_sel2 = local_min_argd[0]
                                in_h1 = oh_stride + kh_sel1 * dilation_h
                                in_w1 = ow_stride + kw_sel1 * dilation_w
                                in_d1 = od_stride + kd_sel1 * dilation_d
                                in_h2 = oh_stride + kh_sel2 * dilation_h
                                in_w2 = ow_stride + kw_sel2 * dilation_w
                                in_d2 = od_stride + kd_sel2 * dilation_d
                                if (
                                    (in_h1 >= 0)
                                    and (in_h1 < in_height)
                                    and (in_w1 >= 0)
                                    and (in_w1 < in_width)
                                    and (in_d1 >= 0)
                                    and (in_d1 < in_depth)
                                ):
                                    T.atomic_add(
                                        dimg_buffer[batch_idx, ic_global, in_h1, in_w1, in_d1],
                                        local_alpha[ic] * local_dout[0],
                                    )
                                if (
                                    (in_h2 >= 0)
                                    and (in_h2 < in_height)
                                    and (in_w2 >= 0)
                                    and (in_w2 < in_width)
                                    and (in_d2 >= 0)
                                    and (in_d2 < in_depth)
                                ):
                                    T.atomic_add(
                                        dimg_buffer[batch_idx, ic_global, in_h2, in_w2, in_d2],
                                        (1 - local_alpha[ic]) * local_dout[0],
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
        in_depth=None,
        kernel_height=None,
        kernel_width=None,
        kernel_depth=None,
        out_height=None,
        out_width=None,
        out_depth=None,
    ):

        MAX_TRANSACTION_SIZE_IN_BITS = 128

        TILE_IC = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_IC = TILE_IC * reduce_threads
        
        @T.prim_func
        def compute_dinput_multi(
            img_buffer : T.Buffer((batch_size, in_channels, in_height, in_width, in_depth), dtype),
            weight_buffer : T.Buffer((out_channels, in_channels, kernel_height, kernel_width, kernel_depth), dtype),
            alpha_buffer : T.Buffer((out_channels, in_channels), dtype),
            dout_buffer : T.Buffer((batch_size, out_channels, out_height, out_width, out_depth), dtype),
            dimg_buffer : T.Buffer((batch_size, in_channels, in_height, in_width, in_depth), dtype),
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
                    od = patch_idx % out_depth
                    tmp0 = patch_idx // out_depth
                    ow = tmp0 % out_width
                    tmp = tmp0 // out_width
                    oh = tmp % out_height
                    oc = tmp // out_height
                    oh_stride = oh * stride_h
                    ow_stride = ow * stride_w
                    od_stride = od * stride_d
                    # local memory
                    local_max = T.alloc_local((1,), dtype)
                    local_neg_max = T.alloc_local((1,), dtype)
                    max_k = T.alloc_local((1,), int_dtype)
                    min_k = T.alloc_local((1,), int_dtype)

                    local_img = T.alloc_local((TILE_IC, kernel_height, kernel_width, kernel_depth), dtype)
                    local_weight = T.alloc_local((TILE_IC, kernel_height, kernel_width, kernel_depth), dtype)
                    local_alpha = T.alloc_local((TILE_IC,), dtype)
                    local_dout = T.alloc_local((1,), dtype)

                    local_val = T.alloc_local((kernel_height, kernel_width, kernel_depth), dtype)

                    local_dout[0] = dout_buffer[batch_idx, oc, oh, ow, od]
                    # Initialize local max and min arrays
                    for bic in T.serial(T.ceildiv(in_channels, BLOCK_IC)):
                        for ic in T.vectorized(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                # load local img
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride+ kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            
                                            if in_h < in_height and in_w < in_width and in_d < in_depth:
                                                local_img[ic, kh, kw, kd] = img_buffer[
                                                    batch_idx,
                                                    ic_global,
                                                    in_h,
                                                    in_w,
                                                    in_d
                                                ]
                                            local_weight[ic, kh, kw, kd] = weight_buffer[
                                                oc,
                                                ic_global,
                                                kh,
                                                kw,
                                                kd
                                            ]
                                local_alpha[ic] = alpha_buffer[oc, ic_global]                                
                                
                        for ic in T.serial(TILE_IC):
                            ic_global = bic * BLOCK_IC + tic * TILE_IC + ic
                            if ic_global < in_channels:
                                local_max[0] = -T.infinity(dtype)
                                local_neg_max[0] = -T.infinity(dtype)
                                max_k[0] = 0
                                min_k[0] = 0

                                # Loop kh, kw over the kernel window for this local tile
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            
                                            if in_h < in_height and in_w < in_width and in_d < in_depth:
                                                local_val[kh, kw, kd] = (
                                                    local_img[ic, kh, kw, kd] + local_weight[ic, kh, kw, kd]
                                                )
                                                if local_val[kh, kw, kd] > local_max[0]:
                                                    local_max[0] = local_val[kh, kw, kd]
                                                if -local_val[kh, kw, kd] > local_neg_max[0]:
                                                    local_neg_max[0] = -local_val[kh, kw, kd]
                                                
                                # local_min[0] = -local_neg_max[0]
                                
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            
                                            if in_h < in_height and in_w < in_width and in_d < in_depth:
                                                if local_val[kh, kw, kd] == local_max[0]:
                                                    max_k[0] += 1
                                                if -local_val[kh, kw, kd] == local_neg_max[0]:
                                                    min_k[0] += 1
                                
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        for kd in T.serial(kernel_depth):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            in_d = od_stride + kd * dilation_d
                                            
                                            if in_h < in_height and in_w < in_width and in_d < in_depth:
                                                if (max_k[0] > 0) and (local_val[kh, kw, kd] == local_max[0]):
                                                    T.atomic_add(
                                                        dimg_buffer[batch_idx, ic_global, in_h, in_w, in_d],
                                                        local_alpha[ic] * local_dout[0] / T.cast(max_k[0], dtype),
                                                    )
                                                if (min_k[0] > 0) and (-local_val[kh, kw, kd] == local_neg_max[0]):
                                                    T.atomic_add(
                                                        dimg_buffer[batch_idx, ic_global, in_h, in_w, in_d],
                                                        (1 - local_alpha[ic]) * local_dout[0] / T.cast(min_k[0], dtype),
                                                    )
        return compute_dinput_multi

    if maxmin_gradient == "single":
        _kernel = _kernel_dinput_single
    else:
        _kernel = _kernel_dinput_multi

    return _kernel()

