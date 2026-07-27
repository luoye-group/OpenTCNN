"""This is the file containing kernels used for MaxPlusSumConv2d operation."""

import os
import tilelang
import tilelang.language as T
import itertools
from tilelang.autotuner import AutoTuner
from ..utils import DataType, maybe_autotune
from ..autotune_env import _get_autotune_env

def min_plus_max_conv2d_kernel(
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
                local_max = T.alloc_local((1,), dtype)

                local_accum = T.alloc_local((1,), dtype)
                
                local_img = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                local_weight = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                
                # shared memory
                sum_val = T.alloc_shared((BLOCK_B, BLOCK_P,), dtype)
                
                if tic == 0:
                    sum_val[tb, tp] = -T.infinity(dtype)
                # T.clear(local_accum)
                local_accum[0] = -T.infinity(dtype)

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
                                local_max[0] = T.infinity(dtype)
                                # Loop kh, kw over the kernel window for this local tile
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        if in_h < in_height and in_w < in_width:
                                            val = (local_img[ic, kh, kw] + local_weight[ic, kh, kw])
                                            if val < local_max[0]:
                                                local_max[0] = val
                                # local_accum[0] += local_max[0] 
                                if local_max[0] > local_accum[0]: # max operation
                                    local_accum[0] = local_max[0]

                    # T.atomic_add(sum_val[tb, tp], local_accum[0])
                    T.atomic_max(sum_val[tb, tp], local_accum[0])
                T.sync_threads()
                if valid_b and valid_p:
                    if tic == 0:
                        output_buffer[batch_idx, oc, oh, ow] = sum_val[tb, tp]
                    # T.atomic_add(output_buffer[batch_idx, oc, oh, ow], local_accum[0])

        return main
    
    return _kernel()



def min_plus_max_conv2d_kernel_save_idx(
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
            index_buffer : T.Buffer((batch_size, out_channels, out_height, out_width, 3), int_dtype), # (ic, kh, kw)
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
                local_max = T.alloc_local((1,), dtype)

                local_accum = T.alloc_local((1,), dtype)
                
                local_img = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                local_weight = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                
                local_max_idx = T.alloc_local((3,), int_dtype) # (ic, kh, kw)
                local_accum_idx = T.alloc_local((3,), int_dtype) # (ic, kh, kw)
                T.clear(local_max_idx)
                
                # shared memory
                sum_val = T.alloc_shared((BLOCK_B, BLOCK_P, reduce_threads), dtype)
                sum_idx = T.alloc_shared((BLOCK_B, BLOCK_P, reduce_threads, 3), int_dtype) # (ic, kh, kw)
                
                if tic == 0:
                    sum_val[tb, tp] = -T.infinity(dtype)
                # T.clear(local_accum)
                local_accum[0] = -T.infinity(dtype)

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
                                local_max[0] = T.infinity(dtype)
                                # Loop kh, kw over the kernel window for this local tile
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        if in_h < in_height and in_w < in_width:
                                            val = (local_img[ic, kh, kw] + local_weight[ic, kh, kw])
                                            if val < local_max[0]:
                                                local_max[0] = val
                                                local_max_idx[0] = ic_global
                                                local_max_idx[1] = kh
                                                local_max_idx[2] = kw
                                # local_accum[0] += local_max[0] 
                                if local_max[0] > local_accum[0]: # max operation
                                    local_accum[0] = local_max[0]
                                    local_accum_idx[0] = local_max_idx[0]
                                    local_accum_idx[1] = local_max_idx[1]
                                    local_accum_idx[2] = local_max_idx[2]

                    # T.atomic_add(sum_val[tb, tp], local_accum[0])
                    # T.atomic_min(sum_val[tb, tp], local_accum[0])
                    sum_val[tb, tp, tic] = local_accum[0]
                    sum_idx[tb, tp, tic, 0] = local_accum_idx[0]
                    sum_idx[tb, tp, tic, 1] = local_accum_idx[1]
                    sum_idx[tb, tp, tic, 2] = local_accum_idx[2]
                T.sync_threads()
                if valid_b and valid_p:
                    if tic == 0:
                        local_accum[0] = -T.infinity(dtype)
                        T.clear(local_accum_idx)
                        for r in T.serial(reduce_threads):
                            if sum_val[tb, tp, r] > local_accum[0]:
                                local_accum[0] = sum_val[tb, tp, r]
                                local_accum_idx[0] = sum_idx[tb, tp, r, 0]
                                local_accum_idx[1] = sum_idx[tb, tp, r, 1]
                                local_accum_idx[2] = sum_idx[tb, tp, r, 2]
                        output_buffer[batch_idx, oc, oh, ow] = local_accum[0]
                        T.copy(index_buffer[batch_idx, oc, oh, ow], local_accum_idx)
                        # index_buffer[batch_idx, oc, oh, ow, 0] = local_accum_idx[0]
                        # index_buffer[batch_idx, oc, oh, ow, 1] = local_accum_idx[1]
                        # index_buffer[batch_idx, oc, oh, ow, 2] = local_accum_idx[2]
                        
                        # output_buffer[batch_idx, oc, oh, ow] = sum_val[tb, tp]
                    # T.atomic_add(output_buffer[batch_idx, oc, oh, ow], local_accum[0])

        return main
    
    return _kernel()


# @tilelang.jit
def min_plus_max_conv2d_kernel_backward_dweight(
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
    maxmin_gradient="multiple" # multiple 1/k for more than one max/max position, single picks one of them (e.i. the first one)
):
    """Compute dweight gradient.
    
    dweight[kh,kw,ic,oc] = Σ_{b,oh,ow} (
        dout[b,oh,ow,oc] * δ_max(ic,oc,b,oh,ow,kh,kw)
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
                                
                                # Find max and max over kernel positions
                                max_val = T.alloc_local((1,), dtype)
                                max_argh = T.alloc_local((1,), int_dtype)
                                max_argw = T.alloc_local((1,), int_dtype)
                                max_val[0] = T.infinity(dtype)
                                max_argh[0] = -1
                                max_argw[0] = -1
                                
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        
                                        # Check bounds
                                        if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width):
                                            # Use the locally cached per-position image value
                                            # to ensure identical behavior to buffer-based computation.
                                            val = (local_img[bhw_offset, kh, kw] + local_weight[kh, kw])

                                            if val < max_val[0]:
                                                max_val[0] = val
                                                max_argh[0] = kh
                                                max_argw[0] = kw
                                
                                if max_argh[0] != -1:
                                    local_dweight_acc[max_argh[0], max_argw[0]] += (local_dout[bhw_offset])

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
                                
                                # 1) max over (kh, kw) for current channel ic_global
                                max_val = T.alloc_local((1,), dtype)
                                max_k = T.alloc_local((1,), int_dtype)
                                max_val[0] = T.infinity(dtype)
                                max_k[0] = 0

                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        
                                        # Check bounds
                                        if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width):
                                            # Use the locally cached per-position image value
                                            # to ensure identical behavior to buffer-based computation.
                                            local_val[kh, kw] = (local_img[bhw_offset, kh, kw] + local_weight[kh, kw])
                                            if local_val[kh, kw] < max_val[0]:
                                                max_val[0] = local_val[kh, kw]
                                
                                # 2) count max ties for current channel
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        
                                        # Check bounds
                                        if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width):
                                            if local_val[kh, kw] == max_val[0]:
                                                max_k[0] += 1

                                # 3) find max over channel-wise minima and its multiplicity
                                final_max = T.alloc_local((1,), dtype)
                                max_ch_k = T.alloc_local((1,), int_dtype)
                                final_max[0] = -T.infinity(dtype)
                                max_ch_k[0] = 0

                                for ic2 in T.serial(in_channels):
                                    ch_max = T.alloc_local((1,), dtype)
                                    ch_max[0] = T.infinity(dtype)
                                    for kh in T.serial(kernel_height):
                                        for kw in T.serial(kernel_width):
                                            in_h = oh_stride + kh * dilation_h
                                            in_w = ow_stride + kw * dilation_w
                                            if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width):
                                                v2 = img_buffer[b, ic2, in_h, in_w] + weight_buffer[oc, ic2, kh, kw]
                                                if v2 < ch_max[0]:
                                                    ch_max[0] = v2
                                    if ch_max[0] > final_max[0]:
                                        final_max[0] = ch_max[0]
                                        max_ch_k[0] = 1
                                    elif ch_max[0] == final_max[0]:
                                        max_ch_k[0] += 1
                                
                                # 4) route gradient only if this channel contributes to max over channels
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        
                                        # Check bounds
                                        if (in_h >= 0) & (in_h < in_height) & (in_w >= 0) & (in_w < in_width):
                                            if (
                                                (max_k[0] > 0)
                                                & (max_ch_k[0] > 0)
                                                & (max_val[0] == final_max[0])
                                                & (local_val[kh, kw] == max_val[0])
                                            ):
                                                local_dweight_acc[kh, kw] += (
                                                    local_dout[bhw_offset]
                                                    / (T.cast(max_k[0], dtype) * T.cast(max_ch_k[0], dtype))
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
def min_plus_max_conv2d_kernel_backward_dinput(
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
    maxmin_gradient="multiple" # multiple 1/k for more than one max/max position, single picks one of them (e.i. the first one)
):
    """Compute gradient w.r.t. input for compound-max-max-plus-sum 2D convolution with
    two mixing parameters.

    Layouts:
    img     : (B, C, H, W) NCHW
    weight  : (OC, IC, KH, KW) OIHW
    dout    : (B, OC, OH, OW)
    dimg    : (B, C, H, W)
    Gradient: for each (b, oc, oh, ow) and input channel ic, contributions are routed to
    the argmax/argmax locations of img + weight over (kh, kw)
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
            _max_block_p = (patches + max_grid_y - 1) // max_grid_y
            if _max_block_p <= 32:
                BLOCK_P = [32]
            elif _max_block_p <= 64:
                BLOCK_P = [64]
            elif _max_block_p <= 128:
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
                    local_max = T.alloc_local((1,), dtype)
                    local_max_argh = T.alloc_local((1,), int_dtype)
                    local_max_argw = T.alloc_local((1,), int_dtype)

                    local_img = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                    local_weight = T.alloc_local((TILE_IC, kernel_height, kernel_width), dtype)
                    local_dout = T.alloc_local((1,), dtype)

                    local_dout[0] = dout_buffer[batch_idx, oc, oh, ow]
                    # Initialize local max and max arrays
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
                                local_max[0] = -T.infinity(dtype)
                                local_max_argh[0] = -1
                                local_max_argw[0] = -1
                                # Loop kh, kw over the kernel window for this local tile
                                for kh in T.serial(kernel_height):
                                    for kw in T.serial(kernel_width):
                                        in_h = oh_stride + kh * dilation_h
                                        in_w = ow_stride + kw * dilation_w
                                        
                                        if in_h < in_height and in_w < in_width:
                                            val = (local_img[ic, kh, kw] + local_weight[ic, kh, kw])
                                            if val > local_max[0]:
                                                local_max[0] = val
                                                local_max_argh[0] = T.cast(kh, int_dtype)
                                                local_max_argw[0] = T.cast(kw, int_dtype)
                            
                                kh_sel2 = local_max_argh[0]
                                kw_sel2 = local_max_argw[0]
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
                        threads=(BLOCK_B, BLOCK_P)) as (bid, pid):
                tb = T.get_thread_binding(0)
                tp = T.get_thread_binding(1)    # each thread handling each patch
                # tic = T.get_thread_binding(2)   # each thread handling each ic?
                
                batch_idx = bid * BLOCK_B + tb
                patch_idx = pid * BLOCK_P + tp
                
                valid_b = batch_idx < batch_size
                valid_patch = patch_idx < patches
                
                # local memory
                max_val = T.alloc_local((in_channels,), dtype)
                max_k = T.alloc_local((in_channels,), int_dtype)
                
                T.fill(max_val, -T.infinity(dtype))
                T.fill(max_k, 0)
                
                # min_val = T.alloc_local((1,), dtype)
                neg_max_val = T.alloc_local((1,), dtype)
                min_k = T.alloc_local((1,), int_dtype)
                
                # min_val[0] = T.infinity(dtype)
                neg_max_val[0] = -T.infinity(dtype)
                min_k[0] = 0

                local_img = T.alloc_local((kernel_height, kernel_width), dtype)
                local_weight = T.alloc_local((kernel_height, kernel_width), dtype)
                local_dout = T.alloc_local((1,), dtype)

                plus_val = T.alloc_local((in_channels, kernel_height, kernel_width), dtype)
                T.clear(plus_val)

                if valid_b and valid_patch:
                    # Decode patch index
                    ow = patch_idx % out_width
                    tmp = patch_idx // out_width
                    oh = tmp % out_height
                    oc = tmp // out_height
                    oh_stride = oh * stride_h
                    ow_stride = ow * stride_w
                    
                    local_dout[0] = dout_buffer[batch_idx, oc, oh, ow]
                    
                    for ic in T.serial(in_channels):
                        # load local img
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                in_h = oh_stride + kh * dilation_h
                                in_w = ow_stride + kw * dilation_w
                                
                                if in_h < in_height and in_w < in_width:
                                    local_img[kh, kw] = img_buffer[
                                        batch_idx,
                                        ic,
                                        in_h,
                                        in_w,
                                    ]
                                local_weight[kh, kw] = weight_buffer[
                                    oc,
                                    ic,
                                    kh,
                                    kw
                                ]

                        # Loop kh, kw over the kernel window for this local tile
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                in_h = oh_stride + kh * dilation_h
                                in_w = ow_stride + kw * dilation_w
                                
                                if in_h < in_height and in_w < in_width:
                                    plus_val[ic, kh, kw] = (
                                        local_img[kh, kw] + local_weight[kh, kw]
                                    )
                                    if plus_val[ic, kh, kw] > max_val[ic]:
                                        max_val[ic] = plus_val[ic, kh, kw]
                        
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                if plus_val[ic, kh, kw] == max_val[ic]:
                                    in_h = oh_stride + kh * dilation_h
                                    in_w = ow_stride + kw * dilation_w
                                    
                                    if in_h < in_height and in_w < in_width:    
                                        max_k[ic] += 1
                        
                        # if max_val[ic] < min_val[0]:
                        #     min_val[0] = max_val[ic]
                        
                    for ic in T.serial(in_channels):
                        if -max_val[ic] > neg_max_val[0]:
                            neg_max_val[0] = -max_val[ic]
                            min_k[0] = 1
                        elif -max_val[ic] == neg_max_val[0]:
                            min_k[0] += 1
                            
                    for ic in T.serial(in_channels):
                        if -max_val[ic] == neg_max_val[0]:
                            for kh in T.serial(kernel_height):
                                for kw in T.serial(kernel_width):
                                    in_h = oh_stride + kh * dilation_h
                                    in_w = ow_stride + kw * dilation_w
                                    if in_h < in_height and in_w < in_width:
                                        if (
                                            (plus_val[ic, kh, kw] == max_val[ic])
                                            and (max_k[0] > 0)
                                            and (min_k[0] > 0)
                                        ):
                                            T.atomic_add(
                                                dimg_buffer[batch_idx, ic, in_h, in_w],
                                                local_dout[0] / (T.cast(max_k[0], dtype) * T.cast(min_k[0], dtype)),
                                            )
        return compute_dinput_multi

    if maxmin_gradient == "single":
        _kernel = _kernel_dinput_single
    else:
        _kernel = _kernel_dinput_multi

    return _kernel()


def min_plus_max_conv2d_kernel_backward_all(
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
    maxmin_gradient="multiple" # multiple 1/k for more than one max/max position, single picks one of them (e.i. the first one)
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
    def _kernel_single(
        BLOCK_B=None,
        BLOCK_P=None,
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
        
        @T.prim_func
        def compute_dinput_dweight_single(
            img_buffer : T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight_buffer : T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
            dout_buffer : T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
            dweight_buffer: T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
            dimg_buffer : T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
        ):
            
            with T.Kernel(T.ceildiv(batches, BLOCK_B), T.ceildiv(patches, BLOCK_P),
                        threads=(BLOCK_B, BLOCK_P)) as (bid, pid):
                tb = T.get_thread_binding(0)
                tp = T.get_thread_binding(1)    # each thread handling each patch
                # tic = T.get_thread_binding(2)   # each thread handling each ic?
                
                batch_idx = bid * BLOCK_B + tb
                patch_idx = pid * BLOCK_P + tp
                
                valid_b = batch_idx < batch_size
                valid_patch = patch_idx < patches
                
                # local memory
                neg_max_val = T.alloc_local((1,), dtype)
                min_h0 = T.alloc_local((1,), int_dtype)
                min_w0 = T.alloc_local((1,), int_dtype)

                final_max = T.alloc_local((1,), dtype)
                sel_ic = T.alloc_local((1,), int_dtype)
                sel_h1 = T.alloc_local((1,), int_dtype)
                sel_w1 = T.alloc_local((1,), int_dtype)

                final_max[0] = -T.infinity(dtype)

                local_img = T.alloc_local((kernel_height, kernel_width), dtype)
                local_weight = T.alloc_local((kernel_height, kernel_width), dtype)
                local_dout = T.alloc_local((1,), dtype)

                plus_val = T.alloc_local((1,), dtype)
                neg_plus_val = T.alloc_local((1,), dtype)
                
                if valid_b and valid_patch:
                    # Decode patch index
                    ow = patch_idx % out_width
                    tmp = patch_idx // out_width
                    oh = tmp % out_height
                    oc = tmp // out_height
                    oh_stride = oh * stride_h
                    ow_stride = ow * stride_w

                    local_dout[0] = dout_buffer[batch_idx, oc, oh, ow]
                    
                    for ic in T.serial(in_channels):
                        # per-channel spatial min via neg-max
                        neg_max_val[0] = -T.infinity(dtype)
                        
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                in_h = oh_stride + kh * dilation_h
                                in_w = ow_stride + kw * dilation_w
                                
                                if in_h < in_height and in_w < in_width:
                                    local_img[kh, kw] = img_buffer[
                                        batch_idx,
                                        ic,
                                        in_h,
                                        in_w,
                                    ]
                                local_weight[kh, kw] = weight_buffer[
                                    oc,
                                    ic,
                                    kh,
                                    kw
                                ]

                        # Loop kh, kw over the kernel window for this local tile
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                in_h = oh_stride + kh * dilation_h
                                in_w = ow_stride + kw * dilation_w
                                
                                if in_h < in_height and in_w < in_width:
                                    plus_val[0] = (
                                        local_img[kh, kw] + local_weight[kh, kw]
                                    )
                                    neg_plus_val[0] = -plus_val[0]
                                    if neg_plus_val[0] > neg_max_val[0]:
                                        neg_max_val[0] = neg_plus_val[0]
                                        min_h0[0] = kh
                                        min_w0[0] = kw
                        
                        # final reduction: max over per-channel mins
                        if -neg_max_val[0] > final_max[0]:
                            final_max[0] = -neg_max_val[0]
                            sel_ic[0] = ic
                            sel_h1[0] = min_h0[0]
                            sel_w1[0] = min_w0[0]
                            
                    in_h = oh_stride + sel_h1[0] * dilation_h
                    in_w = ow_stride + sel_w1[0] * dilation_w
                    if in_h < in_height and in_w < in_width:
                        T.atomic_add(
                            dweight_buffer[oc, sel_ic[0], sel_h1[0], sel_w1[0]],
                            local_dout[0],
                        )
                    
                        T.atomic_add(
                            dimg_buffer[batch_idx, sel_ic[0], in_h, in_w],
                            local_dout[0],
                        )
                        
        return compute_dinput_dweight_single

    
    @maybe_autotune(
        configs=get_configs(),
        warmup=warmup,
        rep=rep,
        enabled=autotune,
    )
    @tilelang.jit(
        target="auto",
    )
    def _kernel_multi(
        BLOCK_B=None,
        BLOCK_P=None,
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
        
        @T.prim_func
        def compute_dinput_dweight_multi(
            img_buffer : T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
            weight_buffer : T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
            dout_buffer : T.Buffer((batch_size, out_channels, out_height, out_width), dtype),
            dweight_buffer: T.Buffer((out_channels, in_channels, kernel_height, kernel_width), dtype),
            dimg_buffer : T.Buffer((batch_size, in_channels, in_height, in_width), dtype),
        ):
            
            with T.Kernel(T.ceildiv(batches, BLOCK_B), T.ceildiv(patches, BLOCK_P),
                        threads=(BLOCK_B, BLOCK_P)) as (bid, pid):
                tb = T.get_thread_binding(0)
                tp = T.get_thread_binding(1)    # each thread handling each patch
                # tic = T.get_thread_binding(2)   # each thread handling each ic?
                
                batch_idx = bid * BLOCK_B + tb
                patch_idx = pid * BLOCK_P + tp
                
                valid_b = batch_idx < batch_size
                valid_patch = patch_idx < patches
                
                # local memory
                neg_max_val = T.alloc_local((in_channels,), dtype)
                min_k = T.alloc_local((in_channels,), int_dtype)

                T.fill(neg_max_val, -T.infinity(dtype))
                T.fill(min_k, 0)

                # final channel-wise max over per-channel mins
                final_max = T.alloc_local((1,), dtype)
                max_ch_k = T.alloc_local((1,), int_dtype)

                final_max[0] = -T.infinity(dtype)
                max_ch_k[0] = 0

                local_img = T.alloc_local((kernel_height, kernel_width), dtype)
                local_weight = T.alloc_local((kernel_height, kernel_width), dtype)
                local_dout = T.alloc_local((1,), dtype)

                plus_val = T.alloc_local((in_channels, kernel_height, kernel_width), dtype)
                
                if valid_b and valid_patch:
                    # Decode patch index
                    ow = patch_idx % out_width
                    tmp = patch_idx // out_width
                    oh = tmp % out_height
                    oc = tmp // out_height
                    oh_stride = oh * stride_h
                    ow_stride = ow * stride_w
                    
                    local_dout[0] = dout_buffer[batch_idx, oc, oh, ow]
                    
                    for ic in T.serial(in_channels):
                        # load local img
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                in_h = oh_stride + kh * dilation_h
                                in_w = ow_stride + kw * dilation_w
                                
                                if in_h < in_height and in_w < in_width:
                                    local_img[kh, kw] = img_buffer[
                                        batch_idx,
                                        ic,
                                        in_h,
                                        in_w,
                                    ]
                                local_weight[kh, kw] = weight_buffer[
                                    oc,
                                    ic,
                                    kh,
                                    kw
                                ]

                        # Loop kh, kw over the kernel window for this local tile
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                in_h = oh_stride + kh * dilation_h
                                in_w = ow_stride + kw * dilation_w
                                
                                if in_h < in_height and in_w < in_width:
                                    plus_val[ic, kh, kw] = (
                                        local_img[kh, kw] + local_weight[kh, kw]
                                    )
                                    if -plus_val[ic, kh, kw] > neg_max_val[ic]:
                                        neg_max_val[ic] = -plus_val[ic, kh, kw]
                        
                        for kh in T.serial(kernel_height):
                            for kw in T.serial(kernel_width):
                                in_h = oh_stride + kh * dilation_h
                                in_w = ow_stride + kw * dilation_w
                                if in_h < in_height and in_w < in_width:
                                    if -plus_val[ic, kh, kw] == neg_max_val[ic]:
                                        min_k[ic] += 1
                        
                    for ic in T.serial(in_channels):
                        if -neg_max_val[ic] > final_max[0]:
                            final_max[0] = -neg_max_val[ic]
                            max_ch_k[0] = 1
                        elif -neg_max_val[ic] == final_max[0]:
                            max_ch_k[0] += 1
                            
                    for ic in T.serial(in_channels):
                        if -neg_max_val[ic] == final_max[0]:
                            for kh in T.serial(kernel_height):
                                for kw in T.serial(kernel_width):
                                    in_h = oh_stride + kh * dilation_h
                                    in_w = ow_stride + kw * dilation_w
                                    if in_h < in_height and in_w < in_width:
                                        if (
                                            (-plus_val[ic, kh, kw] == neg_max_val[ic])
                                            and (min_k[ic] > 0)
                                            and (max_ch_k[0] > 0)
                                        ):
                                            T.atomic_add(
                                                dweight_buffer[oc, ic, kh, kw],
                                                local_dout[0]
                                                / (
                                                    T.cast(min_k[ic], dtype)
                                                    * T.cast(max_ch_k[0], dtype)
                                                ),
                                            )
                                            T.atomic_add(
                                                dimg_buffer[batch_idx, ic, in_h, in_w],
                                                local_dout[0]
                                                / (
                                                    T.cast(min_k[ic], dtype)
                                                    * T.cast(max_ch_k[0], dtype)
                                                ),
                                            )
        return compute_dinput_dweight_multi

    if maxmin_gradient == "single":
        _kernel = _kernel_single
    else:
        _kernel = _kernel_multi

    return _kernel()

