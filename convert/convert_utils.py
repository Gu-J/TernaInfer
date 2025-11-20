import torch
from torch import nn
import ctypes
import numpy as np

from typing import Tuple, List

@torch.jit.script
def reorder_weight(weight: torch.Tensor):
        # 预生成 tile 内索引，只计算一次
    ii = torch.arange(16).view(16, 1).expand(16, 16)
    kk = torch.arange(16).view(1, 16).expand(16, 16)

    # flatten 索引方便 vectorized scatter
    ii_flat = ii.flatten()
    kk_flat = kk.flatten()

    # 预计算 tile 内对应的 reordered 索引
    reordered_i = ii_flat.clone()
    reordered_k = ((kk_flat + 1) // 2).clone()

    # 上半部分规则
    mask_upper = ii_flat < 8
    mask_lower = ~mask_upper

    # 处理上半部分
    need_adjust_upper = ((kk_flat & 2) != 0) & mask_upper
    reordered_i[need_adjust_upper] += 8
    reordered_k[need_adjust_upper] -= 1

    # 下半部分偏移
    reordered_k[mask_lower] += 8
    need_adjust_lower = ((kk_flat & 2) == 0) & mask_lower
    reordered_i[need_adjust_lower] -= 8
    reordered_k[mask_lower & ~need_adjust_lower] -= 1

    # 展平后构造映射矩阵 (tile 内相对位置)
    src_idx = ii_flat * 16 + kk_flat
    dst_idx = reordered_i * 16 + reordered_k

    # 构造 tile 内的重排映射，用于快速 gather/scatter
    reorder_map = torch.empty(256, dtype=torch.int64)
    reorder_map[dst_idx] = src_idx

    assert weight.dtype == torch.int8
    M_Global, K_Global = weight.shape
    assert M_Global % 16 == 0 and K_Global % 16 == 0, "尺寸必须是16的倍数"

    A_h = torch.empty_like(weight, dtype=torch.int8)

    # === 主循环 ===
    for i in range(0, M_Global, 16):
        for k in range(0, K_Global, 16):
            block = weight[i:i+16, k:k+16].flatten()
            reordered_block = block[reorder_map]
            A_h[i:i+16, k:k+16] = reordered_block.view(16, 16)

    return A_h


def compress_weight(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    A_h=reorder_weight(weight)

    M,K=A_h.shape[0],A_h.shape[1]

    tile_M=8
    tile_M_median=16
    tile_M_global=64
    tile_K=8
    tile_K_median=64
    tile_K_global=64

    # tile counts
    num_tiles_M = M // tile_M
    num_tiles_K = K // tile_K
    num_tiles = num_tiles_M * num_tiles_K

    num_median_tiles_M = M // tile_M_median
    num_median_tiles_K = K // tile_K_median
    num_median_tiles = num_median_tiles_M * num_median_tiles_K

    num_global_tiles_M = M // tile_M_global
    num_global_tiles_K = K // tile_K_global
    num_global_tiles = num_global_tiles_M * num_global_tiles_K

    # pre-allocate buffers (oversized, will trim later)
    # compressed values: each group of 32 stored as one uint32 -> worst-case number of groups = ceil(#possible_nnz / 32)
    # upper bound of nonzeros <= M*K
    worst_groups = (M * K + 31) // 32
    compressed_val_buf = torch.zeros(worst_groups, dtype=torch.uint32)  # will fill from index 0...
    TileOffsets = torch.zeros(num_tiles, dtype=torch.int32)             # for debugging, will not return
    # For median offsets we store one entry per median tile processed (the code only pushes for many of them)
    TileOffsets_median_list: List[int] = []
    TileOffsets_global = torch.zeros(num_global_tiles + 1, dtype=torch.int32)
    bitmap = torch.zeros(num_tiles, dtype=torch.uint64)

    val_count = 0                # total number of nonzero entries (counts before padding)
    tile_idx = 0
    median_offset_idx = 0
    global_val_counts = [0] * (num_global_tiles + 1)
    max_nnz_count = 0           # for debugging, will not return

    # main traversal
    for global_tile_m in range(num_global_tiles_M):
        for global_tile_k in range(num_global_tiles_K):
            tmp_vals = []   # collects nonzero float values for this global tile in traversal order
            global_row_start = global_tile_m * tile_M_global
            global_col_start = global_tile_k * tile_K_global
            global_val_count = 0
            median_val_counts_for_global = []  # track median tile counts within this global

            # iterate median tiles inside this global tile
            n_med_m = tile_M_global // tile_M_median
            n_med_k = tile_K_global // tile_K_median
            
            median_val_count = 0

            for median_tile_m in range(n_med_m):
                for median_tile_k in range(n_med_k):
                    median_row_start = global_row_start + median_tile_m * tile_M_median
                    median_col_start = global_col_start + median_tile_k * tile_K_median

                    # process 2x2 small-tile groups within the median tile in the specified order
                    # local_tile_m_group increments by 2, local_tile_k_group increments by 2
                    n_local_m_groups = tile_M_median // tile_M
                    n_local_k_groups = tile_K_median // tile_K
                    for local_tile_m_group in range(0, n_local_m_groups, 2):
                        for local_tile_k_group in range(0, n_local_k_groups, 2):
                            # process 2x2 in column-major order: for j in 0..1 (cols), for i in 0..1 (rows)
                            for j in range(2):
                                for i in range(2):
                                    local_tile_k = local_tile_k_group + j
                                    local_tile_m = local_tile_m_group + i

                                    col_start = median_col_start + local_tile_k * tile_K
                                    row_start = median_row_start + local_tile_m * tile_M

                                    # slice block (tile_M x tile_K), might be inside bounds
                                    # If matrix is exact multiples, bounds always valid, but we keep safe slicing
                                    r0 = row_start
                                    r1 = min(row_start + tile_M, M)
                                    c0 = col_start
                                    c1 = min(col_start + tile_K, K)
                                    block = A_h[r0:r1, c0:c1]

                                    # compute bitmap: positions of non-zero inside the small tile
                                    if block.numel() == 0:
                                        tile_bitmap = 0
                                        local_val_count = 0
                                    else:
                                        # convert to float32 for comparison and for storing values
                                        nz_mask = (block != 0)
                                        if nz_mask.any():
                                            # flatten row-major
                                            nz_flat = nz_mask.reshape(-1)
                                            # compute indices where non-zero
                                            nz_idx = torch.nonzero(nz_flat, as_tuple=False).squeeze(1)  # 1D tensor
                                            # build bitmap as python int for speed
                                            # note: positions are row_offset * tile_K + col_offset
                                            # but block may be smaller on edges; still position mapping keeps natural flatten index
                                            # create uint64 by shifting
                                            tile_bitmap = 0
                                            for pos in nz_idx.tolist():
                                                tile_bitmap |= (1 << int(pos))
                                            # append actual nonzero float values in traversal order
                                            # we must follow same order as loops: row-major inside block
                                            vals = block.reshape(-1)[nz_idx].tolist()
                                            tmp_vals.extend(vals)
                                            local_val_count = nz_idx.numel()
                                        else:
                                            tile_bitmap = 0
                                            local_val_count = 0

                                    # write tile-level outputs
                                    bitmap[tile_idx] = torch.tensor(tile_bitmap, dtype=torch.uint64)
                                    TileOffsets[tile_idx] = int(local_val_count)
                                    tile_idx += 1

                                    val_count += local_val_count
                                    median_val_count += local_val_count
                                    global_val_count += local_val_count

                    # after processing median tile's internal small tiles, push median_val_count to median offsets list
                    # The original C++ increments TileOffsets_median only when (median_tile_m < ... -1) or (median_tile_k < ... -1)
                    # We will follow same condition: add entry unless it's the very last median tile in both dims
                    if (median_tile_m < (tile_M_global // tile_M_median - 1) or median_tile_k < (tile_K_global // tile_K_median - 1)):
                        median_val_counts_for_global.append(median_val_count)
                    # in original code they only write some median offsets conditionally; we will write all and trim later if needed

            # Now pad tmp_vals so that val_count (global across processed tiles so far) is multiple of 32
            # BUT original C++ code increments val_count and global_val_count until val_count%32==0 (padding zeros appended to tmpVec)
            while val_count % 32 != 0:
                tmp_vals.append(0)
                val_count += 1
                global_val_count += 1

            # pack sign-bits per 32 values into compressed_val_buf
            # start index in compressed buffer for this global tile:
            start_group_idx = (val_count - global_val_count) // 32
            # tmp_vals length == global_val_count
            # for each group of 32, compute 32-bit mask: bit tt set if tmp_vals[t+tt] < 0
            for t in range(0, global_val_count, 32):
                res = 0
                base = t
                for tt in range(32):
                    value = tmp_vals[base + tt]
                    if value < 0:
                        res |= (1 << tt)
                compressed_val_buf[start_group_idx] = compressed_val_buf[start_group_idx] | (res & 0xFFFFFFFF)
                start_group_idx += 1

            # update global_val_counts and max
            idx_in_global_list = global_tile_m * num_global_tiles_K + global_tile_k + 1
            global_val_counts[idx_in_global_list] = global_val_count
            if global_val_count > max_nnz_count:
                max_nnz_count = global_val_count

            # append median offsets for this global tile to a global list (matching original behavior)
            # original code pushes some median offsets conditionally; we'll append median_val_counts_for_global sequentially
            TileOffsets_median_list.extend(median_val_counts_for_global)

    # compute TileOffsets_global as prefix sums of global_val_counts
    TileOffsets_global[0] = 0
    running = 0
    for i in range(1, num_global_tiles + 1):
        running += global_val_counts[i]
        TileOffsets_global[i] = running

    # compress buffer trim: actual groups used = ceil(total val_count / 32)
    total_groups_used = (val_count + 31) // 32
    compressed_val = compressed_val_buf[:total_groups_used].clone()

    # convert TileOffsets_median_list to tensor
    TileOffsets_median = torch.tensor(TileOffsets_median_list, dtype=torch.uint16) if TileOffsets_median_list else torch.empty(0, dtype=torch.uint16)

    return compressed_val, TileOffsets_global, TileOffsets_median, bitmap
