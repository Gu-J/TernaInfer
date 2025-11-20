/***************************************************************************
 * Copyright 2025 The TernaInfer Authors. All rights reserved.
 * Copyright 2025 The SpInfer Authors. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include "./SpMM_Kernel.cuh"
#include "./Reduction_Kernel.cuh"

template<typename TilingConfig>
static void mySpMM_SplitK_Kernel_Ex_bitmap_v3(cudaStream_t stream,
                                  const uint32_t* Compressed_A,
                                  const int*   TileOffsets,
                                  const uint16_t* TileOffsets_Median,
                                  const uint64_t*   bitmap,
                                  const int8_t*  B,
                                  int32_t*        Reduction_Workspace,
                                  const int    M_Global,
                                  const int    N_Global,
                                  const int    K_Global,
                                  int          Split_K)
{
    static int SHMEM_SZ = max((TilingConfig::TILE_N * TILE_K) * (sizeof(int8_t)) * 2 + TilingConfig::MY_MAX_INT8_NNZS_INTILE* sizeof(int8_t) + (TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3) * sizeof(uint64_t),
                              (TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C) * TilingConfig::TILE_N * sizeof(int32_t));
    cudaFuncSetAttribute(
        mySpMM_Kernel_bitmap_v3<TilingConfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    int dimN =
        max(N_Global / TilingConfig::TILE_N, 1);  // max(N_Global/TilingConfig::TILE_N,1) used when N=8, TILE_N=16
    int  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3 GridDim(dimN, dimM, 1);  // Grid Size is increased due to SplitK for higher SM occupancy
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);
    mySpMM_Kernel_bitmap_v3<TilingConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        Compressed_A, TileOffsets, TileOffsets_Median, bitmap,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
}

cudaError_t mySpMM_SplitK_API_bitmap_v3(cudaStream_t stream,
                            const uint32_t*  Compressed_A,
                            const int*   TileOffsets,
                            const uint16_t* TileOffsets_Median,
                            const uint64_t* bitmap,
                            const int8_t*  B,
                            __nv_bfloat16*        C,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            int32_t*        Reduction_Workspace,  // Identical workspace for all SpMM kernel launchesSpMM_SplitK_Kernel_Ex_bitmap
                            int          Split_K,
                            __nv_bfloat16*  s,
                            __nv_bfloat16*  ws)
{
    // Batched SpMM
    switch (N_Global) {
        case 8:
            mySpMM_SplitK_Kernel_Ex_bitmap_v3<TilingConfigBitmapV3<4, 1, 1, 1>>(
                stream, Compressed_A, TileOffsets, TileOffsets_Median, bitmap,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
            break;
        case 16:
            mySpMM_SplitK_Kernel_Ex_bitmap_v3<TilingConfigBitmapV3<4, 1, 1>>(
                stream, Compressed_A, TileOffsets, TileOffsets_Median, bitmap,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
            break;
        case 32:
            mySpMM_SplitK_Kernel_Ex_bitmap_v3<TilingConfigBitmapV3<4, 1, 2>>(
                stream, Compressed_A, TileOffsets, TileOffsets_Median, bitmap,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
            break;
        case 64:
            mySpMM_SplitK_Kernel_Ex_bitmap_v3<TilingConfigBitmapV3<4, 1, 4>>(
                stream, Compressed_A, TileOffsets, TileOffsets_Median, bitmap,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
            break;
        case 128:
            mySpMM_SplitK_Kernel_Ex_bitmap_v3<TilingConfigBitmapV3<4, 1, 4>>(
                stream, Compressed_A, TileOffsets, TileOffsets_Median, bitmap,   B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
            break;
        default:
            if (N_Global % 128 == 0)
            mySpMM_SplitK_Kernel_Ex_bitmap_v3<TilingConfigBitmapV3<4, 1, 4>>(stream,
                                                                                     Compressed_A,
                                                                                     TileOffsets,
                                                                                     TileOffsets_Median,
                                                                                     bitmap,
                                                                                     B,
                                                                                     Reduction_Workspace,
                                                                                     M_Global,
                                                                                     N_Global,
                                                                                     K_Global,
                                                                                     Split_K);
            else {
                printf("MM_Sparse_API Error: Unsupported N dimension %d!\n", N_Global);
                return cudaErrorUnknown;
            }
            break;
    }
    
    //
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
        return Error;

    int ws_num=1;         // referring to Bitnet-b1.58-2B-4T
    if(M_Global==3840 && K_Global==2560)ws_num=3;
    else if(M_Global==2560 && K_Global==2560)ws_num=1;
    else if(M_Global==13824 && K_Global==2560)ws_num=2;
    else if(M_Global==2560 && K_Global==6912)ws_num=1;

    dim3 GridDim((M_Global * N_Global) / 256, 1, 1);
    dim3 BlockDim(WARP_SIZE, 1, 1);

    SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace,M_Global, N_Global, Split_K,s,ws,ws_num);
    return cudaGetLastError();
}
