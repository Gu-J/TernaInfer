/***************************************************************************
 * Copyright 2026 The TernaInfer Authors. All rights reserved.
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
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/epilogue/threadblock/fusion/visitor_2x.hpp>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/arch.h>


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
                            int32_t*        Reduction_Workspace,  
                            int          Split_K,
                            __nv_bfloat16*  s,
                            __nv_bfloat16*  ws,
                            int ws_num)
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
                printf("MM_Sparse_API Error: Unsupported N dimension %d! N needs to be padded to a multiple of 8!\n", N_Global);
                return cudaErrorUnknown;
            }
            break;
    }
    
    cudaError_t Error = cudaGetLastError();
    if (Error != cudaSuccess)
    {
        printf("Error: mySpMM_SplitK_Kernel_Ex_bitmap_v3");
        return Error;
    }

    dim3 GridDim((M_Global * N_Global) / 256, 1, 1);
    dim3 BlockDim(WARP_SIZE, 1, 1);

    SplitK_Reduction<<<GridDim, BlockDim, 0, stream>>>(C, Reduction_Workspace,M_Global, N_Global, Split_K,s,ws,ws_num);
    return cudaGetLastError();
}


template<typename TilingConfig>
static void mySpMM_SplitK_Kernel_Ex_bitmap_v3_prefill(cudaStream_t stream,
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
    static int SHMEM_SZ = ( TilingConfig::MY_MAX_INT8_NNZS_INTILE* sizeof(int8_t)*2 + (TilingConfig::TILE_BITMAP_M_V3 * TilingConfig::TILE_BITMAP_K_V3) * sizeof(uint64_t));
    cudaFuncSetAttribute(
        mySpMM_Kernel_bitmap_prefill<TilingConfig>, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    int dimN =1;
    int  dimM = M_Global * Split_K / TilingConfig::TILE_M;
    dim3 GridDim(dimN, dimM, 1);
    dim3 BlockDim(WARP_SIZE * TilingConfig::BLOCK_WARPS, 1, 1);
    mySpMM_Kernel_bitmap_prefill<TilingConfig><<<GridDim, BlockDim, SHMEM_SZ, stream>>>(
        Compressed_A, TileOffsets, TileOffsets_Median, bitmap, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);

}



void cutlass_int8_gemm(
                            cudaStream_t   stream,
                            int8_t*  A,
                            int8_t*  B,  
                            __nv_bfloat16* C,
                            const int      M,
                            const int      N,
                            const int      K)
{
    using ElementInputA = int8_t;
    using ElementInputB = int8_t;
    using ElementOutput = cutlass::bfloat16_t;
    using ElementAccumulator = int32_t;
    using ElementComputeEpilogue = float;

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using TileShape = cutlass::gemm::GemmShape<128, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    static int const kStages = 3;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        TileShape,
        WarpShape,
        cutlass::gemm::GemmShape<16, 8, 32>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator,
            ElementComputeEpilogue>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        kStages>;


    cutlass::gemm::GemmCoord problem_size(M, N, K);

    typename Gemm::Arguments args{
        problem_size,
        {A, K},
        {B, K},
        {(ElementOutput*)C, N},
        {(ElementOutput*)C, N},
        {1.0f, 0.0f},
        1
    };

    Gemm gemm_op;
    gemm_op.initialize(args, NULL, stream);
    gemm_op(stream);
}



/* Optimization Note: Current version has high overhead. Theoretical cost 
 * is near zero when fused with CUTLASS GEMM epilogue. Disabling this 
 * during benchmarks reveals the peak theoretical performance. */
__global__ void scale_kernel(
    __nv_bfloat16* C,          // [M, N]
    int M,
    int N,
    const __nv_bfloat16* s,    // [M, 1]
    const __nv_bfloat16* ws,   // [ws_num]
    int ws_num
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int m = idx / N;
    int n = idx % N;

    int group_size = N / ws_num;
    int group = n / group_size;
    if (group >= ws_num) group = ws_num - 1;

    if(N==3840 && ws_num==3)
    {
        if(n<2560)group=0;
        else if(n<2560+640)group=1;
        else group=2;
    }

    float c  = __bfloat162float(C[idx]);
    float sm = __bfloat162float(s[m]);
    float w  = __bfloat162float(ws[group]);

    c = c / sm * w;

    C[idx] = __float2bfloat16_rn(c);
}


cudaError_t mySpMM_SplitK_API_bitmap_v3_prefill(
                            cudaStream_t stream,
                            const uint32_t*  Compressed_A,
                            const int*   TileOffsets,
                            const uint16_t* TileOffsets_Median,
                            const uint64_t* bitmap,
                            int8_t*  B,
                            __nv_bfloat16*        C,
                            const int    M_Global,
                            const int    N_Global,
                            const int    K_Global,
                            int32_t*        Reduction_Workspace,  
                            int          Split_K,
                            __nv_bfloat16*  s,
                            __nv_bfloat16*  ws,
                            int ws_num)
{
    mySpMM_SplitK_Kernel_Ex_bitmap_v3_prefill<TilingConfigBitmapV3<4, 1, 1>>(
        stream, Compressed_A, TileOffsets, TileOffsets_Median, bitmap,  B, Reduction_Workspace, M_Global, N_Global, K_Global, Split_K);
    
    cutlass_int8_gemm(stream, B, (int8_t*)Reduction_Workspace, C, N_Global, M_Global, K_Global);
    // cutlass_int8_gemm(stream, B, (int8_t*)Reduction_Workspace, C,s, N_Global, M_Global, K_Global);
    
    // return cudaGetLastError();

    int total = M_Global * N_Global;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    scale_kernel<<<blocks, threads, 0, stream>>>(
        C,
        N_Global,
        M_Global,
        s,
        ws,
        ws_num
    );

    return cudaGetLastError();
}

