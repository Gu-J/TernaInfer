#include "AsyncCopy_PTX.cuh"
#include "TilingConfig.h"

// New features: Copy size is X * 64 Uint64, X can be  1 // for BitmapV2
template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void CopyTileFromGlobalToShared_Bitmap_1_64(uint64_t* __restrict__ SharedPTR,
                                                               const uint64_t* GlobalPTR,
                                                               bool        Pred = true)
{
    //
    int lane_id       = threadIdx.x % 32;
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy;  //   
    bool AsyncCopyPredictor = warp_id < TotalNumOfCopyUnit && Pred;  
    const uint64_t* GlobalPTR_Unit        = GlobalPTR;  
    uint64_t* __restrict__ SharedPTR_Unit = SharedPTR; 
    cp_async<16>(SharedPTR_Unit + lane_id * UINT64_PER_128B, 
                     GlobalPTR_Unit + lane_id * UINT64_PER_128B,   
                     AsyncCopyPredictor);
}


template<typename TilingConfig>
__device__ __forceinline__ void
myStoreToSharedMemoryFromRegisterBitmapV3(int32_t (*smem_CFrag)[TilingConfig::TILE_M + PADDING_SHARED_MEM_FOR_C],
                                int32_t c[][REG_PER_C_TENSOR_16_16])
{
    const unsigned int warpId        = threadIdx.x / WARP_SIZE;
    int                Warp_i        = warpId / TilingConfig::BLOCK_COL_WARPS;
    int                Warp_j        = warpId % TilingConfig::BLOCK_COL_WARPS;
    int                Warp_i_offset = Warp_i * (MMA_M * WARP_ROW_TENSORS_BITMAP_V3);
    int                Warp_j_offset = Warp_j * (MMA_N * TilingConfig::WARP_COL_TENSORS);
    //
    int lane_id = threadIdx.x % WARP_SIZE;
//
#pragma unroll
    for (int i = 0; i < WARP_ROW_TENSORS_BITMAP_V3; i++) {
#pragma unroll
        for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
            // Dealing with one 16*16 Tensor
            int RegSetID        = i + j * WARP_ROW_TENSORS_BITMAP_V3;
            int Tensor_i_offset = Warp_i_offset + i * MMA_M;
            int Tensor_j_offset = Warp_j_offset + j * MMA_N;
#pragma unroll
            for (int r = 0; r < REG_PER_C_TENSOR_16_16; r++) {
                int row_offset = lane_id / 4;
                int col_offset = (lane_id % 4) * 2;
                //
                if (r % 2 > 0)
                    col_offset += 1;
                //
                if (r % 4 >= 2)
                    row_offset += 8;
                if (r >= 4)
                    col_offset += 8;
                //
                (*(smem_CFrag + Tensor_j_offset + col_offset))[Tensor_i_offset + row_offset] = c[RegSetID][r];
            }
        }
    }
}

__device__ __forceinline__ void mySpMM_LoadFragAwithBitmapFromShem(uint32_t __restrict__ a[][2],
                                                         const int8_t* __restrict__ ShemVal,
                                                         const uint64_t* __restrict__ SharedBitmap,
                                                         bool        Pred = true)
{
    int lane_id = threadIdx.x % 32;
    int start_pos = 0;
    if (Pred == true) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int lid_offset = lane_id << 1;
                uint64_t bit1 = 1ULL << lid_offset;
                uint64_t bit2 = 2ULL << lid_offset;
                char4 tmp;
                    uint64_t bitmap0 = SharedBitmap[i * 4 + j*2];
                    // Calculate the number of ones before lane_id * 2
                    int num_ones_before0 = __popcll(bitmap0 & ((1ULL << lid_offset) - 1));
                    // Load A_val1 and adjust the offset for A_val2
                    tmp.x = (bitmap0 & bit1) ? ShemVal[start_pos+num_ones_before0++] : 0;
                    tmp.y = (bitmap0 & bit2) ? ShemVal[start_pos+num_ones_before0] : 0;
                start_pos += __popcll(bitmap0); 
                    uint64_t bitmap1 = SharedBitmap[i * 4 + j*2+1];
                    // Calculate the number of ones before lane_id * 2
                    int num_ones_before1 = __popcll(bitmap1 & ((1ULL << lid_offset) - 1));
                    // Load A_val1 and adjust the offset for A_val2
                    tmp.z = (bitmap1 & bit1) ? ShemVal[start_pos+num_ones_before1++] : 0;
                    tmp.w = (bitmap1 & bit2) ? ShemVal[start_pos+num_ones_before1] : 0;
                start_pos += __popcll(bitmap1); 

                a[i][j] = *reinterpret_cast<uint32_t*>(&tmp);
            }
        }
    }
}

template<int NumOfRowsToCopy, typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void myCopyTileFromGlobalToShared_X_64(int8_t* __restrict__ mytmpSharedPTR,
                                                                const int8_t* GlobalPTR,
                                                                const int   GlobalStride,
                                                                bool        Pred = true)
{

    constexpr int INT8_PER_128B=16;
    constexpr int COPY_UNIT_INT8_ROWS=8;    // when TILE_K == 64 
    //
    int lane_id       = threadIdx.x % 32;
    int col           = lane_id % 4;
    int row           = lane_id / 4;    // 0~7
    //
    int       warp_id            = threadIdx.x / 32;
    int       TotalNumOfCopyUnit = NumOfRowsToCopy / COPY_UNIT_INT8_ROWS;
    const int MaxIteration =
        (TotalNumOfCopyUnit - 1) / (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + 1;
#pragma unroll
    for (int i = 0; i < MaxIteration; i++) {
        int  COPY_UNIT_I        = (i * (TilingConfig::BLOCK_ROW_WARPS * TilingConfig::BLOCK_COL_WARPS) + warp_id);
        bool AsyncCopyPredictor = COPY_UNIT_I < TotalNumOfCopyUnit && Pred;  
        const int8_t* GlobalPTR_Unit        = GlobalPTR + COPY_UNIT_I * COPY_UNIT_INT8_ROWS * GlobalStride;
        int8_t* __restrict__ SharedPTR_Unit = mytmpSharedPTR + COPY_UNIT_I * COPY_UNIT_INT8_ROWS * TILE_K;
        // if(AsyncCopyPredictor)
        // {
        //     *reinterpret_cast<int4*>(SharedPTR_Unit + col * INT8_PER_128B + row * TILE_K)=
        //     *reinterpret_cast<const int4*>(GlobalPTR_Unit + col * INT8_PER_128B + row * GlobalStride);
        // }
        cp_async<16>(reinterpret_cast<uint64_t*>(SharedPTR_Unit + col * INT8_PER_128B + row * TILE_K),
                     reinterpret_cast<const uint64_t*>(GlobalPTR_Unit + col * INT8_PER_128B + row * GlobalStride),
                     AsyncCopyPredictor);
    }

}

template<int NumOfTensors, int N8>
__device__ __forceinline__ void myB_FragLoadFromSharedToRegisters(uint32_t __restrict__ Registers[][2],
                                                                int8_t* __restrict__ smem,
                                                                int warp_start_row,
                                                                int k_offset)
{
    //
    int      lane_id             = threadIdx.x % 32;
    int      i                   = lane_id / 4;

    int j = lane_id%4;
    //
    int8_t* smem0 = smem+TILE_K * (warp_start_row + i) + (k_offset);
    int8_t* smem1 = smem+TILE_K * (warp_start_row + i+8) + (k_offset);

//
#pragma unroll
    for (int i = 0; i < NumOfTensors; i++) {
        // asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        //              : "=r"(Registers[i][0]), "=r"(Registers[i][1]), "=r"(Registers[i][2]), "=r"(Registers[i][3])
        //              : "r"(smem_local_ptr));
        Registers[i][0]=*reinterpret_cast<uint32_t*>(&smem0[j*4]);
        Registers[i][1]=*reinterpret_cast<uint32_t*>(&smem1[j*4]);
        smem0 += TILE_K * MMA_N ;
        smem1 += TILE_K * MMA_N ;
    }
}

__device__ __forceinline__ void
MMA_INT8_M16N8K16(uint32_t __restrict__ c[], uint32_t __restrict__* a, uint32_t __restrict__* b)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32"
                 "{ %0, %1, %2, %3},"
                 "{ %4, %5 },"
                 "{ %6 },"
                 "{ %7, %8, %9, %10 };"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]),"r"(a[1])
                   "r"(b[0]),
                   "r"(c[0]),"r"(c[1]),"r"(c[2]),"r"(c[3]));
}

template<typename TilingConfig>
__device__ __forceinline__ void myPipelinedCoreComputationsBitmap(int32_t c[][REG_PER_C_TENSOR_16_16],
                                                          uint32_t __restrict__ a[][2],
                                                          uint32_t __restrict__ b[][2],
                                                          int8_t* __restrict__ SharedMemoryPTR,
                                                          int warp_start_row,
                                                          int warp_start_col)
{
    uint32_t(*c_uint32_t)[REG_PER_C_TENSOR_16_16] = reinterpret_cast<uint32_t(*)[REG_PER_C_TENSOR_16_16]>(c);
    myB_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
        b, SharedMemoryPTR, warp_start_col, 0);
// Sencond loading & first computation, so on
#pragma unroll
    for (int k = 0; k < BLOCK_K_TENSORS; k++) {
        uint32_t __restrict__(*b_read)[2]  = b;
        uint32_t __restrict__(*b_write)[2] = b;
        b_read += ((k) % 2) * TilingConfig::WARP_COL_TENSORS;
        b_write += ((k + 1) % 2) * TilingConfig::WARP_COL_TENSORS;
        // data loading
        if (k + 1 < BLOCK_K_TENSORS) {
            myB_FragLoadFromSharedToRegisters<TilingConfig::WARP_COL_TENSORS, TilingConfig::N8>(
                b_write, SharedMemoryPTR, warp_start_col, (k + 1) * MMA_K);
        }
#pragma unroll
            for (int j = 0; j < TilingConfig::WARP_COL_TENSORS; j++) {
                MMA_INT8_M16N8K16(c_uint32_t[j * WARP_ROW_TENSORS_BITMAP_V1], a[k], b_read[j]);
                if (!TilingConfig::N8)
                    MMA_INT8_M16N8K16(c_uint32_t[j * WARP_ROW_TENSORS_BITMAP_V1] + 4, a[k], b_read[j] + 1);  
            }
    }
}

template<typename TilingConfig>  // NumOfRowsToCopy must be multiple to COPY_UNIT_FP16_ROWS
__device__ __forceinline__ void myCopyTileFromGlobalToShared_Sparse(int8_t* __restrict__ SharedPTR,
                                                               const uint32_t* GlobalPTR,
                                                               const int   NNZ,
                                                               bool        Pred = true)
{
    if(Pred) 
    {
    int threadPerBlock = blockDim.x;
    // 32 bit!
    int NNZ_32 = NNZ/32;
    for(int i = threadIdx.x; i < NNZ_32; i+= threadPerBlock) {
        uint32_t c_val=GlobalPTR[i];
        // for(int j=0;j<32;j++)
        // {
            //     SharedPTR[i*32+j]=(c_val & (1<<j) ) ? -1 : 1;
            // }
        int8_t tmp_128bit[16];
        #pragma unroll
        for(int j=0;j<16;j++)
        {
            tmp_128bit[j]=(c_val & (1<<j) ) ? -1 : 1;
        }
        *reinterpret_cast<int4*>(&SharedPTR[i*32])=*reinterpret_cast<int4*>(tmp_128bit);
        #pragma unroll
        for(int j=0;j<16;j++)
        {
            tmp_128bit[j]=(c_val & (1<<(j+16)) ) ? -1 : 1;
        }
        *reinterpret_cast<int4*>(&SharedPTR[i*32+16])=*reinterpret_cast<int4*>(tmp_128bit);
        
    }
    }
}


