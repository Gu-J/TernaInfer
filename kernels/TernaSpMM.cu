/***************************************************************************
 * Copyright 2026 The TernaInfer Authors. All rights reserved.
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
#include "src/TernaSpMM_API.cuh"

extern "C" void checkLastCudaError(int line)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Last Cuda Error Detected at line: %d, Error: %s.\n", line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}


extern "C" void bitlinear_TernaSpMM(int8_t* input0, 
                                    uint64_t* mybitmap_gpu_v3, int* mybitmap_TileOffsets_gpu_v3,uint16_t* mybitmap_WarpOffsets_gpu_v3, uint32_t* myCompressed_Val_gpu_v3, 
                                    __nv_bfloat16* output0, 
                                    __nv_bfloat16* s, __nv_bfloat16* ws, int ws_num, int M, int N, int K, int SPLIT_K,int32_t* myReduction_Workspace_bitmapv3,cudaStream_t stream){

    int M_Global                    = N;
    int K_Global                    = K;
    int N_Global                    = M;

    if(M<=64)
        mySpMM_SplitK_API_bitmap_v3(stream,
                        myCompressed_Val_gpu_v3, 
                        mybitmap_TileOffsets_gpu_v3, 
                        mybitmap_WarpOffsets_gpu_v3, 
                        mybitmap_gpu_v3, 
                        input0,
                        output0,
                        M_Global,
                        N_Global,
                        K_Global,
                        myReduction_Workspace_bitmapv3,
                        SPLIT_K,
                        s,
                        ws,
                        ws_num
                        );
    else
        mySpMM_SplitK_API_bitmap_v3_prefill(stream,
                myCompressed_Val_gpu_v3, 
                mybitmap_TileOffsets_gpu_v3, 
                mybitmap_WarpOffsets_gpu_v3, 
                mybitmap_gpu_v3, 
                input0,
                output0,
                M_Global,
                N_Global,
                K_Global,
                myReduction_Workspace_bitmapv3,
                SPLIT_K,
                s,
                ws,
                ws_num
                );    
 
    checkLastCudaError(__LINE__);

}



//////////////////////////////////////////////////////////////// Helper Functions ////////////////////////////////////////////////////////////////



extern "C" void bitlinear_TernaSpMM_decode(int8_t* input0, 
                                    uint64_t* mybitmap_gpu_v3, int* mybitmap_TileOffsets_gpu_v3,uint16_t* mybitmap_WarpOffsets_gpu_v3, uint32_t* myCompressed_Val_gpu_v3,
                                    __nv_bfloat16* output0, 
                                    __nv_bfloat16* s, __nv_bfloat16* ws, int ws_num, int M, int N, int K, int SPLIT_K,int32_t* myReduction_Workspace_bitmapv3,cudaStream_t stream){

    int M_Global                    = N;
    int K_Global                    = K;
    int N_Global                    = M;

        mySpMM_SplitK_API_bitmap_v3(stream,
                        myCompressed_Val_gpu_v3, 
                        mybitmap_TileOffsets_gpu_v3, 
                        mybitmap_WarpOffsets_gpu_v3, 
                        mybitmap_gpu_v3, 
                        input0,
                        output0,
                        M_Global,
                        N_Global,
                        K_Global,
                        myReduction_Workspace_bitmapv3,
                        SPLIT_K,
                        s,
                        ws,
                        ws_num
                        );

    checkLastCudaError(__LINE__);

}


extern "C" void bitlinear_TernaSpMM_prefill(int8_t* input0, 
                                    uint64_t* mybitmap_gpu_v3, int* mybitmap_TileOffsets_gpu_v3,uint16_t* mybitmap_WarpOffsets_gpu_v3, uint32_t* myCompressed_Val_gpu_v3,
                                    __nv_bfloat16* output0, 
                                    __nv_bfloat16* s, __nv_bfloat16* ws, int ws_num, int M, int N, int K, int SPLIT_K,int32_t* myReduction_Workspace_bitmapv3,cudaStream_t stream){

    int M_Global                    = N;
    int K_Global                    = K;
    int N_Global                    = M;

        mySpMM_SplitK_API_bitmap_v3_prefill(stream,
                myCompressed_Val_gpu_v3, 
                mybitmap_TileOffsets_gpu_v3, 
                mybitmap_WarpOffsets_gpu_v3, 
                mybitmap_gpu_v3, 
                input0,
                output0,
                M_Global,
                N_Global,
                K_Global,
                myReduction_Workspace_bitmapv3,
                SPLIT_K,
                s,
                ws,
                ws_num
                );    
 
    checkLastCudaError(__LINE__);

}


extern "C" void bitlinear_TernaSpMM_benchmark(
                                              int mode, 
                                              int warmup_iters, 
                                              int bench_iters, 
                                              int8_t* input0, 
                                              uint64_t* mybitmap_gpu_v3,
                                              int* mybitmap_TileOffsets_gpu_v3,
                                              uint16_t* mybitmap_WarpOffsets_gpu_v3,
                                              uint32_t* myCompressed_Val_gpu_v3, 
                                              __nv_bfloat16* output0, 
                                              __nv_bfloat16* s, 
                                              __nv_bfloat16* ws, 
                                              int ws_num,
                                              int M, int N, int K, 
                                              int SPLIT_K,
                                              int32_t* myReduction_Workspace_bitmapv3,
                                              cudaStream_t stream){
                                                
    using TernaSpMMFunc = void (*)(
        int8_t*, uint64_t*, int*, uint16_t*, uint32_t*, 
        __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, 
        int, int, int, int, int, int32_t*, cudaStream_t
    );

    TernaSpMMFunc kernel_variants[] = {
        bitlinear_TernaSpMM,         // mode 0
        bitlinear_TernaSpMM_prefill, // mode 1
        bitlinear_TernaSpMM_decode   // mode 2
    };

    if (mode < 0 || mode > 2) {
        mode=0;
    }

    TernaSpMMFunc target_kernel = kernel_variants[mode];

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        target_kernel(
            input0, 
            mybitmap_gpu_v3,
            mybitmap_TileOffsets_gpu_v3,
            mybitmap_WarpOffsets_gpu_v3,
            myCompressed_Val_gpu_v3, 
            output0, 
            s, 
            ws, 
            ws_num,
            M, N, K, 
            SPLIT_K,
            myReduction_Workspace_bitmapv3,
            stream);
    }
    cudaStreamSynchronize(stream);
    checkLastCudaError(__LINE__);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Benchmark
    cudaEventRecord(start, stream);
    for (int i = 0; i < bench_iters; ++i) {
        target_kernel(
            input0, 
            mybitmap_gpu_v3,
            mybitmap_TileOffsets_gpu_v3,
            mybitmap_WarpOffsets_gpu_v3,
            myCompressed_Val_gpu_v3, 
            output0, 
            s, 
            ws, 
            ws_num,
            M, N, K, 
            SPLIT_K,
            myReduction_Workspace_bitmapv3,
            stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    checkLastCudaError(__LINE__);

    // Calculate and print timing
    float total_ms = 0;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_time_ms = total_ms / bench_iters;
    float avg_time_us = avg_time_ms * 1000.0f;

    printf("Shape(M=%d, N=%d, K=%d): %.2f us (avg over %d iters, warmup=%d)\n", 
           M, N, K, avg_time_us, bench_iters, warmup_iters);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}




extern "C" void reorder_WT_h(__nv_bfloat16* WT_h, __nv_bfloat16* reordered_WT_h, int N, int K)
{
    for(int i=0;i<N;i+=16)
    {
        for(int ii=0;ii<8;ii++)
        {
            for(int k=0;k<K;k+=16)
            {
                for(int kk=0;kk<16;kk++)
                {
                    int reordered_i=i+ii;
                    int reordered_k=k+(kk+1)/2;
                    if(kk&2)
                    {
                        reordered_i+=8;
                        reordered_k-=1;
                    }
                    reordered_WT_h[reordered_i*K+reordered_k]=WT_h[(i+ii)*K+k+kk];
                }
            }
        }
        for(int ii=8;ii<16;ii++)
        {
            for(int k=0;k<K;k+=16)
            {
                for(int kk=0;kk<16;kk++)
                {
                    int reordered_i=i+ii;
                    int reordered_k=k+(kk+1)/2+8;
                    if((kk&2)==0)
                    {
                        reordered_i-=8;
                    }
                    else reordered_k-=1;
                    reordered_WT_h[reordered_i*K+reordered_k]=WT_h[(i+ii)*K+k+kk];
                }
            }
        }
    }
}

extern "C" int convert_reorderd(
    __nv_bfloat16* A_h,
    int N,
    int K,
    int tile_N,  // 64
    int warp_subtile_N,  // 16
    int subtile_N,  // 8
    int tile_K,  // 64
    int warp_subtile_K,  // 64
    int subtile_K,  // 8
    uint64_t** Global_Bitmap,
    int** TileOffsets,
    uint16_t** WarpOffsets,
    uint32_t** Global_Values_Array,
    int& max_nnz_count)
{
    // Calc tile counts
    int num_subtiles_N = N / subtile_N;
    int num_subtiles_K = K / subtile_K;
    int num_subtiles = num_subtiles_N * num_subtiles_K;
    
    int num_warp_subtiles_N = N / warp_subtile_N;
    int num_warp_subtiles_K = K / warp_subtile_K;
    int num_warp_subtiles = num_warp_subtiles_N * num_warp_subtiles_K;

    int num_tiles_N = N / tile_N;
    int num_tiles_K = K / tile_K;
    int num_tiles = num_tiles_N * num_tiles_K;

    // malloc
    *Global_Values_Array = (uint32_t*)malloc(N * K /8);
    *TileOffsets = (int*)malloc((num_tiles + 1) * sizeof(int));
    *WarpOffsets = (uint16_t*)malloc(num_warp_subtiles * (warp_subtile_N / subtile_N * warp_subtile_K / subtile_K)/4*3 * sizeof(uint16_t));
    // *subtile_NNZs = (int*)malloc(num_subtiles * sizeof(int));

    cudaMallocHost((void**)Global_Bitmap, num_subtiles * sizeof(uint64_t));     // testing pinned memory transfer speed

    if (*Global_Values_Array == nullptr || 
        // *subtile_NNZs == nullptr || 
        *WarpOffsets == nullptr || *TileOffsets == nullptr || *Global_Bitmap == nullptr) {
        return -1;
    }

    int val_count = 0;
    int tile_idx = 0;
    int warp_offset_idx = 0;
    std::vector<int> global_val_counts(num_tiles + 1, 0);
    max_nnz_count = 0;

    // Traverse all global tiles
    for (int global_tile_m = 0; global_tile_m < num_tiles_N; ++global_tile_m) {
        for (int global_tile_k = 0; global_tile_k < num_tiles_K; ++global_tile_k) {
            std::vector<float> tmpVec;
            int global_row_start = global_tile_m * tile_N;
            int global_col_start = global_tile_k * tile_K;
            int global_val_count = 0;
            
            int warp_val_count = 0;
            // (*WarpOffsets)[warp_offset_idx++] = 0;  // The starting offset of each median tile is 0

            // Traverse warp-subtiles within the global tile (in row order)
            for (int warp_subtiles_m = 0; warp_subtiles_m < tile_N / warp_subtile_N; ++warp_subtiles_m) {
                for (int warp_subtiles_k = 0; warp_subtiles_k < tile_K / warp_subtile_K; ++warp_subtiles_k) {
                    int warp_row_start = global_row_start + warp_subtiles_m * warp_subtile_N;
                    int warp_col_start = global_col_start + warp_subtiles_k * warp_subtile_K;
                    // Process the 2x2 small subtile groups within the warp-subtile
                    for (int subtile_m_group = 0; subtile_m_group < warp_subtile_N / subtile_N; subtile_m_group += 2) {
                        for (int subtile_k_group = 0; subtile_k_group < warp_subtile_K / subtile_K; subtile_k_group += 2) {
                            // Process the 2x2 small tile groups in column-major order
                            for (int j = 0; j < 2; ++j) {
                                for (int i = 0; i < 2; ++i) {
                                    int subtile_k = subtile_k_group + j;
                                    int subtile_m = subtile_m_group + i;

                                    int col_start = warp_col_start + subtile_k * subtile_K;
                                    int row_start = warp_row_start + subtile_m * subtile_N;

                                    uint64_t tile_bitmap = 0;
                                    // int local_val_count = 0;

                                    // handle small tile
                                    for (int row_offset = 0; row_offset < subtile_N; ++row_offset) {
                                        for (int col_offset = 0; col_offset < subtile_K; ++col_offset) {
                                            int row = row_start + row_offset;
                                            int col = col_start + col_offset;

                                            if (row < N && col < K) {
                                                __nv_bfloat16 val = A_h[row * K + col];
                                                float fvalue=__bfloat162float(val);
                                                if (fvalue != 0.0f) {
                                                    tile_bitmap |= (1ULL << (row_offset * subtile_K + col_offset));
                                                    tmpVec.push_back(fvalue);
                                                    val_count++;
                                                    // local_val_count++;
                                                    warp_val_count++;
                                                    global_val_count++;
                                                }
                                            }
                                        }
                                    }

                                    (*Global_Bitmap)[tile_idx] = tile_bitmap;
                                    // (*subtile_NNZs)[tile_idx] = local_val_count;
                                    ++tile_idx;
                                }
                            }
                        }
                    }
                    if(warp_subtiles_m < (tile_N / warp_subtile_N - 1) or warp_subtiles_k < (tile_K / warp_subtile_K - 1)){
                        // Update WarpOffsets
                        (*WarpOffsets)[warp_offset_idx] = warp_val_count;
                        warp_offset_idx++;
                    } 

                }
            }

            while(val_count%32)
            {
                val_count++;
                global_val_count++;
                tmpVec.push_back(0);
            }

            for(int t=0;t<global_val_count;t+=32)
            {
                uint32_t res=0;
                for(int tt=0;tt<32;tt++)
                {
                    float fvalue=tmpVec[t+tt];
                    if(fvalue<0)
                        res |= 1<<(tt);
                }
                (*Global_Values_Array)[(val_count-global_val_count+t)/32]=res;
            }



            // Update global_val_counts and max_nnz_count
            global_val_counts[global_tile_m * num_tiles_K + global_tile_k + 1] = global_val_count;
            if (global_val_count > max_nnz_count) {
                max_nnz_count = global_val_count;
            }
        }
    }

    // Calculate offsets for global tiles
    (*TileOffsets)[0] = 0;
    for (int i = 1; i <= num_tiles; ++i) {
        global_val_counts[i] += global_val_counts[i - 1];
        (*TileOffsets)[i] = global_val_counts[i];
    }

    // Reduce the size of Global_Values_Array to the actually required size
    *Global_Values_Array = (uint32_t*)realloc(*Global_Values_Array, val_count /8);    // this is counted by byte, not numel!

    return num_tiles;
}


__global__ void kernel_sleep(double us)
{
    if (threadIdx.x == 0)
    {
        double gpu_clock_hz =  1800*1e6;
        unsigned long long start = clock64();
        unsigned long long wait_cycles = (unsigned long long)(us * 1e-6 * gpu_clock_hz);
        while (clock64() - start < wait_cycles);
    }
    __syncthreads();
}

extern "C" void gpu_sleep(double sleep_us,cudaStream_t stream)
{
    kernel_sleep<<<1, 32,0,stream>>>(sleep_us);
}