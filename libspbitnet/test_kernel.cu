// Performs: X = x @ WT, where x is [M,K], WT is [N,K] and X is [M,N]
// Correct input format: ./spmm_test M N K Sparsity SplitK

// nvcc test_kernel.cu -L. -lspbitnet -lcublas -o test_kernel -arch=sm_86
// ./test_kernel 8 2560 2560 45 7

#include <assert.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

#define WARM_UP_ITERATION 10
#define BENCHMARK_ITERATION 1000

#define PROFILE_MEMCPY(name, dst, src, count, kind) do {                     \
    cudaEvent_t start_##name, stop_##name;                                   \
    cudaEventCreate(&start_##name);                                          \
    cudaEventCreate(&stop_##name);                                           \
    size_t size_##name = count;                                              \
    cudaEventRecord(start_##name);                                           \
    cudaMemcpy(dst, src, count, kind);                                       \
    cudaEventRecord(stop_##name);                                            \
    cudaEventSynchronize(stop_##name);                                       \
    float ms_##name = 0.0f;                                                  \
    cudaEventElapsedTime(&ms_##name, start_##name, stop_##name);            \
    if(1)printf("[Memcpy] %-40s Size: %10zu bytes (%.2f MB), Time: %.3f ms\n",    \
           #name, size_##name, size_##name / (1024.0 * 1024.0), ms_##name);  \
    cudaEventDestroy(start_##name);                                          \
    cudaEventDestroy(stop_##name);                                           \
} while (0)



extern "C" void checkLastCudaError(int line);

extern "C" void bitlinear_int8xint2(int8_t* input0, 
                                    uint32_t* myCompressed_Val_gpu_v3, int* mybitmap_TileOffsets_global_gpu_v3,uint16_t*mybitmap_TileOffsets_median_gpu_v3,uint64_t*mybitmap_gpu_v3,
                                    __nv_bfloat16* output0, 
                                    __nv_bfloat16* s, __nv_bfloat16* ws, int M, int N, int K, int SPLIT_K,int32_t* myReduction_Workspace_bitmapv3,cudaStream_t stream);

extern "C" int myInitSparseMatrix_bitmap(
    __nv_bfloat16* A_h,
    int M,
    int K,
    int tile_M,  // 8
    int tile_M_median,  // 16
    int tile_M_global,  // 64
    int tile_K,  // 8
    int tile_K_median,  // 64
    int tile_K_global,  // 64
    uint32_t** Compressed_Val,
    int** TileOffsets,
    uint16_t** TileOffsets_median,
    int** TileOffsets_global,
    uint64_t** bitmap,
    int& max_nnz_count);

extern "C" void reorder_WT_h(__nv_bfloat16* WT_h, __nv_bfloat16* reordered_WT_h, int N, int K);


__host__ void myinit_host_matrices(__nv_bfloat16* x_h, __nv_bfloat16* WT_h, int M, int N, int K, int WEIGHT_SPARSITY)
{
    for (int i = 0; i < M * K; i++)
        // x_h[i] = __float2bfloat16_rn(static_cast<float>((rand() % 256)-128));
        x_h[i] = __float2bfloat16_rn(1);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            int r = rand() % 100;
            if (r >= WEIGHT_SPARSITY)
                WT_h[j + i * K] = __float2bfloat16_rn(static_cast<float>((rand() % 2)*2-1));   // 1 or -1
            else
                WT_h[j + i * K] = __float2bfloat16_rn(0.0f);

        }
    }
}


void PrintPerformance(const char* KernelName, float milliseconds, float tflops, double error)
{
    printf("%-10s \t -> \t\t Time/us: %5.3f \t Performance/TFLOPs: %4.2f \t TotalError: %.2lf\n",
           KernelName,
           milliseconds*1000,
           tflops,
           error);
}


int main(int argc, char** argv)
{
    if (argc != 6) {
        printf("Correct input format: ./spmm_test M N K Sparsity SplitK\n");
        printf("Performs: X = x @ WT, where x is [M,K], WT is [N,K] and X is [M,N]\n");
        return 0;
    }
    int M                    = atoi(argv[1]);
    int N                    = atoi(argv[3]);
    int K                    = atoi(argv[2]);
    int WEIGHT_SPARSITY             = atoi(argv[4]);
    int SPLIT_K                     = atoi(argv[5]);
    cublasStatus_t cublas_status;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Host memory
    __nv_bfloat16* x_h            = NULL;  // row major
    __nv_bfloat16* WT_h            = NULL;  // col major
    // Device memory
    __nv_bfloat16* x            = NULL;
    __nv_bfloat16* WT           = NULL;
    //
    x_h            = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * M * K);
    WT_h           = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * N * K);
    if (x_h == NULL || WT_h == NULL) {
        printf("Error in CPU Malloc!\n");
        exit(-1);
    }
    cudaMalloc(reinterpret_cast<void**>(&x), sizeof(__nv_bfloat16) * M * K);
    cudaMalloc(reinterpret_cast<void**>(&WT), sizeof(__nv_bfloat16) * N * K);
    if (x == NULL || WT == NULL) {
        printf("Error in cudaMalloc!\n");
        exit(-1);
    }
    
    myinit_host_matrices(x_h, WT_h, M, N, K, WEIGHT_SPARSITY);
    
    cudaMemcpy(x, x_h, sizeof(__nv_bfloat16) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(WT, WT_h, sizeof(__nv_bfloat16) * N * K, cudaMemcpyHostToDevice);
    checkLastCudaError(__LINE__);
 
// CUBLAS
/////////////////////////////////////////////////////////////////////////////////////////////////
    printf("Launching CuBlas...\n");
    __nv_bfloat16* X_cublas = NULL;
    cudaMalloc(reinterpret_cast<void**>(&X_cublas), sizeof(__nv_bfloat16) * M * N);
    if (X_cublas == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(X_cublas, 0, sizeof(__nv_bfloat16) * M * N);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, 0);
    const float      alpha     = 1.0;
    const float      beta      = 0.0;
    cublasGemmAlgo_t CuBlasALG = static_cast<cublasGemmAlgo_t>(0);
    // Tensor core enabled
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    cudaDeviceSynchronize();
    for (int i = 0; i < WARM_UP_ITERATION; i++) {
        cublas_status = cublasGemmEx(handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     N,
                                     M,
                                     K,
                                     &alpha,
                                     WT,
                                     CUDA_R_16BF,
                                     K,
                                     x,
                                     CUDA_R_16BF,
                                     K,
                                     &beta,
                                     X_cublas,
                                     CUDA_R_16BF,
                                     N,
                                     CUDA_R_32F,
                                     CuBlasALG);
    }
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        printf("Cublas Error at line %d, Error Code: %d\n", __LINE__, cublas_status);
        exit(EXIT_FAILURE);
    }
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
                        cublasGemmEx(handle,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     N,
                                     M,
                                     K,
                                     &alpha,
                                     WT,
                                     CUDA_R_16BF,
                                     K,
                                     x,
                                     CUDA_R_16BF,
                                     K,
                                     &beta,
                                     X_cublas,
                                     CUDA_R_16BF,
                                     N,
                                     CUDA_R_32F,
                                     CuBlasALG);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //
    float milliseconds_cublas_tc = 0;
    cudaEventElapsedTime(&milliseconds_cublas_tc, start, stop);
    milliseconds_cublas_tc = milliseconds_cublas_tc / BENCHMARK_ITERATION;
    float tflops_cublas_tc = static_cast<double>((static_cast<double>(M) * N * K * 2)
                                                 / (milliseconds_cublas_tc / 1000.))/ 1e12;
    __nv_bfloat16* X_cublas_h = NULL;
    X_cublas_h       = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * M * N);
    if (X_cublas_h == NULL) {
        printf("Error in spmm_test.cu: line %d CPU Malloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemcpy(X_cublas_h, X_cublas, sizeof(__nv_bfloat16) * M * N, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(X_cublas);
    cudaFree(x);
    cublasDestroy(handle);


    ////////////////////////////////////////////////////////////////////////////////////////////////
// my

    __nv_bfloat16* myX_SpMM_bitmapv3 = NULL;
    cudaMalloc(reinterpret_cast<void**>(&myX_SpMM_bitmapv3), sizeof(__nv_bfloat16) * M * N);
    if (myX_SpMM_bitmapv3 == NULL) {
        printf("Error in spmm_test.cu: line %d cudaMalloc falied\n", __LINE__);
        exit(-1);
    }
    cudaMemset(myX_SpMM_bitmapv3, 0, sizeof(__nv_bfloat16) * M * N);

    // Define the output pointer
    uint32_t* myCompressed_Val_cpu_v3 = nullptr;
    int* mybitmap_TileOffsets_cpu_v3 = nullptr;
    // int* mybitmap_TileOffsets_median_cpu_v3 = nullptr;
    // my idea
    uint16_t* mybitmap_TileOffsets_median_cpu_v3 = nullptr;
    int* mybitmap_TileOffsets_global_cpu_v3 = nullptr;
    uint64_t* mybitmap_cpu_v3 = nullptr;
    int mymax_nnz_intilev3 = 0;
    // Call the InitSparseMatrixA_bitmap_v6 function
    
    // my reordering
    __nv_bfloat16* reordered_WT_h=nullptr;
    reordered_WT_h = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * N * K);
    reorder_WT_h(WT_h,reordered_WT_h,N,K);

    auto mynum_gtilesv3 = myInitSparseMatrix_bitmap(reordered_WT_h, N, K, 8, 16, 64, 8, 64, 64, &myCompressed_Val_cpu_v3, &mybitmap_TileOffsets_cpu_v3, &mybitmap_TileOffsets_median_cpu_v3, &mybitmap_TileOffsets_global_cpu_v3, &mybitmap_cpu_v3, mymax_nnz_intilev3);

    auto mylocal_tile_numv3 = 8*8;
    auto mymedian_tile_numv3 = 4*1;
    auto mynum_ltilesv3 = mynum_gtilesv3 * mylocal_tile_numv3;
    auto mynum_mtilesv3 = mynum_gtilesv3 * mymedian_tile_numv3/4*3;
    // The offset of the last tile is equal to the total number of compressed non-zero values
    int myval_count_v3 = mybitmap_TileOffsets_global_cpu_v3[mynum_gtilesv3]; 
    uint16_t myval_count_median_v3 = mybitmap_TileOffsets_median_cpu_v3[mynum_mtilesv3];
    // Adjust max_nnz_intilev3 to a multiple of 64
    if (mymax_nnz_intilev3 % 64 != 0) {
        mymax_nnz_intilev3 = ((mymax_nnz_intilev3 / 64) + 1) * 64;
    }
    printf("num_tiles: %d, bitmap v3 NNZ: %d, bitmap v3 median layer NNZ: %d,  max_nnz_intilev3: %d \n", mynum_gtilesv3, myval_count_v3, myval_count_median_v3, mymax_nnz_intilev3);


    int8_t* myx_h = nullptr;
    cudaMallocHost((void**)&myx_h, sizeof(int8_t) * M * K );
    for(int i=0;i<M * K;i++)
        myx_h[i]=(int8_t)__bfloat162float(x_h[i]);


    uint32_t* myCompressed_Val_gpu_v3 = nullptr;
    uint16_t* mybitmap_TileOffsets_median_gpu_v3 = nullptr;
    int* mybitmap_TileOffsets_global_gpu_v3 = nullptr;
    uint64_t* mybitmap_gpu_v3 = nullptr;
    int8_t* myx=nullptr;
    cudaMalloc(&mybitmap_gpu_v3, sizeof(uint64_t) * (mynum_ltilesv3));
    cudaMalloc(&mybitmap_TileOffsets_median_gpu_v3, sizeof(uint16_t) * (mynum_mtilesv3));
    cudaMalloc(&mybitmap_TileOffsets_global_gpu_v3, sizeof(int) * (mynum_gtilesv3 + 1));
    cudaMalloc(&myx, sizeof(int8_t) * M * K);
    if (myval_count_v3 == 0)
         myval_count_v3 = 1;  // For 100% sparsity, NNZ = 0, malloc will return NULL
    cudaMalloc(&myCompressed_Val_gpu_v3,  myval_count_v3/8+32);    // padding for 128bit fetch
    if (mybitmap_gpu_v3 == NULL || myCompressed_Val_gpu_v3 == NULL || mybitmap_TileOffsets_global_gpu_v3 == NULL) {
        printf("Error in malloc memory from device memory!\n");
        exit(-1);
    }

    PROFILE_MEMCPY(mybitmap_TileOffsets_global_gpu_v3_copy,
                mybitmap_TileOffsets_global_gpu_v3,
                mybitmap_TileOffsets_global_cpu_v3,
                sizeof(int) * (mynum_gtilesv3 + 1),
                cudaMemcpyHostToDevice);
    free(mybitmap_TileOffsets_global_cpu_v3);

    PROFILE_MEMCPY(mybitmap_TileOffsets_median_gpu_v3_copy,
                mybitmap_TileOffsets_median_gpu_v3,
                mybitmap_TileOffsets_median_cpu_v3,
                sizeof(uint16_t) * mynum_mtilesv3,
                cudaMemcpyHostToDevice);
    free(mybitmap_TileOffsets_median_cpu_v3);

    PROFILE_MEMCPY(mybitmap_gpu_v3_copy,
                mybitmap_gpu_v3,
                mybitmap_cpu_v3,
                sizeof(uint64_t) * mynum_ltilesv3,
                cudaMemcpyHostToDevice);
    cudaFreeHost(mybitmap_cpu_v3);

    PROFILE_MEMCPY(myCompressed_Val_gpu_v3_copy,
                myCompressed_Val_gpu_v3,
                myCompressed_Val_cpu_v3,
                myval_count_v3/8,
                cudaMemcpyHostToDevice);
    free(myCompressed_Val_cpu_v3);

    PROFILE_MEMCPY(myx_copy,
                myx,
                myx_h,
                sizeof(int8_t) * M * K,
                cudaMemcpyHostToDevice);
    cudaFreeHost(myx_h);

    printf("SPLIT_K = %d\n", SPLIT_K);
    int32_t* myReduction_Workspace_bitmapv3 = NULL;
    cudaMalloc(reinterpret_cast<void**>(&myReduction_Workspace_bitmapv3), sizeof(int32_t) * M * N * SPLIT_K);
    if (myReduction_Workspace_bitmapv3 == NULL) {
        printf("Error in cudaMalloc\n");
        exit(-1);
    }

    __nv_bfloat16* s_h = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16)*M);
    for (int i = 0; i < M; ++i) {
        s_h[i] = __float2bfloat16(1.0);
    }
    __nv_bfloat16* ws_h = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16)*4);
    for (int i = 0; i < 4; ++i) {
        ws_h[i] = __float2bfloat16(1.0);
    }

    __nv_bfloat16* s;
    cudaMalloc(&s, M * sizeof(__nv_bfloat16));
    cudaMemcpy(s, s_h, M * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    __nv_bfloat16* ws;
    cudaMalloc(&ws, 4 * sizeof(__nv_bfloat16));
    cudaMemcpy(ws, ws_h, 4 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    for (int i = 0; i < WARM_UP_ITERATION; i++)
        bitlinear_int8xint2(
                        myx,
                        myCompressed_Val_gpu_v3, // half
                        mybitmap_TileOffsets_global_gpu_v3, // int
                        mybitmap_TileOffsets_median_gpu_v3, // int
                        mybitmap_gpu_v3, //uint64
                        myX_SpMM_bitmapv3,
                        s,
                        ws,
                        M,
                        N,
                        K,
                        SPLIT_K,
                        myReduction_Workspace_bitmapv3,
                        0
                    );
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATION; i++)
        bitlinear_int8xint2(
                        myx,
                        myCompressed_Val_gpu_v3, // half
                        mybitmap_TileOffsets_global_gpu_v3, // int
                        mybitmap_TileOffsets_median_gpu_v3, // int
                        mybitmap_gpu_v3, //uint64
                        myX_SpMM_bitmapv3,
                        s,
                        ws,
                        M,
                        N,
                        K,
                        SPLIT_K,
                        myReduction_Workspace_bitmapv3,
                        0
                    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    checkLastCudaError(__LINE__);
    // //
    float mymilliseconds_SpMM_bitmapv3 = 0.0f;
    cudaEventElapsedTime(&mymilliseconds_SpMM_bitmapv3, start, stop);
    mymilliseconds_SpMM_bitmapv3 = mymilliseconds_SpMM_bitmapv3 / BENCHMARK_ITERATION;
    float mytflops_SpMM_bitmapv3 =
        static_cast<double>((static_cast<double>(M) * N * K * 2) / (mymilliseconds_SpMM_bitmapv3 / 1000.))
        / 1e12;
    __nv_bfloat16* myX_SpMM_hbitmapv3 = NULL;  // col major
    myX_SpMM_hbitmapv3       = (__nv_bfloat16*)malloc(sizeof(__nv_bfloat16) * M * N);
    cudaMemcpy(myX_SpMM_hbitmapv3, myX_SpMM_bitmapv3, sizeof(__nv_bfloat16) * M * N, cudaMemcpyDeviceToHost);  // Col Major
    cudaFree(myX_SpMM_bitmapv3);
    cudaFree(mybitmap_TileOffsets_global_gpu_v3);
    cudaFree(mybitmap_TileOffsets_median_gpu_v3);
    cudaFree(mybitmap_gpu_v3);
    cudaFree(myCompressed_Val_gpu_v3);
    cudaFree(myReduction_Workspace_bitmapv3);
    cudaFree(s);
    cudaFree(ws);
    /////////////////////////////////////////////////////////////////////////////////////////////////


    double mytotalError_SpMM_bitmapv3 = 0.0;

    for (int i = 0; i < M * N; i++)
        mytotalError_SpMM_bitmapv3 += fabs(__bfloat162float(X_cublas_h[i]) - __bfloat162float(myX_SpMM_hbitmapv3[i]));

    // for(int i=0;i<10;i++)
    //     printf("%.2f, ",__bfloat162float(X_cublas_h[i]));
    // printf("\n");
    // for(int i=0;i<10;i++)
    //     printf("%.2f, ",__bfloat162float(myX_SpMM_hbitmapv3[i]));
    // printf("\n");
      
    free(myX_SpMM_hbitmapv3);
    free(X_cublas_h);
    PrintPerformance("SpBitnet", mymilliseconds_SpMM_bitmapv3, mytflops_SpMM_bitmapv3, mytotalError_SpMM_bitmapv3);
    PrintPerformance("CuBlas_TC", milliseconds_cublas_tc, tflops_cublas_tc, 0.0);

    free(x_h);
    free(WT_h);
    free(reordered_WT_h);
    cudaFree(WT);
    cudaFree(myx);
    free(s_h);
    free(ws_h);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

