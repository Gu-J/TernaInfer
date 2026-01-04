/***************************************************************************
 * Copyright 2026 The TernaInfer Authors. All rights reserved.
 * Copyright 2023 The FLash-LLM Authors. All rights reserved.
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

// used for the reduction of result matrix if Split-K is used
// Reduction_Workspace:     (M_Global, N_Global, Split_K),  column major
// C:                       (M_Global, N_Global),           column major
// Each thread deals with 8 output elements, each elements is the sum of Split_K elements


#define ELEMENT_PER_THREADBLOCK 256

__global__ void SplitK_Reduction(__nv_bfloat16* C, int32_t* Reduction_Workspace, int M_Global, int N_Global, int Split_K,
                                 __nv_bfloat16* s, __nv_bfloat16* ws, int ws_num)
{
    // if(blockIdx.x==0&&threadIdx.x==0)for(int i=0;i<ws_num;i++)printf("ws[%d] = %f\n",i,__bfloat162float(ws[i]));
    int32_t* R_BasePTR_ThisBlock = Reduction_Workspace + ELEMENT_PER_THREADBLOCK * blockIdx.x;
    //
    float Results[HALF_PER_128B];
//
#pragma unroll
    for (int j = 0; j < HALF_PER_128B; j++)
        Results[j] = 0.0f;
    //
    for (int i = 0; i < Split_K; i++) {
#pragma unroll
        for (int j = 0; j < HALF_PER_128B; j++)
            Results[j] += static_cast<float>(R_BasePTR_ThisBlock[threadIdx.x * HALF_PER_128B + j]);
        R_BasePTR_ThisBlock += M_Global * N_Global;
    }

    int thread_begin_ind=ELEMENT_PER_THREADBLOCK * blockIdx.x + threadIdx.x * HALF_PER_128B;
    int row_ind=thread_begin_ind / M_Global;
    int col_ind=thread_begin_ind % M_Global;
    float s_el=__bfloat162float(s[row_ind]);
    float ws_el=__bfloat162float(ws[col_ind/(M_Global/ws_num)]);

#pragma unroll
    for (int j = 0; j < HALF_PER_128B; j++)
    {
        __nv_bfloat16 tmp=__float2bfloat16_rn(Results[j]/s_el*ws_el);
        C[ thread_begin_ind + j] = tmp;
    }
}


__global__ void SplitK_Reduction(__nv_bfloat16* C, int32_t* Reduction_Workspace, int M_Global, int N_Global, int Split_K)
{
    // return;
    int32_t* R_BasePTR_ThisBlock = Reduction_Workspace + ELEMENT_PER_THREADBLOCK * blockIdx.x;
    //
    float Results[HALF_PER_128B];
//
#pragma unroll
    for (int j = 0; j < HALF_PER_128B; j++)
        Results[j] = 0.0f;
    //
    for (int i = 0; i < Split_K; i++) {
#pragma unroll
        for (int j = 0; j < HALF_PER_128B; j++)
            Results[j] += static_cast<float>(R_BasePTR_ThisBlock[threadIdx.x * HALF_PER_128B + j]);
        R_BasePTR_ThisBlock += M_Global * N_Global;
    }

    int thread_begin_ind=ELEMENT_PER_THREADBLOCK * blockIdx.x + threadIdx.x * HALF_PER_128B;

#pragma unroll
    for (int j = 0; j < HALF_PER_128B; j++)
    {
        __nv_bfloat16 tmp=__float2bfloat16_rn(Results[j]);
        C[ thread_begin_ind + j] = tmp;
    }
}


__global__ void SplitK_Reduction(half* C, half* Reduction_Workspace, int M_Global, int N_Global, int Split_K)
{
    // return;
    half* C_BasePTR_ThisBlock = C + ELEMENT_PER_THREADBLOCK * blockIdx.x;
    half* R_BasePTR_ThisBlock = Reduction_Workspace + ELEMENT_PER_THREADBLOCK * blockIdx.x;
    //
    float Results[HALF_PER_128B];
//
#pragma unroll
    for (int j = 0; j < HALF_PER_128B; j++)
        Results[j] = 0.0f;
    //
    for (int i = 0; i < Split_K; i++) {
#pragma unroll
        for (int j = 0; j < HALF_PER_128B; j++)
            Results[j] += __half2float(R_BasePTR_ThisBlock[threadIdx.x * HALF_PER_128B + j]);
        R_BasePTR_ThisBlock += M_Global * N_Global;
    }
#pragma unroll
    for (int j = 0; j < HALF_PER_128B; j++)
        C_BasePTR_ThisBlock[threadIdx.x * HALF_PER_128B + j] = __float2half_rn(Results[j]);
}
