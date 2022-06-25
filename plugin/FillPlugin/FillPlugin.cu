/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
 #include "FillPlugin.h"

using namespace nvinfer1;

PluginFieldCollection FillPluginCreator::fc_{};
std::vector<PluginField> FillPluginCreator::attr_;

template<typename T>
__global__ void fillKernel(T *pInput, T *pOutput)
{
    const int tx = threadIdx.x, index = blockIdx.x * 256 + threadIdx.x;

    if (pInput[index] > (T)0.1 || pInput[index] < (T)-0.1){
        pOutput[index] = (T)-100.0;
    }
    else{
        pOutput[index] = 0.0;
    }
}

int32_t FillPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int nElement = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
    {
        nElement *= inputDesc[0].dims.d[i];
    }

    dim3 grid(CEIL_DIVIDE(nElement, 256), 1, 1), block(256, 1, 1);
    switch (int(inputDesc[0].type)){
        case int(DataType::kFLOAT):{
            fillKernel<float> <<<grid, block, 0, stream>>>((float *)inputs[0], (float *)outputs[0]);
            break;
        }
        case int(DataType::kHALF):{
            fillKernel<half> <<<grid, block, 0, stream>>>((half *)inputs[0], (half *)outputs[0]);
            break;
        }
        default:
            printf("DataType not support!\n");
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(FillPluginCreator);

