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
 
 #include "STReshapePlugin.h"

using namespace nvinfer1;

PluginFieldCollection STReshapePluginCreator::fc_{};
std::vector<PluginField> STReshapePluginCreator::attr_;

__global__ void STReshapeKernel(float *pInput, float *pOutput, int nElement)
{
    const int index = blockIdx.x * 256 + threadIdx.x;
    if (index > nElement){
        return;
    }
    pOutput[index] = pInput[index];
}

int32_t STReshapePlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int nElement = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
    {
        nElement *= inputDesc[0].dims.d[i];
    }

    dim3 grid(CEIL_DIVIDE(nElement, 256), 1, 1), block(256, 1, 1); 
    STReshapeKernel <<<grid, block, 0, stream>>>((float *)inputs[0], (float *)outputs[0], nElement);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(STReshapePluginCreator);

