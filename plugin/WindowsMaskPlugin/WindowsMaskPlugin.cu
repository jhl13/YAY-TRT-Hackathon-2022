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
 
 #include "WindowsMaskPlugin.h"

using namespace nvinfer1;

PluginFieldCollection WindowsMaskPluginCreator::fc_{};
std::vector<PluginField> WindowsMaskPluginCreator::attr_;

__global__ void windowsMaskKernel(float *pInput, int *shape, float *pOutput, int nElement)
{
    const int index = blockIdx.x * 64 + threadIdx.x;
    if (index > nElement){
        return;
    }
    int H = shape[1];
    int W = shape[2];

    int h = index / W;
    int w = index % W;
    
    if (h < (H - 8) && w < (W - 8)){
        pOutput[index] = 0;
    }
    else if (h < (H - 8) && w < (W - 4) && w >= (W - 8)){
        pOutput[index] = 1;
    }
    else if (h < (H - 8) && w >= (W - 4)){
        pOutput[index] = 2;
    }
    else if (h < (H - 4) && h >= (H - 8) && w < (W - 8)){
        pOutput[index] = 3;
    }
    else if (h < (H - 4) && h >= (H - 8) && w < (W - 4) && w >= (W - 8)){
        pOutput[index] = 4;
    }
    else if (h < (H - 4) && h >= (H - 8) && w >= (W - 4)){
        pOutput[index] = 5;
    }
    else if (h >= (H - 4) && w < (W - 8)){
        pOutput[index] = 6;
    }
    else if (h >= (H - 4) && w < (W - 4) && w >= (W - 8)){
        pOutput[index] = 7;
    }
    else if (h >= (H - 4) && w >= (W - 4)){
        pOutput[index] = 8;
    }
}

int32_t WindowsMaskPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    int nElement = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
    {
        nElement *= inputDesc[0].dims.d[i];
    }

    dim3 grid(CEIL_DIVIDE(nElement, 64), 1, 1), block(64, 1, 1); 
    windowsMaskKernel <<<grid, block, 0, stream>>>((float *)inputs[0], (int *)inputs[1], (float *)outputs[0], nElement);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(WindowsMaskPluginCreator);

