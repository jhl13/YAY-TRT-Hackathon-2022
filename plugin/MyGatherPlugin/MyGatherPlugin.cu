#include "MyGatherPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    MyGatherPluginCreator::fc_ {};
std::vector<PluginField> MyGatherPluginCreator::attr_;

__global__ void MyGatherKernel(float *pInput, int nfea, float *pOutput0, float *pOutput1, float *pOutput2)
{
    const int index = blockIdx.x * 256 + threadIdx.x;
    int target_pos = index % nfea;
    int target_dim = index / nfea;
    if (target_dim == 0){
        pOutput0[target_pos] = pInput[index];
    }
    else if (target_dim == 1){
        pOutput1[target_pos] = pInput[index];
    }
    else{
        pOutput2[target_pos] = pInput[index];
    }
}

int32_t MyGatherPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nElement = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
    {
        nElement *= inputDesc[0].dims.d[i];
    }
    int nfea = 1;
    for (int i = 1; i < inputDesc[0].dims.nbDims; i++)
    {
        nfea *= inputDesc[0].dims.d[i];
    }

    dim3 grid(CEIL_DIVIDE(nElement, 256), 1, 1), block(256, 1, 1);
    MyGatherKernel<<<grid, block, 0, stream>>>((float *)inputs[0], nfea, (float *)outputs[0], (float *)outputs[1], (float *)outputs[2]);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(MyGatherPluginCreator);