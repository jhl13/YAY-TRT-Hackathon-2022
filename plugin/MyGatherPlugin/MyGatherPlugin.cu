#include "MyGatherPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    MyGatherPluginCreator::fc_ {};
std::vector<PluginField> MyGatherPluginCreator::attr_;

template<typename T>
__global__ void MyGatherKernel(T *pInput, int nfea, T *pOutput0, T *pOutput1, T *pOutput2, T B)
{
    const int index = blockIdx.x * 256 + threadIdx.x;
    int target_pos = index % nfea;
    int target_dim = index / nfea;
    if (target_dim == 0){
        pOutput0[target_pos] = pInput[index] * (T)0.31622776601683794;
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
    switch (int(inputDesc[0].type)){
        case int(DataType::kFLOAT):{
            MyGatherKernel<float><<<grid, block, 0, stream>>>((float *)inputs[0], nfea, (float *)outputs[0], (float *)outputs[1], (float *)outputs[2], B_);
            break;
        }
        case int(DataType::kHALF):{
            MyGatherKernel<half><<<grid, block, 0, stream>>>((half *)inputs[0], nfea, (half *)outputs[0], (half *)outputs[1], (half *)outputs[2], (half)B_);
            break;
        }
        default:
            printf("DataType not support!\n");
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(MyGatherPluginCreator);