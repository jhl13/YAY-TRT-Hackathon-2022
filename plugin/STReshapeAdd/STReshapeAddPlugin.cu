#include "STReshapeAddPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    STReshapeAddPluginCreator::fc_ {};
std::vector<PluginField> STReshapeAddPluginCreator::attr_;

template<typename T>
__global__ void STReshapeAddKernel(T *pInput, T *pAdd1, T *pAdd2, int num_heads, int windows_pow, T *pOutput)
{    
    const int index = blockIdx.x * 256 + threadIdx.x;
    int hw = num_heads * windows_pow;
    int addIndex1 = index % (num_heads * windows_pow);
    int B_ = index / (num_heads * windows_pow);
    int window_pos = index % windows_pow;
    int addIndex2 = B_ * windows_pow + window_pos;

    pOutput[index] = pInput[index] + pAdd1[addIndex1] + pAdd2[addIndex2];
}

int32_t STReshapeAddPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nElement = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
    {
        nElement *= inputDesc[0].dims.d[i];
    }

    dim3 grid(CEIL_DIVIDE(nElement, 256), 1, 1), block(256, 1, 1);
    int windows_pow = m.window_size_ * m.window_size_ * m.window_size_ * m.window_size_;
    
    switch (int(inputDesc[0].type)){
        case int(DataType::kFLOAT):{
            STReshapeAddKernel<float><<<grid, block, 0, stream>>>((float *)inputs[0], (float *)inputs[1], (float *)inputs[2], m.num_heads_, windows_pow, (float *)outputs[0]);
            break;
        }
        case int(DataType::kHALF):{
            STReshapeAddKernel<half><<<grid, block, 0, stream>>>((half *)inputs[0], (half *)inputs[1], (half *)inputs[2], m.num_heads_, windows_pow, (half *)outputs[0]);
            break;
        }
        default:
            printf("DataType not support!\n");
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(STReshapeAddPluginCreator);