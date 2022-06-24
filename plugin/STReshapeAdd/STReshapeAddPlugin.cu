#include "STReshapeAddPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    STReshapeAddPluginCreator::fc_ {};
std::vector<PluginField> STReshapeAddPluginCreator::attr_;

__global__ void STReshapeAddKernel(float *pInput, float *pAdd, int num_heads, int windows_pow, float *pOutput)
{
    const int index = blockIdx.x * 256 + threadIdx.x;
    int B_ = index / (num_heads * windows_pow);
    int window_pos = index % windows_pow;
    int addIndex = B_ * windows_pow + window_pos;

    pOutput[index] = pInput[index] + pAdd[addIndex];
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
    // printf("input0 dim1: %d, dim2: %d, dim3: %d, dim4: %d\n", inputDesc[0].dims.d[0], inputDesc[0].dims.d[1], inputDesc[0].dims.d[2], inputDesc[0].dims.d[3], inputDesc[0].dims.d[4]);
    // printf("input1 dim1: %d, dim2: %d, dim3: %d, dim4: %d\n", inputDesc[1].dims.d[0], inputDesc[1].dims.d[1], inputDesc[1].dims.d[2], inputDesc[1].dims.d[3], inputDesc[1].dims.d[4]);
    // printf("windows_pow: %d, num_heads: %d \n", windows_pow, m.num_heads_);
    STReshapeAddKernel<<<grid, block, 0, stream>>>((float *)inputs[0], (float *)inputs[1], m.num_heads_, windows_pow, (float *)outputs[0]);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(STReshapeAddPluginCreator);