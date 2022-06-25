#include "STReshapeAddPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    STReshapeAddPluginCreator::fc_ {};
std::vector<PluginField> STReshapeAddPluginCreator::attr_;

__global__ void STReshapeAddKernel(float *pInput, float *pAdd, int num_heads, int windows_pow, float *pOutput)
{
    // const int tx = threadIdx.x;
    // const int index = blockIdx.x * 256 + threadIdx.x;
    
    // int B_ = index / (num_heads * windows_pow);
    // int window_pos = index % windows_pow;
    // int addIndex = B_ * windows_pow + window_pos;

    // __shared__ float tempExponentShared[256];

    // tempExponentShared[tx] = pInput[index] + pAdd[addIndex];
    // __syncthreads();

    // __shared__ float maxNum[4];
    // __shared__ float expSum[4];
    // int flag = tx % 64;
    // if (flag == 0){
    //     maxNum[flag] = tempExponentShared[tx];
    //     for (int i = tx; i < (tx / 64 + 1) * 64; ++i)
    //     {
    //         if (tempExponentShared[i] > maxNum[flag]){
    //             maxNum[flag] = tempExponentShared[i];
    //         }
    //     }
    //     expSum[flag] = 0;
    //     for (int i = tx; i < (tx / 64 + 1) * 64; ++i){
    //         expSum[flag] += __expf(tempExponentShared[i] - maxNum[flag]);
    //     }
    // }
    // __syncthreads();

    // pOutput[index] = __expf(tempExponentShared[tx] - maxNum[flag] - __logf(expSum[flag]));
    
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
    STReshapeAddKernel<<<grid, block, 0, stream>>>((float *)inputs[0], (float *)inputs[1], m.num_heads_, windows_pow, (float *)outputs[0]);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(STReshapeAddPluginCreator);