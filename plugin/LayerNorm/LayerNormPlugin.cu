#include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    LayerNormPluginCreator::fc_ {};
std::vector<PluginField> LayerNormPluginCreator::attr_;

template<typename T, int n>
__global__ void layerNormKernel(T *pInput, T *pInput1, T *pInput2, T *pOutput)
{
    const int tx = threadIdx.x, index = blockIdx.x * n + threadIdx.x;

    float _x = (float)pInput[index];

    __shared__ float mean_shared, var_shared;

    typedef cub::BlockReduce<float, n>               BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    float &                                          ref0 = _x;
    float                                            sum  = BlockReduce(temp).Sum(ref0);
    //__syncthreads();
    if (tx == 0)
        mean_shared = sum / (float)n;
    __syncthreads();

    float  moment = _x - mean_shared, moment2 = moment * moment;
    float &ref1 = moment2;
    float  var  = BlockReduce(temp).Sum(ref1);
    //__syncthreads();
    if (tx == 0)
        var_shared = var / (float)n;
    __syncthreads();

    pOutput[index] = (T)(moment * (float)rsqrtf(var_shared + (float)1e-5) * (float)pInput1[tx] + (float)pInput2[tx]);
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1], nValuePerBlock = inputDesc[0].dims.d[2];

    switch (nValuePerBlock)
    {
    case 60:{
        switch (int(inputDesc[0].type)){
            case int(DataType::kFLOAT):{
                (layerNormKernel<float, 60>)<<<nBlock, nValuePerBlock, 0, stream>>>((float *)inputs[0], (float *)inputs[1], (float *)inputs[2], (float *)outputs[0]);
                break;
            }
            case int(DataType::kHALF):{
                (layerNormKernel<half, 60>)<<<nBlock, nValuePerBlock, 0, stream>>>((half *)inputs[0], (half *)inputs[1], (half *)inputs[2], (half *)outputs[0]);
                break;
            }
            default:
                printf("DataType not support!\n");
        }
        break;
    }
    case 180:{
        switch (int(inputDesc[0].type)){
            case int(DataType::kFLOAT):{
                (layerNormKernel<float, 180>)<<<nBlock, nValuePerBlock, 0, stream>>>((float *)inputs[0], (float *)inputs[1], (float *)inputs[2], (float *)outputs[0]);
                break;
            }
            case int(DataType::kHALF):{
                (layerNormKernel<half, 180>)<<<nBlock, nValuePerBlock, 0, stream>>>((half *)inputs[0], (half *)inputs[1], (half *)inputs[2], (half *)outputs[0]);
                break;
            }
            default:
                printf("DataType not support!\n");
        }
        break;
    }
    default: // shoulf NOT be here
        printf("[LayerNormPlugin::enqueue] nValuePerBlock = %d is not supported\n", nValuePerBlock);
        break;
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);