#include "RollPlugin.h"

using namespace nvinfer1;

PluginFieldCollection    RollPluginCreator::fc_ {};
std::vector<PluginField> RollPluginCreator::attr_;

__global__ void RollVerticalKernel(float *pInput, int fea_b, int fea_h, int fea_w, int fea_c, int shift, float *pOutput)
{
    // B, H, W, C
    const int index = blockIdx.x * 256 + threadIdx.x;
    int h = index / (fea_w * fea_c) % fea_h;
    int b = index / (fea_h * fea_w * fea_c);
    int target_h = (fea_h + shift + h) % fea_h;
    int target_pos = b * (fea_h * fea_w * fea_c) + target_h * (fea_w * fea_c) + index % (fea_w * fea_c);
    pOutput[target_pos] = pInput[index];
}

__global__ void RollHorizontalKernel(float *pInput, int fea_b, int fea_h, int fea_w, int fea_c, int shift, float *pOutput)
{
    const int index = blockIdx.x * 256 + threadIdx.x;
    int w = index / fea_c % fea_w;
    int b_h = index / (fea_w * fea_c);
    int target_w = (fea_w + shift + w) % fea_w;
    int target_pos = b_h * (fea_w * fea_c) + target_w * fea_c + index % fea_c;
    pOutput[target_pos] = pInput[index];
}

int32_t RollPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
{
    WHERE_AM_I();
    int fea_b = inputDesc[0].dims.d[0], fea_h = inputDesc[0].dims.d[1], fea_w = inputDesc[0].dims.d[2], fea_c = inputDesc[0].dims.d[3];
    int nElement = 1;
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++)
    {
        nElement *= inputDesc[0].dims.d[i];
    }

    dim3 grid(CEIL_DIVIDE(nElement, 256), 1, 1), block(256, 1, 1); 
    if (m.direction_ == 0){
        RollVerticalKernel<<<grid, block, 0, stream>>>((float *)inputs[0], fea_b, fea_h, fea_w, fea_c, m.shift_, (float *)outputs[0]);
    }
    else if (m.direction_ == 1){
        RollHorizontalKernel<<<grid, block, 0, stream>>>((float *)inputs[0], fea_b, fea_h, fea_w, fea_c, m.shift_, (float *)outputs[0]);
    }
    else{
        printf("No implement!");
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(RollPluginCreator);