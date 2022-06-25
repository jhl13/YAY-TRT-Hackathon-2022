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
#include <NvInfer.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <string>
#include <vector>


// +------- Debug wrapper --------------------------------------------------------------------------
#if DEBUG
#define WHERE_AM_I() do {printf("[%s]: this=->%p\n",__func__,this);} while(0);
#else
#define WHERE_AM_I()
#endif // DEBUG

#define CEIL_DIVIDE(X, Y) (((X) + (Y)-1) / (Y))
#define CEIL_TO(X, Y)     (CEIL_DIVIDE(X, Y) * (Y))

// +------- Plguin ---------------------------------------------------------------------------------
namespace
{
static const char* PLUGIN_NAME{"STReshape"};
static const char* PLUGIN_VERSION{"1"};
} // namespace

namespace nvinfer1
{

// +------- Plugin body ----------------------------------------------------------------------------
class STReshapePlugin: public IPluginV2DynamicExt
{
private:    
    std::string name_;
    std::string namespace_;
    struct{
        int window_size_;
        int type_;
        int num_heads_;
    }m;
    

public:
    STReshapePlugin(const std::string& name, int window_size, int type, int num_heads) : name_(name)
    {
        m.window_size_ = window_size;
        m.type_        = type;
        m.num_heads_   = num_heads;
        WHERE_AM_I();
    }

    STReshapePlugin(const std::string& name, const void* data, size_t length) : name_(name)
    {
        WHERE_AM_I();
        memcpy(&m, data, sizeof(m));
    }
    
    STReshapePlugin() = delete;

    ~STReshapePlugin()
    {
        WHERE_AM_I();
    }

    size_t getSerializationSize() const noexcept override
    {
        WHERE_AM_I();
        return sizeof(m);
    }
    
    void serialize(void *buffer) const noexcept override
    {
        WHERE_AM_I();
        memcpy(buffer, &m, sizeof(m));
    }
  
    IPluginV2DynamicExt* clone() const noexcept override
    {
        WHERE_AM_I();
        return new STReshapePlugin(name_, m.window_size_, m.type_, m.num_heads_);
    }

    int getNbOutputs() const noexcept override
    {
        WHERE_AM_I();
        return 1;
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override
    {
        WHERE_AM_I();
        DimsExprs out;
        const auto* ws = exprBuilder.constant(m.window_size_);
        const auto* nh = exprBuilder.constant(m.num_heads_);
        const auto* wspow2 = exprBuilder.constant(m.window_size_*m.window_size_);
        
        // B, H, W, C without shift
        if (m.type_ == 0){
            out.nbDims = 6;
            out.d[0]   = inputs[1].d[0];
            out.d[1]   = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[1].d[2], *ws);
            out.d[2]   = exprBuilder.constant(m.window_size_);
            out.d[3]   = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[1].d[3], *ws);
            out.d[4]   = exprBuilder.constant(m.window_size_);
            out.d[5]   = inputs[1].d[1];
        }
        // B, H*W, C without shift
        else if (m.type_ == 1){
            out.nbDims = 3;
            out.d[0]   = inputs[1].d[0];
            out.d[1]   = inputs[1].d[1];
            out.d[2]   = inputs[1].d[2];
        }
        // -1, window_size*window_size, C
        if (m.type_ == 2){
            out.nbDims = 3;
            const auto* tmp = exprBuilder.operation(DimensionOperation::kPROD, *inputs[1].d[0], *inputs[1].d[1]);
            out.d[0]   = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *tmp, *wspow2);
            out.d[1]   = exprBuilder.constant(m.window_size_*m.window_size_);
            out.d[2]   = inputs[1].d[2];
        }
        // B, H // window_szie, W // window_size, window_size, window_size, C
        else if (m.type_ == 3){
            out.nbDims = 6;
            out.d[0]   = inputs[1].d[0];
            out.d[1]   = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[1].d[2], *ws);
            out.d[2]   = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[1].d[3], *ws);
            out.d[3]   = exprBuilder.constant(m.window_size_);
            out.d[4]   = exprBuilder.constant(m.window_size_);
            out.d[5]   = inputs[1].d[1];
        }
        // B, C, H, W
        else if (m.type_ == 4){
            return inputs[1];
        }
        // -1, window_szie*window_szie, 3, num_heads, C // num_heads
        else if (m.type_ == 5){
            out.nbDims = 5;
            out.d[0]   = inputs[1].d[0];
            out.d[1]   = inputs[1].d[1];
            out.d[2]   = exprBuilder.constant(3);
            out.d[3]   = exprBuilder.constant(m.num_heads_);
            out.d[4]   = exprBuilder.operation(DimensionOperation::kFLOOR_DIV, *inputs[1].d[2], *nh);
        }
        // , window_szie*window_szie, C
        else if (m.type_ == 6){
            return inputs[1];
        }
        return out;
    }

    bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        WHERE_AM_I();
        if(inOut[pos].format != TensorFormat::kLINEAR)
        {
            return false;
        }

        bool res = false;
        switch(pos)
        {
        case 0:
            res = (inOut[pos].type == DataType::kFLOAT || inOut[pos].type == DataType::kHALF); break;
        case 1:
            res = (inOut[pos].type == inOut[0].type); break;
        case 2:
            res = (inOut[pos].type == inOut[0].type); break;
        default:// should NOT be here
            res = false;
        }
        return res;
    }
    
    DataType getOutputDataType(int outputIndex, const DataType* inputTypes, int nbInputs) const noexcept override
    {
        WHERE_AM_I();
        return inputTypes[0];
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int32_t nbInputs,const DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override
    {
        WHERE_AM_I();
    }

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int32_t nbInputs, const PluginTensorDesc* outputs,int32_t nbOutputs) const noexcept override
    {
        WHERE_AM_I();
        return 0;
    }

    void setPluginNamespace(const char* szNamespace) noexcept override
    {
        WHERE_AM_I();
        namespace_ = szNamespace;
    }
    const char* getPluginNamespace() const noexcept override
    {
        WHERE_AM_I();
        return namespace_.c_str();
    }
    const char* getPluginType() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_NAME;
    }
    const char* getPluginVersion() const noexcept override
    {
        WHERE_AM_I();
        return PLUGIN_VERSION;
    }
    int initialize() noexcept override
    {
        WHERE_AM_I();
        return 0;
    }
    void terminate() noexcept override
    {
        WHERE_AM_I();
        return;
    }

    void destroy() noexcept override
    {
        WHERE_AM_I();
    }
    
    int32_t enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
}; // class STReshapePlugin

class STReshapePluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection fc_;
    static std::vector<PluginField> attr_;
    std::string namespace_;

public:
    STReshapePluginCreator()
    {
        attr_.emplace_back(PluginField("window_size", nullptr, PluginFieldType::kINT32, 1));
        attr_.emplace_back(PluginField("type", nullptr, PluginFieldType::kINT32, 1));
        attr_.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
        fc_.nbFields = attr_.size();
        fc_.fields   = attr_.data();
    }

    ~STReshapePluginCreator() {}

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
    {
        WHERE_AM_I();
        int window_size {8};
        int type {0};
        int num_heads {6};
        for (int i = 0; i < fc->nbFields; i++)
        {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare("window_size") == 0)
            {
                window_size = *static_cast<const int *>(fc->fields[i].data);
            }
            if (field_name.compare("type") == 0)
            {
                type = *static_cast<const int *>(fc->fields[i].data);
            }
            if (field_name.compare("num_heads") == 0)
            {
                num_heads = *static_cast<const int *>(fc->fields[i].data);
            }
        }
        return new STReshapePlugin(name, window_size, type, num_heads);
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
    {
        return new STReshapePlugin(name, serialData, serialLength);
    }

    void setPluginNamespace(const char* szNamespace) noexcept override
    {
        namespace_ = szNamespace;
    }

    const char* getPluginNamespace() const noexcept override
    {
        return namespace_.c_str();
    }

    const char* getPluginName() const noexcept override
    {
        return PLUGIN_NAME;
    }

    const char* getPluginVersion() const noexcept override
    {
        return PLUGIN_VERSION;
    }

    const PluginFieldCollection* getFieldNames() noexcept override
    {
        return &fc_;
    }
}; // class STReshapePluginCreator

} // namespace nvinfer1

