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
static const char* PLUGIN_NAME{"Roll"};
static const char* PLUGIN_VERSION{"1"};
} // namespace

namespace nvinfer1
{

// +------- Plugin body ----------------------------------------------------------------------------
class RollPlugin: public IPluginV2DynamicExt
{
private:    
    std::string name_;
    std::string namespace_;
    struct{
        int shift_;
        int direction_;
    }m;

public:
    RollPlugin(const std::string& name, int shift, int direction) : name_(name)
    {
        m.shift_ = shift;
        m.direction_ = direction;
        WHERE_AM_I();
    }

    RollPlugin(const std::string& name, const void* data, size_t length) : name_(name)
    {
        WHERE_AM_I();
        memcpy(&m, data, sizeof(m));
    }
    
    RollPlugin() = delete;

    ~RollPlugin()
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
        return new RollPlugin(name_, m.shift_, m.direction_);
    }

    int getNbOutputs() const noexcept override
    {
        WHERE_AM_I();
        return 1;
    }

    DimsExprs getOutputDimensions(int32_t outputIndex, const DimsExprs* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override
    {
        WHERE_AM_I();
        return inputs[0];
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
            res = (inOut[pos].type == DataType::kFLOAT); break;
        case 1:
            res = (inOut[pos].type == DataType::kFLOAT); break;
        default:// should NOT be here
            res = false;
        }
        return res;
    }
    
    DataType getOutputDataType(int outputIndex, const DataType* inputTypes, int nbInputs) const noexcept override
    {
        WHERE_AM_I();
        return DataType::kFLOAT;
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
}; // class RollPlugin

class RollPluginCreator : public IPluginCreator
{
private:
    static PluginFieldCollection fc_;
    static std::vector<PluginField> attr_;
    std::string namespace_;

public:
    RollPluginCreator()
    {
        attr_.emplace_back(PluginField("shift", nullptr, PluginFieldType::kINT32, 1));
        attr_.emplace_back(PluginField("direction", nullptr, PluginFieldType::kINT32, 1));
        fc_.nbFields = attr_.size();
        fc_.fields   = attr_.data();
    }

    ~RollPluginCreator() {}

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override
    {
        WHERE_AM_I();
        int shift {-4};
        int direction {1};
        for (int i = 0; i < fc->nbFields; i++)
        {
            std::string field_name(fc->fields[i].name);
            if (field_name.compare("shift") == 0)
            {
                shift = *static_cast<const int *>(fc->fields[i].data);
            }
            if (field_name.compare("direction") == 0)
            {
                direction = *static_cast<const int *>(fc->fields[i].data);
            }
        }
        return new RollPlugin(name, shift, direction);
    }

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override
    {
        return new RollPlugin(name, serialData, serialLength);
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
}; // class RollPluginCreator

} // namespace nvinfer1

