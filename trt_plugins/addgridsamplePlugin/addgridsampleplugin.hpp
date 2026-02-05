#pragma once
#include <iostream>
#include <string>
#include <vector>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_fp16.h>
#include "addgridsample.cuh"
typedef uint16_t half_type;


namespace nvinfer1
{
    namespace plugin
    {
        class AddGridSamplePlugin : public IPluginV3,
                                    public IPluginV3OneBuildV2,
                                    public IPluginV3OneCore,
                                    public IPluginV3OneRuntime
        {
        public:
            AddGridSamplePlugin(const std::string name,
                                size_t input_channel,
                                size_t input_depth,
                                size_t input_height, 
                                size_t input_width,
                                size_t grid_depth,
                                size_t grid_height,
                                size_t grid_width,
                                bool aligncorners,
                                std::string interpolation_mode,
                                std::string padding_mode,
                                DataType data_type);
            AddGridSamplePlugin(const std::string name,
                                bool aligncorners,
                                std::string interpolation_mode,
                                std::string padding_mode);
            AddGridSamplePlugin() = delete;
            AddGridSamplePlugin(AddGridSamplePlugin const&) = default;
            ~AddGridSamplePlugin() override;
            /***************** IPluginV3 Methods *****************/ 
            IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;
            AddGridSamplePlugin* clone() noexcept override;



            /***************** IPluginV3OneBuild(V2) Methods *****************/
            //创建引擎和执行引擎期间
            int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, 
                                    DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
            //对输出的类型进行定义
            int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs, 
                                       DataType const* inputTypes, int32_t nbInputs) const noexcept override;
            //对输出的形状进行定义
            int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
                                    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;
            //验证节点的支持（线性布局）（Float32/Float16）（类型保持一致）
            bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, 
                                           int32_t nbInputs, int32_t nbOutputs) noexcept override;
            //输出维度
            int32_t getNbOutputs() const noexcept override;
            //存储中间变量要申请的额外空间
            size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                                    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

            /***************** IPluginV3OneCore Methods *****************/
            char const* getPluginName() const noexcept override;
            char const* getPluginVersion() const noexcept override;
            char const* getPluginNamespace() const noexcept override;

            /***************** IPluginV3OneRuntime Methods *****************/

            //可以当回调函数用，有时候需要对特定形状进行单独处理
            int32_t onShapeChange(PluginTensorDesc const* in, int32_t nbInputs, 
                                  PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
            //推理计算
            int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
                            void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
            //绑定上下文，在这里克隆模型，使本身的状态/数据能够复用，如果用到GPU资源记得释放和重新申请
            IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;
            //用onnx中数据序列化引擎的核心函数
            PluginFieldCollection const* getFieldsToSerialize() noexcept override;

            void setPluginNamespace(char const* pluginNamespace) noexcept;

        private:
            const int m_in_num = 2;
            const int m_out_num = 1;
            const std::string m_layername;

            bool m_initialized{false};

            // shape
            size_t m_batch;
            size_t m_input_channel, m_input_depth, m_input_width, m_input_height;
            size_t m_grid_depth, m_grid_width, m_grid_height;

            std::string m_namespace;

            //边角对齐
            int m_align_corners;
            //插值算法
            std::string m_mode;
            //padding类型
            std::string m_padding_mode;

            //数据类型
            nvinfer1::DataType m_data_type;
            // For Serialize() -> 参数数据
            std::vector<nvinfer1::PluginField> m_data_to_serialize;
            nvinfer1::PluginFieldCollection m_FC_to_serialize;
        };


        class AddGridSamplePluginCreator : public nvinfer1::IPluginCreatorV3One
        {
        public:
            AddGridSamplePluginCreator();
            ~AddGridSamplePluginCreator() override = default;
            IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;
            PluginFieldCollection const* getFieldNames() noexcept override;
            //插件名+版本
            char const* getPluginName() const noexcept override;
            char const* getPluginVersion() const noexcept override;

            char const* getPluginNamespace() const noexcept override;
            void setPluginNamespace(char const* t_namespace) noexcept;
        private:
            PluginFieldCollection m_field_collection;
            std::vector<PluginField> m_plugin_attributes;
            std::string m_namespace;
        };
        
        

        

    }
}