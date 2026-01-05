#pragma once
#include "InferenceApi.hpp"
#include <openvino/openvino.hpp>

class OpenVinoInfer:public InferenceApi
{
private:
    ov::Core m_core;
    std::shared_ptr<ov::Model> m_model;
    ov::CompiledModel m_compiled_model;
    ov::InferRequest m_infer_request;
    std::string m_device;

    std::vector<std::pair<std::string,std::vector<size_t>>> m_input_layouts;
    std::vector<std::pair<std::string,std::vector<size_t>>> m_output_layouts;

public:
    OpenVinoInfer();

    /// @brief 初始化当前推理配置
    void CreateInferenceEngine() override;

    /// @brief 获取当前推理引擎类型枚举
    /// @return _device_type::OpenVino
    const _device_type GetInferenceType() const override;
    
    /// @brief 获取输入层名称列表
    ResultData<std::list<std::string>> GetInputNames() override;

    /// @brief 获取输出层名称列表
    ResultData<std::list<std::string>> GetOutputNames() override;

    /// @brief 使用onnx文件根据配置创建引擎
    /// @param layout  例 first:OpenVINO输入输出格式(NCHW\NHWC\NC?\NC...\N...C)    second:{1,3,224,224}(NCHW的话)
    /// @return 引擎文件路径
    ResultData<std::string> LoadModel(std::string file_path,
        std::vector<std::pair<std::string,std::vector<size_t>>>t_input_layouts,
        std::vector<std::pair<std::string,std::vector<size_t>>>t_output_layouts) override;
    
    /// @brief 使用引擎文件初始化识别引擎 设置输入输出配置(OpenVino引擎在LoadModel已初始化过)
    /// @param engine_path 引擎文件路径
    /// @return bool
    ResultData<bool> CreateEngine(std::string& engine_path)override;

    /// @brief 根据输入执行推理
    /// @param data_layout 输入数据布局 例:{{1,3,224,224},{1,3,224,224},{1,3,224,224}..}
    /// @param data 输入数据
    /// @return 输出数据
    ResultData<std::vector<float*>> Infer(std::vector<std::vector<size_t>>data_layout,std::vector<float*> data)override;

    /// @brief 释放资源
    void ReleaseInferenceEngine() override;
    
    ~OpenVinoInfer() override {ReleaseInferenceEngine();}
};


