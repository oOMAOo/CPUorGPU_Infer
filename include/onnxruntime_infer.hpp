#pragma once
#include <inference_api.hpp>
#include <onnxruntime_cxx_api.h>
class OnnxRuntimeInfer : public InferenceApi
{
public:
    OnnxRuntimeInfer() = default;

    /// @brief 初始化当前推理配置
    void CreateInferenceEngine() override;

    /// @brief 获取当前推理引擎类型枚举
    /// @return _device_type::OpenVino
    const DEVICE_TYPE GetInferenceType() const override;
    

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
    /// @param data_layouts 输入数据布局 例:{{1,3,224,224},{1,3,224,224},{1,3,224,224}..}
    /// @param datas 输入数据
    /// @param output_datas 输入数据
    /// @return 输出数据
    bool Infer(const std::vector<float*> &datas,std::vector<std::vector<float>> &output_datas)override;

    /// @brief 释放资源
    void ReleaseInferenceEngine() override;

    ~OnnxRuntimeInfer() override {ReleaseInferenceEngine();}
private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::SessionOptions> session_options;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::MemoryInfo> mem_info;
};

