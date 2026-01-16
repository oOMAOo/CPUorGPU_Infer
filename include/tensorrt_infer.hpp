#pragma once
#include <NvInfer.h>
#include "inference_api.hpp"

class TensorRTInfer : public InferenceApi
{
private:
    /// @brief 使用onnx文件解析到引擎文件中
    /// @param onnx_path onnx来源路径
    /// @param trt_engine_path 引擎保存路径
    /// @param fp16_flag 开启f16模式
    /// @return 
    bool ConvertONNXToTensorRT(const std::string& onnx_path, const std::string& trt_engine_path, bool fp16_flag);
    /// @brief 读取引擎文件二进制数据
    /// @return 数据列表
    std::vector<unsigned char> LoadEngineFile(const std::string& file_name);
    /// @brief 反序列化引擎
    /// @param engine_name 引擎路径
    void DeserializeEngine(std::string& engine_name);

    // 推理相关
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;
    // GPU 输入buffer池
    float** gpu_input_buffers;
    // GPU 输出buffer池
    float** gpu_output_buffers;
public:
    TensorRTInfer() = default;

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
    /// @return bool
    bool Infer(const std::vector<float*> &datas,std::vector<std::vector<float>> &output_datas)override;

    /// @brief 释放资源
    void ReleaseInferenceEngine() override;

    ~TensorRTInfer()override{ReleaseInferenceEngine();}
};


