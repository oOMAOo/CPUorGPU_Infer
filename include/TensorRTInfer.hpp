#pragma once
#include "InferenceApi.hpp"
#include <NvInfer.h>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

class TensorRTInfer : public InferenceApi
{
    
private:
    //输入布局
    std::vector<std::pair<std::string,std::vector<size_t>>> m_input_layouts;
    //输出布局
    std::vector<std::pair<std::string,std::vector<size_t>>> m_output_layouts;
    /// @brief 使用onnx文件解析到引擎文件中
    /// @param onnx_path onnx来源路径
    /// @param trt_engine_path 引擎保存路径
    /// @param fp16_flag 开启f16模式
    /// @return 
    bool convertONNXToTensorRT(const std::string& onnx_path, const std::string& trt_engine_path, bool fp16_flag);
    /// @brief 读取引擎文件二进制数据
    /// @return 数据列表
    std::vector<unsigned char> loadEngineFile(const std::string& file_name);
    /// @brief 反序列化引擎
    /// @param engine_name 引擎路径
    void deserializeEngine(std::string& engine_name);

    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;
    
    // GPU 输入buffer池
    float** gpu_input_buffers;
    // GPU 输出buffer池
    float** gpu_output_buffers;
public:
    TensorRTInfer(){};

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

    ~TensorRTInfer()override{ReleaseInferenceEngine();}
};


