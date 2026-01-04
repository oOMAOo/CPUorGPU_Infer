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
    std::vector<std::pair<std::string,std::vector<size_t>>> m_input_layouts;
    std::vector<std::pair<std::string,std::vector<size_t>>> m_output_layouts;
    bool convertONNXToTensorRT(const std::string& onnx_path, const std::string& trt_engine_path, bool fp16_flag);
    std::vector<unsigned char> loadEngineFile(const std::string& file_name);
    void deserializeEngine(std::string& engine_name);

    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;
    
    float** gpu_input_buffers;
    float** gpu_output_buffers;
public:
    TensorRTInfer(){};
    void CreateInferenceEngine() override;
    const _device_type GetInferenceType() const override;
    ResultData<std::list<std::string>> GetInputNames() override;
    ResultData<std::list<std::string>> GetOutputNames() override;
    ResultData<std::string> LoadModel(
        std::string file_path,
        std::vector<std::pair<std::string,std::vector<size_t>>>t_input_layouts,
        std::vector<std::pair<std::string,std::vector<size_t>>>t_output_layouts) override;
    ResultData<bool> CreateEngine(std::string& engine_path) override;
    ResultData<std::vector<float*>> Infer(std::vector<std::vector<size_t>>data_layout,std::vector<float*> data) override;
    void ReleaseInferenceEngine()override;
    ~TensorRTInfer()override{ReleaseInferenceEngine();}
};


