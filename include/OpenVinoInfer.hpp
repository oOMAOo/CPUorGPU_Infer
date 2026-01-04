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
    void CreateInferenceEngine() override;
    const _device_type GetInferenceType() const override;
    ResultData<std::list<std::string>> GetInputNames() override;
    ResultData<std::list<std::string>> GetOutputNames() override;
    ResultData<std::string> LoadModel(std::string file_path,
        std::vector<std::pair<std::string,std::vector<size_t>>>t_input_layouts,
        std::vector<std::pair<std::string,std::vector<size_t>>>t_output_layouts) override;
    ResultData<bool> CreateEngine(std::string& engine_path)override;
    ResultData<std::vector<float*>> Infer(std::vector<std::vector<size_t>>data_layout,std::vector<float*> data)override;
    void ReleaseInferenceEngine() override;
    ~OpenVinoInfer() override {ReleaseInferenceEngine();}
};


