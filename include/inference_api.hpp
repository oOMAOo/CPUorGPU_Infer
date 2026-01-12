#pragma once
#include "inference_struct.h"
#include <list>
#include <map>

class InferenceApi
{
public:
    InferenceApi() = default;
    virtual void CreateInferenceEngine() = 0;
    //（0：OpenVino 1：TensorRT）
    virtual const DEVICE_TYPE GetInferenceType()  const = 0;

    virtual ResultData<std::string> LoadModel(std::string file_path,
        std::vector<std::pair<std::string,std::vector<size_t>>>input_layouts,
        std::vector<std::pair<std::string,std::vector<size_t>>>output_layouts) = 0;

    virtual ResultData<bool> CreateEngine(std::string& engine_path) = 0;

    virtual bool Infer(const std::vector<float*> &datas,std::vector<std::vector<float>> &output_datas) = 0;

    virtual void ReleaseInferenceEngine() = 0;
    virtual ~InferenceApi() = default;
protected:
    //输入布局
    std::vector<std::pair<std::string,std::vector<size_t>>> m_input_layouts;
    //输出布局
    std::vector<std::pair<std::string,std::vector<size_t>>> m_output_layouts;
};


