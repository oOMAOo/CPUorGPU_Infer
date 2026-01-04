#pragma once
#include "InferenceStruct.h"
#include <list>
#include <map>

class InferenceApi
{
public:
    InferenceApi() = default;
    //初始化推理环境 根据系统环境返回 推理句柄
    virtual void CreateInferenceEngine() = 0;
    //获取当前推理的类型（0：OpenVino 1：TensorRT）
    virtual const _device_type GetInferenceType()  const = 0;
    virtual ResultData<std::list<std::string>> GetInputNames() = 0;
    virtual ResultData<std::list<std::string>> GetOutputNames() = 0;
    //加载模型
    virtual ResultData<std::string> LoadModel(std::string file_path,
        std::vector<std::pair<std::string,std::vector<size_t>>>input_layouts,
        std::vector<std::pair<std::string,std::vector<size_t>>>output_layouts) = 0;
    //创建识别引擎
    virtual ResultData<bool> CreateEngine(std::string& engine_path) = 0;
    //推理
    virtual ResultData<std::vector<float*>> Infer(std::vector<std::vector<size_t>>data_layout,std::vector<float*> data) = 0;


    //释放句柄 释放资源
    virtual void ReleaseInferenceEngine() = 0;
    virtual ~InferenceApi() = default;
};


