#include "onnxruntime_infer.hpp"
#include <format>
#include <numeric>
#include <filesystem>
#include "onnxruntime_cxx_api.h"
#include "inference_common.hpp"


void OnnxRuntimeInfer::CreateInferenceEngine(){
    env = nullptr;
    session_options = nullptr;
    for (auto x :Ort::GetAvailableProviders())
    {
        std::cout << x << std::endl;
    }
    std::cout << "<(*^_^*)> OnnxRuntime Inference Version: ["<< Ort::GetVersionString() << "] Created Successfully" << std::endl;

}

const DEVICE_TYPE OnnxRuntimeInfer::GetInferenceType() const{
    return DEVICE_TYPE::OnnxRuntime;
}

ResultData<std::string> OnnxRuntimeInfer::LoadModel(std::string file_path,
    std::vector<std::pair<std::string,std::vector<size_t>>>t_input_layouts,
    std::vector<std::pair<std::string,std::vector<size_t>>>t_output_layouts){
    m_input_layouts = t_input_layouts;
    m_output_layouts = t_output_layouts;
    ResultData<std::string> return_data;
    inference_common::TryFunction([&](){
        env = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,"onnxruntime_infer");
        session_options = std::make_unique<Ort::SessionOptions>();

        session_options->SetIntraOpNumThreads(1);
        //启用所有可用的优化
        // session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

        if (true) {
            // ============ 设置 TensorRT 提供器 ============
            OrtTensorRTProviderOptionsV2* trt_options = nullptr;
            Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&trt_options));
            
            std::vector<const char*> keys = {
                "device_id",
                "trt_engine_cache_enable",
                "trt_engine_cache_path",
                "trt_extra_plugin_lib_paths"
            };
            std::vector<const char*> values = {
                "0",
                "1",
                "D:\\workplace\\CPUorGPU_Infer\\build\\model\\model",
                "D:\\workplace\\WarpingSpade_SpeedTest\\plugins\\trt_plugins.dll"
            };
            Ort::ThrowOnError(Ort::GetApi().UpdateTensorRTProviderOptions(
                trt_options, 
                keys.data(), 
                values.data(), 
                keys.size()
            ));
            session_options->AppendExecutionProvider_TensorRT_V2(*trt_options);
    
            std::unordered_map<std::string, std::string> options;
            options["device_type"] = "CPU";
            options["disable_dynamic_shapes"] = "true";
            std::string config = R"({
            "CPU": {
                    "INFERENCE_PRECISION_HINT": "f32"
            }
            })";
            options["load_config"] = config;
            session_options->AppendExecutionProvider_OpenVINO_V2(options);
        }
        session_options->SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_DISABLE_ALL);
        return_data.result_info = file_path;
        std::cout << "<(*^_^*)> OnnxRuntime LoadModel(...) Successfully" << std::endl;
    },return_data);
        if(return_data.error_message.empty()) {
        return_data.result_state = true;
    }
    return return_data;
}

ResultData<bool> OnnxRuntimeInfer::CreateEngine(std::string& engine_path){
    ResultData<bool> return_data;
    inference_common::TryFunction([&](){
        std::wstring w_engine_path = std::filesystem::path(std::u8string(engine_path.begin(),engine_path.end())).wstring();
        const ORTCHAR_T* model_path = w_engine_path.c_str();  // wide_path生命周期持续到函数结束

        session = std::make_unique<Ort::Session>(*env, model_path, *session_options);
        if(!session){
            return_data.error_message = "<(E`_`E)> OnnxRuntime Session created failed...";
            return;
        }
        
        
        //输入
        std::vector<std::string> input_node_names = session->GetInputNames();
        MY_ASSERT(input_node_names.size() == m_input_layouts.size(),"input_layouts size != input_node_names size");

        //输出
        std::vector<std::string> output_node_names = session->GetOutputNames();
        MY_ASSERT(output_node_names.size() == m_output_layouts.size(),"output_layouts size != output_node_names size");

        mem_info = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, 
                OrtMemType::OrtMemTypeDefault
            )
        );
        if(!mem_info){
            return_data.error_message = "<(E`_`E)> OnnxRuntime mem_info created failed...";
            return;
        }
    },return_data);
    if(return_data.error_message.empty()) {
        return_data.result_state = true;
    }

    return return_data;
}


bool OnnxRuntimeInfer::Infer(const std::vector<float*> &datas,std::vector<std::vector<float>> &output_datas){
    ResultData<bool> return_data;
    inference_common::TryFunction<bool>([&](){
        if(!session){
            return_data.error_message = "<(E`_`E)> Please CreateEngine() before Infer() ...";
        }
        MY_ASSERT(datas.size() == m_input_layouts.size(), "datas num != Input Layer num");

        std::vector<Ort::Value> input_tensors;
        for (int input_idx = 0; input_idx < m_input_layouts.size(); input_idx++) {
            const std::vector<size_t>& input_layout = m_input_layouts[input_idx].second;
            size_t size_num = std::accumulate(input_layout.begin(), input_layout.end(), 1, std::multiplies<size_t>());
            
            //使用m_input_layouts的形状
            std::vector<int64_t> input_dims;
            std::cout << "<(*^_^*)> Model input "<< std::to_string(input_idx) <<" Shape: ";
            for(const size_t& dim:input_layout){
                std::cout << dim <<  " x ";
                input_dims.emplace_back(static_cast<int64_t>(dim));
            }
            std::cout << std::endl;
            
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                *mem_info, datas[input_idx], size_num,
                input_dims.data(), input_dims.size())
            );
        }
        
        std::vector<Ort::Value> output_tensors;
        output_datas.resize(m_output_layouts.size());
        for (int output_idx = 0; output_idx < m_output_layouts.size(); output_idx++) {
            const std::vector<size_t>& output_layout = m_output_layouts[output_idx].second;
            size_t size_num = std::accumulate(output_layout.begin(), output_layout.end(), 1, std::multiplies<size_t>());

            //使用m_output_layouts的形状
            std::vector<int64_t> output_dims;
            std::cout << "<(*^_^*)> Model output "<< std::to_string(output_idx) <<" Shape: ";
            for(const size_t& dim:output_layout){
                std::cout << dim <<  " x ";
                output_dims.emplace_back(static_cast<int64_t>(dim));
            }
            std::cout << std::endl;
            //给输出data分配空间
            output_datas[output_idx].resize(size_num);
            output_tensors.push_back(Ort::Value::CreateTensor<float>(
                *mem_info, output_datas[output_idx].data(), size_num,
                output_dims.data(), output_dims.size())
            );
        }
        
        std::vector<std::string> input_node_names_str = session->GetInputNames();
        std::vector<std::string> output_node_names_str = session->GetOutputNames();
        std::vector<const char*> input_node_names;
        std::vector<const char*> output_node_names;
        for (int i = 0; i < input_node_names_str.size(); i++)
        {
            input_node_names.push_back(input_node_names_str[i].c_str());
        }
        for (int i = 0; i < output_node_names_str.size(); i++)
        {
            output_node_names.push_back(output_node_names_str[i].c_str());
        }
        
        session->Run(Ort::RunOptions{nullptr}, 
            input_node_names.data(),
            input_tensors.data(), 
            input_tensors.size(), 
            output_node_names.data(),
            output_tensors.data(),
            output_tensors.size());
    },return_data);
    if(return_data.error_message.empty())
        return true;
    return false;

}

void OnnxRuntimeInfer::ReleaseInferenceEngine(){
    if (mem_info) {
        mem_info.reset();
    }
    if (session) {
        session.reset();
    }
    
    // session_options 不依赖其他对象
    if (session_options) {
        session_options.reset();
    }
    if (env) {
        env.reset();
    }
}
