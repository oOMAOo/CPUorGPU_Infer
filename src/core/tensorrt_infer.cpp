#include "tensorrt_infer.hpp"
#include <numeric>
#include <fstream>
#include <filesystem>
#include <format>
#include <NvOnnxParser.h>
#include "inference_common.hpp"
#include "cuda_fun.hpp"
#include "addgridsampleplugin.hpp"
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};


void TensorRTInfer::CreateInferenceEngine(){
    runtime = nullptr;
    engine = nullptr;
    context = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::cout << "<(*^_^*)> TensorRT Inference Created Successfully" << std::endl;
}
const DEVICE_TYPE TensorRTInfer::GetInferenceType() const{
    return DEVICE_TYPE::TensorRT;
};

bool TensorRTInfer::ConvertONNXToTensorRT(
    const std::string& onnx_path, const std::string& trt_engine_path, bool fp16_flag) {
    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    MY_ASSERT(builder,"Convert ONNX to TensorRT : Create builder in error..." );
    // initLibNvInferPlugins(&logger, "");
    nvinfer1::plugin::AddGridSamplePluginCreator* addGridSampleCreator = new nvinfer1::plugin::AddGridSamplePluginCreator();
    auto* pluginRegistry = getPluginRegistry();
    pluginRegistry->registerCreator(*addGridSampleCreator, "MyGridSample");
    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicit_batch));
    MY_ASSERT(network,"Convert ONNX to TensorRT : Create network in error...");

    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    MY_ASSERT(parser,"Convert ONNX to TensorRT : Create onnx parser in error...");
    std::cout << "<(*^_^*)> builder |  network | parser is ready. Start building the engine." << std::endl;
    if (!parser->parseFromFile(std::filesystem::path(std::u8string(onnx_path.begin(),onnx_path.end())).string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "<(E`_`E)> Convert ONNX to TensorRT : Can't parser the ONNX file" << onnx_path << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "    " << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }

    // 创建构建配置
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

    //测试demo
    // {
    //     std::unique_ptr<nvinfer1::IHostMemory> ser_model(builder->buildSerializedNetwork(*network, *config));
    //     // 保存引擎文件
    //     std::ofstream ser_engine_file(trt_engine_path, std::ios::binary);
    //     MY_ASSERT(ser_engine_file.is_open(),std::string("can't create trt file: ") + trt_engine_path);
    //     std::cout << "    Save model path: " << trt_engine_path << std::endl;
    //     ser_engine_file.write(static_cast<const char*>(ser_model->data()), ser_model->size());
    //     ser_engine_file.close();

    //     std::cout << "<(*^_^*)> Inference model (TensorRT) has been loaded successfully." << std::endl;
    //     return false;
    // }

    MY_ASSERT(config,"Convert ONNX to TensorRT : Create builder config in error...");
    if (fp16_flag && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "    Set FP16 model..." << std::endl;
    }
    else if (fp16_flag) {
        std::cout << "   Your GPU don't support FP16 model..." << std::endl;
    }
    MY_ASSERT(network->getNbInputs() == m_input_layouts.size(),"network NbInputs!=shapes.size()");

    bool static_shape_flag = false;
    
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    MY_ASSERT(profile,"Convert ONNX to TensorRT : Create profile in error...");
    for (int input_idx = 0; input_idx < network->getNbInputs(); input_idx++)
    {
        auto shape = m_input_layouts[input_idx].second;
        auto input_name = network->getInput(input_idx)->getName();
        nvinfer1::Dims min_dims = network->getInput(input_idx)->getDimensions();
        nvinfer1::Dims opt_dims = min_dims;
        nvinfer1::Dims max_dims = min_dims;
        std::cout << "<(*^_^*)> Model input "<< std::to_string(input_idx) <<" Shape: ";
        for (int i = 0; i < opt_dims.nbDims; i++)
        {
            static_shape_flag &= (opt_dims.d[i] > 0);
            std::cout << opt_dims.d[i] << (i != opt_dims.nbDims - 1 ? " x " : "");
        }
        std::cout << "   VS    Your input Shape: ";
        for (int i = 0; i < shape.size(); i++)
        {
            std::cout << shape[i] << (i != shape.size() - 1 ? " x " : "");
        }
        std::cout << "   LayOut:" << m_input_layouts[input_idx].first << std::endl;
        //非动态形状直接跳过
        if(static_shape_flag) 
            continue;
        for (int i = 0; i < opt_dims.nbDims; ++i) {
            if (i < static_cast<int>(shape.size())) {
                min_dims.d[i] = shape[i];
                opt_dims.d[i] = shape[i];
                max_dims.d[i] = shape[i];
            }
        }
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, min_dims);
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, opt_dims);
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, max_dims);
        config->addOptimizationProfile(profile);
    }
    for (int output_idx = 0; output_idx < network->getNbOutputs(); output_idx++)
    {
        nvinfer1::Dims output_dims = network->getOutput(output_idx)->getDimensions();
        std::cout << "<(*^_^*)> Model output "<< std::to_string(output_idx) <<" Shape: ";
        for (int i = 0; i < output_dims.nbDims; i++)
        {
            std::cout << output_dims.d[i] << (i != output_dims.nbDims - 1 ? " x " : "");
        }
        std::cout << std::endl;
    }
    // 构建引擎
    std::unique_ptr<nvinfer1::IHostMemory> serialized_model(builder->buildSerializedNetwork(*network, *config));
    MY_ASSERT(serialized_model,"Can't serialized the Network.");

    // 保存引擎文件
    std::ofstream engine_file(trt_engine_path, std::ios::binary);
    MY_ASSERT(engine_file.is_open(),std::string("can't create trt file: ") + trt_engine_path);
    std::cout << "    Save model path: " << trt_engine_path << std::endl;
    engine_file.write(static_cast<const char*>(serialized_model->data()), serialized_model->size());
    engine_file.close();

    std::cout << "<(*^_^*)> Inference model (TensorRT) has been loaded successfully." << std::endl;
    return true;
}

ResultData<std::string> TensorRTInfer::LoadModel(std::string file_path,
    std::vector<std::pair<std::string,std::vector<size_t>>>t_input_layouts,
    std::vector<std::pair<std::string,std::vector<size_t>>>t_output_layouts){
    ResultData<std::string> return_data;
    inference_common::TryFunction<std::string>([&](){
        m_input_layouts = t_input_layouts;
        m_output_layouts = t_output_layouts;
        if (file_path.find(".trt") == std::string::npos) {
            std::string onnx_file_path = file_path;
            int idx = static_cast<int>(file_path.find_last_of('/'));
            if (idx != std::string::npos) {
                idx+=1;
                file_path = file_path.substr(idx,file_path.size()-idx);
            }
            int idx_1 = static_cast<int>(file_path.find_last_of('\\'));
            if (idx_1 != std::string::npos) {
                idx_1+=1;
                file_path = file_path.substr(idx_1,file_path.size()-idx_1);
            }
            std::string trt_engine_path = std::format("{}/{}.trt",MODELPATH,file_path.substr(0,file_path.find_last_of('.')));
            return_data.result_info = trt_engine_path;
            std::cout << "    Save model path: " << trt_engine_path << std::endl;
            if (!std::filesystem::exists(trt_engine_path)) {
                if (!ConvertONNXToTensorRT(onnx_file_path,trt_engine_path,false)){
                    return_data.error_message = "<(E`_`E)> convertONNXToTensorRT() failed...";
                };
            }
    }
    },return_data);
    if (return_data.error_message.empty()) {
        return_data.result_state = true;
        std::cout << "<(*^_^*)> LoadModel(...) Successfully" << std::endl;
    }
    
    return return_data;
    
}

std::vector<unsigned char> TensorRTInfer::LoadEngineFile(const std::string& file_name)
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(std::filesystem::path(std::u8string(file_name.begin(),file_name.end())), std::ios::binary);
    if (!engine_file.is_open()) {
        MY_ASSERT(false, std::string("engine file :") + file_name + " can't open()...");
    }
    engine_file.seekg(0, engine_file.end);
    int length = static_cast<int>(engine_file.tellg());
    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char*>(engine_data.data()), length);
    return engine_data;
}

void TensorRTInfer::DeserializeEngine(std::string& engine_name) {
    auto plan = LoadEngineFile(engine_name);
    Logger logger;
    runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    MY_ASSERT(runtime,"createInferRuntime error...");
    engine = std::unique_ptr<nvinfer1::ICudaEngine>((runtime)->deserializeCudaEngine(plan.data(), plan.size()));
    MY_ASSERT(engine,"deserializeCudaEngine error...");
    context = std::unique_ptr<nvinfer1::IExecutionContext>((engine)->createExecutionContext());
    MY_ASSERT(context,"createExecutionContext error...");
}

ResultData<bool> TensorRTInfer::CreateEngine(std::string& engine_path){
    ResultData<bool> return_data;
    inference_common::TryFunction<bool>([&](){
        DeserializeEngine(engine_path);
        MY_ASSERT(
            m_input_layouts.size()+m_output_layouts.size() == engine->getNbIOTensors(),
            std::string("Input Layer num + Output Layer num != Model Layer num:") + std::to_string(engine->getNbIOTensors())
        );
        //输入输出 GPU 缓存
        gpu_input_buffers = new float*[m_input_layouts.size()];
        gpu_output_buffers = new float*[m_output_layouts.size()];
        for (int input_idx = 0; input_idx < m_input_layouts.size(); input_idx++) {
            
            int size_num = accumulate(m_input_layouts[input_idx].second.begin(), m_input_layouts[input_idx].second.end(), 1, std::multiplies<int>());
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu_input_buffers[input_idx]), size_num * sizeof(float)));
            const char* input_name = engine->getIOTensorName(input_idx);
            bool state = context->setInputTensorAddress(input_name, gpu_input_buffers[input_idx]);
            if (state) {
                std::cout << std::format("    Input node [name:{}] has bind buffer [{}/{}].\n",input_name,input_idx+1,m_input_layouts.size());
            }else{
                return_data.error_message = std::format("<(E`_`E)> Input node [name:{}] can't bind gpu buffers.\n",input_name);
                return;
            }
            nvinfer1::Dims dims = engine->getTensorShape(input_name);
            MY_ASSERT(dims.nbDims == m_input_layouts[input_idx].second.size(),"Input node: " + input_name + ":Please check your input_layout dims");
            //默认不需要设置形状
            bool static_shape_flag = true;
            for (int i = 0; i < dims.nbDims; i++)
            {   
                static_shape_flag &= (dims.d[i] > 0);
                MY_ASSERT((dims.d[i]<=0 || dims.d[i] == m_input_layouts[input_idx].second[i]),"Model input size mismatch. Please delete the TRT engine file and rebuild.");
                dims.d[i] = m_input_layouts[input_idx].second[i];
            }
            //模型使用动态形状 需要设置形状
            if(!static_shape_flag){
                state = context->setInputShape(input_name, dims);
                if (state) {
                std::cout << std::format("    Input node [name:{}] has bind its shape.\n",input_name);
                }else{
                    return_data.error_message = std::format("<(E`_`E)> Input node [name:{}] can't bind its shape.\n",input_name);
                    return;
                }
            }

        }
        for (int output_idx = 0; output_idx < m_output_layouts.size(); output_idx++) {
            int size_num = accumulate(m_output_layouts[output_idx].second.begin(), m_output_layouts[output_idx].second.end(), 1, std::multiplies<int>());
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu_output_buffers[output_idx]), size_num * sizeof(float)));
            int real_idx = static_cast<int>(m_input_layouts.size()) + output_idx;
            const char* output_name = engine->getIOTensorName(real_idx);
            bool state = context->setOutputTensorAddress(output_name, gpu_output_buffers[output_idx]);
            if (state) {
                std::cout << std::format("    Output node [name:{}] has bind buffer [{}/{}].\n",output_name,output_idx+1,m_output_layouts.size());
            }else{
                return_data.error_message = std::format("<(E`_`E)> Output node [name:{}] can't bind gpu buffers.\n",output_name);
            }
            
        }
    },return_data);
    if (return_data.error_message.empty()) {
        std::cout << "<(*^_^*)> CreateEngine(...) Successfully" << std::endl;
        return_data.result_state = true;
    }
    return return_data;
    
}

bool TensorRTInfer::Infer(const std::vector<float*> &datas,std::vector<std::vector<float>> &output_datas){
    ResultData<bool> return_data;
    inference_common::TryFunction<bool>([&](){
        MY_ASSERT(datas.size() == m_input_layouts.size(),"Please check the size() of your input data...");
        for (int input_idx = 0; input_idx < m_input_layouts.size(); input_idx++)
        {
            //为GPU准备数据
            int size_num = accumulate(m_input_layouts[input_idx].second.begin(), m_input_layouts[input_idx].second.end(), 1, std::multiplies<int>());
            CUDA_CHECK(cudaMemcpyAsync(gpu_input_buffers[input_idx], datas[input_idx], size_num * sizeof(float), cudaMemcpyHostToDevice, stream));
            std::cout << std::format("    GPU input buffer [{}/{}] is ready.\n",input_idx+1,m_input_layouts.size());
        }
        
        if (!context->enqueueV3(stream)) {
            return_data.error_message = "<(E`_`E)> Error: Inference run failed!";
            return;
        }
        for (int output_idx = 0; output_idx < m_output_layouts.size(); output_idx++)
        {
            //整理推理结果
            int size_num = accumulate(m_output_layouts[output_idx].second.begin(), m_output_layouts[output_idx].second.end(), 1, std::multiplies<int>());
            float* cpu_output_buffer= new float[size_num];
            CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_output_buffers[output_idx], size_num * sizeof(float), cudaMemcpyDeviceToHost, stream));
            std::vector<float> output_copy(size_num);
            std::copy_n(cpu_output_buffer, size_num, output_copy.data());
            output_datas.push_back(std::move(output_copy));
            std::cout << std::format("    GPU output buffer [{}/{}] is ready.\n",output_idx+1,m_output_layouts.size());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    },return_data);
    if (return_data.error_message.empty()) {
        return_data.result_state = true;
        std::cout << "<(*^_^*)> Infer(...) Successfully" << std::endl;
    }
    
    return return_data.result_state;

}



void TensorRTInfer::ReleaseInferenceEngine(){
    std::cout << "<(*-_-*)> Releasing TensorRT inference engine..." << std::endl;
    for (int i = 0; i < m_input_layouts.size(); i++)
    {
        cudaFree(gpu_input_buffers[i]);
    }
    delete[] gpu_input_buffers;
    gpu_input_buffers = nullptr;

    for (int i = 0; i < m_output_layouts.size(); i++)
    {
        cudaFree(gpu_output_buffers[i]);
    }
    delete[] gpu_output_buffers;
    gpu_output_buffers = nullptr;
    context = nullptr;
    engine = nullptr;
    runtime = nullptr;
    
    
}

