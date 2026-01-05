#include "TensorRTInfer.hpp"
#include "InferenceCommon.hpp"
#include "CudaFun.hpp"
#include <NvOnnxParser.h>
#include <numeric>
#include <fstream>
#include <filesystem>
#include <format>

void TensorRTInfer::CreateInferenceEngine(){
    runtime = nullptr;
    engine = nullptr;
    context = nullptr;
    std::cout << "<(*^_^*)> TensorRT Inference Created Successfully" << std::endl;
}
const DEVICE_TYPE TensorRTInfer::GetInferenceType() const{
    return DEVICE_TYPE::TensorRT;
};

ResultData<std::list<std::string>> TensorRTInfer::GetInputNames(){
    ResultData<std::list<std::string>> return_data;
    if(!engine){
        return_data.error_message = "<(E`_`E)> engine has not been cteated...";
    }else{
        for (size_t i_idx = 0; i_idx < m_input_layouts.size(); i_idx++)
        {
            const char* input_name = engine->getIOTensorName(i_idx);
            return_data.result_info.push_back(input_name);
        }
        return_data.result_state = true;
    }
    return return_data;
}
ResultData<std::list<std::string>> TensorRTInfer::GetOutputNames(){
    ResultData<std::list<std::string>> return_data;
        if(!engine){
        return_data.error_message = "<(E`_`E)> engine has not been cteated...";
    }else{
        for (size_t i_idx = 0; i_idx < m_output_layouts.size(); i_idx++)
        {
            const char* input_name = engine->getIOTensorName(m_input_layouts.size() + i_idx);
            return_data.result_info.push_back(input_name);
        }
        return_data.result_state = true;
    }
    return return_data;
}

bool TensorRTInfer::convertONNXToTensorRT(
    const std::string& onnx_path, const std::string& trt_engine_path, bool fp16_flag) {
    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    MY_ASSERT(builder,"Convert ONNX to TensorRT : Create builder in error..." );

    // 显式 1 batch
    const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(explicit_batch));
    MY_ASSERT(network,"Convert ONNX to TensorRT : Create network in error...");

    // 解析ONNX文件
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    MY_ASSERT(parser,"Convert ONNX to TensorRT : Create onnx parser in error...");

    std::cout << "<(*^_^*)> builder |  network | parser is ready. Start building the engine." << std::endl;
    std::string u8p = std::filesystem::u8path(onnx_path.c_str()).string();
    if (!parser->parseFromFile(u8p.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "<(E`_`E)> Convert ONNX to TensorRT : Can't parser the ONNX file" << onnx_path.c_str() << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "    " << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }

    // 创建构建配置
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    MY_ASSERT(config,"Convert ONNX to TensorRT : Create builder config in error...");
    // 查看是否要设置FP16
    if (fp16_flag && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "    Set FP16 model..." << std::endl;
    }
    else if (fp16_flag) {
        std::cout << "    Your GPU don't support FP16 model..." << std::endl;
    }
    //创建配置项
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    MY_ASSERT(profile,"Convert ONNX to TensorRT : Create profile in error...");
    auto shapes = m_input_layouts;
    MY_ASSERT(network->getNbInputs() == shapes.size(),"network NbInputs!=shapes.size()");
    // 获取输入名称
    for (size_t i_idx = 0; i_idx < network->getNbInputs(); i_idx++)
    {
        auto shape = shapes[i_idx].second;
        auto input_name = network->getInput(i_idx)->getName();
        nvinfer1::Dims min_dims = network->getInput(i_idx)->getDimensions();
        nvinfer1::Dims opt_dims = network->getInput(i_idx)->getDimensions();
        nvinfer1::Dims max_dims = network->getInput(i_idx)->getDimensions();
        std::cout << "<(*^_^*)> Model input "<< std::to_string(i_idx) <<" Shape: ";
        for (size_t i = 0; i < opt_dims.nbDims; i++)
        {
            std::cout << opt_dims.d[i] << (i != opt_dims.nbDims - 1 ? " x " : "");
        }
        std::cout << "   VS    Your input Shape: ";
        for (size_t i = 0; i < shape.size(); i++)
        {
            std::cout << shape[i] << (i != shape.size() - 1 ? " x " : "");
        }
        std::cout << "   LayOut:" << shapes[i_idx].first << std::endl;

        for (int i = 0; i < min_dims.nbDims; ++i) {
            if (i < static_cast<int>(shape.size())) {
                min_dims.d[i] = shape[i];
                opt_dims.d[i] = shape[i];
                max_dims.d[i] = shape[i];
            }
        }
        // 设置动态形状范围
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, min_dims);
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, opt_dims);
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, max_dims);
        config->addOptimizationProfile(profile);
    }
    for (size_t o_idx = 0; o_idx < network->getNbOutputs(); o_idx++)
    {
        nvinfer1::Dims output_dims = network->getOutput(o_idx)->getDimensions();
        std::cout << "<(*^_^*)> Model output "<< std::to_string(o_idx) <<" Shape: ";
        for (size_t i = 0; i < output_dims.nbDims; i++)
        {
            std::cout << output_dims.d[i] << (i != output_dims.nbDims - 1 ? " x " : "");
        }
        std::cout << std::endl;
    }

    //最高水平优化
    config->setBuilderOptimizationLevel(5);

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
    if(file_path.empty()){
        return_data.error_message = "<(E`_`E)> Model path is empty, please check your path.";
        return return_data;
    }
    InferenceCommon::TryFunction<std::string>([&](){
    m_input_layouts = t_input_layouts;
    m_output_layouts = t_output_layouts;

    if(file_path.find(".trt") == std::string::npos){
        std::string onnx_file_path = file_path;
        int idx = file_path.find_last_of('/');
        if (idx != std::string::npos){
            idx+=1;
            file_path = file_path.substr(idx,file_path.size()-idx);
        }
        int idx_1 = file_path.find_last_of('\\');
        if (idx_1 != std::string::npos){
            idx_1+=1;
            file_path = file_path.substr(idx_1,file_path.size()-idx_1);
        }
        std::string trt_engine_path = "./model/" + file_path.substr(0,file_path.find_last_of('.'))+ ".trt";
        return_data.result_info = trt_engine_path;
        std::cout << "    Save model path: " << trt_engine_path << std::endl;
        if (!InferenceCommon::IsFileExist(trt_engine_path.c_str())) {
            if(!convertONNXToTensorRT(onnx_file_path,trt_engine_path,false)){
                return_data.error_message = "<(E`_`E)> convertONNXToTensorRT() failed...";
            };
        }
    }
    },return_data);
    if(return_data.error_message.empty()){
        return_data.result_state = true;
    }
    std::cout << "<(*^_^*)> LoadModel(...) Successfully" << std::endl;
    return return_data;
    
}

std::vector<unsigned char> TensorRTInfer::loadEngineFile(const std::string& file_name)
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(file_name, std::ios::binary);
    if (!engine_file.is_open()) {
        MY_ASSERT(false, std::string("engine file :") + file_name + " can't open()...");
    }
    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char*>(engine_data.data()), length);
    return engine_data;
}

void TensorRTInfer::deserializeEngine(std::string& engine_name) {
    auto plan = loadEngineFile(engine_name);
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
    InferenceCommon::TryFunction<bool>([&](){
        //反序列化引擎文件
        deserializeEngine(engine_path);
        CUDA_CHECK(cudaStreamCreate(&stream));
        //输入输出头数量必须等于模型定义数量
        MY_ASSERT(m_input_layouts.size()+m_output_layouts.size() == engine->getNbIOTensors(),
                std::string("Input Layer num + Output Layer num != Model Layer num:") + std::to_string(engine->getNbIOTensors()));
            
        //准备 GPU 输入输出 Buffer数据
        gpu_input_buffers = new float*[m_input_layouts.size()];
        gpu_output_buffers = new float*[m_output_layouts.size()];
        for (size_t i_idx = 0; i_idx < m_input_layouts.size(); i_idx++)
        {
            
            int size_num = accumulate(m_input_layouts[i_idx].second.begin(), m_input_layouts[i_idx].second.end(), 1, std::multiplies<int>());
            // 分配 输入 GPU内存
            CUDA_CHECK(cudaMalloc((void**)&gpu_input_buffers[i_idx], size_num * sizeof(float)));
            //输入节点名称
            const char* input_name = engine->getIOTensorName(i_idx);
            //绑定输入Buffer
            context->setInputTensorAddress(input_name, gpu_input_buffers[i_idx]);
            //定义输入形状
            nvinfer1::Dims dims = engine->getTensorShape(input_name);
            MY_ASSERT(dims.nbDims == m_input_layouts[i_idx].second.size(),"Input node: " + input_name + ":Please check your input_layout dims");
            for (size_t i = 0; i < dims.nbDims; i++)
            {
                MY_ASSERT((dims.d[i]<=0 || dims.d[i] == m_input_layouts[i_idx].second[i]),"Model input size mismatch. Please delete the TRT engine file and rebuild.");
                dims.d[i] = m_input_layouts[i_idx].second[i];
            }
            context->setInputShape(input_name, dims);
            std::cout << std::format("    Input node [name:{}] has bind buffer [{}/{}].\n",input_name,i_idx+1,m_input_layouts.size());
            
        }
        for (size_t o_idx = 0; o_idx < m_output_layouts.size(); o_idx++)
        {
            //分配输出头GPU内存
            int size_num = accumulate(m_output_layouts[o_idx].second.begin(), m_output_layouts[o_idx].second.end(), 1, std::multiplies<int>());
            CUDA_CHECK(cudaMalloc((void**)&gpu_output_buffers[o_idx], size_num * sizeof(float)));
            int real_idx = m_input_layouts.size() + o_idx;
            //输出节点名称
            const char* output_name = engine->getIOTensorName(real_idx);
            context->setOutputTensorAddress(output_name, gpu_output_buffers[o_idx]);
            std::cout << std::format("    Output node [name:{}] has bind buffer [{}/{}].\n",output_name,o_idx+1,m_output_layouts.size());
        }
    },return_data);
    if(return_data.error_message.empty()){
        std::cout << "<(*^_^*)> CreateEngine(...) Successfully" << std::endl;
        return_data.result_state = true;
    }
    return return_data;
    
}

bool TensorRTInfer::Infer(const std::vector<std::vector<size_t>> &data_layouts,const std::vector<float*> &datas,std::vector<std::vector<float>> &output_datas){
    //图像数据拷贝进 GPU 输入内存块
    ResultData<std::vector<float*>> return_data;
    InferenceCommon::TryFunction<std::vector<float*>>([&](){
        MY_ASSERT(m_input_layouts.size() == datas.size(),"Please check the size() of your input data...");
        for (size_t i_idx = 0; i_idx < m_input_layouts.size(); i_idx++)
        {
            //计算每个 输入 头内存用量
            int size_num = accumulate(m_input_layouts[i_idx].second.begin(), m_input_layouts[i_idx].second.end(), 1, std::multiplies<int>());
            //CPU -> GPU
            CUDA_CHECK(cudaMemcpyAsync(gpu_input_buffers[i_idx], datas[i_idx], size_num * sizeof(float), cudaMemcpyHostToDevice, stream));
            std::cout << std::format("    GPU input buffer [{}/{}] is ready.\n",i_idx+1,m_input_layouts.size());
        }
        //执行推理
        if (!context->enqueueV3(stream)) {
            //推理失败
            return_data.error_message = "<(E`_`E)> Error: Inference run failed!";
            return;
        }
        for (size_t o_idx = 0; o_idx < m_output_layouts.size(); o_idx++)
        {
            //计算每个 输出 头内存用量
            int size_num = accumulate(m_output_layouts[o_idx].second.begin(), m_output_layouts[o_idx].second.end(), 1, std::multiplies<int>());
            // 分配CPU内存
            float* cpu_output_buffer= new float[size_num];
            //GPU -> CPU
            CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_output_buffers[o_idx], size_num * sizeof(float), cudaMemcpyDeviceToHost));
  
            std::vector<float> output_copy(size_num);
            std::copy_n(cpu_output_buffer, size_num, output_copy.data());
            output_datas.push_back(std::move(output_copy));

            std::cout << std::format("    GPU output buffer [{}/{}] is ready.\n",o_idx+1,m_output_layouts.size());
            return_data.result_info.push_back(cpu_output_buffer);
        }
        cudaStreamSynchronize(stream);
    },return_data);
    if(return_data.error_message.empty()){
        return_data.result_state = true;
    }
    std::cout << "<(*^_^*)> Infer(...) Successfully" << std::endl;
    return return_data.result_state;

}



void TensorRTInfer::ReleaseInferenceEngine(){
    std::cout << "<(*-_-*)> Releasing TensorRT inference engine..." << std::endl;
    for (size_t i = 0; i < m_input_layouts.size(); i++)
    {
        cudaFree(gpu_input_buffers[i]);
    }
    delete[] gpu_input_buffers;
    gpu_input_buffers = nullptr;

    for (size_t i = 0; i < m_output_layouts.size(); i++)
    {
        cudaFree(gpu_output_buffers[i]);
    }
    delete[] gpu_output_buffers;
    gpu_output_buffers = nullptr;
    context = nullptr;
    engine = nullptr;
    runtime = nullptr;
    
    
}

