
#include <direct.h>
#include <io.h>

#include <iostream>
#include <fstream>
#include <ctime>

#include <opencv2/opencv.hpp>

#include "cuda_fun.hpp"
#include "inference_api.hpp" 
#include "openvino_infer.hpp"
#include "tensorrt_infer.hpp"
#include "inference_common.hpp"

#define INPUT_W 1050
#define INPUT_H 1550
int main(){
    std::string input_img = "C:\\Users\\影擎星图\\Desktop\\nafnet_demo\\CUDA-13.1.png";
    std::string model_path = "C:\\Users\\影擎星图\\Desktop\\nafnet_demo\\nafnet_ep304_20251117.onnx";

    if(!std::filesystem::is_regular_file(std::filesystem::u8path(input_img)) || !std::filesystem::is_regular_file(std::filesystem::u8path(model_path))){
        std::cerr << "<(E`_`E)> Model path/Image path is invalid, please check your path." << std::endl;;
        return -1;
    }
    InferenceApi* api_tool = nullptr;

    //使用CUDA函数检测是否存在NVIDIA设备且计算能力需要是8.9（后续再补全）  目前仅做简单检测
    //TODO 增加显卡版本检测
    bool availableCUDA = inference_common::GetAvailableCUDA();
    std::unique_ptr<CudaFun> cuda_tool = nullptr;
    if(availableCUDA){
        api_tool = new TensorRTInfer();
        cuda_tool = std::make_unique<CudaFun>(INPUT_H,INPUT_W,3);
    }else{
        api_tool = new OpenVinoInfer();
    }

    // ==================== 1. 创建基础推理引擎 ====================
    api_tool->CreateInferenceEngine();

    
    using LayoutShape = std::pair<std::string,std::vector<size_t>>;
    std::vector<LayoutShape> input_layouts;
    input_layouts.emplace_back(LayoutShape{"NCHW", {1,3,INPUT_H,INPUT_W}});
    std::vector<LayoutShape> output_layouts;
    output_layouts.emplace_back(LayoutShape{"NCHW", {1,3,INPUT_H,INPUT_W}});
    if (!std::filesystem::exists(MODELPATH))
    {
        if(!std::filesystem::create_directory(MODELPATH)){
            std::cerr << std::format("(E`_`E)> create directory [{}] in error...",MODELPATH) << std::endl;
            return -1;
        }
    }

    // ==================== 2. 加载模型文件 ====================
    ResultData<std::string> load_state = api_tool->LoadModel(model_path,input_layouts,output_layouts);
    if(!load_state.result_state){
        return -1;
    }
    // ==================== 3. 加载引擎文件，创建识别引擎 ====================
    ResultData<bool> engine_state = api_tool->CreateEngine(load_state.result_info);
    if(!engine_state.result_state){
        return -1;
    }

    // ==================== 4. 准备输入数据 ====================
    std::vector<std::vector<float>> output_datas;
    auto infer_start = std::chrono::steady_clock::now();
    cv::Mat image = cv::imread(std::filesystem::u8path(input_img).string(), cv::IMREAD_COLOR_RGB);
    if (image.empty()) {
        std::cout << std::format("<(E`_`E)> Can't Open the image: {}\n", input_img);
        return -1;
    }
    std::cout << std::format("\n(I_I)>>>>> Input image size : {} x {}\n", INPUT_W, INPUT_H);
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0f/255.0f,cv::Size(INPUT_W,INPUT_H));
    std::vector<std::vector<size_t>>data_layout;
    data_layout.emplace_back(std::vector<size_t>{1,3,INPUT_H,INPUT_W});

    // ==================== 5. 推理 ====================
    bool output_state = api_tool->Infer(data_layout,{reinterpret_cast<float*>(blob.data)},output_datas);
    if(!output_state){
        return -1;
    }
    auto infer_end = std::chrono::steady_clock::now();
    double infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    std::cout << std::format("----------------- Infer image used time : {} ms -----------------\n", infer_time);

    // ==================== 6. 整理结果 ====================
    for (int i = 0; i < output_datas.size(); i++)
    {
        float* out_data = output_datas[i].data();
        auto tran_start = std::chrono::steady_clock::now();
        auto output_image_opt = cuda_tool ? cuda_tool->DatatoImage(&out_data,INPUT_H*INPUT_W*3,255.0f) : inference_common::DatatoImage(&out_data,INPUT_H,INPUT_W,3,255.0f);
        auto tran_end = std::chrono::steady_clock::now();
        double tran_time = std::chrono::duration<double, std::milli>(tran_end - tran_start).count();
        std::cout << std::format("----------------- transfor image used time : {} ms -----------------\n", tran_time);
        output_datas[i].clear();
        output_datas[i].shrink_to_fit();
        if(output_image_opt.has_value()){
            cv::Mat output_image = output_image_opt.value();
            cv::cvtColor(output_image, output_image, cv::COLOR_RGB2BGR);
            // 显示
            // cv::imshow("input", image);
            cv::resize(output_image,output_image,image.size());
            cv::imshow("Output", output_image);
            cv::waitKey(0);
        }else {
            std::cerr << "DatatoImage Error" << std::endl;
        }
    }
    output_datas.clear();
    output_datas.shrink_to_fit();

    delete api_tool;
    return 0;
}