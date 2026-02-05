
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
#include "onnxruntime_infer.hpp"
#include "inference_common.hpp"

#define INPUT_W 512
#define INPUT_H 512
//
//1862.054ms
int main(){
    std::string input_img = "C:\\Users\\影擎星图\\Desktop\\nafnet_demo\\CUDA-13.1.png";
    std::string model_path = "C:\\Users\\影擎星图\\Desktop\\nafnet_demo\\warping_spade1.onnx";

    if(!std::filesystem::is_regular_file(std::filesystem::path(std::u8string(input_img.begin(),input_img.end()))) || !std::filesystem::is_regular_file(std::u8string(model_path.begin(),model_path.end()))){
        std::cerr << "<(E`_`E)> Model path/Image path is invalid, please check your path." << std::endl;;
        return -1;
    }
    InferenceApi* api_tool = nullptr;

    //使用CUDA函数检测是否存在NVIDIA设备且计算能力需要是8.9（后续再补全）  目前仅做简单检测
    //TODO 增加显卡版本检测
    bool availableCUDA = inference_common::GetAvailableCUDA();
    std::unique_ptr<CudaFun> cuda_tool = nullptr;
    // if(availableCUDA){
        // api_tool = new TensorRTInfer();
    //     cuda_tool = std::make_unique<CudaFun>(INPUT_H,INPUT_W,3);
    // }else{
        api_tool = new OpenVinoInfer();
    // }
    // api_tool = new OnnxRuntimeInfer();

    // ==================== 1. 创建基础推理引擎 ====================
    api_tool->CreateInferenceEngine();

    using LayoutShape = std::pair<std::string,std::vector<size_t>>;
    std::vector<LayoutShape> input_layouts;
    input_layouts.emplace_back(LayoutShape{"NC...", {1,32,16,64,64}});
    input_layouts.emplace_back(LayoutShape{"NC...", {1,21,3}});
    input_layouts.emplace_back(LayoutShape{"NC...", {1,21,3}});
    // input_layouts.emplace_back(LayoutShape{"NCHW", {1,3,INPUT_H,INPUT_W}});

    std::vector<LayoutShape> output_layouts;
    output_layouts.emplace_back(LayoutShape{"NCHW", {1,3,INPUT_H,INPUT_W}});
    if (!std::filesystem::exists(MODELPATH))
    {
        if(!std::filesystem::create_directory(MODELPATH)){
            std::cerr << std::format("(E`_`E)> create directory [{}] in error...",MODELPATH) << std::endl;
            return -1;
        }
    }
    //0.19562 | 0.163566 | 0.196407 | 0.194865 | 0.202482 | 0.204522 | 0.191374 | 0.181597 | 0.198211 | 0.191047
    //0.195589 | 0.163502 | 0.196259 | 0.195076 | 0.202707 | 0.204838 | 0.191625 | 0.181881 | 0.198311 | 0.191297
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
    cv::Mat image = cv::imread(std::filesystem::path(std::u8string(input_img.begin(),input_img.end())).string(), cv::IMREAD_COLOR_RGB);
    if (image.empty()) {
        std::cout << std::format("<(E`_`E)> Can't Open the image: {}\n", input_img);
        return -1;
    }
    std::cout << std::format("\n(I_I)>>>>> Input image size : {} x {}\n", INPUT_W, INPUT_H);
    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0f/255.0f,cv::Size(INPUT_W,INPUT_H));

    // 测试数据
    float *input1 = new float[1*32*16*64*64]{0};
    float *input2 = new float[1*21*3]{0};
    float *input3 = new float[1*21*3]{0};
    const std::vector<float*> input_datas{input1,input2,input3};
    
    // ==================== 5. 推理 ====================

    // 测试推理 200 次
    double min_time = std::numeric_limits<double>::max();
    for (size_t i = 0; i < 200; i++)
    {
        auto start = std::chrono::steady_clock::now();
        // bool output_state = api_tool->Infer({reinterpret_cast<float*>(blob.data)},output_datas);
        bool output_state = api_tool->Infer(input_datas,output_datas);
        auto end = std::chrono::steady_clock::now();
        double cost_time = std::chrono::duration<double,std::milli>(end - start).count();
        min_time = std::min(min_time, cost_time);
        if(!output_state){
            return -1;
        }
    }
    std::cout << std::format("----------------- Infer image used time : {} ms -----------------\n", min_time);

    auto infer_start = std::chrono::steady_clock::now();
    // bool output_state = api_tool->Infer({reinterpret_cast<float*>(blob.data)},output_datas);
    bool output_state = api_tool->Infer(input_datas,output_datas);
    auto infer_end = std::chrono::steady_clock::now();
    double infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    std::cout << std::format("----------------- Infer image used time : {} ms -----------------\n", infer_time);
    
    //========================== 推理 ==========================
    for (size_t i = 990; i < 1000; i++)
    {
        std::cout << (output_datas[0][i]) << (i==999 ? "" : " | ");
    }
    // ==================== 6. 整理结果 ====================
    // for (int i = 0; i < output_datas.size(); i++)
    // {
    //     float* out_data = output_datas[i].data();
    //     auto tran_start = std::chrono::steady_clock::now();
    //     auto output_image_opt = cuda_tool ? cuda_tool->DatatoImage(&out_data,INPUT_H*INPUT_W*3,255.0f) : inference_common::DatatoImage(&out_data,INPUT_H,INPUT_W,3,255.0f);
    //     auto tran_end = std::chrono::steady_clock::now();
    //     double tran_time = std::chrono::duration<double, std::milli>(tran_end - tran_start).count();
    //     std::cout << std::format("----------------- transfor image used time : {} ms -----------------\n", tran_time);
    //     output_datas[i].clear();
    //     output_datas[i].shrink_to_fit();
    //     if(output_image_opt.has_value()){
    //         cv::Mat output_image = output_image_opt.value();
    //         cv::cvtColor(output_image, output_image, cv::COLOR_RGB2BGR);
    //         // 显示
    //         // cv::imshow("input", image);
    //         cv::resize(output_image,output_image,image.size());
    //         cv::imshow("Output", output_image);
    //         cv::waitKey(0);
    //     }else {
    //         std::cerr << "DatatoImage Error" << std::endl;
    //     }
    // }
    delete[] input1;
    delete[] input2;
    delete[] input3;
    output_datas.clear();
    output_datas.shrink_to_fit();

    delete api_tool;
    return 0;
}