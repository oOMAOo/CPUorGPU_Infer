
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
#include "infer_math.hpp"
#define INPUT_W 512
#define INPUT_H 512
//
//1862.054ms
int main(){
    std::string input_img = "C:\\Users\\影擎星图\\Desktop\\nafnet_demo\\CUDA-13.1.png";
    std::string model_path_1 = "C:\\Users\\影擎星图\\Desktop\\nafnet_demo\\split_spade.onnx";
    std::string model_path_2 = "C:\\Users\\影擎星图\\Desktop\\nafnet_demo\\split_spade2.onnx";
    std::string model_path_3 = "C:\\Users\\影擎星图\\Desktop\\nafnet_demo\\split_spade3.onnx";

    if(!std::filesystem::is_regular_file(std::u8string(input_img.begin(),input_img.end())) || 
       !std::filesystem::is_regular_file(std::u8string(model_path_1.begin(),model_path_1.end())) ||
       !std::filesystem::is_regular_file(std::u8string(model_path_2.begin(),model_path_2.end())) ||
       !std::filesystem::is_regular_file(std::u8string(model_path_3.begin(),model_path_3.end()))
    ){
        std::cerr << "<(E`_`E)> Model path/Image path is invalid, please check your path." << std::endl;;
        return -1;
    }
    InferenceApi* api_tool_1 = nullptr;
    InferenceApi* api_tool_2 = nullptr;
    InferenceApi* api_tool_3 = nullptr;

    //使用CUDA函数检测是否存在NVIDIA设备且计算能力需要是8.9（后续再补全）  目前仅做简单检测
    //TODO 增加显卡版本检测
    bool availableCUDA = inference_common::GetAvailableCUDA();
    std::unique_ptr<CudaFun> cuda_tool = nullptr;
    if(availableCUDA){
        api_tool_1 = new TensorRTInfer();
        api_tool_2 = new TensorRTInfer();
        api_tool_3 = new TensorRTInfer();
        cuda_tool = std::make_unique<CudaFun>(INPUT_H,INPUT_W,3);
    }else{
        api_tool_1 = new OpenVinoInfer();
        api_tool_2 = new OpenVinoInfer();
        api_tool_3 = new OpenVinoInfer();
    }
    // api_tool_1 = new OnnxRuntimeInfer();
    // api_tool_2 = new OnnxRuntimeInfer();
    // api_tool_3 = new OnnxRuntimeInfer();

    // ==================== 1. 创建基础推理引擎 ====================
    api_tool_1->CreateInferenceEngine();
    api_tool_2->CreateInferenceEngine();
    api_tool_3->CreateInferenceEngine();



    using LayoutShape = std::pair<std::string,std::vector<size_t>>;
    std::vector<LayoutShape> input_layouts_1;
    input_layouts_1.emplace_back(LayoutShape{"NC...", {1, 32, 16, 64, 64}});
    input_layouts_1.emplace_back(LayoutShape{"NC...", {1, 21, 3}});
    input_layouts_1.emplace_back(LayoutShape{"NC...", {1, 21, 3}});
    std::vector<LayoutShape> output_layouts_1;
    output_layouts_1.emplace_back(LayoutShape{"NC...", {22, 4, 16, 64, 64}});
    output_layouts_1.emplace_back(LayoutShape{"NC...", {22, 16, 64, 64, 3}});
    output_layouts_1.emplace_back(LayoutShape{"NC...", {1, 21, 16, 64, 64}});
    output_layouts_1.emplace_back(LayoutShape{"NC...", {1, 22, 16, 64, 64, 3}});


    std::vector<LayoutShape> input_layouts_2;
    input_layouts_2.emplace_back(LayoutShape{"NC...", {22, 4, 16, 64, 64}});
    input_layouts_2.emplace_back(LayoutShape{"NC...", {1, 21, 16, 64, 64}});
    input_layouts_2.emplace_back(LayoutShape{"NC...", {1, 22, 16, 64, 64, 3}});
    std::vector<LayoutShape> output_layouts_2;
    output_layouts_2.emplace_back(LayoutShape{"NC...", {1, 142, 16, 64, 64}});
    output_layouts_2.emplace_back(LayoutShape{"NC...", {1, 16, 64, 64, 3}});

    std::vector<LayoutShape> input_layouts_3;
    input_layouts_3.emplace_back(LayoutShape{"NC...", {1, 142, 16, 64, 64}});
    input_layouts_3.emplace_back(LayoutShape{"NC...", {1, 32, 16, 64, 64}});
    std::vector<LayoutShape> output_layouts_3;
    output_layouts_3.emplace_back(LayoutShape{"NC...", {1, 3, 512, 512}});


    if (!std::filesystem::exists(MODELPATH))
    {
        if(!std::filesystem::create_directory(MODELPATH)){
            std::cerr << std::format("(E`_`E)> create directory [{}] in error...",MODELPATH) << std::endl;
            return -1;
        }
    }
    // ==================== 2. 加载模型文件 ====================
    ResultData<std::string> load_state_1 = api_tool_1->LoadModel(model_path_1,input_layouts_1,output_layouts_1);
    ResultData<std::string> load_state_2 = api_tool_2->LoadModel(model_path_2,input_layouts_2,output_layouts_2);
    ResultData<std::string> load_state_3 = api_tool_3->LoadModel(model_path_3,input_layouts_3,output_layouts_3);
    if(!(load_state_1.result_state && load_state_2.result_state && load_state_3.result_state) ){
        return -1;
    }
    // ==================== 3. 加载引擎文件，创建识别引擎 ====================
    ResultData<bool> engine_state = api_tool_1->CreateEngine(load_state_1.result_info);
    engine_state.result_state &= api_tool_2->CreateEngine(load_state_2.result_info).result_state;
    engine_state.result_state &= api_tool_3->CreateEngine(load_state_3.result_info).result_state;
    if(!engine_state.result_state){
        return -1;
    }


    // ==================== 4. 准备输入数据 ====================
    std::vector<std::vector<float>> output_datas;
    // 测试数据
    constexpr int model_1_input1_size = 1*32*16*64*64;
    constexpr int model_1_input2_size = 1*21*3;
    constexpr int model_1_input3_size = 1*21*3;
    float *model_1_input1 = new float[model_1_input1_size]{0};
    float *model_1_input2 = new float[model_1_input2_size]{0};
    float *model_1_input3 = new float[model_1_input3_size]{0};
    const std::vector<float*> model_1_input_datas{model_1_input1,model_1_input2,model_1_input3};

    constexpr int model_2_input1_size = 22*4*16*64*64;
    constexpr int model_2_input2_size = 1*21*16*64*64;
    constexpr int model_2_input3_size = 1*22*16*64*64*3;
    float *model_2_input1 = new float[model_2_input1_size]{0};  //reshape3
    float *model_2_input2 = new float[model_2_input2_size]{0};
    float *model_2_input3 = new float[model_2_input3_size]{0};
    const std::vector<float*> model_2_input_datas{model_2_input1,model_2_input2,model_2_input3};

    constexpr int model_3_input1_size = 1*142*16*64*64;
    constexpr int model_3_input2_size = 1*32*16*64*64;
    float *model_3_input1 = new float[model_3_input1_size]{0};
    float *model_3_input2 = new float[model_3_input2_size]{0};
    const std::vector<float*> model_3_input_datas{model_3_input1,model_3_input2};
    
    // ==================== 5. 推理 (Model 1)====================
    auto infer_start = std::chrono::steady_clock::now();
    bool output_state = api_tool_1->Infer(model_1_input_datas,output_datas);
    auto infer_end = std::chrono::steady_clock::now();
    double infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    std::cout << std::format("----------------- Model_1 Infer image used time : {} ms -----------------\n", infer_time);

    assert(output_datas.size() == output_layouts_1.size());//Model 1 output
    // output_datas[0]  /dense_motion_network/Reshape_3_output_0
    // output_datas[1]  /dense_motion_network/Reshape_4_output_0
    // output_datas[2]  /dense_motion_network/Sub_3_output_0
    // output_datas[3]  /dense_motion_network/Concat_1_output_0
    InferMath::Tensor3D GridSample1_input(22, 4, 16, 64, 64);
    GridSample1_input.setData(output_datas[0].data(),output_datas[0].size());
    InferMath::Grid3D GridSample1_grid(22, 16, 64, 64);
    GridSample1_grid.setData(output_datas[1].data(),output_datas[1].size());
    auto GridSample1 = InferMath::mathGridsample3D(GridSample1_input,GridSample1_grid,false);
    errno_t result = memcpy_s(model_2_input1,model_2_input1_size,GridSample1.ptr(),model_2_input1_size);
    result += memcpy_s(model_2_input2,model_2_input2_size,output_datas[2].data(),model_2_input2_size);
    result += memcpy_s(model_2_input3,model_2_input3_size,output_datas[3].data(),model_2_input3_size);
    if (result != 0) {
        std::cout << std::format("Memory copy failed with error code: %d\n", result) << std::endl;
    }
    output_datas.clear();
    output_datas.shrink_to_fit();

    // ==================== 5. 推理 (Model 2)====================
    infer_start = std::chrono::steady_clock::now();
    output_state = api_tool_2->Infer(model_2_input_datas,output_datas);
    infer_end = std::chrono::steady_clock::now();
    infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    std::cout << std::format("----------------- Model_2 Infer image used time : {} ms -----------------\n", infer_time);

    assert(output_datas.size() == output_layouts_2.size());//Model 2 output
    // output_datas[0]  /dense_motion_network/hourglass/decoder/Relu_output_0
    // output_datas[1]  /dense_motion_network/Transpose_1_output_0
    InferMath::Tensor3D GridSample2_input(1, 32, 16, 64, 64);
    GridSample2_input.setData(model_1_input1,model_1_input1_size);
    InferMath::Grid3D GridSample2_grid(1, 16, 64, 64);
    GridSample2_grid.setData(output_datas[1].data(),output_datas[1].size());
    
    auto GridSample2 = InferMath::mathGridsample3D(GridSample2_input,GridSample2_grid,false);
    result = memcpy_s(model_3_input1,model_3_input1_size,output_datas[0].data(),model_3_input1_size);
    result += memcpy_s(model_3_input2,model_3_input2_size,GridSample2.ptr(),model_3_input2_size);
    if (result != 0) {
        std::cout << std::format("Memory copy failed with error code: %d\n", result) << std::endl;
    }


    output_datas.clear();
    output_datas.shrink_to_fit();
    // ==================== 5. 推理 (Model 3)====================
    infer_start = std::chrono::steady_clock::now();
    output_state = api_tool_3->Infer(model_3_input_datas,output_datas);
    infer_end = std::chrono::steady_clock::now();
    infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    std::cout << std::format("----------------- Model_3 Infer image used time : {} ms -----------------\n", infer_time);
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
            cv::resize(output_image,output_image,cv::Size(INPUT_W,INPUT_H));
            cv::imshow("Output", output_image);
            cv::waitKey(0);
        }else {
            std::cerr << "DatatoImage Error" << std::endl;
        }
        std::cout << std::format("OutPutData:{}  ->   first data [{}]\n",i,*out_data);
    }
    output_datas.clear();
    output_datas.shrink_to_fit();

    //清理测试数据
    for (size_t i = 0; i < model_1_input_datas.size(); i++)
    {
       delete[] model_1_input_datas[i];
    }
        for (size_t i = 0; i < model_2_input_datas.size(); i++)
    {
       delete[] model_2_input_datas[i];
    }
        for (size_t i = 0; i < model_3_input_datas.size(); i++)
    {
       delete[] model_2_input_datas[i];
    }
    
    output_datas.clear();
    output_datas.shrink_to_fit();

    delete api_tool_1;
    delete api_tool_2;
    delete api_tool_3;
    return 0;
}