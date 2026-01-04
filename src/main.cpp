#include <opencv2/opencv.hpp>
#include <iostream>
#include "InferenceApi.hpp" 
#include "OpenVinoInfer.hpp"
#include "TensorRTInfer.hpp"
#include "InferenceCommon.hpp"
#include <fstream>
#include <ctime>
#include <direct.h>
#include <io.h>
using namespace cv;
#define Input_W 1050
#define Input_H 1550

int main(){
    bool AvailableCUDA = false;
    std::string input_img = "C:\\Users\\影擎星图\\Desktop\\nafnet_demo\\input.jpg";
    std::string model_path = "C:\\Users\\影擎星图\\Desktop\\nafnet_demo\\nafnet_ep304_20251117.onnx";
    InferenceApi* api_tool = nullptr;
#ifdef USE_CUDA
    int Nvidia_version = InferenceCommon::GetNVIDIADriverVersion();
    if(Nvidia_version >= 58000){
        std::cout << std::format("Nvidia driver [{}.{}]\n",Nvidia_version/100,Nvidia_version%100);
        AvailableCUDA = true;
        api_tool = new TensorRTInfer();
    }else if(Nvidia_version>0){
        std::cout << std::format("Nvidia driver [{}.{}] is too old (need 580.xx+ for CUDA 13.x), use CPU\n" ,Nvidia_version/100,Nvidia_version%100);
    }else{
        std::cout << "Please check if you have the Nvidia driver installed?" << std::endl;
    }
#endif

    if(!api_tool){
        api_tool = new OpenVinoInfer();
    }
    api_tool->CreateInferenceEngine();
    std::pair<std::string,std::vector<size_t>>input_layout;
    input_layout.first = "NCHW";
    input_layout.second = {1,3,Input_H,Input_W};

    std::pair<std::string,std::vector<size_t>>output_layout;
    output_layout.first = "NCHW";
    output_layout.second = {1,3,Input_H,Input_W};

    if (_access("model", 0) != 0)
    {
        int ret = _mkdir("model");
    }

    ResultData<std::string> load_state = api_tool->LoadModel(model_path,{input_layout},{output_layout});
    if(!load_state.result_state){
        return -1;
    }
    //这里会给GPU分配显存
    ResultData<bool> engine_state = api_tool->CreateEngine(load_state.result_info);
    if(!engine_state.result_state){
        return -1;
    }


    for (size_t loop_i = 0; loop_i < 500; loop_i++)
    {
        auto infer_start = std::chrono::steady_clock::now().time_since_epoch();
        std::string p = std::filesystem::u8path(input_img).string();
        Mat image = imread(p.c_str(), IMREAD_COLOR);
        if (image.empty()) {
            std::cout << std::format("<(E`_`E)> Can't Open the image: {}\n", input_img.c_str());
            return -1;
        }
        std::cout << std::format("\n(I_I)>>>>> Input image size : {} x {}\n", (int)Input_W, (int)Input_H);
        Mat blob = cv::dnn::blobFromImage(image, 1.0f/255.0f, Size(Input_W, Input_H), Scalar(0, 0, 0), true, false, CV_32F); // NCHW
        std::vector<std::vector<size_t>>data_layout;
        data_layout.push_back({1,3,Input_H,Input_W});
        ResultData<std::vector<float*>> output_state = api_tool->Infer(data_layout,{(float*)(blob.data)});
        if(!output_state.result_state){
            return -1;
        }
        auto infer_end = std::chrono::steady_clock::now().time_since_epoch();
        double infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
        std::cout << std::format("----------------- {}/{} Infer image used time : {} ms -----------------\n",loop_i+1,500, infer_time);
        for (size_t i = 0; i < output_state.result_info.size(); i++)
        {
            float* out_data = output_state.result_info[i];
            auto tran_start = std::chrono::steady_clock::now().time_since_epoch();
            auto output_image_opt = InferenceCommon::Data2Image(&out_data,Input_H,Input_W,3,255.0f,AvailableCUDA);
            auto tran_end = std::chrono::steady_clock::now().time_since_epoch();
            double tran_time = std::chrono::duration<double, std::milli>(tran_end - tran_start).count();
            std::cout << std::format("----------------- {}/{} transfor image used time : {} ms -----------------\n",loop_i+1,500, tran_time);
            delete[] output_state.result_info[i];
            out_data = nullptr;
            if(output_image_opt.has_value()){
                cv::Mat output_image = output_image_opt.value();
                cvtColor(output_image, output_image, COLOR_RGB2BGR);
                // 显示
                // cv::imshow("input", image);
                // cv::imshow("Output", output_image);
                // cv::waitKey(0);
            }
            _sleep(1*1000);
        }
    }
    

    

    

    
    


    delete api_tool;
    return 0;
}