#include "InferenceCommon.hpp"
#include "CudaFun.hpp"
std::optional<cv::Mat> InferenceCommon::Data2Image(float** data_p,int H,int W, int C,float scale,bool available_cuda){
    try
    {
        if(available_cuda){
            cv::Mat output_image(H, W, CV_32FC3);
            CUDA_1CHW_2_1HWC(data_p,H,W,C,scale);
            float* data = *data_p;
            output_image.data = (uchar*)(data);
            cv::Mat output_image_8u;
            output_image.convertTo(output_image_8u, CV_8UC3, 1.0f);
            return output_image_8u;}
        else{
            cv::Mat output_image(H, W, CV_8UC3);
            float* data = *data_p;
            for (size_t i = 0; i < H; i++)
            {
                for (size_t j = 0; j < W; j++) {
                    for (size_t k = 0; k < C; k++)
                    {
                        float e = std::max(std::min((data[k * H * W + i * W + j] * 255.0f), 255.0f), 0.0f);
                        output_image.at<cv::Vec3b>(i, j)[k] = e;
                    }
                }
            }
            cv::Mat output_image_8u;
            output_image.convertTo(output_image_8u, CV_8UC3, 1.0f);
            return output_image_8u;
        }
        
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return std::nullopt;
    }
}

// std::optional<float**> InferenceCommon::Image2Data(cv::Mat image){
//     try
//     {
//         #ifdef USE_CUDA
//             cv::Mat input_image;
//             image.convertTo(input_image,CV_32FC1,1.0f);
//             float* img_data = (float*)input_image.data;
//             CUDA_1HWC_2_1CHW(&img_data,image.rows,image.cols,3,1.0f/255.0f);
//             return &img_data;
//         #else
//         //TODO
//         #endif
//     }
//     catch(const std::exception& e)
//     {
//         std::cerr << e.what() << std::endl;
//         return std::nullopt;
//     }
// }
int InferenceCommon::GetNVIDIADriverVersion()
{
    std::unique_ptr<FILE, decltype(&_pclose)> pipe(
        _popen("nvidia-smi --query-gpu=driver_version --format=csv,noheader", "r"),
        _pclose
    );

    if (pipe) {
        char buffer[32];
        if (fgets(buffer, sizeof(buffer), pipe.get())) {
            // 解析版本，如 "535.98" -> 53598
            std::string version(buffer);
            size_t dot = version.find('.');
            if (dot != std::string::npos) {
                try {
                    int major = std::stoi(version.substr(0, dot));
                    int minor = std::stoi(version.substr(dot + 1));
                    return major * 100 + minor;
                } catch (...) {
                }
            }
        }
    }
    return 0;
}

