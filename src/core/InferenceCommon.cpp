#include "InferenceCommon.hpp"
#include "CudaFun.hpp"

std::optional<cv::Mat> InferenceCommon::DatatoImage(float** data_p,int H,int W, int C,float scale){
    try
    {
        cv::Mat output_image(H, W, CV_8UC3);
        float* data = *data_p;
        for (int i = 0; i < H; i++)
        {
            for (int j = 0; j < W; j++) 
            {
                for (int k = 0; k < C; k++)
                {
                    //cv::saturate_cast 强制转换为指定类型，并确保转换过程中像素值不会超出取值范围
                    output_image.at<cv::Vec3b>(i, j)[k] = cv::saturate_cast<uchar>(data[k * H * W + i * W + j] * scale);
                }
            }
        }
        return output_image;
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

bool InferenceCommon::GetAvailableCUDA()
{
    try
    {
        int driver_count;
        CUDA_CHECK(cudaGetDeviceCount(&driver_count));

        if (driver_count == 0) {
            std::cout << "Can't find a NVIDIA GPU" << std::endl;
        }
        int runtime_version;
        cudaRuntimeGetVersion(&runtime_version);
        std::cout << "Detect " << driver_count << " NVIDIA GPU \nCUDA Runtime Version:"<< runtime_version << std::endl;
        //主要不是0个，就取第一个设备
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, 0);
        std::cout << "=== Device Name :" << device_prop.name << " ===" << std::endl;
        std::cout << "Compute Capability:" << device_prop.major << "." << device_prop.minor << std::endl;
        std::cout << "GlobalMem:" << device_prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "CUDA Processor Count:" << device_prop.multiProcessorCount << std::endl;
        std::cout << std::endl;

        if(device_prop.major*10 + device_prop.minor == 89)
        {
            return true;
        }
            
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return false;
}

