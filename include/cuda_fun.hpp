#pragma once
#include <cuda_runtime_api.h>
#include <string>

#include <opencv2/opencv.hpp>
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    do {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            throw std::runtime_error(std::string("<(E`_`E)> CUDA error Code:[") + std::to_string(error_code) + \
            "] in function " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "  >>>  " + cudaGetErrorString(error_code)); \
        }\
    } while(0)
#endif


class CudaFun
{
private:
    float* m_gpu_buffer;
    float* m_transform_buffer;
    cudaStream_t m_stream;
    int m_height;
    int m_width;
    int m_channel;
    void CUDA_CHWtoHWC(float** data_ptr,float scale);
    void CUDA_HWCtoCHW(float** data_ptr,float scale);
public:
    CudaFun(int H, int W, int C);
    std::optional<cv::Mat> DatatoImage(float** data_ptr, int data_size, float scale);
    int GetHeight(){return m_height;};
    int GetWidth(){return m_width;};
    int GetChannel(){return m_channel;};
    ~CudaFun();

};


