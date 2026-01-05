#pragma once
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <string>
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
    float* gpu_buffer_;
    float* transform_buffer_;
    cudaStream_t stream_;
    int H_;
    int W_;
    int C_;
    void CUDA_CHWtoHWC(float** data,float scale);
    void CUDA_HWCtoCHW(float** data,float scale);
public:
    CudaFun(int H, int W, int C);
    std::optional<cv::Mat> Data2Image(float** data_p, int data_size, float scale);
    int Get_W(){return W_;};
    int Get_H(){return H_;};
    int Get_C(){return C_;};
    ~CudaFun();

};


