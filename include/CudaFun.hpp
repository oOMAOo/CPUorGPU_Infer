#pragma once
#include <cuda_runtime_api.h>
#include <string>
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            throw std::runtime_error(std::string("<(E`_`E)> CUDA error Code:[") + std::to_string(error_code) + \
            "] in function " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "  >>>  " + cudaGetErrorString(error_code)); \
        }\
    }
#endif
void CUDA_1CHW_2_1HWC(float** data,int H,int W, int C,float scale);
void CUDA_1HWC_2_1CHW(float** data,int H,int W, int C,float scale);