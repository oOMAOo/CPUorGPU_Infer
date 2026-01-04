#pragma once
#include <string>
#include <functional>
#include <iostream>
#include "InferenceStruct.h"
#include "opencv2/opencv.hpp"
#include <optional>
// 自定义异常
#define MY_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error(std::string("<(E`_`E)> ") + message); \
        } \
    } while(false)

namespace InferenceCommon
{
    template<typename T>
    inline void TryFunction(std::function<void(void)> func,ResultData<T> &state){
        try
        {
            func();
        }
        catch(const std::exception& e)
        {
            state.error_message = e.what();
            std::cerr << state.error_message << std::endl;
        }
    }
    inline bool isFileExist(const char* path){
        struct stat buffer;
        return (stat(path, &buffer) == 0);
    }
    std::optional<cv::Mat> Data2Image(float** data,int H,int W, int C,float scale,bool available_cuda);
    // std::optional<float**> InferenceCommon::Image2Data(cv::Mat image);
    int GetNVIDIADriverVersion();
};

