#pragma once
#include <string>
#include <functional>
#include <iostream>
#include <optional>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "inference_struct.h"



#define MODELPATH "./model"
// 自定义异常
#define MY_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error(std::string("<(E`_`E)> ") + message); \
        } \
    } while(false)

namespace inference_common
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
        catch(const Ort::Exception& e)
        {
            state.error_message = e.what();
            std::cerr << state.error_message << std::endl;
        }
    }
    std::optional<cv::Mat> DatatoImage(float** data_p,int H,int W, int C,float scale);
    // std::optional<float**> inference_common::Image2Data(cv::Mat image);
    bool GetAvailableCUDA();
};

