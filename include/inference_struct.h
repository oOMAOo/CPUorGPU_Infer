#pragma once
#include <string>
#include <functional>
#include <iostream>

enum DEVICE_TYPE{
    OpenVino = 0,
    TensorRT = 1,
    OnnxRuntime = 2
};

template<typename T>
struct ResultData
{
    bool result_state;
    std::string error_message;
    T result_info;
    ResultData(){
        result_state = false;
        error_message = "";
    }
};

