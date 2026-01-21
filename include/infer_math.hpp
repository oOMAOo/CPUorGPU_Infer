#pragma once
#include <string>
#include <iostream>
#include <vector>
#include <cassert>
namespace InferMath{

    struct Tensor3D {
        int N, C, D, H, W;
        Tensor3D(int n, int c, int d, int h, int w) 
            : N(n), C(c), D(d), H(h), W(w), data(n * c * d * h * w) {}
        bool setData(const std::vector<float> &t_data){
            if(t_data.size() == data.size()){
                data = t_data;
                return true;
            }
            return false;
        }
        bool setData(float* t_data_ptr,size_t size){
            assert(size == data.size());
            errno_t code =memcpy_s(data.data(),size*sizeof(float),t_data_ptr,size*sizeof(float));
            assert(code == 0);
            return true;
        }
        float& at(int n, int c, int d, int h, int w) {
            return data[((n * C + c) * D + d) * H * W + h * W + w];
        }
        float* ptr(int n, int c, int d, int h, int w){
            return &data[((n * C + c) * D + d) * H * W + h * W + w];
        }
        float* ptr(){
            return data.data();
        }
        size_t size(){
            return data.size();
        }
    private:
        // 数据在这里
        std::vector<float> data;
    };

    struct Grid3D {
        int N, D, H, W;
        const int C{3};
        Grid3D(int n, int d, int h, int w) 
            : N(n), D(d), H(h), W(w), data(n * d * h * w * C) {}
        bool setData(const std::vector<float> &t_data){
            if(t_data.size() == data.size()){
                data = t_data;
                return true;
            }
            return false;
        }
        bool setData(float* t_data_ptr,size_t size){
            assert(size == data.size());
            errno_t code =memcpy_s(data.data(),size*sizeof(float),t_data_ptr,size*sizeof(float));
            assert(code == 0);
            return true;


        }
        float& at(int n, int d, int h, int w, int c) {
            return data[((n * D + d) * H + h) * W * C + w * C + c];
        }
        float* ptr(){
            return data.data();
        }
    private:
        // 数据在这里
        std::vector<float> data;
    };

    enum class InterpolationMode
    { 
        Bilinear, 
        Nearest
    };
    enum PaddingMode
    { 
        Zeros, 
        Border, 
        Reflection
    };

    Tensor3D mathGridsample3D(Tensor3D input,Grid3D grid,bool align_corners);

} //namespace


