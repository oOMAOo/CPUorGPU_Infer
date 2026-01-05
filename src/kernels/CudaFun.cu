#include "CudaFun.hpp"
#include <device_launch_parameters.h>
#include <iostream>
#include <optional>
__global__ void CHW2HWC(float* from_buffer, float* to_buffer, int H, int W, int C,float scale) {
    int h_i = blockIdx.y * blockDim.y + threadIdx.y; // H 序号
    int w_i = blockIdx.x * blockDim.x + threadIdx.x; // W 序号

    if (h_i < H && w_i < W) {

        // NCHW   N  RRRR...GGGG...BBBB... -> RGBRGBRGB
        to_buffer[h_i * W* C + w_i*C + 0] = float(from_buffer[0*H*W + h_i*W + w_i])*scale;
        to_buffer[h_i * W* C + w_i*C + 1] = float(from_buffer[1*H*W + h_i*W + w_i])*scale;
        to_buffer[h_i * W* C + w_i*C + 2] = float(from_buffer[2*H*W + h_i*W + w_i])*scale;
    }
}

__global__ void HWC2CHW(float* from_buffer, float* to_buffer, int H, int W, int C,float scale) {
    int h_i = blockIdx.y * blockDim.y + threadIdx.y; // H 序号
    int w_i = blockIdx.x * blockDim.x + threadIdx.x; // W 序号

    if (h_i < H && w_i < W) {

        // RGBRGBRGB... -> CHW   RRRR...GGGG...BBBB...
        for (size_t c_i = 0; c_i < 3; c_i++)
        {
            int from_idx = h_i * W* C + w_i*C + c_i; // image 索引
            int to_idx = c_i*H*W + h_i*W + w_i; // buffer 索引
            to_buffer[to_idx] = float(from_buffer[from_idx])*scale;
        }
    }
}

CudaFun::CudaFun(int H,int W, int C): H_(H), W_(W), C_(C)
{
    CUDA_CHECK(cudaMalloc((void**)&transform_buffer_, H_ * W_ * C_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&gpu_buffer_, H_ * W_ * C_ * sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

CudaFun::~CudaFun()
{
    cudaFree(gpu_buffer_);
    cudaFree(transform_buffer_);
}

void CudaFun::CUDA_CHWtoHWC(float** data,float scale){
   size_t data_size = static_cast<size_t>(H_) * W_ * C_ * sizeof(float);
    
    CUDA_CHECK(cudaMemcpyAsync(gpu_buffer_, *data, data_size,cudaMemcpyHostToDevice, stream_));
    dim3 blockSize(32, 8);
    dim3 gridSize(
        (W_ + blockSize.x - 1) / blockSize.x,
        (H_ + blockSize.y - 1) / blockSize.y,
        1
    );
    CHW2HWC<<<gridSize, blockSize, 0, stream_>>>(gpu_buffer_, transform_buffer_, H_, W_, C_, scale);
    //检查内核启动错误
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(*data, transform_buffer_, data_size,
                              cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void CudaFun::CUDA_HWCtoCHW(float** data,float scale){

    dim3 blockSize(24, 24); // CUDA网格 256个线程每块 处理 16H * 16W 数据

    int dim_x = (W_ / blockSize.x) + (W_ % blockSize.x == 0 ? 0 : 1);
    int dim_y = (H_ / blockSize.y) + (H_ % blockSize.y == 0 ? 0 : 1);
    dim3 gridSize(
        dim_x, dim_y,1
    );
    cudaMemcpy(gpu_buffer_, *data, H_ * W_ * C_ * sizeof(float), cudaMemcpyHostToDevice);
    // 调用内核
    HWC2CHW<<<gridSize, blockSize>>> (gpu_buffer_, transform_buffer_, H_, W_, C_,scale);
    cudaMemcpy(*data, transform_buffer_, H_ * W_ * C_ * sizeof(float), cudaMemcpyDeviceToHost);
    // 同步和错误检查
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

std::optional<cv::Mat> CudaFun::Data2Image(float** data_p,int data_size,float scale){
    try
    {
        if(data_size != C_*H_*W_){
            //数据大小不一致，尺寸不对
            return std::nullopt;
        }
        CUDA_CHWtoHWC(data_p,scale);
        return std::nullopt;
        cv::Mat output_image(H_, W_, CV_32FC3,*data_p);
        cv::Mat output_image_8u;
        output_image.convertTo(output_image_8u, CV_8UC3, 1.0f);
        return output_image_8u;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return std::nullopt;
    }
}