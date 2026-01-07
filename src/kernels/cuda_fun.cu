#include "cuda_fun.hpp"

#include <iostream>
#include <optional>

#include <device_launch_parameters.h>
__global__ void CHW2HWC(float* from_buffer, float* to_buffer, int H, int W, int C,float scale) {
    int h_i = blockIdx.y * blockDim.y + threadIdx.y;
    int w_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_i < H && w_i < W) {

        // NCHW   N  RRRR...GGGG...BBBB... -> RGBRGBRGB
        to_buffer[h_i * W* C + w_i*C + 0] = float(from_buffer[0*H*W + h_i*W + w_i])*scale;
        to_buffer[h_i * W* C + w_i*C + 1] = float(from_buffer[1*H*W + h_i*W + w_i])*scale;
        to_buffer[h_i * W* C + w_i*C + 2] = float(from_buffer[2*H*W + h_i*W + w_i])*scale;
    }
}

__global__ void HWC2CHW(float* from_buffer, float* to_buffer, int H, int W, int C,float scale) {
    int h_i = blockIdx.y * blockDim.y + threadIdx.y;
    int w_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_i < H && w_i < W) {

        // RGBRGBRGB... -> CHW   RRRR...GGGG...BBBB...
        for (int c_i = 0; c_i < 3; c_i++)
        {
            int from_idx = h_i * W* C + w_i*C + c_i;
            int to_idx = c_i*H*W + h_i*W + w_i;
            to_buffer[to_idx] = float(from_buffer[from_idx])*scale;
        }
    }
}

CudaFun::CudaFun(int height,int width, int channel): m_height(height), m_width(width), m_channel(channel)
{
    CUDA_CHECK(cudaMalloc((void**)&m_transform_buffer, m_height*m_width*m_channel*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&m_gpu_buffer, m_height*m_width*m_channel*sizeof(float)));
    CUDA_CHECK(cudaStreamCreate(&m_stream));
}

CudaFun::~CudaFun()
{
    cudaFree(m_gpu_buffer);
    cudaFree(m_transform_buffer);
}

void CudaFun::CUDA_CHWtoHWC(float** data_ptr,float scale){
   size_t data_size = m_height*m_width*m_channel*sizeof(float);
    
    CUDA_CHECK(cudaMemcpyAsync(m_gpu_buffer, *data_ptr, data_size,cudaMemcpyHostToDevice, m_stream));
    dim3 blockSize(32, 8);
    dim3 gridSize(
        (m_width + blockSize.x - 1) / blockSize.x,
        (m_height + blockSize.y - 1) / blockSize.y,
        1
    );
    CHW2HWC<<<gridSize, blockSize, 0, m_stream>>>(m_gpu_buffer, m_transform_buffer, m_height, m_width, m_channel, scale);
    //检查内核启动错误
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(*data_ptr, m_transform_buffer, data_size,
                              cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));
}

void CudaFun::CUDA_HWCtoCHW(float** data,float scale){

    dim3 blockSize(24, 24); // CUDA网格 256个线程每块 处理 16H * 16W 数据

    int dim_x = (m_width / blockSize.x) + (m_width % blockSize.x == 0 ? 0 : 1);
    int dim_y = (m_height / blockSize.y) + (m_height % blockSize.y == 0 ? 0 : 1);
    dim3 gridSize(
        dim_x, dim_y,1
    );
    cudaMemcpy(m_gpu_buffer, *data, m_height*m_width*m_channel * sizeof(float), cudaMemcpyHostToDevice);
    // 调用内核
    HWC2CHW<<<gridSize, blockSize>>> (m_gpu_buffer, m_transform_buffer, m_height, m_width, m_channel,scale);
    cudaMemcpy(*data, m_transform_buffer, m_height*m_width*m_channel * sizeof(float), cudaMemcpyDeviceToHost);
    // 同步和错误检查
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
}

std::optional<cv::Mat> CudaFun::DatatoImage(float** data_ptr,int data_size,float scale){
    try
    {
        if(data_size != m_channel*m_height*m_width){
            //数据大小不一致，尺寸不对
            return std::nullopt;
        }
        CUDA_CHWtoHWC(data_ptr,scale);
        cv::Mat output_image(m_height, m_width, CV_32FC3,*data_ptr);
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