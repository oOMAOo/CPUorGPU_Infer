#include "CudaFun.hpp"
#include <iostream>
__global__ void CHW2HWC(float* from_buffer, float* to_buffer, int H, int W, int C,float scale) {
    int h_i = blockIdx.y * blockDim.y + threadIdx.y; // H 序号
    int w_i = blockIdx.x * blockDim.x + threadIdx.x; // W 序号

    if (h_i < H && w_i < W) {

        // NCHW   N  RRRR...GGGG...BBBB... -> RGBRGBRGB
        for (size_t c_i = 0; c_i < 3; c_i++)
        {
            int from_idx = c_i*H*W + h_i*W + w_i; // buffer 索引
            int to_idx = h_i * W* C + w_i*C + c_i; // image 索引
            to_buffer[to_idx] = float(from_buffer[from_idx])*scale;
        }
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

void CUDA_1CHW_2_1HWC(float** data,int H,int W, int C,float scale){
    dim3 blockSize(24, 24); // CUDA网格 256个线程每块 处理 16H * 16W 数据

    int dim_x = (W / blockSize.x) + (W % blockSize.x == 0 ? 0 : 1);
    int dim_y = (H / blockSize.y) + (H % blockSize.y == 0 ? 0 : 1);
    dim3 gridSize(
        dim_x, dim_y,1
    );
    
    float* transform_data;
    cudaMalloc(&transform_data, H * W * C * sizeof(float));
   
    float* gpu_data;
    cudaMalloc(&gpu_data, H * W * C * sizeof(float));

    cudaMemcpy(gpu_data, *data, H * W * C * sizeof(float), cudaMemcpyHostToDevice);

    // 调用内核
    CHW2HWC<<<gridSize, blockSize>>> (gpu_data, transform_data, H, W, C,scale);
        // 同步和错误检查
    cudaDeviceSynchronize();
    cudaMemcpy(*data, transform_data, H * W * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(gpu_data);
    cudaFree(transform_data);
    CUDA_CHECK(cudaGetLastError());
}

void CUDA_1HWC_2_1CHW(float** data,int H,int W, int C,float scale){
    dim3 blockSize(24, 24); // CUDA网格 256个线程每块 处理 16H * 16W 数据

    int dim_x = (W / blockSize.x) + (W % blockSize.x == 0 ? 0 : 1);
    int dim_y = (H / blockSize.y) + (H % blockSize.y == 0 ? 0 : 1);
    dim3 gridSize(
        dim_x, dim_y,1
    );
    
    float* transform_data;
    cudaMalloc(&transform_data, H * W * C * sizeof(float));
   
    float* gpu_data;
    cudaMalloc(&gpu_data, H * W * C * sizeof(float));

    cudaMemcpy(gpu_data, *data, H * W * C * sizeof(float), cudaMemcpyHostToDevice);

    // 调用内核
    HWC2CHW<<<gridSize, blockSize>>> (gpu_data, transform_data, H, W, C,scale);
    cudaMemcpy(*data, transform_data, H * W * C * sizeof(float), cudaMemcpyDeviceToHost);
    // 同步和错误检查
    cudaDeviceSynchronize();
    cudaFree(gpu_data);
    cudaFree(transform_data);
    CUDA_CHECK(cudaGetLastError());
}