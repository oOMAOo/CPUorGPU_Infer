#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <CudaFun.hpp>
static inline double now_ms()
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

int main(int argc, char **argv){

    if (argc != 3)
    {
        std::cout << "need: " << " <width> <height> agrs" << argc << std::endl;
        return -1;
    }
    const int W = std::atoi(argv[1]);
    const int H = std::atoi(argv[2]);
    size_t plane_size = static_cast<size_t>(H) * W;
    constexpr int C = 3;
    constexpr int LOOP_COUNT = 1000;
    std::vector<float> chw_data(C * plane_size);

    for (auto &v : chw_data)
    {
        v = static_cast<float>(std::rand()) / RAND_MAX;
    }
    float* cpu_buffer = chw_data.data();

    //分配GPU buffer
    // float* gpu_buffer;
    // cudaMalloc((void**)&gpu_buffer, C * plane_size * sizeof(float));
    // double cpu_2_gpu_sum = 0;

    double HWC_2_1CHW_sum = 0;
    for (size_t i = 0; i < LOOP_COUNT; i++)
    {
        double t0 = now_ms();
        CUDA_1HWC_2_1CHW(&cpu_buffer,H,W,C,1.0f/255.0f);
        double t1 = now_ms();
        if(i>=9){
            HWC_2_1CHW_sum += (t1-t0);
        }
    }
    std::cout << "HWC_2_1CHW cost time: " << HWC_2_1CHW_sum/990.0 << "ms" << std::endl;



}