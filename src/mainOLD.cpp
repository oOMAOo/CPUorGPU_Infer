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
        v = static_cast<float>(std::rand() % 255) ;
    }
    float* cpu_buffer = chw_data.data();

    // double blob_from_image_sum = 0;
    // for (int i = 0; i < LOOP_COUNT; i++)
    // {
    //     cv::Mat img(H,W,CV_32FC3,cpu_buffer);
    //     double t0 = now_ms();
    //     cv::Mat blob = cv::dnn::blobFromImage(img,1.0f/255.0f,cv::Size(W,H));
    //     double t1 = now_ms();
    //     if(i>=9){
    //         blob_from_image_sum += (t1-t0);
    //     }
    // }
    // std::cout << "blob_from_image cost time: " << blob_from_image_sum/990.0 << "ms" << std::endl;

    // float* gpu_buffer;
    // cudaMalloc((void**)&gpu_buffer, H * W * C * sizeof(float));
    // double CUDA_CPU_to_GPU_sum = 0;
    // for (int i = 0; i < LOOP_COUNT; i++)
    // {
    //     double t0 = now_ms();
    //     cudaMemcpyAsync(gpu_buffer,cpu_buffer,H * W * C * sizeof(float),cudaMemcpyHostToDevice);
    //     cudaDeviceSynchronize();
    //     double t1 = now_ms();
    //     if(i>=9){
    //         CUDA_CPU_to_GPU_sum += (t1-t0);
    //     }
    // }
    // std::cout << "cudaMemcpyAsync CPU_to_GPU cost time: " << CUDA_CPU_to_GPU_sum/990.0 << "ms" << std::endl;


    // double CUDA_GPU_to_CPU_sum = 0;
    // for (int i = 0; i < LOOP_COUNT; i++)
    // {
    //     double t0 = now_ms();
    //     cudaMemcpyAsync(cpu_buffer,gpu_buffer,H * W * C * sizeof(float),cudaMemcpyDeviceToHost);
    //     cudaDeviceSynchronize();
    //     double t1 = now_ms();
    //     if(i>=9){
    //         CUDA_GPU_to_CPU_sum += (t1-t0);
    //     }
    // }
    // std::cout << "cudaMemcpyAsync GPU_to_CPU cost time: " << CUDA_GPU_to_CPU_sum/990.0 << "ms" << std::endl;
    auto prosser = std::make_unique<CudaFun>(H,W,C);
    double min_time = std::numeric_limits<double>::max();
    // double CHW_2_1HWC_sum = 0;
    for (int i = 0; i < LOOP_COUNT; i++)
    {
        double t0 = now_ms();
        prosser->Data2Image(&cpu_buffer,H*W*C,1.0f/255.0f);
        double t1 = now_ms();
        double elapsed = t1 - t0;
        min_time = std::min(min_time, elapsed);
        // if(i>=9){
        //     CHW_2_1HWC_sum += (t1-t0);
        // }
    }
    std::cout << "CHW_2_1HWC cost time: " << min_time << "ms" << std::endl;




    // double HWC_2_1CHW_sum = 0;
    // for (int i = 0; i < LOOP_COUNT; i++)
    // {
    //     double t0 = now_ms();
    //     CUDA_1HWC_2_1CHW(&cpu_buffer,H,W,C,1.0f/255.0f);
    //     double t1 = now_ms();
    //     if(i>=9){
    //         HWC_2_1CHW_sum += (t1-t0);
    //     }
    // }
    // std::cout << "HWC_2_1CHW cost time: " << HWC_2_1CHW_sum/990.0 << "ms" << std::endl;



}