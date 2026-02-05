#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "infer_math.hpp"
#include <Eigen/Eigen>
#include "trt_plugins/addgridsamplePlugin/addgridsample.cuh"
using namespace std;
using namespace InferMath;
//余弦相似度
double cosineSimilarity(cv::Mat& feature1, cv::Mat& feature2) {
    CV_Assert(feature1.rows == 1 && feature2.rows == 1 && feature1.cols == feature2.cols);
    double dot = feature1.dot(feature2);
    double norm1 = cv::norm(feature1);
    double norm2 = cv::norm(feature2);
    return dot / (norm1 * norm2);
}

inline vector<float> readDataFromFile(const char* file_path){
    ifstream file(file_path);
    if(!file.is_open()){
        std::cout << file_path << " Open in Error!" << std::endl;
    }
    vector<float> data;
    string line;
    while (getline(file,line))
    {
        data.push_back(atof(line.c_str()));
    }
    file.close();
    return std::move(data);
}
bool evaluate(float* input_data,float *grid_data,float* output_data){

    const vector<size_t> input_shape = {22,4,16,64,64};
    const vector<size_t> grid_shape = {22,16,64,64,3};
    
    const size_t N = input_shape[0];
    const size_t C = input_shape[1];
    const size_t I_D = input_shape[2];
    const size_t I_H = input_shape[3];
    const size_t I_W = input_shape[4];
    const size_t D = grid_shape[1];
    const size_t H = grid_shape[2];
    const size_t W = grid_shape[3];
    constexpr size_t G_C = 3;
    // input 转成N*C个 Egine 对象 （值copy）修改矩阵数据不影响源数据
    std::vector<Eigen::MatrixXf> input_slices;
    for (size_t n = 0; n < N; n++)
    {
        for (size_t c = 0; c < C; c++)
        {
            for (size_t d = 0; d < I_D; d++)
            {
                //偏移量
                int input_mat_shift = ((n * C + c) * I_D + d) * I_H * I_W;
                input_slices.emplace_back(Eigen::Map<Eigen::MatrixXf>((input_data + input_mat_shift), I_W, I_H));
            }
        }
    }
    //输出形状(D H W)看 grid
    // Tensor3D output(N, C, D, H, W);

    //遍历Grid
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int od = 0; od < D; ++od) {
                // int output_mat_shift = ((n * C + c) * D + d) * H * W;
                // Eigen::Map<Eigen::MatrixXf> output_map(output.ptr(n, c, od, 0, 0), W, H);
                // int grid_mat_shift = (n * D + od) * H  * W * G_C;
                // cv::Mat grid_mat(H,W,CV_32FC3,(grid_data + grid_mat_shift));
                // std::vector<cv::Mat>grid_channels;
                // cv::split(grid_mat,grid_channels);
                // cv::Mat x_channel = grid_channels[0];
                // cv::Mat y_channel = grid_channels[1];
                // cv::Mat z_channel = grid_channels[2];
                // x_channel = ((x_channel + 1.0f) * I_W - 1.0f) / 2.0f;
                // y_channel = ((y_channel + 1.0f) * I_H - 1.0f) / 2.0f;
                // z_channel = ((z_channel + 1.0f) * I_D - 1.0f) / 2.0f;
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        // 获取网格坐标
                        int grid_mat_shift = (((n * D + od) * H + h) * W + w) * G_C;
                        float grid_x = *(grid_data + grid_mat_shift);
                        float grid_y = *(grid_data + grid_mat_shift + 1);
                        float grid_z = *(grid_data + grid_mat_shift + 2);

                        
                        //线性函数 计算坐标
                        float ix, iy, iz;
                        // [-0.5, I_W-0.5] 
                        ix = ((grid_x + 1.0f) * I_W - 1.0f) / 2.0f;
                        iy = ((grid_y + 1.0f) * I_H - 1.0f) / 2.0f;
                        iz = ((grid_z + 1.0f) * I_D - 1.0f) / 2.0f;
                        // float ix = x_channel.at<float>(h,w);
                        // float iy = y_channel.at<float>(h,w);
                        // float iz = z_channel.at<float>(h,w);

                        int ix0 = static_cast<int>(std::floor(ix));//小x坐标
                        int ix1 = ix0 + 1;//大x坐标

                        int iy0 = static_cast<int>(std::floor(iy));//小y坐标
                        int iy1 = iy0 + 1;//大y坐标

                        int iz0 = static_cast<int>(std::floor(iz));//小z坐标
                        int iz1 = iz0 + 1;//大z坐标

                        // 双线性插值 算法
                        //value=(1−Δx)(1−Δy)⋅V_(i,j)  +  Δx(1−Δy)⋅V_(i+1,j)  +  (1−Δx)Δy⋅V_(i,j+1)  +  ΔxΔy⋅V_(i+1,j+1)    Δx 是ix的小数部分（ix-ix0）
                        // ↓
                        //value = (1-(ix-ix0))*(1−(iy-iy0))*get_value(ix0,iy0) + ... 
                        
                        // 根据坐标 计算权重
                        float wx1 = ix - ix0;// Δx
                        float wx0 = 1.0f - wx1;// 1-Δx
                        float wy1 = iy - iy0;// Δy
                        float wy0 = 1.0f - wy1;// 1-Δy
                        float wz1 = iz - iz0;// Δz
                        float wz0 = 1.0f - wz1;// 1-Δz
                        
                        // 输入切片的索引
                        int slice_idx_base = n * C * I_D + c * I_D;
                        
                        // 边界检查函数
                        auto get_value = [&](int x_idx,int y_idx,int z_idx) -> float {
                            if(z_idx>=0 && z_idx<I_D && y_idx >=0 && y_idx<I_H && x_idx >=0 && x_idx<I_W){
                                int slice_idx = slice_idx_base + z_idx;
                                return input_slices[slice_idx](x_idx,y_idx);
                            }else{
                                return 0.f;
                            }
                        };

                        //获取输入矩阵的原坐标内容
                        float v000 = wx0 * wy0 * wz0;
                        float v001 = wx0 * wy0 * wz1;
                        float v010 = wx0 * wy1 * wz0;
                        float v011 = wx0 * wy1 * wz1;
                        float v100 = wx1 * wy0 * wz0;
                        float v101 = wx1 * wy0 * wz1;
                        float v110 = wx1 * wy1 * wz0;
                        float v111 = wx1 * wy1 * wz1;

                        auto val = 
                            v000 * get_value(ix0,iy0,iz0)+
                            v001 * get_value(ix0,iy0,iz1)+
                            v010 * get_value(ix0,iy1,iz0)+
                            v011 * get_value(ix0,iy1,iz1)+
                            v100 * get_value(ix1,iy0,iz0)+
                            v101 * get_value(ix1,iy0,iz1)+
                            v110 * get_value(ix1,iy1,iz0)+
                            v111 * get_value(ix1,iy1,iz1);
                        int output_shift = ((n * C + c) * D + od) * H * W + h*W + w;
                        *(output_data + output_shift) = val;
                    }
                }
            }
        }
    }
    
    return true;
}

void printData(float* output_datas){
    for (size_t i = 990; i < 1000; i++)
    {
        std::cout << *(output_datas+i) << (i==999 ? "" : " | ");
    }
    std::cout << std::endl;
}

int main(){
    
    string grid("..\\data\\grid.txt");
    string input("..\\data\\input.txt");
    string output("..\\data\\output.txt");
    vector<float> input_data = readDataFromFile(input.c_str());
    vector<float> grid_data = readDataFromFile(grid.c_str());
    vector<float> output_data = readDataFromFile(output.c_str());
    std::cout << "Shape:  \\\\\\  [22* 4* 16* 64* 64]*[22, 16, 64, 64, 3]  ///" << std::endl;
    // torch.nn.functional.grid_sample 计算结果
    assert(output_data.size() == 22* 4* 16* 64* 64);
    cv::Mat output_mat(1,output_data.size(),CV_32FC1,output_data.data());
    printData(output_data.data());

    // C++ mathGridsample3D 计算结果
    auto start = std::chrono::steady_clock::now();
    Tensor3D gridsample_input(22, 4, 16, 64, 64);
    Grid3D gridsample_grid(22, 16, 64, 64);
    gridsample_input.setData(input_data);
    gridsample_grid.setData(grid_data);
    Tensor3D gridsample_output = mathGridsample3D(gridsample_input,gridsample_grid,false);
    auto end = std::chrono::steady_clock::now();
    double cost_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << std::format("GridSample3D : ----------------- C++ used time : {} ms -----------------",cost_time) << std::endl;
    cv::Mat gridsample_mat(1,output_data.size(),CV_32FC1,gridsample_output.ptr());
    printData(gridsample_output.ptr());
    // CUDA grid_sample_3d_cuda 计算结果
    start = std::chrono::steady_clock::now();
    float* cuda_out = new float[22* 4* 16* 64* 64];
    float* gpuinbuffer[2];
    float* gpuoutbuffer;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMalloc(&gpuinbuffer[0],input_data.size() * sizeof(float));
    cudaMalloc(&gpuinbuffer[1],grid_data.size() * sizeof(float));
    cudaMalloc(&gpuoutbuffer,output_data.size() * sizeof(float));
    cudaMemcpyAsync(gpuinbuffer[0], input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(gpuinbuffer[1], grid_data.data(), grid_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    int state_code = grid_sample_3d_cuda<float>(gpuinbuffer[0],gpuinbuffer[1],22, 4, 16, 64, 64,16, 64, 64,0,"bilinear","zeros",gpuoutbuffer,stream);
    if(state_code!=0){
        delete[] cuda_out;
        cudaFree(gpuinbuffer[0]);
        cudaFree(gpuinbuffer[1]);
        cudaFree(gpuoutbuffer);
        return -1;
    }
    cudaMemcpyAsync(cuda_out, gpuoutbuffer, output_data.size() * sizeof(float), cudaMemcpyDeviceToHost, stream);
    end = std::chrono::steady_clock::now();
    cost_time = std::chrono::duration<double, std::milli>(end - start).count();
    cv::Mat cuda_mat(1,output_data.size(),CV_32FC1,cuda_out);
    printData(cuda_out);
    std::cout << std::format("GridSample3D : ----------------- CUDA used time : {} ms  -----------------",cost_time) << std::endl;


    // Openvino grid_sample 计算结果
    start = std::chrono::steady_clock::now();
    std::vector<float> ov_out(22*4*16*64*64);
    evaluate(input_data.data(),grid_data.data(),ov_out.data());
    end = std::chrono::steady_clock::now();
    cost_time = std::chrono::duration<double, std::milli>(end - start).count();
    cv::Mat ov_mat(1,ov_out.size(),CV_32FC1,ov_out.data());
    printData(ov_out.data());
    std::cout << std::format("GridSample3D : ----------------- OpenVino used time : {} ms  -----------------",cost_time) << std::endl;




    //相似度
    std::cout << "cosine_similarity  Torch VS C++  :" << cosineSimilarity(output_mat,gridsample_mat) << std::endl;
    std::cout << "cosine_similarity  Torch VS CUDA :" << cosineSimilarity(output_mat,cuda_mat) << std::endl;
    std::cout << "cosine_similarity  C++   VS CUDA :" << cosineSimilarity(gridsample_mat,cuda_mat) << std::endl;
    std::cout << "cosine_similarity  C++   VS opencv :" << cosineSimilarity(gridsample_mat,ov_mat) << std::endl;

    delete[] cuda_out;
    cudaFree(gpuinbuffer[0]);
    cudaFree(gpuinbuffer[1]);
    cudaFree(gpuoutbuffer);
    return 0;
}