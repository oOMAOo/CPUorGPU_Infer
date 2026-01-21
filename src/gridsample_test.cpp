#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "infer_math.hpp"
#include "addgridsample.cuh"
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

int main(){
    
    string grid(".\\shape4.txt");
    string input(".\\shape3.txt");
    string output(".\\out.txt");
    vector<float> input_data = readDataFromFile(input.c_str());
    vector<float> grid_data = readDataFromFile(grid.c_str());
    vector<float> output_data = readDataFromFile(output.c_str());
    std::cout << "Shape:  \\\\\\  [22* 4* 16* 64* 64]*[22, 16, 64, 64, 3]  ///" << std::endl;
    // torch.nn.functional.grid_sample 计算结果
    assert(output_data.size() == 22* 4* 16* 64* 64);
    cv::Mat output_mat(1,output_data.size(),CV_32FC1,output_data.data());

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
    std::cout << std::format("GridSample3D : ----------------- CUDA used time : {} ms  -----------------",cost_time) << std::endl;


    cv::Mat cuda_mat(1,output_data.size(),CV_32FC1,cuda_out);

    //相似度
    std::cout << "cosine_similarity  Torch VS C++  :" << cosineSimilarity(output_mat,gridsample_mat) << std::endl;
    std::cout << "cosine_similarity  Torch VS CUDA :" << cosineSimilarity(output_mat,cuda_mat) << std::endl;
    std::cout << "cosine_similarity  C++   VS CUDA :" << cosineSimilarity(gridsample_mat,cuda_mat) << std::endl;

    delete[] cuda_out;
    cudaFree(gpuinbuffer[0]);
    cudaFree(gpuinbuffer[1]);
    cudaFree(gpuoutbuffer);
    return 0;
}