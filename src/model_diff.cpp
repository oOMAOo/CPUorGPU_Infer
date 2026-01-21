
#include <direct.h>
#include <io.h>

#include <iostream>
#include <fstream>
#include <ctime>
#include <cassert>
#include <opencv2/opencv.hpp>

#include "cuda_fun.hpp"
#include "inference_api.hpp" 
#include "openvino_infer.hpp"
#include "tensorrt_infer.hpp"
#include "onnxruntime_infer.hpp"
#include "inference_common.hpp"
#include "infer_math.hpp"
#include "addgridsample.cuh"
#define INPUT_W 512
#define INPUT_H 512
class MyLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

double cosineSimilarity(cv::Mat& feature1, cv::Mat& feature2) {
    CV_Assert(feature1.rows == 1 && feature2.rows == 1 && feature1.cols == feature2.cols);
    double dot = feature1.dot(feature2);
    double norm1 = cv::norm(feature1);
    double norm2 = cv::norm(feature2);
    return dot / (norm1 * norm2);
}

inline std::vector<float> readDataFromFile(const char* file_path){
    std::ifstream file(file_path);
    if(!file.is_open()){
        std::cout << file_path << " Open in Error!" << std::endl;
    }
    std::vector<float> data;
    std::string line;
    while (std::getline(file,line))
    {
        data.push_back(atof(line.c_str()));
    }
    file.close();
    return std::move(data);
}

// 加载模型文件
std::vector<unsigned char> loadEngineFile(const std::string& file_name)
{
    std::vector<unsigned char> engine_data;
    std::ifstream engine_file(file_name, std::ios::binary);
    assert(engine_file.is_open() && "Unable to load engine file.");
    engine_file.seekg(0, engine_file.end);
    int length = engine_file.tellg();
    engine_data.resize(length);
    engine_file.seekg(0, engine_file.beg);
    engine_file.read(reinterpret_cast<char*>(engine_data.data()), length);
    return engine_data;
}

void deserializeEngine(std::string& engine_name, nvinfer1::IRuntime** runtime, nvinfer1::ICudaEngine** engine, nvinfer1::IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    auto plan = loadEngineFile(engine_name);
    MyLogger logger;
    *runtime = nvinfer1::createInferRuntime(logger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(plan.data(), plan.size());
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    plan.clear();
}


int main(){
    //模型 带插件
    auto plugin_Model_start = std::chrono::steady_clock::now();
    float* plugin_cpu_output = new float[1*3*512*512];
    float* split_cpu_output = new float[1*3*512*512];
    {
        std::string plugin_model = "model/warping_spade1.trt";
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
        //反序列化引擎文件
        deserializeEngine(plugin_model, &runtime, &engine, &context);
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));



        // 输入数据
        constexpr int model_1_input1_size = 1*32*16*64*64;
        constexpr int model_1_input2_size = 1*21*3;
        constexpr int model_1_input3_size = 1*21*3;
        std::string feature_3d("..\\data\\feature_3d.txt");
        std::string kp_driving("..\\data\\kp_driving.txt");
        std::string kp_source("..\\data\\kp_source.txt");
        std::vector<float> feature_3d_data = readDataFromFile(feature_3d.c_str());
        std::vector<float> kp_driving_data = readDataFromFile(kp_driving.c_str());
        std::vector<float> kp_source_data = readDataFromFile(kp_source.c_str());

        assert(feature_3d_data.size() == model_1_input1_size && kp_driving.size() == model_1_input2_size && kp_source_data.size() == model_1_input3_size);

        //准备 GPU 输入输出 Buffer
        float* input_gpu_buffers[3];
        float* output_gpu_buffer[1];

        //输入 GPU显存
        cudaMalloc((void**)&input_gpu_buffers[0], model_1_input1_size * sizeof(float));
        cudaMalloc((void**)&input_gpu_buffers[1], model_1_input2_size * sizeof(float));
        cudaMalloc((void**)&input_gpu_buffers[2], model_1_input3_size * sizeof(float));
        cudaMemcpyAsync(input_gpu_buffers[0], feature_3d_data.data(), model_1_input1_size * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(input_gpu_buffers[1], kp_driving_data.data(), model_1_input2_size * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(input_gpu_buffers[2], kp_source_data.data(), model_1_input3_size * sizeof(float), cudaMemcpyHostToDevice, stream);
        context->setInputTensorAddress("feature_3d", input_gpu_buffers[0]);
        context->setInputTensorAddress("kp_driving", input_gpu_buffers[1]);
        context->setInputTensorAddress("kp_source", input_gpu_buffers[2]);

        //输出 GPU显存
        cudaMalloc((void**)&output_gpu_buffer[0], 1*3*512*512 * sizeof(float));
        context->setOutputTensorAddress("out", output_gpu_buffer[0]);
        std::cout << "Plugin : Model start running enqueue..." << std::endl;
        if (!context->enqueueV3(stream)) {
            std::cerr << "Error: Inference run failed!" << std::endl;
            return -1;
        }
        //Copy输出
        cudaMemcpyAsync(plugin_cpu_output, output_gpu_buffer[0], 1*3*512*512 * sizeof(float), cudaMemcpyDeviceToHost,stream);

        cudaStreamSynchronize(stream);

        cudaFree(input_gpu_buffers[0]);
        cudaFree(input_gpu_buffers[1]);
        cudaFree(input_gpu_buffers[2]);
        cudaFree(output_gpu_buffer[0]);
    }
    auto plugin_Model_end = std::chrono::steady_clock::now();
    double plugin_Model_cost_time = std::chrono::duration<double,std::milli>(plugin_Model_end-plugin_Model_start).count();
    std::cout << std::format("******************* Plugin Model cost time : {} ms *******************", plugin_Model_cost_time) << std::endl;


    auto Split_Model_start = std::chrono::steady_clock::now();
    std::cout << "\n================================ Split Model ====================================" << std::endl;
    {
        //=========================== 阶段1 ==============================
        std::string plugin_model = "model/split_spade.trt";
        nvinfer1::IRuntime* runtime = nullptr;
        nvinfer1::ICudaEngine* engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
        //反序列化引擎文件
        deserializeEngine(plugin_model, &runtime, &engine, &context);
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        constexpr int model_1_input1_size = 1*32*16*64*64;
        constexpr int model_1_input2_size = 1*21*3;
        constexpr int model_1_input3_size = 1*21*3;
        std::string feature_3d("..\\data\\feature_3d.txt");
        std::string kp_driving("..\\data\\kp_driving.txt");
        std::string kp_source("..\\data\\kp_source.txt");
        std::vector<float> feature_3d_data = readDataFromFile(feature_3d.c_str());
        std::vector<float> kp_driving_data = readDataFromFile(kp_driving.c_str());
        std::vector<float> kp_source_data = readDataFromFile(kp_source.c_str());
        assert(feature_3d_data.size() == model_1_input1_size && kp_driving.size() == model_1_input2_size && kp_source_data.size() == model_1_input3_size);

        //准备 GPU 输入输出 Buffer
        float* input_gpu_buffers[3];
        float* output_gpu_buffer[4];

        //输入 GPU显存
        cudaMalloc((void**)&input_gpu_buffers[0], model_1_input1_size * sizeof(float));
        cudaMalloc((void**)&input_gpu_buffers[1], model_1_input2_size * sizeof(float));
        cudaMalloc((void**)&input_gpu_buffers[2], model_1_input3_size * sizeof(float));
        cudaMemcpyAsync(input_gpu_buffers[0], feature_3d_data.data(), model_1_input1_size * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(input_gpu_buffers[1], kp_driving_data.data(), model_1_input2_size * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(input_gpu_buffers[2], kp_source_data.data(), model_1_input3_size * sizeof(float), cudaMemcpyHostToDevice, stream);
        context->setInputTensorAddress("feature_3d", input_gpu_buffers[0]);
        context->setInputTensorAddress("kp_driving", input_gpu_buffers[1]);
        context->setInputTensorAddress("kp_source", input_gpu_buffers[2]);

        //输出 GPU显存
        cudaMalloc((void**)&output_gpu_buffer[0], 22*4*16*64*64 * sizeof(float));
        cudaMalloc((void**)&output_gpu_buffer[1], 22*16*64*64*3 * sizeof(float));
        cudaMalloc((void**)&output_gpu_buffer[2], 1*21*16*64*64 * sizeof(float));
        cudaMalloc((void**)&output_gpu_buffer[3], 1*22*16*64*64*3 * sizeof(float));
        context->setOutputTensorAddress("/dense_motion_network/Reshape_3_output_0", output_gpu_buffer[0]);
        context->setOutputTensorAddress("/dense_motion_network/Reshape_4_output_0", output_gpu_buffer[1]);
        context->setOutputTensorAddress("/dense_motion_network/Sub_3_output_0", output_gpu_buffer[2]);
        context->setOutputTensorAddress("/dense_motion_network/Concat_1_output_0", output_gpu_buffer[3]);

        //推理
        std::cout << "Step 1 : Model start running enqueue..." << std::endl;
        auto step1_start = std::chrono::steady_clock::now();
        if (!context->enqueueV3(stream)) {
            std::cerr << "Error: Step 1 Inference run failed!" << std::endl;
            return -1;
        }
        auto step1_end = std::chrono::steady_clock::now();
        double step1_cost_time = std::chrono::duration<double, std::milli>(step1_end - step1_start).count();
        std::cout << std::format("Step 2 : ----------------- Infer image used time : {} ms -----------------",step1_cost_time) << std::endl;

        //接收数据
        float* cpu_step1_input = new float[22*4*16*64*64];
        float* cpu_step1_grid = new float[22*16*64*64*3];
        cudaMemcpyAsync(cpu_step1_input, output_gpu_buffer[0], 22*4*16*64*64 * sizeof(float), cudaMemcpyDeviceToHost,stream);
        cudaMemcpyAsync(cpu_step1_grid, output_gpu_buffer[1], 22*16*64*64*3 * sizeof(float), cudaMemcpyDeviceToHost,stream);

        cudaStreamSynchronize(stream);

        cudaFree(input_gpu_buffers[0]);
        cudaFree(input_gpu_buffers[1]);
        cudaFree(input_gpu_buffers[2]);
        cudaFree(output_gpu_buffer[0]);
        cudaFree(output_gpu_buffer[1]);
        // cudaFree(output_gpu_buffer[2]); 由于阶段2使用，暂时不释放
        // cudaFree(output_gpu_buffer[3]); 由于阶段2使用，暂时不释放
        delete context;
        delete engine;
        delete runtime;
        InferMath::Tensor3D GridSample_step1_input(22, 4, 16, 64, 64);
        GridSample_step1_input.setData(cpu_step1_input,22*4*16*64*64);
        InferMath::Grid3D GridSample_step1_grid(22, 16, 64, 64);
        GridSample_step1_grid.setData(cpu_step1_grid,22*16*64*64*3);
        delete[] cpu_step1_input;
        delete[] cpu_step1_grid;

        
        //=========================== 阶段2 ==============================
        cudaStreamSynchronize(stream);
        plugin_model = "model/split_spade2.trt";
        deserializeEngine(plugin_model, &runtime, &engine, &context);

        //使用 第一阶段的数据
        auto GridSample1_start = std::chrono::steady_clock::now();
        auto GridSample1 = InferMath::mathGridsample3D(GridSample_step1_input,GridSample_step1_grid,false);
        auto GridSample1_end = std::chrono::steady_clock::now();
        double GridSample1_cost_time = std::chrono::duration<double, std::milli>(GridSample1_end - GridSample1_start).count();
        std::cout << std::format("GridSample1 : ----------------- mathGridsample3D used time : {} ms  [22, 4, 16, 64, 64]*[22, 16, 64, 64, 3] -----------------",GridSample1_cost_time) << std::endl;
        //准备 GPU 输入输出 Buffer
        float* model2_input_gpu_buffers[1];
        float* model2_output_gpu_buffer[2];

        //输入 GPU显存
        cudaMalloc((void**)&model2_input_gpu_buffers[0], GridSample1.size() * sizeof(float));
        cudaMemcpyAsync(model2_input_gpu_buffers[0], GridSample1.ptr(), GridSample1.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
        context->setInputTensorAddress("/dense_motion_network/GridSample_output_0", model2_input_gpu_buffers[0]);
        context->setInputTensorAddress("/dense_motion_network/Sub_3_output_0", output_gpu_buffer[2]);//复用
        context->setInputTensorAddress("/dense_motion_network/Concat_1_output_0", output_gpu_buffer[3]);//复用
        //输出 GPU显存
        cudaMalloc((void**)&model2_output_gpu_buffer[0], 1*142*16*64*64 * sizeof(float));
        cudaMalloc((void**)&model2_output_gpu_buffer[1], 1*16*64*64*3 * sizeof(float));
        context->setOutputTensorAddress("/dense_motion_network/hourglass/decoder/Relu_output_0", model2_output_gpu_buffer[0]);
        context->setOutputTensorAddress("/dense_motion_network/Transpose_1_output_0", model2_output_gpu_buffer[1]);

        //推理
        std::cout << "Step 2 : Model start running enqueue..." << std::endl;
        auto step2_start = std::chrono::steady_clock::now();
        if (!context->enqueueV3(stream)) {
            std::cerr << "Error: Step 2 Inference run failed!" << std::endl;
            return -1;
        }
        auto step2_end = std::chrono::steady_clock::now();
        double step2_cost_time = std::chrono::duration<double, std::milli>(step2_end - step2_start).count();
        std::cout << std::format("Step 2 : ----------------- Infer image used time : {} ms -----------------",step2_cost_time) << std::endl;

        //接收数据
        float* cpu_step2_grid = new float[1*16*64*64*3];
        cudaMemcpyAsync(cpu_step2_grid, model2_output_gpu_buffer[1], 1*16*64*64*3 * sizeof(float), cudaMemcpyDeviceToHost,stream);
        
        cudaStreamSynchronize(stream);

        cudaFree(output_gpu_buffer[2]);  //阶段1的显存
        cudaFree(output_gpu_buffer[3]);  //阶段1的显存
        cudaFree(model2_input_gpu_buffers[0]);
        // cudaFree(model2_output_gpu_buffer[0]); 由于阶段3使用，暂时不释放
        cudaFree(model2_output_gpu_buffer[1]);

        delete context;
        delete engine;
        delete runtime;
        InferMath::Tensor3D GridSample_step2_input(1, 32, 16, 64, 64);
        GridSample_step2_input.setData(feature_3d_data.data(),feature_3d_data.size());
        InferMath::Grid3D GridSample_step2_grid(1, 16, 64, 64);
        GridSample_step2_grid.setData(cpu_step2_grid,1*16*64*64*3);
        delete[] cpu_step2_grid;

        //=========================== 阶段3 ==============================
        cudaStreamSynchronize(stream);
        plugin_model = "model/split_spade3.trt";

        deserializeEngine(plugin_model, &runtime, &engine, &context);

        float* model3_input_gpu_buffers[1];//存放GridSample计算结果
        float* model3_output_gpu_buffer[1];
        auto GridSample2_start = std::chrono::steady_clock::now();
        auto GridSample2 = InferMath::mathGridsample3D(GridSample_step2_input,GridSample_step2_grid,false);
        auto GridSample2_end = std::chrono::steady_clock::now();
        double GridSample2_cost_time = std::chrono::duration<double, std::milli>(GridSample2_end - GridSample2_start).count();
        std::cout << std::format("GridSample2 : ----------------- mathGridsample3D used time : {} ms  [1, 32, 16, 64, 64]*[1, 16, 64, 64, 3] -----------------",GridSample2_cost_time) << std::endl;
        cudaMalloc((void**)&model3_input_gpu_buffers[0], GridSample2.size() * sizeof(float));
        cudaMemcpyAsync(model3_input_gpu_buffers[0], GridSample2.ptr(), GridSample2.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
        context->setInputTensorAddress("/dense_motion_network/hourglass/decoder/Relu_output_0", model2_output_gpu_buffer[0]);//复用
        context->setInputTensorAddress("/GridSample_output_0", model3_input_gpu_buffers[0]);

        cudaMalloc((void**)&model3_output_gpu_buffer[0], 1*3*512*512 * sizeof(float));
        context->setOutputTensorAddress("out", model3_output_gpu_buffer[0]);

        std::cout << "Step 3 : Model start running enqueue..." << std::endl;
        auto step3_start = std::chrono::steady_clock::now();
        if (!context->enqueueV3(stream)) {
            std::cerr << "Error: Step 2 Inference run failed!" << std::endl;
            return -1;
        }
        auto step3_end = std::chrono::steady_clock::now();
        double step3_cost_time = std::chrono::duration<double, std::milli>(step3_end - step3_start).count();
        std::cout << std::format("Step 3 : ----------------- Infer image used time : {} ms -----------------",step2_cost_time) << std::endl;
        cudaMemcpyAsync(split_cpu_output, model3_output_gpu_buffer[0], 1*3*512*512 * sizeof(float), cudaMemcpyDeviceToHost,stream);

        cudaStreamSynchronize(stream);


        cudaFree(model2_output_gpu_buffer[0]);  //阶段2的显存
        cudaFree(model3_input_gpu_buffers[0]);
        cudaFree(model3_output_gpu_buffer[0]);


    }
    auto Split_Model_end = std::chrono::steady_clock::now();
    double Split_Model_cost_time = std::chrono::duration<double,std::milli>(Split_Model_end-Split_Model_start).count();
    std::cout << std::format("******************* Split Model cost time : {} ms *******************\n", Split_Model_cost_time) << std::endl;


    //验证输出结果
    cv::Mat mat7(1,1*512*512*3,CV_32F,split_cpu_output);
    cv::Mat mat8(1,1*512*512*3,CV_32F,plugin_cpu_output);
    std::cout << "@@@ Finally cosineSimilarity: " << cosineSimilarity(mat7,mat8) << std::endl;
    auto output_image_opt = inference_common::DatatoImage(&split_cpu_output,512,512,3,255.0f);
    if(output_image_opt.has_value()){
        cv::Mat output_image = output_image_opt.value();
        cv::cvtColor(output_image, output_image, cv::COLOR_RGB2BGR);
        // 显示
        // cv::imshow("input", image);
        // cv::resize(output_image,output_image,image.size());
        cv::resize(output_image,output_image,cv::Size(512,512));
        cv::imshow("Output", output_image);
        cv::waitKey(0);
    }else {
        std::cerr << "DatatoImage Error" << std::endl;
    }

    delete[] split_cpu_output;
    delete[] plugin_cpu_output;
}