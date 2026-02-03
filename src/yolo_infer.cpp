
#include <direct.h>
#include <io.h>

#include <iostream>
#include <fstream>
#include <ctime>

#include <opencv2/opencv.hpp>

#include "cuda_fun.hpp"
#include "inference_api.hpp" 
#include "openvino_infer.hpp"
#include "tensorrt_infer.hpp"
#include "onnxruntime_infer.hpp"
#include "inference_common.hpp"
#include <Eigen/Dense>

#define INPUT_W 1280
#define INPUT_H 1280
#define SEG_W 320
#define SEG_H 320

#define CLASSES 3
#define SEG_CHANNELS 32
#define BOX_NUM 33600
#define SCORE_THRESHOLD 0.8
#define NMS_THRESHOLD 0.8

struct OutputSeg
{
    int id;
    float confidence;
    cv::Rect box;
    cv::Mat boxMask;
};
void drawPred(cv::Mat& img,std::vector<OutputSeg> result) {
    //给每个类别创建颜色
    std::vector<cv::Scalar> color;
    for (int i = 0; i < CLASSES; i++) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        color.push_back(cv::Scalar(b, g, r));
    }
    cv::Mat mask = img.clone();
    for (int i = 0; i < result.size(); i++) {
        cv::rectangle(img, result[i].box, color[result[i].id], 2, 8);
        std::cout << result[i].boxMask.size() << " | " << result[i].boxMask.channels() << std::endl;
        std::cout << img.size() << " | " << img.channels() << std::endl;
        mask(result[i].box).setTo(color[result[i].id], result[i].boxMask(result[i].box));
    }
    
    cv::addWeighted(img, 0.5, mask, 0.8, 1, img); //将mask加在原图上面

}

int main(){
    std::string input_img = "D:\\workplace\\YOLO_Python\\input.jpg";
    std::string model_path = "D:\\workplace\\YOLO_Python\\model_openvino_model\\model.xml";

    if(!std::filesystem::is_regular_file(std::filesystem::path(std::u8string(input_img.begin(),input_img.end()))) || !std::filesystem::is_regular_file(std::u8string(model_path.begin(),model_path.end()))){
        std::cerr << "<(E`_`E)> Model path/Image path is invalid, please check your path." << std::endl;;
        return -1;
    }
    InferenceApi* api_tool = nullptr;

    //使用CUDA函数检测是否存在NVIDIA设备且计算能力需要是8.9（后续再补全）  目前仅做简单检测
    //TODO 增加显卡版本检测
    api_tool = new OpenVinoInfer();
    
    // api_tool = new OnnxRuntimeInfer();

    // ==================== 1. 创建基础推理引擎 ====================
    api_tool->CreateInferenceEngine();

    using LayoutShape = std::pair<std::string,std::vector<size_t>>;
    std::vector<LayoutShape> input_layouts;
    input_layouts.emplace_back(LayoutShape{"NCHW", {1,3,INPUT_H,INPUT_W}});
    // input_layouts.emplace_back(LayoutShape{"NCHW", {1,3,INPUT_H,INPUT_W}});

    std::vector<LayoutShape> output_layouts;
    const size_t net_length = CLASSES + 4 + SEG_CHANNELS; // 39 
    output_layouts.emplace_back(LayoutShape{"NC...", {1,net_length,BOX_NUM}});
    output_layouts.emplace_back(LayoutShape{"NCHW", {1,SEG_CHANNELS,SEG_H,SEG_W}});
    if (!std::filesystem::exists(MODELPATH))
    {
        if(!std::filesystem::create_directory(MODELPATH)){
            std::cerr << std::format("(E`_`E)> create directory [{}] in error...",MODELPATH) << std::endl;
            return -1;
        }
    }
    // ==================== 2. 加载模型文件 ====================
    ResultData<std::string> load_state = api_tool->LoadModel(model_path,input_layouts,output_layouts);
    if(!load_state.result_state){
        return -1;
    }
    // ==================== 3. 加载引擎文件，创建识别引擎 ====================
    ResultData<bool> engine_state = api_tool->CreateEngine(load_state.result_info);
    if(!engine_state.result_state){
        return -1;
    }

    // ==================== 4. 准备输入数据 ====================
    std::vector<std::vector<float>> output_datas;
    cv::Mat src = cv::imread(std::filesystem::path(std::u8string(input_img.begin(),input_img.end())).string(), cv::IMREAD_COLOR_RGB);
    cv::Mat image= src.clone();
    if (image.empty()) {
        std::cout << std::format("<(E`_`E)> Can't Open the image: {}\n", input_img);
        return -1;
    }
    std::cout << std::format("\n(I_I)>>>>> Input image size : {} x {}\n", INPUT_W, INPUT_H);

    int pad_r = 0;
    int pad_b = 0;
    cv::Size resize_size;
    inference_common::resizeImageWithPadding(image,cv::Size(INPUT_W, INPUT_H),resize_size,pad_r,pad_b);

    cv::Mat blob = cv::dnn::blobFromImage(image, 1.0f/255.0f,cv::Size(INPUT_W, INPUT_H));

    //恢复比例系数
    const float radio = std::max(static_cast<float>(src.cols) / INPUT_W,static_cast<float>(src.rows) / INPUT_H);
    // ==================== 5. 推理 ====================

    auto infer_start = std::chrono::steady_clock::now();
    bool output_state = api_tool->Infer({reinterpret_cast<float*>(blob.data)},output_datas);
    auto infer_end = std::chrono::steady_clock::now();
    double infer_time = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    std::cout << std::format("----------------- Infer image used time : {} ms -----------------\n", infer_time);

    //======================================= Use Eigen ==========================================
    //每个分类做一个Mat
    std::vector<Eigen::MatrixXf> type_bounding_boxs;
    // 33600行 39列 33600*39   32* (320*320)
    auto start = std::chrono::steady_clock::now();
    Eigen::Map<Eigen::MatrixXf>out1_mat(output_datas[0].data(), BOX_NUM, net_length);
    for (int i = 4; i < 4+CLASSES; i++)
    {
        // 实际类别索引
        int type = i-4;
        Eigen::Array<bool, BOX_NUM, 1> condition_cols = (out1_mat.col(i).array() > SCORE_THRESHOLD);
        int count = condition_cols.count();
        Eigen::MatrixXf result(count, net_length);
        int index = 0;
        for (size_t r = 0; r < BOX_NUM; r++)
        {
            if(condition_cols(r)){
                result.row(index++) = out1_mat.row(r);
            }
        }
        type_bounding_boxs.push_back(result);
    }
    Eigen::Map<Eigen::MatrixXf>mask_mat(output_datas[1].data(), SEG_H*SEG_W, SEG_CHANNELS);//320*320  * 32
    std::cout << mask_mat.rows() << "*" << mask_mat.cols() <<std::endl;
    std::vector<int> _ids;//结果每个id对应置信度数组
    std::vector<float> _scores;//结果每个id对应置信度数组
    std::vector<cv::Rect> _boxes;//每个id矩形框
    std::vector<Eigen::MatrixXf> _proposals;  //mask权重 1*32

    for (int type_id = 0; type_id < type_bounding_boxs.size(); type_id++)
    {
        Eigen::MatrixXf& proposal_mat = type_bounding_boxs[type_id];
        for (int row_i = 0; row_i < type_bounding_boxs[type_id].rows(); row_i++)
        {
            
            float cx = (proposal_mat(row_i,0)) * radio;  //cx
            float cy = (proposal_mat(row_i,1)) * radio;  //cy
            float w = proposal_mat(row_i,2) * radio;  //w
            float h = proposal_mat(row_i,3)  * radio;  //h
            int left = MAX((cx - 0.5 * w), 0);
            int top = MAX((cy - 0.5 * h), 0);
            int width = static_cast<int>(std::round(w));
            int height = static_cast<int>(std::round(h));
            if (width <= 0 || height <= 0) { continue; }
            _ids.emplace_back(type_id);
            _scores.emplace_back(proposal_mat(row_i,4+type_id));
            _boxes.emplace_back(cv::Rect(left, top, width, height));
            _proposals.emplace_back(proposal_mat.block(row_i,7,1,SEG_CHANNELS));
        }
    }
    //NMS消除冗余重叠框
    std::vector<int> _nms_result;
    cv::dnn::NMSBoxes(_boxes, _scores, SCORE_THRESHOLD, NMS_THRESHOLD, _nms_result);
    cv::Rect holeImgRect(0, 0, src.cols, src.rows);
    std::vector<OutputSeg> output;
    for (size_t i = 0; i < _nms_result.size(); i++)
    {
        int idx = _nms_result[i];
        OutputSeg result;
        result.id = _ids[idx];
        result.confidence = _scores[idx];
        result.box = _boxes[idx]&holeImgRect;

        Eigen::MatrixXf result_mask_mat = mask_mat * _proposals[idx].transpose();
        cv::Mat mask_image(SEG_H,SEG_W,CV_32F,result_mask_mat.data());
        cv::Mat mask;
        cv::resize(mask_image, mask, cv::Size(image.cols*radio, image.rows*radio), cv::INTER_NEAREST);
        
        mask = mask(cv::Rect(0,0,src.cols,src.rows));
        cv::Rect temp_rect = result.box;
        mask.convertTo(mask,CV_8U);
        result.boxMask = mask;
        output.push_back(result);
    }
    auto end = std::chrono::steady_clock::now();
    double cost_time = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << std::format("----------------- postprocess used time : {} ms -----------------", cost_time) << std::endl;
    cv::cvtColor(src,src,cv::COLOR_RGB2BGR);
    drawPred(src,output);
    // cv::imshow("result.jpg",src);
    // cv::waitKey(0);
    

    //======================================= Not Use Eigen ==========================================
    
    // auto postprocess_start = std::chrono::steady_clock::now();
    // std::vector<int> class_ids;//结果id数组
    // std::vector<float> confidences;//结果每个id对应置信度数组
    // std::vector<cv::Rect> boxes;//每个id矩形框
    // std::vector<cv::Mat> proposals;  //后续计算mask

    // // 39行 33600列 每一列都是一个box
    // cv::Mat out1 = cv::Mat(net_length, BOX_NUM, CV_32F, output_datas[0].data());

    // for (int i = 0; i < BOX_NUM; i++) {
    //     //类别置信度  4 ~ 4+CLASSES
    //     cv::Mat scores = out1(cv::Rect(i, 4, 1, CLASSES)).clone();
    //     cv::Point classIdPoint;
    //     double max_class_socre;
    //     //得到最高置信度
    //     cv::minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
    //     max_class_socre = (float)max_class_socre;
    //     if (max_class_socre >= SCORE_THRESHOLD) {
    //         //1*32
    //         cv::Mat temp_proto = out1(cv::Rect(i, 4 + CLASSES, 1, SEG_CHANNELS)).clone();
    //         proposals.push_back(temp_proto.t());
    //         //转换到原图的坐标
    //         float cx = (out1.at<float>(0, i) - pad_l) * radio;  //cx
    //         float cy = (out1.at<float>(1, i) - pad_t) * radio;  //cy
    //         float w = out1.at<float>(2, i) * radio;  //w
    //         float h = out1.at<float>(3, i) * radio;  //h
    //         int left = MAX((cx - 0.5 * w), 0);
    //         int top = MAX((cy - 0.5 * h), 0);
    //         int width = static_cast<int>(std::round(w));
    //         int height = static_cast<int>(std::round(h));
    //         if (width <= 0 || height <= 0) { continue; }
    //         class_ids.push_back(classIdPoint.y);
    //         confidences.push_back(max_class_socre);
    //         boxes.push_back(cv::Rect(left, top, width, height));
    //     }
    // }
    // //NMS消除冗余重叠框
    // std::vector<int> nms_result;
    // cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    // std::vector<cv::Mat> mask_proposals;
    // // std::vector<OutputSeg> output;
    // // cv::Rect holeImgRect(0, 0, src.cols, src.rows);
    // for (int i = 0; i < nms_result.size(); ++i) {
    //     int idx = nms_result[i];
    //     OutputSeg result;
    //     result.id = class_ids[idx];
    //     result.confidence = confidences[idx];
    //     result.box = boxes[idx]&holeImgRect;
    //     output.push_back(result);
    //     //1*32
    //     mask_proposals.push_back(proposals[idx]);
    // }
    // cv::Mat mask_proposals_mat;
    // // n*32
    // for (int i = 0; i < mask_proposals.size(); ++i) {
    //     mask_proposals_mat.push_back(mask_proposals[i]);
    // }

    // cv::Mat protos = cv::Mat(SEG_CHANNELS, SEG_W * SEG_H, CV_32F, output_datas[1].data());

    // //n*32 32*33600 => n*33600 => 33600*n 每个候选框作为一个通道
    // cv::Mat matmul_result = (mask_proposals_mat * protos).t();
    // cv::Mat masks = matmul_result.reshape(output.size(), { SEG_W,SEG_H });//320*320*n

    // //分离候选框
    // std::vector<cv::Mat> maskChannels;
    // cv::split(masks, maskChannels);

    // for (int i = 0; i < output.size(); ++i) {
    //     cv::Mat dest, mask;
    //     //sigmoid => 0~1
    //     cv::exp(-maskChannels[i], dest);
    //     dest = 1.0 / (1.0 + dest);

    //     cv::resize(dest, mask, cv::Size(image.cols*radio, image.rows*radio), cv::INTER_NEAREST);
    //     int paddint_left = static_cast<float>(std::round(pad_l*radio));
    //     int paddint_top = static_cast<float>(std::round(pad_t*radio));

    //     mask = mask(cv::Rect(paddint_left,paddint_top,mask.cols-paddint_left,mask.rows-paddint_top));
    //     cv::Rect temp_rect = output[i].box;
    //     mask = mask(temp_rect) > 0.8;
    //     output[i].boxMask = mask;
    // }
    // auto postprocess_end = std::chrono::steady_clock::now();
    // double postprocess_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
    // std::cout << std::format("----------------- postprocess used time : {} ms -----------------", postprocess_time) << std::endl;;
    // cv::cvtColor(src,src,cv::COLOR_RGB2BGR);
    // drawPred(src,output);
    // cv::imwrite("result.jpg",src);
    output_datas.clear();
    output_datas.shrink_to_fit();
    delete api_tool;
    return 0;
}
