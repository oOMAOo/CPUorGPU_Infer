#include "addgridsampleplugin.hpp"
#include "Eigen/Eigen"
#include "opencv2/opencv.hpp"
namespace TemplateExtension{
    
AddGridSamplePlugin::AddGridSamplePlugin(
        const ov::Output<ov::Node>& input, 
        const ov::Output<ov::Node>& grid) : ov::op::Op({input,grid}) {
    constructor_validate_and_infer_types();
}

void AddGridSamplePlugin::validate_and_infer_types(){
    auto output_type = get_input_element_type(0);
    const ov::PartialShape& input_shape = get_input_partial_shape(0);
    const ov::PartialShape& grid_shape = get_input_partial_shape(1);


    set_output_type(
        0, 
        output_type, 
        {
            input_shape[0],// N
            input_shape[1],// C
            grid_shape[1], // D
            grid_shape[2], // H
            grid_shape[3]  // W
        }
    );
}
std::shared_ptr<ov::Node> AddGridSamplePlugin::clone_with_new_inputs(const ov::OutputVector& inputs) const{
    OPENVINO_ASSERT(inputs.size() == 2, "Incorrect number of new arguments");
    return std::make_shared<AddGridSamplePlugin>(inputs[0],inputs[1]);
}
bool AddGridSamplePlugin::visit_attributes(ov::AttributeVisitor& visitor){
    return true;
}
bool AddGridSamplePlugin::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const{
    ov::Tensor& out_0 = outputs[0];
    float *input_data = static_cast<float *>(inputs[0].data());
    float *grid_data = static_cast<float *>(inputs[1].data());
    const auto& input_shape = inputs[0].get_shape();
    const auto& grid_shape = inputs[1].get_shape();
    
    float* output_data = static_cast<float*>(out_0.data());
    const size_t N = input_shape[0];
    const size_t C = input_shape[1];
    const size_t I_D = input_shape[2];
    const size_t I_H = input_shape[3];
    const size_t I_W = input_shape[4];
    const size_t O_D = grid_shape[1];
    const size_t O_H = grid_shape[2];
    const size_t O_W = grid_shape[3];
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
            for (int d = 0; d < O_D; ++d) {
                int grid_mat_shift = (n * O_D + d) * O_H  * O_W * G_C;
                cv::Mat grid_mat(O_H,O_W,CV_32FC3,(grid_data + grid_mat_shift));
                std::vector<cv::Mat>grid_channels;
                cv::split(grid_mat,grid_channels);
                cv::Mat x_channel = grid_channels[0];
                cv::Mat y_channel = grid_channels[1];
                cv::Mat z_channel = grid_channels[2];
                x_channel = ((x_channel + 1.0f) * I_W - 1.0f) / 2.0f;
                y_channel = ((y_channel + 1.0f) * I_H - 1.0f) / 2.0f;
                z_channel = ((z_channel + 1.0f) * I_D - 1.0f) / 2.0f;
                for (int h = 0; h < O_H; ++h) {
                    for (int w = 0; w < O_W; ++w) {
                        float ix = x_channel.at<float>(h,w);
                        float iy = y_channel.at<float>(h,w);
                        float iz = z_channel.at<float>(h,w);

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
                        int output_shift = (((n * C + c) * O_D + d) * O_H + h) * O_W + w;
                        *(output_data + output_shift) = val;
                    }
                }
            }
        }
    }
    
    out_0.set_shape({N,C,O_D,O_H,O_W});
    return true;
}
bool AddGridSamplePlugin::has_evaluate() const {
    return true;
}

}  // namespace TemplateExtension