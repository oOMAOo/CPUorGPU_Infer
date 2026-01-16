#include "infer_math.hpp"
#include <cassert>
#include <cmath>
#include <Eigen/Dense>
namespace InferMath{

    inline float clamp(float x, float max_val, float min_val){
        return std::max(min_val,std::min(max_val,x));
    }

    //input :NCDHW grid:N DHW 3
    Tensor3D mathGridsample3D(Tensor3D input,Grid3D grid,bool align_corners){
        using namespace Eigen;
        assert(input.D == grid.D && input.H == grid.H && input.W == grid.W);
        int N = input.N;
        int C = input.C;
        int I_D = input.D;
        int I_H = input.H;
        int I_W = input.W;

        int D = grid.D;
        int H = grid.H;
        int W = grid.W;
        // input 转成N*C个 Egine 对象 （值copy）修改矩阵数据不影响源数据
        std::vector<MatrixXf> input_slices;
        for (size_t n = 0; n < N; n++)
        {
            for (size_t c = 0; c < C; c++)
            {
                for (size_t d = 0; d < I_D; d++)
                {
                    int mat_idx = n * C * I_D + c * I_D + d;
                    input_slices.emplace_back(Eigen::Map<Eigen::MatrixXf>(input.ptr(n, c, d, 0, 0), I_H, I_W));
                }
            }
        }
        //输出形状看 grid
        Tensor3D output(N, C, D, H, W);
        //遍历Grid
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int od = 0; od < D; ++od) {
                    Eigen::Map<Eigen::MatrixXf> output_map(output.ptr(n, c, od, 0, 0), H, W);
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            // 获取网格坐标
                            float grid_x = grid.at(n,od,h,w,0);
                            float grid_y = grid.at(n,od,h,w,1);
                            float grid_z = grid.at(n,od,h,w,2);
                            
                            //线性函数 计算坐标
                            float ix, iy, iz;
                            if(align_corners){
                                //[-1, 1] -> [0, I_W-1]
                                ix = ((grid_x + 1.0f) / 2.0f) * (I_W - 1.0f);
                                iy = ((grid_y + 1.0f) / 2.0f) * (I_H - 1.0f);
                                iz = ((grid_z + 1.0f) / 2.0f) * (I_D - 1.0f);
                            }else{
                                // [-0.5, I_W-0.5] 
                                ix = ((grid_x + 1.0f) * I_W - 1.0f) / 2.0f;
                                iy = ((grid_y + 1.0f) * I_H - 1.0f) / 2.0f;
                                iz = ((grid_z + 1.0f) * I_D - 1.0f) / 2.0f;
                            }
 
                            int ix0 = static_cast<int>(std::floor(ix));//小x坐标
                            int ix1 = ix0 + 1;//大x坐标

                            int iy0 = static_cast<int>(std::floor(iy));//小y坐标
                            int iy1 = iy0 + 1;//大y坐标

                            int iz0 = static_cast<int>(std::floor(iz));//小z坐标
                            int iz1 = iz0 + 1;//大z坐标
                            
                            // 根据坐标 计算权重
                            float wx0 = ix - ix0;// 距离小x的比例
                            float wx1 = 1.0f - wx0;// 距离大x的比例
                            float wy0 = iy - iy0;// 距离小y的比例
                            float wy1 = 1.0f - wy0;// 距离大y的比例
                            float wz0 = iz - iz0;// 距离小z的比例
                            float wz1 = 1.0f - wz0;// 距离大z的比例
                            
                            // 输入切片的索引
                            int slice_idx_base = n * C * I_D + c * I_D;
                            
                            // 边界检查函数
                            auto get_value = [&](int d_idx, int y_idx, int x_idx) -> float {
                                d_idx = std::clamp(d_idx, 0, I_D - 1);
                                y_idx = std::clamp(y_idx, 0, I_H - 1);
                                x_idx = std::clamp(x_idx, 0, I_W - 1);
                                int slice_idx = slice_idx_base + d_idx;
                                return input_slices[slice_idx](y_idx, x_idx);
                            };

                            //获取输入矩阵的原坐标内容
                            float v000 = get_value(iz0, iy0, ix0);
                            float v001 = get_value(iz0, iy0, ix1);
                            float v010 = get_value(iz0, iy1, ix0);
                            float v011 = get_value(iz0, iy1, ix1);
                            float v100 = get_value(iz1, iy0, ix0);
                            float v101 = get_value(iz1, iy0, ix1);
                            float v110 = get_value(iz1, iy1, ix0);
                            float v111 = get_value(iz1, iy1, ix1);
                            
                            output_map(h, w) = 
                                wz0 * wy0 * wx0 * v000 +
                                wz0 * wy0 * wx1 * v001 +
                                wz0 * wy1 * wx0 * v010 +
                                wz0 * wy1 * wx1 * v011 +
                                wz1 * wy0 * wx0 * v100 +
                                wz1 * wy0 * wx1 * v101 +
                                wz1 * wy1 * wx0 * v110 +
                                wz1 * wy1 * wx1 * v111;
                        }
                    }
                }
            }
        }
        return output;
    }
}