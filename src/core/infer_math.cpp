#include "infer_math.hpp"
#include <cassert>
#include <cmath>
#include <Eigen/Dense>
namespace InferMath{

    inline float clamp(float x, float max_val, float min_val){
        return std::max(min_val,std::min(max_val,x));
    }

    //input :NCDHcdW grid:N DHW 3
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
                    input_slices.emplace_back(Eigen::Map<Eigen::MatrixXf>(input.ptr(n, c, d, 0, 0), I_W, I_H));
                }
            }
        }
        //输出形状(D H W)看 grid
        Tensor3D output(N, C, D, H, W);
        //遍历Grid
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                for (int od = 0; od < D; ++od) {
                    Eigen::Map<Eigen::MatrixXf> output_map(output.ptr(n, c, od, 0, 0), W, H);
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
                            if(n==5 && od==9 && h==3 && c==1 && w>=50 && w<60){
                                // printf("CPU first Input / Grid Data W:[%d] grid_x[%.6f] grid_y[%.6f]  grid_z[%.6f] Idx[%d]\n",w,grid_x,grid_y,grid_z,((n * D + od) * H + h) * W * 3 + w * 3);
                                // printf("CPU first Input / Grid Data W:[%d] ix0[%d] iy0[%d]  iz0[%d] \n",w,ix0,iy0,iz0);
                                //printf("CPU first Input / Grid Data W:[%d] InputValue[%.6f] GridValue[%.6f]  OutputVal[%.6f] \n",w,*(input.ptr()+w),*(grid.ptr()+w),val);
                            }
                            output_map(w, h) = val;
                        }
                    }
                }
            }
        }
        return output;
    }
}