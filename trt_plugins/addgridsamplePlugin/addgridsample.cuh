#pragma once
#include <string>

#include <cuda_runtime.h>
//双线性插值 临近插值
enum class MyGridSampleInterpolationMode{ Bilinear, Nearest};
enum class MyGridSamplePaddingMode{ Zeros, Border, Reflection};
//全精 半精
enum class MyGridSampleDataType {GFLOAT, GHALF};

template <typename scalar_t>
int grid_sample_3d_cuda(
    const scalar_t* input,
    const scalar_t* grid,
    size_t N, size_t C, size_t D_in, size_t H_in, size_t W_in,
    size_t D_grid, size_t H_grid, size_t W_grid,
    int align_corners,
    std::string interpolationMode,
    std::string paddingMode,
    scalar_t* output,
    cudaStream_t stream
);


