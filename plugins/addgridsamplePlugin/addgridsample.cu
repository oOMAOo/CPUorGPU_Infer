

#include "addgridsample.cuh"
#include "cuda_common.cuh"

#define NUM_THREADS 128

using half = __half;

inline int get_num_blocks(int n) {
    return (n + NUM_THREADS - 1) / NUM_THREADS;
}

template<typename scalar_t>
static __forceinline__ __device__
scalar_t reflect_coordinates(scalar_t in, int twice_low,
                                           int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = ::fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = ::fmod(in, span);
  int flips = static_cast<int>(::floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

template <typename scalar_t>
static __forceinline__ __device__
scalar_t compute_index(
    const scalar_t coord,
    const int size,
    const MyGridSamplePaddingMode padding_mode, 
    const int align_corners
) {
    // unnormalize coord from [-1, 1] to [0, size - 1] if align_corners = False
    // else unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    scalar_t coord_;
    if(align_corners > 0) {
        coord_ = ((coord + 1.f) / 2) * (size - 1);
    }else {
        coord_ = ((coord + 1.f) * size - 1) / 2;
    }

    // check if the coord_ is out of input boundary,
    // if so, make it back to the boundary based on padding_mode
    if (padding_mode == MyGridSamplePaddingMode::Border) { // border mode, clip to [0, size-1]
        coord_ = ::min(static_cast<scalar_t>(size-1), ::max(coord_, static_cast<scalar_t>(0)));
    } else if (padding_mode == MyGridSamplePaddingMode::Reflection) { // reflection mode
        if(align_corners > 0) {
            coord_ = reflect_coordinates(coord_, 0, 2 * (size - 1));
        } else {
            coord_ = reflect_coordinates(coord_, -1, 2 * size - 1);
        }
        coord_ = ::min(static_cast<scalar_t>(size-1), ::max(coord_, static_cast<scalar_t>(0)));
    }

    return coord_;
}



template <typename scalar_t>
__global__ void grid_sample_3d_nearest_kernel(
    const scalar_t* input,
    const scalar_t* grid,
    size_t N, size_t C, size_t D_in, size_t H_in, size_t W_in,
    size_t input_stride_N, size_t input_stride_C, size_t input_stride_D, size_t input_stride_H, size_t input_stride_W,
    size_t D_grid, size_t H_grid, size_t W_grid,
    size_t grid_stride_N, size_t grid_stride_D, size_t grid_stride_H, size_t grid_stride_W, size_t grid_stride_XYZ,
    size_t output_stride_N, size_t output_stride_C, size_t output_stride_D, size_t output_stride_H, size_t output_stride_W,
    int align_corners,
    MyGridSamplePaddingMode padding_mode_type,
    scalar_t* output    
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= N * D_grid * H_grid * W_grid) {
        return;
    }

    auto n = tid / (D_grid * H_grid * W_grid);
    auto d = (tid / (H_grid * W_grid)) % D_grid;
    auto h = (tid / W_grid) % H_grid;
    auto w = tid % W_grid;

    const scalar_t* input_N_offset = input + n * input_stride_N;
    const scalar_t* grid_N_offset = grid + n * grid_stride_N;
    scalar_t* output_N_offset = output + n * output_stride_N;

    const scalar_t* grid_NDHW_offset = grid_N_offset + d * grid_stride_D + h * grid_stride_H + w * grid_stride_W;
    const scalar_t x = *grid_NDHW_offset;
    const scalar_t y = *(grid_NDHW_offset + grid_stride_XYZ);
    const scalar_t z = *(grid_NDHW_offset + 2 * grid_stride_XYZ);

    // MyGridSamplePaddingMode padding_mode_type = padding_mode == "border" ? MyGridSamplePaddingMode::Border : 
    //                                             padding_mode == "reflection" ? MyGridSamplePaddingMode::Reflection : 
    //                                             MyGridSamplePaddingMode::Zeros;
    scalar_t ix = compute_index<scalar_t>(x, W_in, padding_mode_type, align_corners);
    scalar_t iy = compute_index<scalar_t>(y, H_in, padding_mode_type, align_corners);
    scalar_t iz = compute_index<scalar_t>(z, D_in, padding_mode_type, align_corners);    

    int ix_nearest = static_cast<int>(::roundf(ix));
    int iy_nearest = static_cast<int>(::roundf(iy));
    int iz_nearest = static_cast<int>(::roundf(iz));

    scalar_t *input_NC_offset = const_cast<scalar_t *>(input_N_offset);
    scalar_t *output_NCDHW_offset = output_N_offset + d * output_stride_D + h * output_stride_H + w * output_stride_W;
    for (auto c = 0; c < C; c++) {
        if(ix_nearest >= 0 && ix_nearest < W_in && iy_nearest >= 0 && iy_nearest < H_in && iz_nearest >= 0 && iz_nearest < D_in) {
            *output_NCDHW_offset = input_NC_offset[ix_nearest * input_stride_W + iy_nearest * input_stride_H + iz_nearest * input_stride_D];
        } else {
            *output_NCDHW_offset = static_cast<scalar_t>(0);
        }
    }
}

template <typename scalar_t>
__global__ void grid_sample_3d_bilinear_kernel(
    const scalar_t* input,
    const scalar_t* grid,
    size_t N, size_t C, size_t D_in, size_t H_in, size_t W_in,
    size_t input_stride_N, size_t input_stride_C, size_t input_stride_D, size_t input_stride_H, size_t input_stride_W,
    size_t D_grid, size_t H_grid, size_t W_grid,
    size_t grid_stride_N, size_t grid_stride_D, size_t grid_stride_H, size_t grid_stride_W, size_t grid_stride_XYZ,
    size_t output_stride_N, size_t output_stride_C, size_t output_stride_D, size_t output_stride_H, size_t output_stride_W,
    int align_corners,
    MyGridSamplePaddingMode padding_mode_type,
    scalar_t* output
) {

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= N * D_grid * H_grid * W_grid) {
        return;
    }

    auto n = tid / (D_grid * H_grid * W_grid);
    auto d = (tid / (H_grid * W_grid)) % D_grid;
    auto h = (tid / W_grid) % H_grid;
    auto w = tid % W_grid;

    const scalar_t* input_N_offset = input + n * input_stride_N;
    const scalar_t* grid_N_offset = grid + n * grid_stride_N;
    scalar_t* output_N_offset = output + n * output_stride_N;

    const scalar_t* grid_NDHW_offset = grid_N_offset + d * grid_stride_D + h * grid_stride_H + w * grid_stride_W;
    const scalar_t x = *grid_NDHW_offset;
    const scalar_t y = *(grid_NDHW_offset + grid_stride_XYZ);
    const scalar_t z = *(grid_NDHW_offset + 2 * grid_stride_XYZ);


    scalar_t ix = compute_index(x, W_in, padding_mode_type, align_corners);
    scalar_t iy = compute_index(y, H_in, padding_mode_type, align_corners);
    scalar_t iz = compute_index(z, D_in, padding_mode_type, align_corners);
    
    int x0 = static_cast<int>(floor(ix));
    int y0 = static_cast<int>(floor(iy));
    int z0 = static_cast<int>(floor(iz));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;

    scalar_t v000 = (ix                        - x0) * (iy - y0)                        * (iz - z0);
    scalar_t v100 = (static_cast<scalar_t>(x1) - ix) * (iy - y0)                        * (iz - z0);
    scalar_t v010 = (ix - x0)                        * (static_cast<scalar_t>(y1) - iy) * (iz - z0);
    scalar_t v110 = (static_cast<scalar_t>(x1) - ix) * (static_cast<scalar_t>(y1) - iy) * (iz - z0);
    scalar_t v001 = (ix - x0)                        * (iy - y0)                        * (static_cast<scalar_t>(z1) - iz);
    scalar_t v101 = (static_cast<scalar_t>(x1) - ix) * (iy - y0)                        * (static_cast<scalar_t>(z1) - iz);
    scalar_t v011 = (ix - x0)                        * (static_cast<scalar_t>(y1) - iy) * (static_cast<scalar_t>(z1) - iz);
    scalar_t v111 = (static_cast<scalar_t>(x1) - ix) * (static_cast<scalar_t>(y1) - iy) * (static_cast<scalar_t>(z1) - iz);

    scalar_t *input_NC_offset = const_cast<scalar_t *>(input_N_offset);
    scalar_t *output_NCDHW_offset = output_N_offset + d * output_stride_D + h * output_stride_H + w * output_stride_W;

    for(auto c = 0; c < C; c++) {
        scalar_t value = static_cast<scalar_t>(0);
        if(x1 >= 0 && x1 < W_in && y1 >= 0 && y1 < H_in && z1 >= 0 && z1 < D_in) {
            value += v000 * input_NC_offset[x1 * input_stride_W + y1 * input_stride_H + z1 * input_stride_D];   
        }
        if(x0 >= 0 && x0 < W_in && y1 >= 0 && y1 < H_in && z1 >= 0 && z1 < D_in) {
            value += v100 * input_NC_offset[x0 * input_stride_W + y1 * input_stride_H + z1 * input_stride_D];
        }
        if(x1 >= 0 && x1 < W_in && y0 >= 0 && y0 < H_in && z1 >= 0 && z1 < D_in) {
            value += v010 * input_NC_offset[x1 * input_stride_W + y0 * input_stride_H + z1 * input_stride_D];
        }
        if(x0 >= 0 && x0 < W_in && y0 >= 0 && y0 < H_in && z1 >= 0 && z1 < D_in) {
            value += v110 * input_NC_offset[x0 * input_stride_W + y0 * input_stride_H + z1 * input_stride_D];
        }
        if(x1 >= 0 && x1 < W_in && y1 >= 0 && y1 < H_in && z0 >= 0 && z0 < D_in) {
            value += v001 * input_NC_offset[x1 * input_stride_W + y1 * input_stride_H + z0 * input_stride_D];
        }
        if(x0 >= 0 && x0 < W_in && y1 >= 0 && y1 < H_in && z0 >= 0 && z0 < D_in) {
            value += v101 * input_NC_offset[x0 * input_stride_W + y1 * input_stride_H + z0 * input_stride_D];
        }
        if(x1 >= 0 && x1 < W_in && y0 >= 0 && y0 < H_in && z0 >= 0 && z0 < D_in) {
            value += v011 * input_NC_offset[x1 * input_stride_W + y0 * input_stride_H + z0 * input_stride_D];
        }
        if(x0 >= 0 && x0 < W_in && y0 >= 0 && y0 < H_in && z0 >= 0 && z0 < D_in) {
            value += v111 * input_NC_offset[x0 * input_stride_W + y0 * input_stride_H + z0 * input_stride_D];
        }
        *output_NCDHW_offset = value;
        input_NC_offset += input_stride_C;
        output_NCDHW_offset += output_stride_C;
          
    }
    
}

template <typename scalar_t>
int grid_sample_3d_cuda(
    const scalar_t* input,
    const scalar_t* grid,
    size_t N, size_t C, size_t D_in, size_t H_in, size_t W_in,
    size_t D_grid, size_t H_grid, size_t W_grid,
    int align_corners,
    std::string mode,
    std::string padding_mode,
    scalar_t* output,
    cudaStream_t stream
) {
    

    size_t totalThreads = N * D_grid * W_grid * H_grid;
    dim3 dimBlock(NUM_THREADS);
    dim3 dimGrid(get_num_blocks(totalThreads));

    size_t input_stride_N = C * D_in * H_in * W_in;
    size_t input_stride_C = D_in * H_in * W_in;
    size_t input_stride_D = H_in * W_in;
    size_t input_stride_H = W_in;
    size_t input_stride_W = 1;

    size_t grid_stride_N = D_grid * H_grid * W_grid * 3;
    size_t grid_stride_D = H_grid * W_grid * 3;
    size_t grid_stride_H = W_grid * 3;
    size_t grid_stride_W = 3;
    size_t grid_stride_XYZ = 1;

    size_t output_stride_N = C * D_grid * H_grid * W_grid;
    size_t output_stride_C = D_grid * H_grid * W_grid;
    size_t output_stride_D = H_grid * W_grid;
    size_t output_stride_H = W_grid;
    size_t output_stride_W = 1;

    MyGridSamplePaddingMode padding_mode_type = padding_mode == "border" ? MyGridSamplePaddingMode::Border : 
                                                padding_mode == "reflection" ? MyGridSamplePaddingMode::Reflection : 
                                                MyGridSamplePaddingMode::Zeros;

    if(mode == "bilinear") {
        grid_sample_3d_bilinear_kernel<scalar_t><<<dimGrid, dimBlock, 0, stream>>>(
            input,
            grid,
            N, C, D_in, H_in, W_in,
            input_stride_N, input_stride_C, input_stride_D, input_stride_H, input_stride_W,
            D_grid, H_grid, W_grid,
            grid_stride_N, grid_stride_D, grid_stride_H, grid_stride_W, grid_stride_XYZ,
            output_stride_N, output_stride_C, output_stride_D, output_stride_H, output_stride_W,
            align_corners,
            padding_mode_type,
            output
        );
    } else if(mode == "nearest") {
        grid_sample_3d_nearest_kernel<scalar_t><<<dimGrid, dimBlock, 0, stream>>>(
            input,
            grid,
            N, C, D_in, H_in, W_in,
            input_stride_N, input_stride_C, input_stride_D, input_stride_H, input_stride_W,
            D_grid, H_grid, W_grid,
            grid_stride_N, grid_stride_D, grid_stride_H, grid_stride_W, grid_stride_XYZ,
            output_stride_N, output_stride_C, output_stride_D, output_stride_H, output_stride_W,
            align_corners,
            padding_mode_type,
            output
        );        
    } else {
        return 1;
    }

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("Error in grid_sample_3d_cuda: %s\n", cudaGetErrorString(err));
    }

    return cudaGetLastError() != cudaSuccess;
}

// 显式实例化
template int grid_sample_3d_cuda<float>(
    const float* input,
    const float* grid,
    size_t N, size_t C, size_t D_in, size_t H_in, size_t W_in,
    size_t D_grid, size_t H_grid, size_t W_grid,
    int align_corners,
    std::string interpolationMode,
    std::string paddingMode,
    float* output,
    cudaStream_t stream
);

template int grid_sample_3d_cuda<half>(
    const half* input,
    const half* grid,
    size_t N, size_t C, size_t D_in, size_t H_in, size_t W_in,
    size_t D_grid, size_t H_grid, size_t W_grid,
    int align_corners,
    std::string interpolationMode,
    std::string paddingMode,
    half* output,
    cudaStream_t stream
);

