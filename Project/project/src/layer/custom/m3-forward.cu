#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <mma.h>
using namespace nvcuda;

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void float_to_half(const float *input, half *output, int input_row, int input_col, int output_row, int output_col)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < static_cast<size_t>(output_row * output_col)) 
    {
        size_t out_row = idx / output_col;
        size_t out_col = idx % output_col;
        if (out_row >= input_row || out_col >= input_col) output[idx] = __float2half(0.0f); 
        else output[idx] = __float2half(input[out_row * input_col + out_col]);
    }
}

__global__ void padding_to_output(float *input, float *output, int input_row, int input_col, int output_row, int output_col)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < static_cast<size_t>(output_row * output_col)) 
    {
        size_t in_row = idx / input_col;
        size_t in_col = idx % input_col;
        if (in_row >= output_row || in_col >= output_col) return;
        else output[in_row * output_col + in_col] = __float2half(input[in_row * input_col + in_col]);
    }
}

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define out_3d(i2, i1, i0) output[(i2) * (Height_out * Width_out) + (i1) * (Batch * Height_out * Width_out) + i0]

    // TODO: Insert your input matrix unrolling kernel code here
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   
    int w = idx % Width_out;
    int h = idx / Width_out;
    int b = blockIdx.y;
    int c = blockIdx.z;

    if (idx < Height_out * Width_out)
    {
        size_t w_unroll = static_cast<size_t>(h) * Width_out + w;
        size_t w_base = static_cast<size_t>(c) * K * K;
        for (int p = 0; p < K; p++) 
        {
            for (int q = 0; q < K; q++) 
            {
                size_t h_unroll = w_base + static_cast<size_t>(p) * K + q;
                out_3d(b, h_unroll, w_unroll) = in_4d(b, c, h + p, w + q);
            }
        }
    }

    #undef in_4d
    #undef out_3d
}

// This code includes modifications or adaptations based on source code 
// originally provided by NVIDIA Corporation, as found in:
// NVIDIA Developer Blog, "Programming Tensor Cores in CUDA 9",
// https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9
__global__ void wmma_mul(half *A, half *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    // fragment
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;   // m x k
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;   // k x n
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;             // m x n
    wmma::fill_fragment(c_frag, 0.0f);

    // row, col in C
    // int col = blockIdx.x * WMMA_M;
    // int row = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_N;
    size_t col = static_cast<size_t>(blockIdx.x) * WMMA_M;
    size_t row = (static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y) * WMMA_N;

    for (size_t i = 0; i < numAColumns; i += WMMA_K)
    {
        size_t a_row = row;
        size_t a_col = i;
        size_t b_row = i;
        size_t b_col = col;

        wmma::load_matrix_sync(a_frag, A + a_row * numAColumns + a_col, numAColumns);
        wmma::load_matrix_sync(b_frag, B + b_row * numBColumns + b_col, numBColumns);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    // store
    wmma::store_matrix_sync(C + row * numCColumns + col, c_frag, numCColumns, wmma::mem_row_major);
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    int input_size =  (Batch * Channel * Height * Width) * sizeof(float);
    int output_size = (Batch * Map_out * Height_out * Width_out) * sizeof(float);
    int mask_size = (Map_out * Channel * K * K) * sizeof(float);

    cudaMalloc((void **)device_input_ptr, input_size);
    cudaMalloc((void **)device_output_ptr, output_size);
    cudaMalloc((void **)device_mask_ptr, mask_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    dim3 unroll_kernel_grid_dim(ceil(1.0 * Height_out * Width_out / BLOCK_SIZE), Batch, Channel);
    dim3 unroll_kernel_block_dim(BLOCK_SIZE, 1, 1);
    matrix_unrolling_kernel<<<unroll_kernel_grid_dim, unroll_kernel_block_dim>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);
    // TODO: Set the kernel dimensions and call the matmul kernel
    int numARows = Map_out;
    int numAColumns = Channel * K * K;
    int numBRows = Channel * K * K;
    int numBColumns = Height_out * (Batch * Width_out);
    int numCRows = numARows;
    int numCColumns = numBColumns;

    // Tensor cores
    // type change and matrix padding
    int numARows_half = ceil(1.0 * numARows / WMMA_M) * WMMA_M;
    int numAColumns_half = ceil(1.0 * numAColumns / WMMA_K) * WMMA_K;
    int numBRows_half = numAColumns_half;
    int numBColumns_half = ceil(1.0 * numBColumns / WMMA_N) * WMMA_N;
    int numCRows_half = numARows_half;
    int numCColumns_half = numBColumns_half;

    half *device_unrolled_half;
    half *device_mask_half;
    float *device_output_padding;

    cudaMalloc((void**)&device_mask_half, (size_t) numARows_half * numAColumns_half * sizeof(half));
    cudaMalloc((void**)&device_unrolled_half, (size_t) numBRows_half * numBColumns_half * sizeof(half));
    cudaMalloc((void**)&device_output_padding, (size_t) numCRows_half * numCColumns_half * sizeof(float));
    // mask: float to half
    dim3 mask_float_to_half_grid_dim(ceil(1.0 * numARows_half * numAColumns_half / BLOCK_SIZE), 1, 1);
    dim3 mask_float_to_half_block_dim(BLOCK_SIZE, 1, 1);
    float_to_half<<<mask_float_to_half_grid_dim, mask_float_to_half_block_dim>>>(device_mask, device_mask_half, numARows, 
        numAColumns, numARows_half, numAColumns_half);
    // unrolled matrix: float to half
    dim3 unrolled_float_to_half_grid_dim(ceil(1.0 * numBRows_half * numBColumns_half / BLOCK_SIZE), 1, 1);
    dim3 unrolled_float_to_half_block_dim(BLOCK_SIZE, 1, 1);
    float_to_half<<<unrolled_float_to_half_grid_dim, unrolled_float_to_half_block_dim>>>(unrolled_matrix, device_unrolled_half, 
        numBRows, numBColumns, numBRows_half, numBColumns_half);
    // launch kernel
    dim3 matmul_kernel_grid_dim(ceil(1.0 * numCColumns/ TILE_WIDTH), ceil(1.0 * numCRows/ TILE_WIDTH), 1);
    dim3 matmul_kernel_block_dim(WARP_SIZE, 1, 1);
    wmma_mul<<<matmul_kernel_grid_dim, matmul_kernel_block_dim>>>(device_mask_half, device_unrolled_half, device_output_padding, numARows_half, 
        numAColumns_half, numBRows_half, numBColumns_half, numCRows_half, numCColumns_half);
    // copy padding to output
    dim3 padding_to_output_grid_dim(ceil(1.0 * numCRows_half * numCColumns_half / BLOCK_SIZE), 1, 1);
    dim3 padding_to_output_block_dim(BLOCK_SIZE, 1, 1);
    padding_to_output<<<padding_to_output_grid_dim, padding_to_output_block_dim>>>(device_output_padding, 
        matmul_output, numCRows_half, numCColumns_half, numCRows, numCColumns);
    // Permute the result of matrix multiplication
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int output_size = (Batch * Map_out * Height_out * Width_out) * sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // TODO: Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}