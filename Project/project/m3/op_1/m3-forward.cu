#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256


__global__ void matrix_unrolling_kernel(const float * __restrict__ input, float * __restrict__ output,
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

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float * __restrict__ input, float * __restrict__ output, int Map_out,
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

__host__ void GPUInterface::conv_forward_gpu_prolog(const float * __restrict__ host_output, const float * __restrict__ host_input, 
    const float * __restrict__ host_mask, float ** __restrict__ device_output_ptr, float ** __restrict__ device_input_ptr, 
    float ** __restrict__ device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
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


__host__ void GPUInterface::conv_forward_gpu(float * __restrict__ device_output, const float * __restrict__ device_input, 
    const float * __restrict__ device_mask, const int Batch, const int Map_out, 
    const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // const int Height_unrolled = Channel * K * K;
    // const int Width_unrolled = Batch * Height_out * Width_out;

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

    dim3 matmul_kernel_grid_dim(ceil(1.0 * numCColumns/ TILE_WIDTH), ceil(1.0 * numCRows/ TILE_WIDTH), 1);
    dim3 matmul_kernel_block_dim(TILE_WIDTH, TILE_WIDTH, 1);
    matrixMultiplyShared<<<matmul_kernel_grid_dim, matmul_kernel_block_dim>>>(device_mask, unrolled_matrix, matmul_output, numARows, numAColumns,
        numBRows, numBColumns, numCRows, numCColumns);

    // Permute the result of matrix multiplication
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float * __restrict__ host_output, float * __restrict__ device_output, 
    float * __restrict__ device_input, float * __restrict__ device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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