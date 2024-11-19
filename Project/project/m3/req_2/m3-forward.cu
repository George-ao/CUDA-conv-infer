#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__global__ void matrix_mul_built_in_unrolling_kernel(float *device_output, const float *device_input, const float *device_mask, 
    const int Batch, const int Map_out, const int Channel, 
    const int Height, const int Width, const int K)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // load and unroll input 
    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int cur_batch = blockIdx.z; 

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    #define in_4d(i3, i2, i1, i0) device_input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define out_3d(i3, i2, i1) device_output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + i1]

    // perform multiplication
    float val = 0;
    const float *A = device_mask;
    
    int numARows = Map_out;
    int numAColumns = Channel * K * K;
    int numBRows = Channel * K * K;
    int numBColumns = Height_out * Width_out;
    int numCRows = numARows;
    int numCColumns = numBColumns;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            size_t cur_row = tileId * TILE_WIDTH + ty;
            int w = col % Width_out;
            int h = col / Width_out;
            int c = cur_row / (K * K);
            int offset = cur_row % (K * K);
            int p = offset / K;
            int q = offset % K;
            tileB[ty][tx] = in_4d(cur_batch, c, h + p, w + q);
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

    if (row < numCRows && col < numCColumns) out_3d(cur_batch, row, col) = val;

    #undef in_4d
    #undef out_3d
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

    int W_grid = ceil(1.0 * Height_out * Width_out / TILE_WIDTH);
    int H_grid = ceil(1.0 * Map_out / TILE_WIDTH);

    dim3 Dimblock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 Dimgrid(W_grid, H_grid, Batch);

    matrix_mul_built_in_unrolling_kernel<<<Dimgrid, Dimblock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
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