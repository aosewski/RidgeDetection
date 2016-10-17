#include <iostream>
#include <assert.h>

__host__ __device__ __forceinline__ cudaError_t Debug(
    cudaError_t     error,
    const char*     filename,
    int             line)
{
    if (error)
    {
    #if (__CUDA_ARCH__ == 0)
        fprintf(stderr, "CUDA error %d [%s, %d]: %s\n",
            error, filename, line, cudaGetErrorString(error));
        fflush(stderr);
    #elif (__CUDA_ARCH__ >= 200)
        printf("CUDA error %d [block (%3d,%3d,%3d) thread (%3d,%3d,%3d), %s, %d]\n",
            error, blockIdx.z, blockIdx.y, blockIdx.x,
            threadIdx.z, threadIdx.y, threadIdx.x, filename, line);
    #endif
    }
    return error;
}

/**
 * @brief Macros for error checking.     
 */
#ifndef devCheckCall
    #define devCheckCall(e) if ( Debug((e), __FILE__, __LINE__) ) { assert(0); }
#endif

#ifndef checkCudaErrors
    #define checkCudaErrors(e) if ( Debug((e), __FILE__, __LINE__) ) { cudaDeviceReset(); exit(1); }
#endif

//----------------------------------------------------------
//
//----------------------------------------------------------

#define RD_WARP_MASK       31
#define RD_WARP_SIZE       32

/**
 * @return Thread inner-warp index.
 */
__device__ __forceinline__ int rdLaneId() 
{ 
    // return threadIdx.x % RD_WARP_SIZE; 
    return threadIdx.x & RD_WARP_MASK; 
}

// template <typename T>
// __device__ __forceinline__ void rdBroadcast(T &arg, const int srcLaneId)
// {
//     arg = __shfl(arg, srcLaneId);
// }

/**
 * \brief Returns the warp lane ID of the calling thread
 */
__device__ __forceinline__ unsigned int cubLaneId()
{
    unsigned int ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret) );
    return ret;
}

/**
 * \brief Returns the warp lane mask of all lanes less than the calling thread
 */
__device__ __forceinline__ unsigned int cubLaneMaskLt()
{
    unsigned int ret;
    asm volatile("mov.u32 %0, %lanemask_lt;" : "=r"(ret) );
    return ret;
}


//----------------------------------------------------------
//
//----------------------------------------------------------


__global__ void testKernel1(float *  in)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[x];
    int mask = __ballot(val == 1);
    if (val == 1)
    {
        in[x] = __popc(mask & (1 << rdLaneId()) - 1);
    }
}

__global__ void testKernel2(float *  in)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[x];
    int mask = __ballot(val == 1);
    if (val == 1)
    {
        in[x] = __popc(mask & (1 << cubLaneId()) - 1);
    }
}

__global__ void testKernel3(float *  in)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    float val = in[x];
    int mask = __ballot(val == 1);
    if (val == 1)
    {
        in[x] = __popc(mask & cubLaneMaskLt());
    }
}


static constexpr int BLOCK_THREADS  = 128;
static constexpr int POINTS_NUM     = 1 << 16;
static constexpr int DIM            = 2;

int main(void)
{

    float *h_data = new float[POINTS_NUM * DIM];
    float *d_out1 = new float[POINTS_NUM * DIM];
    float *d_out2 = new float[POINTS_NUM * DIM];
    float *d_out3 = new float[POINTS_NUM * DIM];
    float *d_in1, *d_in2, *d_in3;

    checkCudaErrors(cudaSetDevice(0));

    checkCudaErrors(cudaMalloc(&d_in1, POINTS_NUM * DIM * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_in2, POINTS_NUM * DIM * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_in3, POINTS_NUM * DIM * sizeof(float)));

    for (int k = 0; k < POINTS_NUM * DIM; ++k)
    {
        h_data[k] = k * 0.2f;
    }

    checkCudaErrors(cudaMemcpy(d_in1, h_data, POINTS_NUM * DIM * sizeof(float),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_in2, h_data, POINTS_NUM * DIM * sizeof(float),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_in3, h_data, POINTS_NUM * DIM * sizeof(float),
        cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_THREADS);
    dim3 dimGrid(POINTS_NUM / BLOCK_THREADS);

    testKernel1<<<dimGrid, dimBlock>>>(d_in1);
    testKernel2<<<dimGrid, dimBlock>>>(d_in2);
    testKernel3<<<dimGrid, dimBlock>>>(d_in3);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(d_out1, d_in1, POINTS_NUM * DIM * sizeof(float),
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(d_out2, d_in2, POINTS_NUM * DIM * sizeof(float),
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(d_out3, d_in3, POINTS_NUM * DIM * sizeof(float),
        cudaMemcpyDeviceToHost));

    for (int k = 0; k < POINTS_NUM * DIM; ++k)
    {
        if (d_out1[k] != d_out2[k] ||
            d_out1[k] != d_out3[k])
        {
            std::cout << "d_out1["<<k<<"]: " << d_out1[k]
                << ", d_out2["<<k<<"]: " << d_out2[k]
                << ", d_out3["<<k<<"]: " << d_out3[k] << std::endl;
        }
    }

    checkCudaErrors(cudaDeviceReset());

    return 0;
}