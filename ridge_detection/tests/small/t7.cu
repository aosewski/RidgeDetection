
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

#ifndef devCheckCall
    #define devCheckCall(e) if ( Debug((e), __FILE__, __LINE__) ) { assert(0); }
#endif

#ifndef checkCudaErrors
    #define checkCudaErrors(e) if ( Debug((e), __FILE__, __LINE__) ) { cudaDeviceReset(); exit(1); }
#endif


//----------------------------------------------------------
//
//----------------------------------------------------------

// static constexpr int g_iterations   = 100;
static constexpr int g_iterations   = 1;
static constexpr int POINTS_NUM     = (1 << 18) + 357;
__constant__ static char dConstBuff[128];

//----------------------------------------------------------
//
//----------------------------------------------------------

__device__ __forceinline__ unsigned int LaneId()
{
    unsigned int ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret) );
    return ret;
}

//----------------------------------------------------------
//
//----------------------------------------------------------

__global__ void memsetKernel(char * out, int size, int value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // get number of 4-element vectors to cover size
    const int m = (size + 3) / 4;
    // round up to blockDim.x multiply
    const int n = (m + blockDim.x - 1) & ~(blockDim.x - 1);

    char4 * ptr = reinterpret_cast<char4*>(out);
    char oval = value;

    for (; i < n - blockDim.x; i += gridDim.x * blockDim.x)
    {
        ptr[i] = make_char4(oval, oval, oval, oval);
    }

    // last (probably not full) tile of data
    if (i < m - 1)
    {
        ptr[i] = make_char4(oval, oval, oval, oval);
    }

    // last output vector to store
    if (i == m - 1)
    {
        i *= 4;
        for (; i < size; ++i)
        {
            out[i] = char(value);
        }
    }
}


__global__ void memcpyKernel(char *  out, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = (size + 3) / 4;
    const int n = (m + blockDim.x - 1) & ~(blockDim.x - 1);

    char4 * outPtr = reinterpret_cast<char4*>(out);
    char4 const * __restrict__ inPtr = reinterpret_cast<char4 const *>(dConstBuff);

    for (; i < n - blockDim.x; i += gridDim.x * blockDim.x)
    {
        outPtr[i] = inPtr[LaneId()];
    }

    // last (probably not full) tile of data
    if (i < m - 1)
    {
        outPtr[i] = inPtr[LaneId()];
    }

    // last output vector to store
    if (i == m - 1)
    {
        i *= 4;
        for (; i < size; ++i)
        {
            out[i] = dConstBuff[LaneId()];
        }
    }
}


//----------------------------------------------------------
//
//----------------------------------------------------------


int main(void)
{

    int value = 1;

    char *out1 = new char[POINTS_NUM];
    char *out2 = new char[POINTS_NUM];
    char *constBuff = new char[128];
    char *d_in1, *d_in2;

    for (int k = 0; k < 128; ++k)
    {
        constBuff[k] = value;
    }
    checkCudaErrors(cudaMemcpyToSymbol(dConstBuff, constBuff, 128 * sizeof(char)));

    int devId = 0;
    checkCudaErrors(cudaSetDevice(devId));

    checkCudaErrors(cudaMalloc(&d_in1, POINTS_NUM * sizeof(char)));
    checkCudaErrors(cudaMalloc(&d_in2, POINTS_NUM * sizeof(char)));

    int dimBlockMemset(256);
    int dimGridMemset(256);

    int dimBlockMemcpy(256);
    int dimGridMemcpy(256);

    // Get SM count
    int smCount;
    checkCudaErrors(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 
        devId));

    // Get SM occupancy
    int memsetSmOccupancy;
    int memcpySmOccupancy;

    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &memsetSmOccupancy, memsetKernel, dimBlockMemset, 0));

    checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &memcpySmOccupancy, memcpyKernel, dimBlockMemcpy, 0));

    std::cout << "memsetSmOccupancy: " << memsetSmOccupancy 
            << ", memcpySmOccupancy: " << memcpySmOccupancy << std::endl;
    
    dimGridMemset = memsetSmOccupancy * smCount;
    dimGridMemcpy = memcpySmOccupancy * smCount;

    // checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&dimGridMemset, 
    //     &dimBlockMemset, memsetKernel));
    // checkCudaErrors(cudaOccupancyMaxPotentialBlockSize(&dimGridMemcpy, 
    //     &dimBlockMemcpy, memcpyKernel));

    // std::cout << "Memset: dimBlock: " << dimBlockMemset 
    //     << ", dimGrid: " << dimGridMemset << std::endl;
    // std::cout << "Memcpy: dimBlock: " << dimBlockMemcpy 
    //     << ", dimGrid: " << dimGridMemcpy << std::endl;

    checkCudaErrors(cudaMemset(d_in1, 0, POINTS_NUM * sizeof(char)));
    checkCudaErrors(cudaMemset(d_in2, 0, POINTS_NUM * sizeof(char)));

    for (int k = 0; k < g_iterations; ++k)
    {
        memsetKernel<<<dimGridMemset, dimBlockMemset>>>(d_in1, POINTS_NUM, value);
        checkCudaErrors(cudaGetLastError());
    }

    // dimGrid = memcpySmOccupancy * smCount;

    for (int k = 0; k < g_iterations; ++k)
    {
        memcpyKernel<<<dimGridMemcpy, dimBlockMemcpy>>>(d_in2, POINTS_NUM);
        checkCudaErrors(cudaGetLastError());
    }

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(out1, d_in1, POINTS_NUM * sizeof(char),
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(out2, d_in2, POINTS_NUM * sizeof(char),
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    for (int k = 0; k < POINTS_NUM; ++k)
    {
        if (int(out1[k]) != value ||
            int(out2[k]) != value)
        {
            std::cout << "--- ERROR!---- out1["<<k<<"]: (" << int(out1[k])
                << "), out2["<<k<<"]: (" << int(out2[k])
                << ")" << std::endl;
            break;
        }
    }

    checkCudaErrors(cudaDeviceReset());

    return 0;
}