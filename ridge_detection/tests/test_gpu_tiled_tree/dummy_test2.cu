
#include <assert.h>
#include <stdio.h>

#include "rd/utils/memory.h"
#include "rd/gpu/device/bounding_box.cuh"
#include "rd/gpu/device/device_spatial_histogram.cuh"

static constexpr int POINTS_NUM     = 1 << 20;
static constexpr int DIM            = 2;

//-------------------------------------------------------
//      Debug utilities
//-------------------------------------------------------

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

#ifndef rdDebug
    #define rdDebug(e) Debug((e), __FILE__, __LINE__)
#endif

//----------------------------------------------
// stateful CUDA thread block abstraction for some algorithm
//----------------------------------------------

class AlgorithmImpl
{

    // per-thread fields
    float const * __restrict__ inputData;
    int someParameter; 

public:
    __device__ __forceinline__ AlgorithmImpl(
        float const * __restrict__  in,
        int                         param)
    :
        inputData(in),
        someParameter(param)
    {
    }

    __device__ __forceinline__ void getHistogram(
        int * pointHist,
        int * ngbrHist, 
        int pointNum,
        int (&binCnt)[DIM])
    {
        rd::gpu::BoundingBox<DIM, float> bbox;
        devCheckCall(bbox.findBounds<rd::ROW_MAJOR>(inputData, pointNum));
        bbox.calcDistances();

        __syncthreads();
        bbox.print();

        cudaStream_t histStream;
        devCheckCall(cudaStreamCreateWithFlags(&histStream, cudaStreamNonBlocking));

        void * tempStorage = nullptr;
        size_t tempStorageBytes = 0;

        cudaError err = cudaSuccess;
        // query for temporary storage
        err = rd::gpu::DeviceHistogram::spatialHistogram<DIM, rd::ROW_MAJOR>(
            tempStorage, tempStorageBytes, inputData, pointHist, pointNum, binCnt, bbox, 1,
            histStream);
        devCheckCall(err);

        tempStorage = new char[tempStorageBytes];
        assert(tempStorage != nullptr);
        
        err = rd::gpu::DeviceHistogram::spatialHistogram<DIM, rd::ROW_MAJOR>(
            tempStorage, tempStorageBytes, inputData, pointHist, pointNum, binCnt, bbox, 1,
            histStream);
        devCheckCall(err);

        devCheckCall(cudaDeviceSynchronize());

        for (int k = 0; k < someParameter; ++k)
        {
            printf("pointHist[%d]: %d \n", k, pointHist[k]);
        }

        err = rd::gpu::DeviceHistogram::spatialHistogram<DIM, rd::ROW_MAJOR>(
            tempStorage, tempStorageBytes, inputData, ngbrHist, pointNum, binCnt, bbox, 1,
            histStream);
        devCheckCall(err);

        devCheckCall(cudaDeviceSynchronize());

        for (int k = 0; k < someParameter; ++k)
        {
            printf("ngbrHist[%d]: %d\n", k, ngbrHist[k]);
        }

        devCheckCall(cudaStreamDestroy(histStream));
        delete[] tempStorage;
    }

    __device__ __forceinline__ void build(
        int             pointNum,
        cudaStream_t    stream)
    {

        // do some crazy stuff..
        int * d_tilesPointsHist, *d_tilesNeighboursHist;
        
        d_tilesPointsHist       = new int[someParameter];
        d_tilesNeighboursHist   = new int[someParameter];
        
        assert(d_tilesPointsHist     != nullptr);
        assert(d_tilesNeighboursHist != nullptr);

        devCheckCall(cudaMemsetAsync(d_tilesPointsHist, 0, someParameter * sizeof(int),
            stream));
        devCheckCall(cudaMemsetAsync(d_tilesNeighboursHist, 0, someParameter * sizeof(int),
            stream));
        devCheckCall(cudaDeviceSynchronize());

        int binCnt[DIM];
        binCnt[0] = someParameter;
        binCnt[1] = 1;

        getHistogram(d_tilesPointsHist, d_tilesNeighboursHist, pointNum, binCnt);
        devCheckCall(cudaDeviceSynchronize());

        delete[] d_tilesPointsHist;
        delete[] d_tilesNeighboursHist;
    }

};

__launch_bounds__ (1)
__global__ void kernel(
    float const * __restrict__  in,
    int                         pointNum,
    int                         dummyParameter)
{
    cudaStream_t stream;
    devCheckCall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    AlgorithmImpl * ptr = new AlgorithmImpl(in, dummyParameter);
    assert(ptr != nullptr);

    ptr->build(pointNum, stream);

    devCheckCall(cudaStreamDestroy(stream));
    delete ptr;
}

int main(void)
{
    
    float *h_data = new float[POINTS_NUM * DIM];
    float *d_in;

    printf("Num elements: %d\n", POINTS_NUM * DIM);

    checkCudaErrors(cudaSetDevice(0));

    int parameter = 16;
    size_t neededMemSize = POINTS_NUM * DIM * sizeof(float);
    checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, neededMemSize));
    printf("Reserved %f Mb for malloc heap\n",
            float(neededMemSize) / 1024.f / 1024.f);

    checkCudaErrors(cudaMalloc(&d_in, POINTS_NUM * DIM * sizeof(float)));

    for (int k = 0; k < POINTS_NUM; ++k)
    {
        for (int d = 0; d < DIM; ++d)
        {
            h_data[k * DIM + d] = k * 0.1f + d * 0.03;
        }
    }

    checkCudaErrors(cudaMemcpy(d_in, h_data, POINTS_NUM * DIM * sizeof(float),
        cudaMemcpyHostToDevice));

    kernel<<<1, 1>>>(d_in, POINTS_NUM, parameter);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaDeviceReset());

    return 0;
}