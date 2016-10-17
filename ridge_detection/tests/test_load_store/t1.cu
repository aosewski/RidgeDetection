
#include <helper_cuda.h>
#include <iostream>

#include "rd/gpu/block/block_tile_load_store4.cuh"
#include "rd/gpu/util/dev_memcpy.cuh"
#include "tests/test_util.hpp"
#include "cub/test_util.h"

static constexpr int POINTS_PER_THREAD  = 8;
static constexpr int DIM                = 4;
static constexpr int SAMPLES_PER_THREAD = POINTS_PER_THREAD * DIM;


//-----------------------------------------------------------------------
//      LOAD & STORE 
//-----------------------------------------------------------------------

template <
    typename BlockTileLoadT,
    // typename BlockTileStoreT,
    typename T>
__global__ void ls_row_major(T const *in, T *out, int inStride/*, int outStride*/)
{
    int inOffset = blockIdx.x * blockDim.x * SAMPLES_PER_THREAD;
    int outOffset = blockIdx.x * blockDim.x * POINTS_PER_THREAD;

    typename BlockTileLoadT::ThreadPrivatePoints<rd::ROW_MAJOR> samples;
    BlockTileLoadT::loadTile2RowM(in, samples.data, inOffset, inStride);

    #pragma unroll
    for (int i = 0; i < POINTS_PER_THREAD; ++i)
    {
        T outVal = 1;
        #pragma unroll
        for (int d = 0; d < POINTS_PER_THREAD; ++d)
        {
            outVal *= samples.data[i][d];
        }
        out[outOffset + blockDim.x * i + threadIdx.x] = outVal;
    }

    // BlockTileStoreT::storeTileFromRowM(out, samples.data, outOffset, outStride);
}


template <
    typename BlockTileLoadT,
    // typename BlockTileStoreT,
    typename T>
__global__ void ls_col_major(T const *in, T *out, int inStride/*, int outStride*/)
{
    int inOffset = blockIdx.x * blockDim.x * SAMPLES_PER_THREAD;
    int outOffset = blockIdx.x * blockDim.x * POINTS_PER_THREAD;

    typename BlockTileLoadT::ThreadPrivatePoints<rd::COL_MAJOR> samples;
    BlockTileLoadT::loadTile2ColM(in, samples.data, inOffset, inStride);

    #pragma unroll
    for (int i = 0; i < POINTS_PER_THREAD; ++i)
    {
        T outVal = 1;
        #pragma unroll
        for (int d = 0; d < POINTS_PER_THREAD; ++d)
        {
            outVal *= samples.data[i][d];
        }
        out[outOffset + blockDim.x * i + threadIdx.x] = outVal;
    }

    // BlockTileStoreT::storeTileFromColM(out, samples.data, outOffset, outStride);
}

//-----------------------------------------------------------------------
//      TEST 
//-----------------------------------------------------------------------

template <typename T>
void test()
{
    const int nPoints = 1 << 18;
    const int nElements = nPoints * SAMPLES_PER_THREAD;

    std::cout << "nPoints: " << nPoints;
    std::cout << "\nnElements: " << nElements;
    std::cout << "\nmemSize: " << nElements * sizeof(T) / 1024.0 / 1024.0 << "(MB)\n";

    T *d_in_RM, *d_in_CM, *d_out;
    T *h_in, *h_out;

    h_in = new T[nElements];
    h_out = new T[nPoints];

    int d_in_CM_stride;
    size_t pitch;
    checkCudaErrors(cudaMallocPitch(&d_in_CM, &pitch, nPoints * sizeof(T), DIM));
    d_in_CM_stride = pitch / sizeof(T);

    checkCudaErrors(cudaMalloc((void**)&d_in_RM, nElements * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_out, nElements * sizeof(T)));

    for (int k = 0; k < nPoints; ++k)
    {
        for (int d = 0; d < DIM; ++d)
        {
            h_in[k*DIM + d] = d+1;
        }
    }

    checkCudaErrors(cudaMemcpy(d_in_RM, h_in, nElements * sizeof(T), cudaMemcpyHostToDevice));
    rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
        d_in_CM, h_in, DIM, nPoints, pitch, DIM * sizeof(T));

    dim3 gridSize(1);
    const int blockThreads = 64;
    gridSize.x = nPoints / blockThreads / POINTS_PER_THREAD;

    typedef rd::gpu::BlockTileLoadPolicy<
            blockThreads,
            POINTS_PER_THREAD, 
            cub::LOAD_LDG> 
        BlockTileLoadPolicyT;

    //------------------------------------------
    // CUB ROW_MAJOR
    //------------------------------------------

    typedef rd::gpu::BlockTileLoad<
                BlockTileLoadPolicyT,
                DIM,
                rd::ROW_MAJOR, 
                rd::gpu::IO_BACKEND_CUB,
                T, 
                int>
            TileLoaderRMCUB;

    ls_row_major<TileLoaderRMCUB><<<gridSize, blockThreads>>>(d_in_RM, d_out, DIM);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_out, d_out, nElements * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    for (int k = 0; k < nPoints; ++k)
    {
        T value = 1;
        for (int d = 0; d < DIM; ++d)
        {
            value *= h_in[k * DIM + d];
        }
        if (h_out[k] != value)
        {
            std::cout << "ERROR!: is: " << h_out[k] << ", should be: " << value << "\n";
            exit(1);
        }
    }

    std::cout << "CUB ROW_MAJOR - Success!" << std::endl;

    //------------------------------------------
    // CUB COL_MAJOR
    //------------------------------------------

    typedef rd::gpu::BlockTileLoad<
                BlockTileLoadPolicyT,
                DIM,
                rd::COL_MAJOR, 
                rd::gpu::IO_BACKEND_CUB,
                T, 
                int>
            TileLoaderCMCUB;

    ls_col_major<TileLoaderCMCUB><<<gridSize, blockThreads>>>(d_in_CM, d_out, d_in_CM_stride);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_out, d_out, nElements * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    for (int k = 0; k < nPoints; ++k)
    {
        T value = 1;
        for (int d = 0; d < DIM; ++d)
        {
            value *= h_in[k * DIM + d];
        }
        if (h_out[k] != value)
        {
            std::cout << "ERROR!: is: " << h_out[k] << ", should be: " << value << "\n";
            exit(1);
        }
    }

    std::cout << "CUB COL_MAJOR - Success!" << std::endl;

    //------------------------------------------
    // trove ROW_MAJOR
    //------------------------------------------

    typedef rd::gpu::BlockTileLoadPolicy<
            blockThreads,
            POINTS_PER_THREAD, 
            cub::LOAD_LDG> 
        BlockTileLoadPolicyT;

    typedef rd::gpu::BlockTileLoad<
                BlockTileLoadPolicyT,
                DIM,
                rd::ROW_MAJOR, 
                rd::gpu::IO_BACKEND_TROVE,
                T, 
                int>
            TileLoaderRMtrove;

    ls_row_major<TileLoaderRMtrove><<<gridSize, blockThreads>>>(d_in_RM, d_out, DIM);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_out, d_out, nElements * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    for (int k = 0; k < nPoints; ++k)
    {
        T value = 1;
        for (int d = 0; d < DIM; ++d)
        {
            value *= h_in[k * DIM + d];
        }
        if (h_out[k] != value)
        {
            std::cout << "ERROR!: is: " << h_out[k] << ", should be: " << value << "\n";
            exit(1);
        }
    }

    std::cout << "trove ROW_MAJOR - Success!" << std::endl;

    //------------------------------------------
    // trove COL_MAJOR
    //------------------------------------------

    typedef rd::gpu::BlockTileLoad<
                BlockTileLoadPolicyT,
                DIM,
                rd::COL_MAJOR, 
                rd::gpu::IO_BACKEND_TROVE,
                T, 
                int>
            TileLoaderCMtrove;

    ls_col_major<TileLoaderCMtrove><<<gridSize, blockThreads>>>(d_in_CM, d_out, d_in_CM_stride);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_out, d_out, nElements * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    for (int k = 0; k < nPoints; ++k)
    {
        T value = 1;
        for (int d = 0; d < DIM; ++d)
        {
            value *= h_in[k * DIM + d];
        }
        if (h_out[k] != value)
        {
            std::cout << "ERROR!: is: " << h_out[k] << ", should be: " << value << "\n";
            exit(1);
        }
    }

    std::cout << "trove COL_MAJOR - Success!" << std::endl;

    delete[] h_in;
    delete[] h_out;

    checkCudaErrors(cudaFree(d_in_RM));
    checkCudaErrors(cudaFree(d_in_CM));
    checkCudaErrors(cudaFree(d_out));
}

int main()
{
    checkCudaErrors(deviceInit());

    test<float>();

    checkCudaErrors(cudaDeviceReset());

    return 0;
}