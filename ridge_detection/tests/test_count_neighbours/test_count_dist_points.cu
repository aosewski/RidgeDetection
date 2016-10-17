/**
 * @file benchmark_count_neighbours.cu
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is conducted under the supervision of prof. dr hab. inż. Marek
 * Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */

#include <helper_cuda.h>
#ifdef RD_PROFILE
#include <cuda_profiler_api.h>
#endif

#include <iostream>

#include "rd/gpu/device/brute_force/test/decimate_dist_mtx.cuh"

#include "rd/utils/utilities.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "cub/test_util.h"


#if defined(RD_PROFILE) || defined(RD_DEBUG)
static const int g_iterations = 1;
#else
static const int g_iterations = 100;
#endif

static int g_devId;

static int cpuCountPoints(
    float const *   data,
    char  const *   mask,
    int             size,
    float           compare,
    const int       threshold,
    const int       blockSize)
{
    int pCount = 0;
    for (int i = 0; i < size; ++i)
    {
        if (mask[i] && data[i] < compare)
        {
            pCount++;
            if ( ((i+1) % blockSize) == 0 && pCount >= threshold)
            {
                return pCount;
            }
        }
    }
    return pCount;
}

template <int BLOCK_SIZE>
__launch_bounds__ (BLOCK_SIZE)
__global__ void countPointsKernel(
    float const * __restrict__  data,
    char const * __restrict__   mask,
    int                         size,
    float                       compare,
    int                         threshold,
    int *                       pCount)
{
    int out = rd::gpu::bruteForce::countPoints<BLOCK_SIZE>(
        data, size, mask, compare, threshold);
    if (threadIdx.x == 0)
    {
        *pCount = out;
    }
}

template <int BLOCK_SIZE>
static void test(
    float const * d_in,
    char  const * d_mask,
    int     size,
    float   compare,
    int     threshold,
    int *   d_pCount,
    const int pCountGold)
{
    GpuTimer timer;
    float kernelTime;

    //warm-up & correctnes check
    countPointsKernel<BLOCK_SIZE><<<1,BLOCK_SIZE>>>(
        d_in, d_mask, size, compare, threshold, d_pCount);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int h_outPCount = 0;
    checkCudaErrors(cudaMemcpy(&h_outPCount, d_pCount, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    if (h_outPCount != pCountGold)
    {
        std::cerr << "ERROR! Incorrect pt count! Is: " << h_outPCount << ", should be: " 
            << pCountGold << std::endl;
        exit(1);
    }

    #ifdef RD_PROFILE
    cudaProfilerStart();
    #endif
    timer.Start();
    for (int i = 0; i < g_iterations; ++i)
    {
        countPointsKernel<BLOCK_SIZE><<<1,BLOCK_SIZE>>>(
            d_in, d_mask, size, compare, threshold, d_pCount);
        checkCudaErrors(cudaGetLastError());
    }
    timer.Stop();
    kernelTime = timer.ElapsedMillis();
    checkCudaErrors(cudaDeviceSynchronize());
    #ifdef RD_PROFILE
    cudaProfilerStop();
    #endif

    kernelTime = kernelTime  / static_cast<float>(g_iterations);
    std::cout << " " << BLOCK_SIZE << " " << kernelTime 
        << " " << h_outPCount << std::endl;

}

static void test(int size, int threshold)
{
    std::cout << "size: " << size << std::endl;

    const float compare = 16.f;
    int pCountGold = 0; 

    float * h_data = rd::createRandomDataTable(size_t(size), 5.f, 250.f);
    // float * h_data = rd::createTable<float>(size_t(size), 0.f);
    char  * h_mask = rd::createTable<char >(size, char (1));

    // for (int i = 0; i < size; ++i)
    // {
    //     h_data[i] = 0.1f * i;
    // }

    for (int i = 0; i < size / 2; ++i)
    {
        h_mask[rd::getRandIndex(size)] = 0;
    }


    float * d_in;
    char  * d_mask;
    int * d_pCount;

    checkCudaErrors(cudaMalloc(&d_in, size * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_mask, size * sizeof(char )));
    checkCudaErrors(cudaMalloc(&d_pCount, sizeof(int)));

    checkCudaErrors(cudaMemcpy(d_in, h_data, size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_mask, h_mask, size * sizeof(char ), cudaMemcpyHostToDevice));

    pCountGold = cpuCountPoints(h_data, h_mask, size, compare, threshold, 64);
    test<64>(d_in, d_mask, size, compare, threshold, d_pCount, pCountGold);
    pCountGold = cpuCountPoints(h_data, h_mask, size, compare, threshold, 96);
    test<96>(d_in, d_mask, size, compare, threshold, d_pCount, pCountGold);
    pCountGold = cpuCountPoints(h_data, h_mask, size, compare, threshold, 128);
    test<128>(d_in, d_mask, size, compare, threshold, d_pCount, pCountGold);
    pCountGold = cpuCountPoints(h_data, h_mask, size, compare, threshold, 160);
    test<160>(d_in, d_mask, size, compare, threshold, d_pCount, pCountGold);
    pCountGold = cpuCountPoints(h_data, h_mask, size, compare, threshold, 192);
    test<192>(d_in, d_mask, size, compare, threshold, d_pCount, pCountGold);

    delete[] h_data;
    delete[] h_mask;

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_mask));
    checkCudaErrors(cudaFree(d_pCount));
}

int main(int argc, char const **argv)
{

    int size = 0;
    int threshold = 10;
    g_devId = 0;

    // Initialize command line
    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help")) 
    {
        printf("%s \n"
            "\t\t[--size=<size>]\n"
            "\t\t[--t=<threshold>]\n"
            "\t\t[--d=<device id>]\n"
            "\n", argv[0]);
        exit(0);
    }

    if (args.CheckCmdLineFlag("size")) 
    {
        args.GetCmdLineArgument("size", size);
    }
    if (args.CheckCmdLineFlag("d")) 
    {
        args.GetCmdLineArgument("d", g_devId);
    }
    if (args.CheckCmdLineFlag("t")) 
    {
        args.GetCmdLineArgument("t", threshold);
    }

    deviceInit(g_devId);

    test(size, threshold);

    checkCudaErrors(cudaDeviceReset());

    std::cout << "END!" << std::endl;

    return 0;
}
