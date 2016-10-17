/**
 * @file test_count_neighbours.cu
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

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <string>

#include "rd/gpu/device/brute_force/rd_globals.cuh"
#include "rd/gpu/device/samples_generator.cuh"
#include "rd/gpu/block/cta_count_neighbour_points.cuh"
#include "rd/cpu/brute_force/rd_inner.hpp"

#include "rd/gpu/util/data_order_traits.hpp"
#include "rd/gpu/util/dev_memcpy.cuh"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/rd_samples.cuh"

#include "cub/test_util.h"
#include "cub/util_device.cuh"

#include "rd/utils/rd_params.hpp"

#include <helper_cuda.h>


static const int TEST_DIM = 2;

template <typename T>
void testCountNeighboursKernel(rd::RDParams<T> &rdp,
                      rd::RDSpiralParams<T> const &rds);

int main(int argc, char const **argv)
{

    rd::RDParams<double> dParams;
    rd::RDSpiralParams<double> dSParams;
    rd::RDParams<float> fParams;
    rd::RDSpiralParams<float> fSParams;

    dSParams.sigma = 1;
    fSParams.sigma = 1.f;

    dParams.np = 5000;
    dParams.r1 = 20;
    dParams.r2 = 20;
    
    fParams.np = 5000;
    fParams.r1 = 20;
    fParams.r2 = 20;

    //-----------------------------------------------------------------

    // Initialize command line
    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help")) 
    {
        printf("%s \n"
            "\t\t[--np=<P size>]\n"
            "\t\t[--r1=<r1 param>]\n"
            "\t\t[--d=<device id>]\n"
            "\t\t[--v <verbose>]\n"
            "\n", argv[0]);
        exit(0);
    }

    args.GetCmdLineArgument("r1", dParams.r1);
    args.GetCmdLineArgument("r1", fParams.r1);


    args.GetCmdLineArgument("np", dParams.np);
    args.GetCmdLineArgument("np", fParams.np);

    if (args.CheckCmdLineFlag("d")) 
    {
        args.GetCmdLineArgument("d", fParams.devId);
        args.GetCmdLineArgument("d", dParams.devId);
    }

    if (args.CheckCmdLineFlag("v")) 
    {
        fParams.verbose = true;
        dParams.verbose = true;
    }

    deviceInit(fParams.devId);

    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT: " << std::endl;
    testCountNeighboursKernel<float>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE: " << std::endl;
    testCountNeighboursKernel<double>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;

    deviceReset();

    std::cout << "END!" << std::endl;
    return 0;
}

template <typename T>
bool countNeighboursGold(
    rd::RDParams<T> &rdp,
    T const *S,
    T const *origin,
    int treshold)
{
    return rd::countNeighbouringPoints(S, rdp.np, origin,
         TEST_DIM, rdp.r1 * rdp.r1, treshold);
}

template <
    int         BLOCK_SIZE,
    typename    T>
__global__ void __dispatch_count_neighbours_row_major(
        T const *  points,
        int        np,
        T const *  srcP,
        int        dim,
        T          r2,
        int        threshold,
        int *      result)
{
    int res = rd::gpu::ctaCountNeighbouringPoints<T, BLOCK_SIZE>(points, np, srcP, dim, r2,
                 threshold, rd::gpu::rowMajorOrderTag());
    if (threadIdx.x == 0)
        *result = res;

}

template <typename T>
void testCountNeighboursRowMajorOrder(
    rd::RDParams<T> &   rdp,
    T const *           d_S,
    T const *           d_origin,
    int                 NEIGHBOURS_THRESHOLD, 
    bool                hasNeighbours)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testCountNeighboursRowMajorOrder:" << std::endl;

    checkCudaErrors(cudaDeviceSynchronize());

    int *d_result, h_result;
    checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(int)));

    __dispatch_count_neighbours_row_major<256><<<1, 256>>>(d_S, rdp.np, d_origin, TEST_DIM,
         rdp.r1 * rdp.r1, NEIGHBOURS_THRESHOLD, d_result);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    if (static_cast<bool>(h_result) != hasNeighbours)
    {
        std::cout << "[ERROR!]";
    }
    else 
    {
        std::cout << "[SUCCESS!]";
    }

    std::cout << std::boolalpha << " is: <" << static_cast<bool>(h_result) << 
            ">, and should be: <" << hasNeighbours << ">" << std::endl;

    checkCudaErrors(cudaFree(d_result));
}

template <
    int         DIM,
    int         BLOCK_SIZE,
    typename    T>
__global__ void __dispatch_count_neighbours_row_major_v2(
        T const *  points,
        int        np,
        T const *  srcP,
        T          r2,
        int        threshold,
        int *      result)
{
    int res = rd::gpu::ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(points, np, srcP, r2,
                 threshold, rd::gpu::rowMajorOrderTag());
    if (threadIdx.x == 0)
    {
        *result = (res >= threshold) ? 1 : 0;
    }
}

template <typename T>
void testCountNeighboursRowMajorOrder_v2(
    rd::RDParams<T> const & rdp,
    T const *           d_S,
    T const *           d_origin,
    int                 NEIGHBOURS_THRESHOLD, 
    bool                hasNeighbours)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testCountNeighboursRowMajorOrder_v2:" << std::endl;

    checkCudaErrors(cudaDeviceSynchronize());

    int *d_result, h_result;

    checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(int)));

    __dispatch_count_neighbours_row_major_v2<TEST_DIM, 256><<<1, 256>>>(d_S, rdp.np, d_origin, 
         rdp.r1 * rdp.r1, NEIGHBOURS_THRESHOLD, d_result);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    if (static_cast<bool>(h_result) != hasNeighbours)
    {
        std::cout << "[ERROR!]";
    }
    else 
    {
        std::cout << "[SUCCESS!]";
    }
    std::cout << std::boolalpha << " is: <" << static_cast<bool>(h_result) << 
            ">, and should be: <" << hasNeighbours << ">" << std::endl;

    checkCudaErrors(cudaFree(d_result));
}


template <
    int         DIM,
    int         BLOCK_SIZE,
    typename    T>
__global__ void __dispatch_count_neighbours_mixed_order_v2(
        T const *  points,
        int        np,
        T const *  srcP,
        int        stride,
        T          r2,
        int        threshold,
        int *      result)
{
    int res = rd::gpu::ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(points, np, srcP, stride, r2,
                 threshold, rd::gpu::rowMajorOrderTag());
    if (threadIdx.x == 0)
    {
        *result = (res >= threshold) ? 1 : 0;
    }
}

template <typename T>
void testCountNeighboursMixedOrder_v2(
    rd::RDParams<T> const & rdp,
    T const *           d_S,
    T const *           d_origin,
    int                 NEIGHBOURS_THRESHOLD, 
    bool                hasNeighbours)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testCountNeighboursMixedOrder_v2:" << std::endl;

    checkCudaErrors(cudaDeviceSynchronize());

    int *d_result, h_result;

    checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(int)));

    __dispatch_count_neighbours_mixed_order_v2<TEST_DIM, 256><<<1, 256>>>(d_S, rdp.np, d_origin, 
         1, rdp.r1 * rdp.r1, NEIGHBOURS_THRESHOLD, d_result);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    if (static_cast<bool>(h_result) != hasNeighbours)
    {
        std::cout << "[ERROR!]";
    }
    else 
    {
        std::cout << "[SUCCESS!]";
    }
    std::cout << std::boolalpha << " is: <" << static_cast<bool>(h_result) << 
            ">, and should be: <" << hasNeighbours << ">" << std::endl;

    checkCudaErrors(cudaFree(d_result));
}


template <
    int         DIM,
    int         BLOCK_SIZE,
    typename    T>
__global__ void __dispatch_count_neighbours_col_major_order_v2(
        T const *  points,
        int        np,
        int        pStride,
        T const *  srcP,
        int        sStride,
        T          r2,
        int        threshold,
        int *      result)
{
    int res = rd::gpu::ctaCountNeighbouringPoints_v2<DIM, BLOCK_SIZE>(points, np, pStride, srcP,
                 sStride, r2, threshold, rd::gpu::colMajorOrderTag());
    if (threadIdx.x == 0)
    {
        *result = (res >= threshold) ? 1 : 0;
    }

}

template <typename T>
void testCountNeighboursColMajorOrder_v2(
    rd::RDParams<T> const & rdp,
    T const *           d_SInitial,
    T const *           d_origin,
    int                 NEIGHBOURS_THRESHOLD, 
    bool                hasNeighbours)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testCountNeighboursColMajorOrder_v2:" << std::endl;

    checkCudaErrors(cudaDeviceSynchronize());

    int *d_result, h_result;
    T *d_S, *aux;

    aux = new T[rdp.np * TEST_DIM];

    checkCudaErrors(cudaMalloc((void**)&d_result, sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&d_S, rdp.np * TEST_DIM * sizeof(T)));

    rd::gpu::rdMemcpy<TEST_DIM, rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyDeviceToHost>(
        aux, d_SInitial, rdp.np, rdp.np, TEST_DIM);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(d_S, aux, rdp.np * TEST_DIM * sizeof(T), cudaMemcpyHostToDevice));

    __dispatch_count_neighbours_col_major_order_v2<TEST_DIM, 256><<<1, 256>>>(d_S, rdp.np, rdp.np, d_origin, 
         1, rdp.r1 * rdp.r1, NEIGHBOURS_THRESHOLD, d_result);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    if (static_cast<bool>(h_result) != hasNeighbours)
    {
        std::cout << "[ERROR!]";
    }
    else 
    {
        std::cout << "[SUCCESS!]";
    }
    std::cout << std::boolalpha << " is: <" << static_cast<bool>(h_result) << 
            ">, and should be: <" << hasNeighbours << ">" << std::endl;

    delete[] aux;

    checkCudaErrors(cudaFree(d_S));
    checkCudaErrors(cudaFree(d_result));
}

template <typename T>
void testCountNeighboursKernel(rd::RDParams<T> &rdp,
                      rd::RDSpiralParams<T> const &sp)
{
    const int NEIGHBOURS_THRESHOLD = 100;
    const int oldCount = rdp.np;
    const int closerRadius = rdp.r1;
    rdp.np += NEIGHBOURS_THRESHOLD;
    rdp.r1 += 10;

    std::cout << "Samples: " << std::endl;
    std::cout <<  "\t dimension: " << TEST_DIM << std::endl;
    std::cout <<  "\t n_samples: " << rdp.np << std::endl;
    std::cout <<  "\t r1: " << rdp.r1 << std::endl;
    std::cout <<  "\t sigma: " << sp.sigma << std::endl; 

    rd::GraphDrawer<T> gDrawer;

    T *d_S, *d_origin;
    T *h_S, *h_origin;

    checkCudaErrors(cudaMalloc((void**)&d_S, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_origin, TEST_DIM * sizeof(T)));

    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMemset(d_origin, 0, TEST_DIM * sizeof(T)));

    h_S = new T[rdp.np * TEST_DIM];
    h_origin = new T[TEST_DIM];

    for (int d = 0; d < TEST_DIM; ++d)
        h_origin[d] = 0;

    switch(TEST_DIM)
    {
        case 2:
            rd::gpu::SamplesGenerator<T>::template circle<rd::ROW_MAJOR>(
                oldCount, T(0), T(0), rdp.r1, sp.sigma, d_S);
            rd::gpu::SamplesGenerator<T>::template circle<rd::ROW_MAJOR>(
                NEIGHBOURS_THRESHOLD, T(0), T(0), closerRadius, sp.sigma, d_S + oldCount * TEST_DIM);
            break;
        case 3:
            rd::gpu::SamplesGenerator<T>::template sphere<rd::ROW_MAJOR>(
                oldCount, T(0), T(0), T(0), rdp.r1, sp.sigma, d_S);
            rd::gpu::SamplesGenerator<T>::template sphere<rd::ROW_MAJOR>(
                NEIGHBOURS_THRESHOLD, T(0), T(0), T(0), closerRadius, sp.sigma, d_S + oldCount * TEST_DIM);
            break;
        default:
            throw std::logic_error("Not supported dimension!");
    }

    // decrease radius for not taking larger circle into consideration 
    rdp.r1 -= 5;

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_S, d_S, rdp.np * TEST_DIM * sizeof(T), 
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    std::ostringstream os;
    if (rdp.verbose)
    {
        os << typeid(T).name() << "_" << TEST_DIM;
        os << "D_initial_samples_set_";
        gDrawer.showPoints(os.str(), h_S, rdp.np, TEST_DIM);
        os.clear();
        os.str(std::string());
    }

    //---------------------------------------------------
    //               REFERENCE COUNT_NEIGHBOURS 
    //---------------------------------------------------

    bool hasNeighbours = countNeighboursGold(rdp, h_S, h_origin, NEIGHBOURS_THRESHOLD);

    //---------------------------------------------------
    //               GPU COUNT_NEIGHBOURS 
    //---------------------------------------------------

    rdp.devId = (rdp.devId != -1) ? rdp.devId : 0;

    // int smVersion;
    // checkCudaErrors(cub::SmVersion(smVersion, rdp.devId));

    // testCountNeighboursRowMajorOrder(rdp, d_S, d_origin, NEIGHBOURS_THRESHOLD, hasNeighbours);
    testCountNeighboursRowMajorOrder_v2(rdp, d_S, d_origin, NEIGHBOURS_THRESHOLD, hasNeighbours);
    testCountNeighboursColMajorOrder_v2(rdp, d_S, d_origin, NEIGHBOURS_THRESHOLD, hasNeighbours);
    testCountNeighboursMixedOrder_v2(rdp, d_S, d_origin, NEIGHBOURS_THRESHOLD, hasNeighbours);

    // clean-up
    delete[] h_S;
    delete[] h_origin;

    checkCudaErrors(cudaFree(d_S));
    checkCudaErrors(cudaFree(d_origin));
}

