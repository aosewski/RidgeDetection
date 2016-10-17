/**
 * @file test_dev_evolve.cu
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
#include <limits>
#ifdef RD_USE_OPENMP
#include <omp.h>
#endif

#include <helper_cuda.h>

#include "rd/gpu/device/brute_force/evolve.cuh"
#include "rd/gpu/device/device_evolve.cuh"
#include "rd/gpu/device/dispatch/dispatch_evolve.cuh"
#include "rd/gpu/device/brute_force/rd_globals.cuh"

#include "rd/cpu/samples_generator.hpp"
#include "rd/cpu/brute_force/choose.hpp"

#include "rd/gpu/util/dev_memcpy.cuh"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/memory.h"
#include "rd/utils/rd_params.hpp"
#include "evolve_gold.hpp"

#include "cub/test_util.h"
#include "cub/util_arch.cuh"

static const int CCSC_ITEMS_PER_THREAD   = 6;
static const int STMC_ITEMS_PER_THREAD   = 2;
static const int BLOCK_SIZE              = 128;
static const int BLOCK_COUNT             = 672;

static bool showResultDiff               = false;

template <int DIM, typename T>
void testEvolveKernel(rd::RDParams<T> &rdp,
                      rd::RDSpiralParams<T> const &rds);

int main(int argc, char const **argv)
{

    rd::RDParams<double> dParams;
    rd::RDSpiralParams<double> dSParams;
    rd::RDParams<float> fParams;
    rd::RDSpiralParams<float> fSParams;

    //-----------------------------------------------------------------

    // Initialize command line
    rd::CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help") && args.ParsedArgc() < 6) 
    {
        printf("%s \n"
            "\t\t[--np=<P size>]\n"
            "\t\t[--r1=<r1 param>]\n"
            "\t\t[--r2=<r2 param>]\n"
            "\t\t[--a=<spiral param>]\n"
            "\t\t[--b=<spiral param>]\n"
            "\t\t[--s=<spiral noise sigma>]\n"
            "\t\t[--d=<device id>]\n"
            "\t\t[--v <verbose>]\n"
            "\t\t[--diff <show results diff>]\n"
            "\n", argv[0]);
        exit(0);
    }

    args.GetCmdLineArgument("r1", dParams.r1);
    args.GetCmdLineArgument("r2", dParams.r2);

    args.GetCmdLineArgument("r1", fParams.r1);
    args.GetCmdLineArgument("r2", fParams.r2);


    args.GetCmdLineArgument("np", dParams.np);
    args.GetCmdLineArgument("np", fParams.np);

    if (args.CheckCmdLineFlag("a")) 
    {
        args.GetCmdLineArgument("a", fSParams.a);
        args.GetCmdLineArgument("a", dSParams.a);
    }
    if (args.CheckCmdLineFlag("b")) 
    {
        args.GetCmdLineArgument("b", fSParams.b);
        args.GetCmdLineArgument("b", dSParams.b);
    }
    if (args.CheckCmdLineFlag("s")) 
    {
        args.GetCmdLineArgument("s", fSParams.sigma);
        args.GetCmdLineArgument("s", dSParams.sigma);
    }
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
    if (args.CheckCmdLineFlag("diff")) 
    {
        showResultDiff = true;
    }

    fParams.dim = 2;
    dParams.dim = 2;

    checkCudaErrors(deviceInit(fParams.devId));

    std::cout << rd::HLINE << std::endl;
    std::cout << "2D: " << std::endl;
    std::cout << rd::HLINE << std::endl;

    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT: " << std::endl;
    testEvolveKernel<2>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE: " << std::endl;
    testEvolveKernel<2>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;

    std::cout << rd::HLINE << std::endl;
    std::cout << "3D: " << std::endl;
    std::cout << rd::HLINE << std::endl;

    fParams.dim = 3;
    dParams.dim = 3;

    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT: " << std::endl;
    testEvolveKernel<3>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE: " << std::endl;
    testEvolveKernel<3>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;

    checkCudaErrors(deviceReset());

    std::cout << "END!" << std::endl;
    return 0;
}

/**
 * @brief      Test calculate closest sphere center kernel.
 *
 * @param      rdp                        Simulation parameters
 * @param[in]  h_ns                       Number of chosen samples
 * @param      d_P                        All samples in device memory.
 * @param      d_S                        Chosen samples in device memory.
 * @param      d_cordSums                 Device memory ptr to store cord sums
 * @param      d_spherePointCount         Device memory ptr to store sphere point count
 * @param      h_cordSums                 Host memory ptr to store gpu results
 * @param      h_spherePointCount         Host memory ptr to store gpu results
 * @param      h_cordSumsGold             Reference values of cord sums
 * @param      h_spherePointCountGold     Reference values of speres point count
 *
 * @tparam     AgentClosestSpherePolicyT  Parameterized policy type for evoking kernel
 * @tparam     DIM                        Data dimensionality
 * @tparam     IN_MEM_LAYOUT              Input memory layout (data samples and cord sums)
 * @tparam     OUT_MEM_LAYOUT             Output memory layout (chosen samples)
 * @tparam     T                          Samples data type.
 */
template <
    typename    AgentClosestSpherePolicyT,
    int         DIM, 
    rd::DataMemoryLayout IN_MEM_LAYOUT,
    rd::DataMemoryLayout OUT_MEM_LAYOUT,
    typename    T>
void test_ccsc(
    rd::RDParams<T> const &rdp, 
    int h_ns,
    T const * d_P,
    T const * d_S,
    T * d_cordSums,
    int * d_spherePointCount,
    T const * h_cordSumsGold,
    int const * h_spherePointCountGold,
    int pStride,
    int sStride,
    int csStride)
{

    T * h_cordSums = rd::createTable(h_ns * DIM, T(0));
    int * h_spherePointCount = rd::createTable(h_ns, 0);

    std::cout << "--------__calc_closest_sphere_center--------" << std::endl;

    dim3 gridSize(BLOCK_COUNT);
    dim3 blockSize(BLOCK_SIZE);

    rd::gpu::bruteForce::detail::DeviceClosestSphereKernel<AgentClosestSpherePolicyT, 
        DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT><<<gridSize, blockSize>>>(
            d_P, d_S, d_cordSums, d_spherePointCount, rdp.np, h_ns, rdp.r1, pStride, csStride,
            sStride);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if (csStride != DIM)
    {
        checkCudaErrors(cudaMemcpy2D(h_cordSums, h_ns * sizeof(T), d_cordSums, csStride * sizeof(T),
            h_ns * sizeof(T), DIM, cudaMemcpyDeviceToHost));
    }
    else
    {
        checkCudaErrors(cudaMemcpy(h_cordSums, d_cordSums, h_ns * DIM * sizeof(T), 
            cudaMemcpyDeviceToHost));
    }

    checkCudaErrors(cudaMemcpy(h_spherePointCount, d_spherePointCount, h_ns * sizeof(int), 
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    //---------------------------------------------------------
    
    std::cout << "<<<< compare cordSums: " << std::endl;
    rd::checkResult(h_cordSumsGold, h_cordSums, h_ns * DIM, showResultDiff);
    std::cout << "<<<< compare spherePointCount: " << std::endl;
    rd::checkResult(h_spherePointCountGold, h_spherePointCount, h_ns, showResultDiff);

    delete[] h_cordSums;
    delete[] h_spherePointCount;
}

/**
 * @brief      Test shift toward mass center kernel.
 *
 * @param      rdp                      Simulation parameters
 * @param[in]  h_ns                     Number of chosen samples
 * @param      d_S                      Chosen samples in device memory.
 * @param      d_cordSums               Device memory ptr to store cord sums
 * @param      d_spherePointCount       Device memory ptr to store sphere point
 *                                      count
 * @param      h_S                      Host memory ptr to store gpu results
 * @param      h_SGold                  Reference CPU results
 * @param      sStride                  Distance between consecutive coordinates.
 *
 * @tparam     AgentShiftSpherePolicyT  Parameterized policy type for evoking
 *                                      kernel
 * @tparam     DIM                      Data dimensionality
 * @tparam     IN_MEM_LAYOUT            Input memory layout (data samples and
 *                                      cord sums)
 * @tparam     OUT_MEM_LAYOUT           Output memory layout (chosen samples)
 * @tparam     T                        Samples data type.
 */
template <
    typename    AgentShiftSpherePolicyT,
    int         DIM, 
    rd::DataMemoryLayout IN_MEM_LAYOUT,
    rd::DataMemoryLayout OUT_MEM_LAYOUT,
    typename    T>
void test_stmc(
    int h_ns,
    T * d_S,
    T const * d_cordSums,
    int const * d_spherePointCount,
    T const * h_SGold,
    int sStride,
    int csStride)
{
    T * h_S = rd::createTable(h_ns * DIM, T(0));

    dim3 gridSize(BLOCK_COUNT);
    dim3 blockSize(BLOCK_SIZE);

    std::cout << "--------__shift_toward_mass_center--------" << std::endl;

    rd::gpu::bruteForce::detail::DeviceShiftSphereKernel<AgentShiftSpherePolicyT, 
        DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT><<<gridSize, blockSize>>>(
        d_S, d_cordSums, d_spherePointCount, h_ns, csStride, sStride);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if (sStride != DIM)
    {
        checkCudaErrors(cudaMemcpy2D(h_S, h_ns * sizeof(T), d_S, sStride * sizeof(T), 
            h_ns * sizeof(T), DIM, cudaMemcpyDeviceToHost));
    }
    else
    {
        checkCudaErrors(cudaMemcpy(h_S, d_S, h_ns * DIM * sizeof(T), cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "<<<< compare h_S: " << std::endl;
    rd::checkResult(h_SGold, h_S, h_ns * DIM, showResultDiff);

    delete[] h_S;
}

/**
 * @brief      Test whole evolve pass
 *
 * @param      rdp                 Simulation parameters
 * @param[in]  h_ns                Number of chosen samples
 * @param      d_P                 All samples in device memory.
 * @param      d_S                 Chosen samples in device memory.
 * @param      d_cordSums          Device memory ptr to store cord sums
 * @param      d_spherePointCount  Device memory ptr to store sphere point count
 * @param      h_S                 Host memory ptr to store gpu results
 * @param      h_SGold             Reference CPU results
 * @param[in]  pStride             Distance between consecutive coordinates in
 *                                 @p d_P table.
 * @param[in]  sStride             Distance between consecutive coordinates in
 *                                 @p d_S table.
 *
 * @tparam     DIM                 Data dimensionality
 * @tparam     IN_MEM_LAYOUT       Input memory layout (data samples and cord
 *                                 sums)
 * @tparam     OUT_MEM_LAYOUT      Output memory layout (chosen samples)
 * @tparam     T                   Samples data type.
 */
template <
    int         DIM,
    rd::DataMemoryLayout  IN_MEM_LAYOUT,
    rd::DataMemoryLayout  OUT_MEM_LAYOUT,
    typename    T>
void test_evolve(
    rd::RDParams<T> const &rdp, 
    int h_ns,
    T const * d_P,
    T * d_S,
    T * d_cordSums,
    int * d_spherePointCount,
    T * h_initChosenPoints,
    T const * h_SGold,
    int pStride,
    int sStride,
    int csStride)
{
    // col-major layout, thus aligned 2D mem
    if (sStride != DIM)
    {
        checkCudaErrors(cudaMemcpy2D(d_S, sStride * sizeof(T), h_initChosenPoints, h_ns * sizeof(T),
            h_ns * sizeof(T), DIM, cudaMemcpyHostToDevice));
    } 
    else
    {
        checkCudaErrors(cudaMemcpy(d_S, h_initChosenPoints, h_ns * DIM * sizeof(T), 
            cudaMemcpyHostToDevice));
    }
    // input col-major layout, thus aligned 2d mem
    if (csStride != DIM)
    {
        checkCudaErrors(cudaMemset2D(d_cordSums, csStride * sizeof(T), 0, h_ns * sizeof(T), DIM));
    }
    else
    {
        checkCudaErrors(cudaMemset(d_cordSums, 0, h_ns * DIM * sizeof(T)));
    }
    checkCudaErrors(cudaMemset(d_spherePointCount, 0, h_ns * sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    T * h_S = rd::createTable(h_ns * DIM, T(0));

    std::cout << "-------- start! evolve --------" << std::endl;

    cudaError_t err = cudaSuccess;
    err = rd::gpu::bruteForce::DeviceEvolve::evolve<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
        d_P, d_S, d_cordSums, d_spherePointCount, rdp.np, h_ns, rdp.r1, pStride, sStride,
        csStride, nullptr, false);
    checkCudaErrors(err);
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "-------- end! evolve --------" << std::endl;

    if (sStride != DIM)
    {
        checkCudaErrors(cudaMemcpy2D(h_S, h_ns * sizeof(T), d_S, sStride * sizeof(T), 
            h_ns * sizeof(T), DIM, cudaMemcpyDeviceToHost));
    }
    else
    {
        checkCudaErrors(cudaMemcpy(h_S, d_S, h_ns * DIM * sizeof(T), cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "<<<< compare h_S: " << std::endl;
    rd::checkResult(h_SGold, h_S, h_ns * DIM, showResultDiff);

    delete[] h_S;
}

template <
    int         DIM,
    typename    ccsc_kernelPtrT,
    typename    stmc_kernelPtrT,
    typename    T>
void test_individual_kernels(
    rd::RDParams<T> const & rdp, 
    int                     h_ns,
    T const *               d_P,
    T *                     d_S,
    T *                     d_cordSums,
    int *                   d_spherePointCount,
    T const *               h_initChosenPoints,     // row-major
    T const *               h_SGold,                // row-major
    T const *               h_cordSumsGold,
    int const *             h_spherePointCountGold,
    int                     pStride,
    int                     sStride,
    int                     csStride,
    ccsc_kernelPtrT         ccscTestKernelPtr,
    stmc_kernelPtrT         stmcTestKernelPtr)
{
    // col-major layout, thus aligned 2D mem
    if (sStride != DIM)
    {
        checkCudaErrors(cudaMemcpy2D(d_S, sStride * sizeof(T), h_initChosenPoints, h_ns * sizeof(T),
            h_ns * sizeof(T), DIM, cudaMemcpyHostToDevice));
    } 
    else
    {
        checkCudaErrors(cudaMemcpy(d_S, h_initChosenPoints, h_ns * DIM * sizeof(T), 
            cudaMemcpyHostToDevice));
    }
    // input col-major layout, thus aligned 2d mem
    if (csStride != DIM)
    {
        checkCudaErrors(cudaMemset2D(d_cordSums, csStride * sizeof(T), 0, h_ns * sizeof(T), DIM));
    }
    else
    {
        checkCudaErrors(cudaMemset(d_cordSums, 0, h_ns * DIM * sizeof(T)));
    }
    checkCudaErrors(cudaMemset(d_spherePointCount, 0, h_ns * sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    ccscTestKernelPtr(
        rdp, h_ns, d_P, d_S, d_cordSums, d_spherePointCount,
        h_cordSumsGold, h_spherePointCountGold, pStride, sStride, csStride);

    stmcTestKernelPtr(
        h_ns, d_S, d_cordSums, d_spherePointCount, h_SGold, sStride, csStride);
}

template <
    int DIM,
    rd::DataMemoryLayout  IN_MEM_LAYOUT,
    rd::DataMemoryLayout  OUT_MEM_LAYOUT,
    typename T>
void test_individual_kernels(
    rd::RDParams<T> const &rdp, 
    int h_ns,
    T const * h_initChosenPoints,        // row-major
    T const * h_PGold)                   // row-major
{
    T *d_S, *d_P, *d_cordSums, *h_cordSumsGold, *h_SGold;
    int *d_spherePointCount, *h_spherePointCountGold;
    int pStride = DIM, sStride = DIM, csStride = DIM;
    size_t pPitch = 1, sPitch = 1, csPitch = 1;

    // allocate needed memory
    if (IN_MEM_LAYOUT == rd::COL_MAJOR)
    {
        checkCudaErrors(cudaMallocPitch(&d_P, &pPitch, rdp.np * sizeof(T), DIM));
        checkCudaErrors(cudaMallocPitch(&d_cordSums, &csPitch, h_ns * sizeof(T), DIM));
    }
    else 
    {
        checkCudaErrors(cudaMalloc((void**)&d_P, rdp.np * DIM * sizeof(T)));
        checkCudaErrors(cudaMalloc((void**)&d_cordSums, h_ns * DIM * sizeof(T)));
    }

    if (OUT_MEM_LAYOUT == rd::COL_MAJOR)
    {
        checkCudaErrors(cudaMallocPitch(&d_S, &sPitch, h_ns * sizeof(T), DIM));
    }
    else
    {
        checkCudaErrors(cudaMalloc((void**)&d_S, h_ns * DIM * sizeof(T)));
    }
    checkCudaErrors(cudaMalloc((void**)&d_spherePointCount, h_ns * sizeof(int)));
    
    h_SGold = new T[h_ns * DIM];
    h_cordSumsGold = new T[h_ns * DIM];
    h_spherePointCountGold = new int[h_ns];


    // initialize data
    rd::fillTable(h_cordSumsGold, T(0), h_ns * DIM);
    rd::fillTable(h_spherePointCountGold, 0, h_ns);
    rd::copyTable(h_initChosenPoints, h_SGold, h_ns * DIM);

    // typedefs
    typedef rd::gpu::bruteForce::AgentClosestSpherePolicy<
        BLOCK_SIZE, CCSC_ITEMS_PER_THREAD> 
    AgentClosestSpherePolicyT;

    typedef rd::gpu::bruteForce::AgentShiftSpherePolicy<
        BLOCK_SIZE, STMC_ITEMS_PER_THREAD> 
    AgentShiftSpherePolicyT;

    /************************************************************************
     *      REFERENCE VERSION
     ************************************************************************/

     ccsc_gold(h_PGold, h_SGold, h_cordSumsGold, h_spherePointCountGold,
        rdp.r1, rdp.np, h_ns, DIM);
     stmc_gold(h_SGold, h_cordSumsGold, h_spherePointCountGold, h_ns, DIM);

    /************************************************************************
     *      GPU VERSION respective kernels
     ************************************************************************/

    // prepare data in appropriate memory layout
    if (IN_MEM_LAYOUT == rd::COL_MAJOR)
    {
        pStride = pPitch / sizeof(T);
        csStride = csPitch / sizeof(T);

        rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_P, h_PGold, DIM, rdp.np, pPitch, DIM * sizeof(T));
        rd::transposeInPlace(h_cordSumsGold, h_cordSumsGold + h_ns * DIM, DIM);

        checkCudaErrors(cudaDeviceSynchronize());
    }
    else
    {
        checkCudaErrors(cudaMemcpy(d_P, h_PGold, rdp.np * DIM * sizeof(T), 
            cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
    }
    
    T * chosenPoints = new T[h_ns * DIM];

    if (OUT_MEM_LAYOUT == rd::COL_MAJOR)
    {
        sStride = sPitch / sizeof(T);

        rd::copyTable(h_initChosenPoints, chosenPoints, h_ns * DIM);
        rd::transposeInPlace(chosenPoints, chosenPoints + h_ns * DIM, DIM);
        rd::transposeInPlace(h_SGold, h_SGold + h_ns * DIM, DIM);
        
        checkCudaErrors(cudaDeviceSynchronize());
    }
    else
    {
        rd::copyTable(h_initChosenPoints, chosenPoints, h_ns * DIM);
    }

    std::cout << "\n//-----------------------------------------------\n"
              << "//  version 1\n" 
              << "//-----------------------------------------------\n" << std::endl;

    test_individual_kernels<DIM>(
        rdp, h_ns, d_P, d_S, d_cordSums, d_spherePointCount, chosenPoints,
        h_SGold, h_cordSumsGold, h_spherePointCountGold, pStride, sStride, csStride,
        test_ccsc<AgentClosestSpherePolicyT, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, T>,
        test_stmc<AgentShiftSpherePolicyT, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, T>);

    // clean-up
    delete[] h_cordSumsGold;
    delete[] h_SGold;
    delete[] h_spherePointCountGold;
    delete[] chosenPoints;

    checkCudaErrors(cudaFree(d_S));
    checkCudaErrors(cudaFree(d_P));
    checkCudaErrors(cudaFree(d_cordSums));
    checkCudaErrors(cudaFree(d_spherePointCount));
}

template <
    int DIM,
    rd::DataMemoryLayout  IN_MEM_LAYOUT,
    rd::DataMemoryLayout  OUT_MEM_LAYOUT,
    typename T>
void test_evolve_whole_pass(
    rd::RDParams<T> const &rdp, 
    int h_ns,
    T const * h_initChosenPoints,        // row-major
    T const * h_PGold,                   // row-major
    T const * h_SGold)                   // row-major
{
    T *d_S, *d_P, *d_cordSums, *h_SResult;
    int *d_spherePointCount;
    int pStride = DIM, sStride = DIM, csStride = DIM;
    size_t pPitch, sPitch, csPitch; 

    // allocate needed memory
    if (IN_MEM_LAYOUT == rd::COL_MAJOR)
    {
        checkCudaErrors(cudaMallocPitch(&d_P, &pPitch, rdp.np * sizeof(T), DIM));
        checkCudaErrors(cudaMallocPitch(&d_cordSums, &csPitch, h_ns * sizeof(T), DIM));
    }
    else
    {
        checkCudaErrors(cudaMalloc((void**)&d_P, rdp.np * DIM * sizeof(T)));
        checkCudaErrors(cudaMalloc((void**)&d_cordSums, h_ns * DIM * sizeof(T)));
    }
    if (OUT_MEM_LAYOUT == rd::COL_MAJOR)
    {
        checkCudaErrors(cudaMallocPitch(&d_S, &sPitch, h_ns * sizeof(T), DIM));
    }
    else
    {
        checkCudaErrors(cudaMalloc((void**)&d_S, h_ns * DIM * sizeof(T)));
    }
    checkCudaErrors(cudaMalloc((void**)&d_spherePointCount, h_ns * sizeof(int)));
    
    h_SResult = new T[h_ns * DIM];
    rd::copyTable(h_SGold, h_SResult, h_ns * DIM);
    std::cout << "prepare... " << std::endl;

    /************************************************************************
     *      GPU VERSION 
     ************************************************************************/

    // prepare data in appropriate memory layout
    if (IN_MEM_LAYOUT == rd::COL_MAJOR)
    {
        pStride = pPitch / sizeof(T);
        csStride = csPitch / sizeof(T);

        rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_P, h_PGold, DIM, rdp.np, pPitch, DIM * sizeof(T));

        checkCudaErrors(cudaDeviceSynchronize());
    }
    else
    {
        checkCudaErrors(cudaMemcpy(d_P, h_PGold, rdp.np * DIM * sizeof(T), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
    }
    
    T * chosenPoints = new T[h_ns * DIM];

    if (OUT_MEM_LAYOUT == rd::COL_MAJOR)
    {
        sStride = sPitch / sizeof(T);

        rd::copyTable(h_initChosenPoints, chosenPoints, h_ns * DIM);
        rd::transposeInPlace(chosenPoints, chosenPoints + h_ns * DIM, DIM);
        rd::transposeInPlace(h_SResult, h_SResult + h_ns * DIM, DIM);
        
        checkCudaErrors(cudaDeviceSynchronize());
    }
    else
    {
        rd::copyTable(h_initChosenPoints, chosenPoints, h_ns * DIM);
    }

    test_evolve<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(rdp, h_ns, d_P, d_S, d_cordSums, 
        d_spherePointCount, chosenPoints, h_SResult, pStride, sStride, csStride);

    // clean-up
    delete[] h_SResult;
    delete[] chosenPoints;

    checkCudaErrors(cudaFree(d_S));
    checkCudaErrors(cudaFree(d_P));
    checkCudaErrors(cudaFree(d_cordSums));
    checkCudaErrors(cudaFree(d_spherePointCount));
}

/**
 * @brief      Invoke tests for individual kernels.
 *
 * @param      rdp                 Simulation parameters
 * @param[in]  h_ns                Number of chosen samples
 * @param      h_initChosenPoints  The initialize chosen points
 * @param      h_PGold             All samples in host memory.
 *
 * @tparam     DIM                 Point dimension
 * @tparam     T                   Samples data type.
 */
template < int DIM, typename T>
void test(rd::RDParams<T> const &rdp, 
    int h_ns,
    T const *h_initChosenPoints,        // row-major
    T const *h_PGold)                   // row-major
{
    std::cout << rd::HLINE << std::endl;
    std::cout << ">>>> ROW-MAJOR - ROW-MAJOR" << std::endl;
    test_individual_kernels<DIM, rd::ROW_MAJOR, rd::ROW_MAJOR>(
        rdp, h_ns, h_initChosenPoints, h_PGold);

    std::cout << rd::HLINE << std::endl;
    std::cout << ">>>> COL-MAJOR - ROW-MAJOR" << std::endl;
    test_individual_kernels<DIM, rd::COL_MAJOR, rd::ROW_MAJOR>(
        rdp, h_ns, h_initChosenPoints, h_PGold);

    std::cout << rd::HLINE << std::endl;
    std::cout << ">>>> COL_MAJOR - COL_MAJOR" << std::endl;
    test_individual_kernels<DIM, rd::COL_MAJOR, rd::COL_MAJOR>(
        rdp, h_ns, h_initChosenPoints, h_PGold);

    /************************************************************************
     *      GPU VERSION whole evolve pass
     ************************************************************************/


    /************************************************************************
     *      REFERENCE VERSION
     ************************************************************************/
    T * h_SGold = new T[h_ns * DIM];
    // initialize data
    rd::copyTable(h_initChosenPoints, h_SGold, h_ns * DIM);

    std::cout << "evolve gold start... " << std::endl;
    evolve_gold(h_PGold, h_SGold, rdp.r1, rdp.np, h_ns, DIM);
    
    std::cout << "end!... " << std::endl;

    std::cout << rd::HLINE << std::endl;
    std::cout << "\nTest evolve whole pass...\n";

    //-----------------------------------------------------------------------

    // ------- ROW-MAJOR - ROW-MAJOR-----------------------------------------
    std::cout << rd::HLINE << std::endl;
    std::cout << ">>>> ROW-MAJOR - ROW-MAJOR" << std::endl;
    test_evolve_whole_pass<DIM, rd::ROW_MAJOR, rd::ROW_MAJOR>(
        rdp, h_ns, h_initChosenPoints, h_PGold, h_SGold);

    // ------- COL-MAJOR - ROW-MAJOR-----------------------------------------
    std::cout << rd::HLINE << std::endl;
    std::cout << ">>>> COL-MAJOR - ROW-MAJOR" << std::endl;
    test_evolve_whole_pass<DIM, rd::COL_MAJOR, rd::ROW_MAJOR>(
        rdp, h_ns, h_initChosenPoints, h_PGold, h_SGold);

    // ------- COL-MAJOR - COL-MAJOR-----------------------------------------
    std::cout << rd::HLINE << std::endl;
    std::cout << ">>>> COL-MAJOR - COL-MAJOR" << std::endl;
    test_evolve_whole_pass<DIM, rd::COL_MAJOR, rd::COL_MAJOR>(
        rdp, h_ns, h_initChosenPoints, h_PGold, h_SGold);

    //----------------------------------------------------------------------
    delete[] h_SGold;
}

template <int DIM, typename T>
void testEvolveKernel(rd::RDParams<T> &rdp,
                      rd::RDSpiralParams<T> const &sp)
{
    std::cout << "Samples: " << std::endl;
    std::cout <<  "\t dimension: " << rdp.dim << std::endl;
    std::cout <<  "\t n_samples: " << rdp.np << std::endl;
    std::cout <<  "\t r1: " << rdp.r1 << std::endl;
    std::cout <<  "\t r2: " << rdp.r2 << std::endl;

    std::cout << "Spiral params: " << std::endl;
    std::cout <<  "\t a: " << sp.a << std::endl;
    std::cout <<  "\t b: " << sp.b << std::endl;
    std::cout <<  "\t sigma: " << sp.sigma << std::endl; 

    rd::GraphDrawer<T> gDrawer;

    int *d_ns;
    T *h_P, *h_S;
    h_P = rd::createTable<T>(rdp.np * DIM, T(0));
    h_S = rd::createTable<T>(rdp.np * DIM, T(0));

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));

    switch(DIM)
    {
        case 2: rd::genSpiral2D(rdp.np, sp.a, sp.b, sp.sigma, h_P); break;
        case 3: rd::genSpiral3D(rdp.np, sp.a, sp.b, sp.sigma, h_P); break;
        default: rd::genSegmentND(rdp.np, DIM, sp.sigma, h_P, sp.a); break;
            
    }


    // {
    //     // rd::genCircle2D(rdp.np, T(0), T(0), rdp.r1 * 1.9, 0, h_P);

    //     int chosenPtsCnt = 130;
    //     // int chosenPtsCnt = rdp.np * 0.005;
    //     int ptsCntPerCircle = rdp.np / chosenPtsCnt;

    //     // rd::genCircle2D(rdp.np, T(0), T(0), rdp.r1 * T(0.7), T(0), h_P);

    //     T r = rdp.r1 * 0.7;
    //     T x0 = 0, y0 = 0;
    //     T * ptr = h_P;
    //     for (int k = 0; k < chosenPtsCnt; ++k)
    //     {
    //         h_S[2 * k] = x0 + 0.1;
    //         h_S[2 * k + 1] = y0 - 0.1;
    //         rd::genCircle2D(ptsCntPerCircle, x0, y0, r, T(0), ptr);
            
    //         if (k & 1)
    //         {
    //             x0 += 4.0 * rdp.r1;
    //         }
    //         else
    //         {
    //             y0 += 4.0 * rdp.r1;
    //         }
    //         ptr += 2 * ptsCntPerCircle;
    //     }
    //     // rd::genCircle2D(chosenPtsCnt, T(0), T(0), rdp.r1, 0, h_S);
    //     rdp.ns = chosenPtsCnt;
    // }


    //---------------------------------------------------
    //               CHOOSE 
    //---------------------------------------------------

    {
        std::list<T*> csList;
        rd::choose(h_P, h_S, csList, rdp.np, rdp.ns, size_t(DIM), rdp.r1);
    }

    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &rdp.ns, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Chosen count: " << rdp.ns << std::endl;

    if (rdp.verbose && DIM <= 3)
    {
        std::ostringstream os;
        os << typeid(T).name() << "_" << DIM << "D_gpu_initial_chosen_set";
        gDrawer.startGraph(os.str(), rdp.dim);
        gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
             h_P, rd::GraphDrawer<T>::POINTS, rdp.np);
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#38abe0' ps 1.3 ",
             h_S, rd::GraphDrawer<T>::POINTS, rdp.ns);
        gDrawer.endGraph();
        os.clear();
        os.str(std::string());
    }
    //---------------------------------------------------
    //          TEST EVOLVE
    //---------------------------------------------------

    rdp.devId = (rdp.devId != -1) ? rdp.devId : 0;

    /*
     * h_S - row-major
     * h_P - row-major
     */

    test<DIM>(rdp, rdp.ns, h_S, h_P);

    // clean-up
    delete[] h_P;
    delete[] h_S;
}

