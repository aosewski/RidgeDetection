/**
 * @file test_decimate.cu
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

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <list>

#define BLOCK_TILE_LOAD_V4 1

#include "rd/cpu/brute_force/choose.hpp"
#include "rd/cpu/brute_force/decimate.hpp"

#include "rd/gpu/device/brute_force/decimate.cuh"
#include "rd/gpu/device/device_decimate.cuh"
#include "rd/gpu/device/brute_force/rd_globals.cuh"
#include "rd/gpu/device/samples_generator.cuh"
#include "rd/gpu/util/data_order_traits.hpp"
#include "rd/gpu/util/dev_memcpy.cuh"

#include "rd/gpu/device/brute_force/test/decimate_nan_marker1.cuh"
#include "rd/gpu/device/brute_force/decimate_dist_mtx.cuh"

#include "rd/utils/memory.h"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/rd_samples.cuh"
#include "rd/utils/rd_params.hpp"
#include "rd/utils/name_traits.hpp"

#include "cub/test_util.h"
#include "cub/util_device.cuh"
#include "cub/util_type.cuh"

static const int TEST_DIM = 2;
static const int BLOCK_SIZE = 512;

template <typename T>
void testDecimateKernel(rd::RDParams<T> &rdp,
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
    if (args.CheckCmdLineFlag("help")) 
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

    deviceInit(fParams.devId);

    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT: " << std::endl;
    testDecimateKernel<float>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    // std::cout << "DOUBLE: " << std::endl;
    // testDecimateKernel<double>(dParams, dSParams);
    // std::cout << rd::HLINE << std::endl;

    deviceReset();

    std::cout << "END!" << std::endl;
    return 0;
}

template <typename T>
void decimateGold(
    rd::RDParams<T> &rdp,
    T *P,
    T *S,
    T *chosenS,
    int &chosenCount)
{
    std::list<T*> csList;
    rd::choose(P, S, csList, rdp.np, rdp.ns, TEST_DIM, rdp.r1);

    chosenCount = rdp.ns;
    rd::copyTable(S, chosenS, chosenCount * TEST_DIM);

    std::cout << "Chosen count: " << rdp.ns << std::endl;
    std::ostringstream os;
    rd::GraphDrawer<T> gDrawer;

    // rd::printTable(chosenS, TEST_DIM, chosenCount, "initial chosen smpl");

    if (rdp.verbose)
    {
        os << typeid(T).name() << "_" << TEST_DIM << "D_ref_chosen_set";
        gDrawer.startGraph(os.str(), TEST_DIM);
        // gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
        //      P, rd::GraphDrawer<T>::POINTS, rdp.np);
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#38abe0' ps 1.3 ",
             S, rd::GraphDrawer<T>::POINTS, rdp.ns);
        gDrawer.endGraph();
        os.clear();
        os.str(std::string());
    }

    rd::decimate(S, csList, rdp.ns, TEST_DIM, rdp.r2);

    std::cout << "Decimate count: " << rdp.ns << std::endl;
    // rd::printTable(S, TEST_DIM, rdp.ns, "cpu decimate smpl");

    if (rdp.verbose)
    {
        os << typeid(T).name() << "_" << TEST_DIM << "D_ref_decimate";
        gDrawer.startGraph(os.str(), TEST_DIM);
        // gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
        //      P, rd::GraphDrawer<T>::POINTS, rdp.np);
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#38abe0' ps 1.3 ",
             S, rd::GraphDrawer<T>::POINTS, rdp.ns);
        gDrawer.endGraph();
        os.clear();
        os.str(std::string());
    }
    
}

template <
    rd::DataMemoryLayout    MEM_LAYOUT,
    typename                DecimateKernelPtr,
    typename                T>
void testDecimateNaNMark(
    rd::RDParams<T> const &rdp,
    T *                 d_S,
    T *                 h_chosenS,
    int                 h_chosenCount,
    T const *           S_gold,
    DecimateKernelPtr   kernelPtr)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testDecimateNaNMark: (" 
        << rd::DataMemoryLayoutNameTraits<MEM_LAYOUT>::name << ")" << std::endl;

    T *S_gpu;
    int *d_ns, h_ns;

    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));

    // get chosen samples to device memory properly ordered
    rd::gpu::rdMemcpy<MEM_LAYOUT, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
        d_S, h_chosenS, TEST_DIM, h_chosenCount,
        (MEM_LAYOUT == rd::COL_MAJOR) ? h_chosenCount : TEST_DIM,
        TEST_DIM);

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    int stride = (MEM_LAYOUT == rd::COL_MAJOR) ? h_chosenCount : TEST_DIM;

    kernelPtr<<<1, BLOCK_SIZE>>>(d_S, d_ns, rdp.r2, stride, cub::Int2Type<MEM_LAYOUT>());

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyFromSymbol(&h_ns, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    if ((int)rdp.ns != h_ns)
    {
        std::cout << "[ERROR]Incorrect number of chosen samples!" << std::endl;
        std::cout << "Is: " << h_ns << " and should be: " << rdp.ns << std::endl;
    }

    S_gpu = new T[h_chosenCount * TEST_DIM];
    rd::gpu::rdMemcpy<rd::ROW_MAJOR, MEM_LAYOUT, cudaMemcpyDeviceToHost>(
        S_gpu, d_S, (MEM_LAYOUT == rd::COL_MAJOR) ? h_chosenCount : TEST_DIM,
        (MEM_LAYOUT == rd::COL_MAJOR) ? TEST_DIM : h_chosenCount,
        TEST_DIM, (MEM_LAYOUT == rd::COL_MAJOR) ? h_chosenCount : TEST_DIM);
    checkCudaErrors(cudaDeviceSynchronize());

    if (rdp.verbose)
    {
        rd::GraphDrawer<T> gDrawer;
        std::ostringstream os;
        os << typeid(T).name() << "_" << TEST_DIM << "D_gpu_decimate";
        gDrawer.startGraph(os.str(), TEST_DIM);
        // gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
        //     P, rd::GraphDrawer<T>::POINTS, rdp.np);
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#38abe0' ps 1.3 ",
            S_gpu, rd::GraphDrawer<T>::POINTS, h_ns);
        gDrawer.endGraph();
        os.clear();
        os.str(std::string());
    }

    T * aux = new T[h_chosenCount * TEST_DIM];
    rd::copyTable_omp(S_gold, aux, rdp.ns * TEST_DIM);

    std::sort(aux, aux + rdp.ns * TEST_DIM);
    std::sort(S_gpu, S_gpu + rdp.ns * TEST_DIM);

    rd::checkResult(aux, S_gpu, rdp.ns * TEST_DIM, rdp.verbose);

    delete[] S_gpu;
}


template <
    typename                DecimateKernelPtr,
    typename                T>
void testDecimateNaNMark_alignedMem(
    rd::RDParams<T> const &rdp,
    T *                 h_chosenS,
    int                 h_chosenCount,
    T const *           S_gold,
    DecimateKernelPtr   kernelPtr)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testDecimateNaNMark_alignedMem: (COL_MAJOR)" << std::endl;

    size_t pitch;
    T * d_S;
    checkCudaErrors(cudaMallocPitch(&d_S, &pitch, h_chosenCount * sizeof(T), TEST_DIM));

    T *S_gpu;
    int *d_ns, h_ns;

    checkCudaErrors(cudaMemset2D(d_S, pitch, 0, h_chosenCount, TEST_DIM));

    // get chosen samples to device memory properly ordered
    rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
        d_S, h_chosenS, TEST_DIM, h_chosenCount, pitch, TEST_DIM * sizeof(T));

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    int stride = pitch / sizeof(T);

    kernelPtr<<<1, BLOCK_SIZE>>>(d_S, d_ns, rdp.r2, stride, cub::Int2Type<rd::COL_MAJOR>());

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyFromSymbol(&h_ns, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    if ((int)rdp.ns != h_ns)
    {
        std::cout << "[ERROR]Incorrect number of chosen samples!" << std::endl;
        std::cout << "Is: " << h_ns << " and should be: " << rdp.ns << std::endl;
    }

    S_gpu = new T[h_chosenCount * TEST_DIM];
    rd::gpu::rdMemcpy2D<rd::ROW_MAJOR, rd::COL_MAJOR, cudaMemcpyDeviceToHost>(
        S_gpu, d_S, h_chosenCount, TEST_DIM, TEST_DIM * sizeof(T), pitch);
    checkCudaErrors(cudaDeviceSynchronize());

    if (rdp.verbose)
    {
        rd::GraphDrawer<T> gDrawer;
        std::ostringstream os;
        os << typeid(T).name() << "_" << TEST_DIM << "D_gpu_decimate";
        gDrawer.startGraph(os.str(), TEST_DIM);
        // gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
        //     P, rd::GraphDrawer<T>::POINTS, rdp.np);
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#38abe0' ps 1.3 ",
            S_gpu, rd::GraphDrawer<T>::POINTS, h_ns);
        gDrawer.endGraph();
        os.clear();
        os.str(std::string());
    }


    T * aux = new T[h_chosenCount * TEST_DIM];
    rd::copyTable_omp(S_gold, aux, rdp.ns * TEST_DIM);

    std::sort(aux, aux + rdp.ns * TEST_DIM);
    std::sort(S_gpu, S_gpu + rdp.ns * TEST_DIM);

    rd::checkResult(aux, S_gpu, rdp.ns * TEST_DIM, rdp.verbose);

    delete[] S_gpu;
    checkCudaErrors(cudaFree(d_S));
}

template <typename T>
static void getChosenPtsFromMask(
    char const *    h_gpuMask, 
    T const *       h_chosenS, 
    T *             S_gpu, 
    int             h_chosenCount)
{

    int index = 0;
    for (int i = 0; i < h_chosenCount; ++i)
    {
        if (h_gpuMask[i])
        {
            for (int d = 0; d < TEST_DIM; ++d)
            {
                S_gpu[index * TEST_DIM + d] = h_chosenS[i * TEST_DIM + d];
            }
            index++;
        }
    }

}

template <
    rd::DataMemoryLayout    MEM_LAYOUT,
    typename                DecimateKernelPtr,
    typename                T>
void testDecimateDistMtx(
    rd::RDParams<T> const &rdp,
    T *                 h_chosenS,
    int                 h_chosenCount,
    T const *           S_gold,
    DecimateKernelPtr   kernelPtr)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testDecimateDistMtx: ("
        << rd::DataMemoryLayoutNameTraits<MEM_LAYOUT>::name << ")" << std::endl;

    size_t sPitch, distMtxPitch;
    T * d_S, *d_distMtx;
    char *d_mask;
    if (MEM_LAYOUT == rd::ROW_MAJOR)
    {
        checkCudaErrors(cudaMalloc(&d_S, TEST_DIM * h_chosenCount * sizeof(T)));
        sPitch = TEST_DIM * sizeof(T);
        checkCudaErrors(cudaMemset(d_S, 0, h_chosenCount * TEST_DIM * sizeof(T)));
        checkCudaErrors(cudaMemcpy(d_S, h_chosenS, h_chosenCount * TEST_DIM * sizeof(T),
            cudaMemcpyHostToDevice));
    }
    else if (MEM_LAYOUT == rd::COL_MAJOR)
    {
        checkCudaErrors(cudaMallocPitch(&d_S, &sPitch, h_chosenCount * sizeof(T), TEST_DIM));
        checkCudaErrors(cudaMemset2D(d_S, sPitch, 0, h_chosenCount * sizeof(T), TEST_DIM));
        // get chosen samples to device memory properly ordered
        rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
            d_S, h_chosenS, TEST_DIM, h_chosenCount, sPitch, TEST_DIM * sizeof(T));
    }
    else 
    {
        throw std::runtime_error("Unsupported memory layout!");
    }
    checkCudaErrors(cudaMallocPitch(&d_distMtx, &distMtxPitch, h_chosenCount * sizeof(T), 
        h_chosenCount));
    checkCudaErrors(cudaMalloc(&d_mask, h_chosenCount * sizeof(char)));

    T *S_gpu;
    int *d_ns, h_ns;

    // checkCudaErrors(cudaMemset2D(d_distMtx, distMtxPitch, 0, h_chosenCount * sizeof(T), 
    //     h_chosenCount));
    // checkCudaErrors(cudaMemset(d_mask, 1, h_chosenCount * sizeof(char)));

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    int sStride = sPitch / sizeof(T);
    int distMtxStride = distMtxPitch / sizeof(T);

    kernelPtr<<<1, BLOCK_SIZE>>>(d_S, d_ns, sStride, d_distMtx, distMtxStride, d_mask, rdp.r2);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyFromSymbol(&h_ns, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    if ((int)rdp.ns != h_ns)
    {
        std::cout << "[ERROR]Incorrect number of chosen samples!" << std::endl;
        std::cout << "Is: " << h_ns << " and should be: " << rdp.ns << std::endl;

        checkCudaErrors(cudaFree(d_S));
        checkCudaErrors(cudaFree(d_distMtx));
        checkCudaErrors(cudaFree(d_mask));
        return;
    }

    S_gpu = new T[h_ns * TEST_DIM];
    if (MEM_LAYOUT == rd::COL_MAJOR)
    {
        rd::gpu::rdMemcpy2D<rd::ROW_MAJOR, rd::COL_MAJOR, cudaMemcpyDeviceToHost>(
            S_gpu, d_S, h_ns, TEST_DIM, TEST_DIM * sizeof(T), sPitch);
    }
    else if (MEM_LAYOUT == rd::ROW_MAJOR)
    {
        checkCudaErrors(cudaMemcpy(S_gpu, d_S, h_ns * TEST_DIM * sizeof(T), 
            cudaMemcpyDeviceToHost));
    }
    else
    {
        throw std::runtime_error("Unsupported memory layout!");
    }
    checkCudaErrors(cudaDeviceSynchronize());

    if (rdp.verbose)
    {
        rd::GraphDrawer<T> gDrawer;
        std::ostringstream os;
        os << typeid(T).name() << "_" << TEST_DIM << "D_gpu_decimate";
        gDrawer.startGraph(os.str(), TEST_DIM);
        // gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
        //     P, rd::GraphDrawer<T>::POINTS, rdp.np);
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#38abe0' ps 1.3 ",
            S_gpu, rd::GraphDrawer<T>::POINTS, h_ns);
        gDrawer.endGraph();
        os.clear();
        os.str(std::string());
    }

    char * h_gpuMask = new char[h_chosenCount];
    T * S_gpu2 = new T[h_ns * TEST_DIM];
    checkCudaErrors(cudaMemcpy(h_gpuMask, d_mask, h_chosenCount * sizeof(char), 
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    getChosenPtsFromMask(h_gpuMask, h_chosenS, S_gpu2, h_chosenCount);

    T * aux = new T[h_chosenCount * TEST_DIM];
    rd::copyTable_omp(S_gold, aux, rdp.ns * TEST_DIM);

    std::sort(aux, aux + rdp.ns * TEST_DIM);
    std::sort(S_gpu, S_gpu + rdp.ns * TEST_DIM);
    std::sort(S_gpu2, S_gpu2 + rdp.ns * TEST_DIM);

    rd::checkResult(aux, S_gpu2, rdp.ns * TEST_DIM, rdp.verbose);
    rd::checkResult(aux, S_gpu, rdp.ns * TEST_DIM, rdp.verbose);

    delete[] S_gpu;
    delete[] S_gpu2;
    delete[] h_gpuMask;
    delete[] aux;
    checkCudaErrors(cudaFree(d_S));
    checkCudaErrors(cudaFree(d_distMtx));
    checkCudaErrors(cudaFree(d_mask));
}

template <typename T>
void testDecimateKernel(rd::RDParams<T> &rdp,
                      rd::RDSpiralParams<T> const &sp)
{

    std::cout << "Samples: " << std::endl;
    std::cout <<  "\t dimension: " << TEST_DIM << std::endl;
    std::cout <<  "\t n_samples: " << rdp.np << std::endl;
    std::cout <<  "\t r1: " << rdp.r1 << std::endl;
    std::cout <<  "\t r2: " << rdp.r2 << std::endl;

    std::cout << "Spiral params: " << std::endl;
    std::cout <<  "\t a: " << sp.a << std::endl;
    std::cout <<  "\t b: " << sp.b << std::endl;
    std::cout <<  "\t sigma: " << sp.sigma << std::endl; 

    T *d_P, *d_S;
    T *h_P, *h_S, *h_chosenS;

    checkCudaErrors(cudaMalloc((void**)&d_P, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_S, rdp.np * TEST_DIM * sizeof(T)));

    checkCudaErrors(cudaMemset(d_P, 0, rdp.np * TEST_DIM * sizeof(T)));

    // h_P = rd::createTable<T>(rdp.np * TEST_DIM, T(1));
    h_P = new T[rdp.np * TEST_DIM];
    h_S = new T[rdp.np * TEST_DIM];
    h_chosenS = new T[rdp.np * TEST_DIM];

    // for (size_t i = 0; i < rdp.np; ++i)
    // {
    //     h_P[i * TEST_DIM] = i;
    // }

    switch(TEST_DIM)
    {
        case 2:
            rd::gpu::SamplesGenerator<T>::template spiral2D<rd::COL_MAJOR>(
                rdp.np, sp.a, sp.b, sp.sigma, d_P);
            break;
        case 3:
            rd::gpu::SamplesGenerator<T>::template spiral3D<rd::COL_MAJOR>(
                rdp.np, sp.a, sp.b, sp.sigma, d_P);
            break;
        default:
            throw std::logic_error("Not supported dimension!");
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_P, d_P, rdp.np * TEST_DIM * sizeof(T), 
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    rd::transposeInPlace(h_P, h_P + rdp.np * TEST_DIM, rdp.np);

    if (rdp.verbose)
    {
        std::ostringstream os;
        rd::GraphDrawer<T> gDrawer;
        os << typeid(T).name() << "_" << TEST_DIM;
        os << "D_initial_samples_set_";
        gDrawer.showPoints(os.str(), h_P, rdp.np, TEST_DIM);
        os.clear();
        os.str(std::string());
    }

    //---------------------------------------------------
    //               REFERENCE DECIMATE 
    //---------------------------------------------------

    int h_chosenCount;
    decimateGold(rdp, h_P, h_S, h_chosenS, h_chosenCount);

    //---------------------------------------------------
    //               GPU DECIMATE 
    //---------------------------------------------------

    rdp.devId = (rdp.devId != -1) ? rdp.devId : 0;

    //---------------------------------------------------
    //               1st version - basic NaN marker version
    //---------------------------------------------------

    typedef void (*DecimateKernelPtr1_RM)(T*, int*, T const, int, cub::Int2Type<rd::ROW_MAJOR>);
    typedef void (*DecimateKernelPtr1_CM)(T*, int*, T const, int, cub::Int2Type<rd::COL_MAJOR>);

    DecimateKernelPtr1_RM kernelPtr1_RM = rd::gpu::bruteForce::decimateNanMarker1<BLOCK_SIZE, TEST_DIM, T>;
    DecimateKernelPtr1_CM kernelPtr1_CM = rd::gpu::bruteForce::decimateNanMarker1<BLOCK_SIZE, TEST_DIM, T>;

    testDecimateNaNMark<rd::ROW_MAJOR>(rdp, d_S, h_chosenS, h_chosenCount, h_S, kernelPtr1_RM);
    testDecimateNaNMark<rd::COL_MAJOR>(rdp, d_S, h_chosenS, h_chosenCount, h_S, kernelPtr1_CM);


    //---------------------------------------------------
    //               3rd version - just mtx row alignment in memory for 1st version
    //---------------------------------------------------

    typedef void (*DecimateKernelPtr3_CM)(T*, int*, T const, int, cub::Int2Type<rd::COL_MAJOR>);
    DecimateKernelPtr3_CM kernelPtr3_CM = rd::gpu::bruteForce::decimateNanMarker1<BLOCK_SIZE, TEST_DIM, T>;

    // assume COL_MAJOR
    testDecimateNaNMark_alignedMem(rdp, h_chosenS, h_chosenCount, h_S, kernelPtr3_CM);


    //---------------------------------------------------
    //               4th version - completly new way, 
    // 1) compute dist mtx, 2) reduce dist mtx, 3) reduce points
    //---------------------------------------------------

    auto kernelPtr4_RM = rd::gpu::bruteForce::decimateDistMtx<TEST_DIM, BLOCK_SIZE, rd::ROW_MAJOR, T>;
    auto kernelPtr4_CM = rd::gpu::bruteForce::decimateDistMtx<TEST_DIM, BLOCK_SIZE, rd::COL_MAJOR, T>;

    testDecimateDistMtx<rd::ROW_MAJOR>(rdp, h_chosenS, h_chosenCount, h_S, kernelPtr4_RM);
    testDecimateDistMtx<rd::COL_MAJOR>(rdp, h_chosenS, h_chosenCount, h_S, kernelPtr4_CM);

    // clean-up
    delete[] h_P;
    delete[] h_S;
    delete[] h_chosenS;

    checkCudaErrors(cudaFree(d_P));
    checkCudaErrors(cudaFree(d_S));
}

