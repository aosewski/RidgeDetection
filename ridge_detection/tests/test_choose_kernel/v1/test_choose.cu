/**
 * @file test_choose.cu
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
#include <list>

#include "rd/cpu/brute_force/choose.hpp"
#include "rd/gpu/device/brute_force/choose.cuh"
#include "rd/gpu/device/device_choose.cuh"
#include "rd/gpu/device/brute_force/rd_globals.cuh"
#include "rd/gpu/device/samples_generator.cuh"
#include "rd/gpu/util/data_order_traits.hpp"

#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/rd_samples.cuh"
#include "cub/test_util.h"

#include "rd/utils/rd_params.hpp"

static const int TEST_DIM       = 2;
static const int ZERO           = 0;

template <typename T>
void testChooseKernel(rd::RDParams<T> &rdp,
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

    fParams.dim = TEST_DIM;
    dParams.dim = TEST_DIM;

    deviceInit(fParams.devId);

    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT: " << std::endl;
    testChooseKernel<float>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    // std::cout << "DOUBLE: " << std::endl;
    // testChooseKernel<double>(dParams, dSParams);
    // std::cout << rd::HLINE << std::endl;

    deviceReset();

    std::cout << "END!" << std::endl;
    return 0;
}

template <typename T>
void choose_gold(
    rd::RDParams<T> &rdp,
    T *P,
    T *S)
{
    std::list<T*> csList;
    rd::choose(P, S, csList, rdp.np, rdp.ns, TEST_DIM, rdp.r1);
}

template <typename T>
void testChooseMixedOrder(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    T *d_S,
    T const *S_gold)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testChooseMixedOrder:" << std::endl;

    T *S_gpu;
    int *d_ns, h_ns;

    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    __choose_kernel_v1<T, 320><<<1, 320>>>(d_P, d_S, rdp.np, rdp.r1, d_ns, TEST_DIM);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyFromSymbol(&h_ns, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    S_gpu = new T[h_ns * TEST_DIM];
    checkCudaErrors(cudaMemcpy(S_gpu, d_S, h_ns * TEST_DIM * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    if ((int)rdp.ns != h_ns)
    {
        std::cout << "[ERROR]Incorrect number of chosen samples!" << std::endl;
        std::cout << "Is: " << h_ns << " and should be: " << rdp.ns << std::endl;
    }

    rd::checkResult(S_gold, S_gpu, rdp.ns * TEST_DIM);

    delete[] S_gpu;
}


template <typename T>
void testChooseRowMajorOrder(
    rd::RDParams<T> const &rdp,
    T *d_S,
    T const *S_gold,
    T const *h_P)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testChooseRowMajorOrder:" << std::endl;

    T *S_gpu, *d_PRowMajor;
    int *d_ns, h_ns;

    checkCudaErrors(cudaMalloc((void**)&d_PRowMajor, rdp.np * TEST_DIM * sizeof(T)));

    checkCudaErrors(cudaMemcpy(d_PRowMajor, h_P, rdp.np * TEST_DIM * sizeof(T), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    __choose_kernel_v1<T, 320><<<1, 320>>>(d_PRowMajor, d_S, rdp.np, rdp.r1, d_ns, TEST_DIM, rd::gpu::rowMajorOrderTag());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyFromSymbol(&h_ns, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    S_gpu = new T[h_ns * TEST_DIM];
    checkCudaErrors(cudaMemcpy(S_gpu, d_S, h_ns * TEST_DIM * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    if ((int)rdp.ns != h_ns)
    {
        std::cout << "[ERROR]Incorrect number of chosen samples!" << std::endl;
        std::cout << "Is: " << h_ns << " and should be: " << rdp.ns << std::endl;
    }


    rd::checkResult(S_gold, S_gpu, rdp.ns * TEST_DIM);

    delete[] S_gpu;
    checkCudaErrors(cudaFree(d_PRowMajor));
}


template <typename T>
void testChooseColMajorOrder(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    T *d_S,
    T const *S_gold)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testChooseColMajorOrder:" << std::endl;

    T *S_gpu;
    int *d_ns, h_ns;

    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    __choose_kernel_v1<T, 320><<<1, 320>>>(d_P, d_S, rdp.np, rdp.r1, d_ns, TEST_DIM, rd::gpu::colMajorOrderTag());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyFromSymbol(&h_ns, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    S_gpu = new T[rdp.np * TEST_DIM];
    checkCudaErrors(cudaMemcpy(S_gpu, d_S, rdp.np * TEST_DIM * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    rd::transposeInPlace(S_gpu, S_gpu + rdp.np * TEST_DIM, rdp.np);

    if ((int)rdp.ns != h_ns)
    {
        std::cout << "[ERROR]Incorrect number of chosen samples!" << std::endl;
        std::cout << "Is: " << h_ns << " and should be: " << rdp.ns << std::endl;
    }

    rd::checkResult(S_gold, S_gpu, rdp.ns * TEST_DIM);

    delete[] S_gpu;
}


template <typename T>
void testChooseSOA(
    rd::RDParams<T> const &rdp,
    T const *h_P,
    T const *h_S_gold)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testChooseSOA:" << std::endl;

    typedef rd::ColMajorDeviceSamples<T, TEST_DIM> SamplesDevT;

    T *S_gpu;
    SamplesDevT *d_P, *d_S;
    int *d_ns, h_ns;

    d_P = new SamplesDevT(rdp.np);
    d_S = new SamplesDevT(rdp.np);

    T *h_aux = new T[rdp.np * TEST_DIM];
    rd::copyTable(h_P, h_aux, rdp.np * TEST_DIM);
    rd::transposeInPlace(h_aux, h_aux + rdp.np * TEST_DIM, TEST_DIM);
    d_P->copyFromContinuousData(h_aux, rdp.np);

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    __choose_kernel_v1<T, 320><<<1, 320>>>(d_P->dSamples, d_S->dSamples, rdp.np, rdp.r1, d_ns, TEST_DIM);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyFromSymbol(&h_ns, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    S_gpu = new T[rdp.np * TEST_DIM];
    d_S->copyToContinuousData(S_gpu);
    checkCudaErrors(cudaDeviceSynchronize());

    rd::transposeInPlace(S_gpu, S_gpu + rdp.np * TEST_DIM, rdp.np);

    if ((int)rdp.ns != h_ns)
    {
        std::cout << "[ERROR]Incorrect number of chosen samples!" << std::endl;
        std::cout << "Is: " << h_ns << " and should be: " << rdp.ns << std::endl;
    }

    rd::checkResult(h_S_gold, S_gpu, rdp.ns * TEST_DIM);

    delete[] S_gpu;
    delete[] h_aux;
    delete d_P;
    delete d_S;
}

template <typename T>
void testChooseMixedOrderNewApi(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    T *d_S,
    T const *S_gold)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testChooseMixedOrderNewApi:" << std::endl;

    T *S_gpu;
    int *d_ns, h_ns;

    size_t pPitch = 0;
    T *d_P2;
    checkCudaErrors(cudaMallocPitch(&d_P2, &pPitch, rdp.np * sizeof(T), TEST_DIM));
    checkCudaErrors(cudaMemcpy2D(d_P2, pPitch, d_P, rdp.np * sizeof(T), rdp.np * sizeof(T), 
        TEST_DIM, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    rd::gpu::bruteForce::DeviceChoose::choose<TEST_DIM, rd::COL_MAJOR, rd::ROW_MAJOR>(
            d_P2, d_S, rdp.np, d_ns, rdp.r1, pPitch / sizeof(T), TEST_DIM, nullptr, true);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyFromSymbol(&h_ns, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    S_gpu = new T[h_ns * TEST_DIM];
    checkCudaErrors(cudaMemcpy(S_gpu, d_S, h_ns * TEST_DIM * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    if ((int)rdp.ns != h_ns)
    {
        std::cout << "[ERROR]Incorrect number of chosen samples!" << std::endl;
        std::cout << "Is: " << h_ns << " and should be: " << rdp.ns << std::endl;
    }

    bool result = rd::checkResult(S_gold, S_gpu, rdp.ns * TEST_DIM);
    std::cout << " Test " << ((result) ? "PASS" : "FAIL") << std::endl;

    delete[] S_gpu;
    checkCudaErrors(cudaFree(d_P2));
}


template <typename T>
void testChooseRowMajorOrderNewApi(
    rd::RDParams<T> const &rdp,
    T *d_S,
    T const *S_gold,
    T const *h_P)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testChooseRowMajorOrderNewApi:" << std::endl;

    T *S_gpu, *d_PRowMajor;
    int *d_ns, h_ns;

    checkCudaErrors(cudaMalloc((void**)&d_PRowMajor, rdp.np * TEST_DIM * sizeof(T)));

    checkCudaErrors(cudaMemcpy(d_PRowMajor, h_P, rdp.np * TEST_DIM * sizeof(T), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    rd::gpu::bruteForce::DeviceChoose::choose<TEST_DIM, rd::ROW_MAJOR, rd::ROW_MAJOR>(
        d_PRowMajor, d_S, rdp.np, d_ns, rdp.r1);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyFromSymbol(&h_ns, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    S_gpu = new T[h_ns * TEST_DIM];
    checkCudaErrors(cudaMemcpy(S_gpu, d_S, h_ns * TEST_DIM * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    if ((int)rdp.ns != h_ns)
    {
        std::cout << "[ERROR]Incorrect number of chosen samples!" << std::endl;
        std::cout << "Is: " << h_ns << " and should be: " << rdp.ns << std::endl;
    }


    bool result = rd::checkResult(S_gold, S_gpu, rdp.ns * TEST_DIM);
    std::cout << " Test " << ((result) ? "PASS" : "FAIL") << std::endl;

    delete[] S_gpu;
    checkCudaErrors(cudaFree(d_PRowMajor));
}


template <typename T>
void testChooseColMajorOrderNewApi(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    T *d_S,
    T const *S_gold)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testChooseColMajorOrderNewApi:" << std::endl;

    T *S_gpu;
    int *d_ns, h_ns;

    size_t pPitch, sPitch;
    T * d_P2, *d_S2;

    checkCudaErrors(cudaMallocPitch(&d_P2, &pPitch, rdp.np * sizeof(T), TEST_DIM));
    checkCudaErrors(cudaMallocPitch(&d_S2, &sPitch, rdp.np * sizeof(T), TEST_DIM));
    checkCudaErrors(cudaMemset2D(d_S2, sPitch, 0, rdp.np * sizeof(T), TEST_DIM));
    checkCudaErrors(cudaMemcpy2D(d_P2, pPitch, d_P, rdp.np * sizeof(T), rdp.np * sizeof(T), 
        TEST_DIM, cudaMemcpyDeviceToDevice));

    // checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &ZERO, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    rd::gpu::bruteForce::DeviceChoose::choose<TEST_DIM, rd::COL_MAJOR, rd::COL_MAJOR>(
        d_P2, d_S2, rdp.np, d_ns, rdp.r1, pPitch / sizeof(T), sPitch / sizeof(T), nullptr, true);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyFromSymbol(&h_ns, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    if ((int)rdp.ns != h_ns)
    {
        std::cout << "[ERROR]Incorrect number of chosen samples!" << std::endl;
        std::cout << "Is: " << h_ns << " and should be: " << rdp.ns << std::endl;
    }

    S_gpu = new T[rdp.np * TEST_DIM];
    // checkCudaErrors(cudaMemcpy(S_gpu, d_S, rdp.np * TEST_DIM * sizeof(T), 
    //      cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy2D(S_gpu, rdp.np * sizeof(T), d_S2, sPitch, rdp.np * sizeof(T), 
        TEST_DIM, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    rd::transposeInPlace(S_gpu, S_gpu + rdp.np * TEST_DIM, rdp.np);

    bool result = rd::checkResult(S_gold, S_gpu, rdp.ns * TEST_DIM);
    std::cout << " Test " << ((result) ? "PASS" : "FAIL") << std::endl;

    delete[] S_gpu;

    checkCudaErrors(cudaFree(d_P2));
    checkCudaErrors(cudaFree(d_S2));
}


template <typename T>
void testChooseKernel(rd::RDParams<T> &rdp,
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

    rd::GraphDrawer<T> gDrawer;

    T *d_P, *d_S;
    T *h_P, *h_S;

    checkCudaErrors(cudaMalloc((void**)&d_P, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_S, rdp.np * TEST_DIM * sizeof(T)));

    checkCudaErrors(cudaMemset(d_P, 0, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));

    h_P = new T[rdp.np * TEST_DIM];
    h_S = new T[rdp.np * TEST_DIM];

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
            rd::gpu::SamplesGenerator<T>::segmentND(rdp.np, TEST_DIM, sp.sigma,
                T(0.005f * rdp.np), d_P);
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_P, d_P, rdp.np * TEST_DIM * sizeof(T), 
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    rd::transposeInPlace(h_P, h_P + rdp.np * TEST_DIM, rdp.np);

    std::ostringstream os;
    if (rdp.verbose)
    {
        os << typeid(T).name() << "_" << TEST_DIM;
        os << "D_initial_samples_set_";
        gDrawer.showPoints(os.str(), h_P, rdp.np, TEST_DIM);
        os.clear();
        os.str(std::string());
    }

    //---------------------------------------------------
    //               REFERENCE CHOOSE 
    //---------------------------------------------------

    choose_gold(rdp, h_P, h_S);

    std::cout << "Chosen count: " << rdp.ns << std::endl;

    if (rdp.verbose)
    {
        os << typeid(T).name() << "_" << TEST_DIM << "D_ref_chosen_set";
        gDrawer.startGraph(os.str(), TEST_DIM);
        gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
             h_P, rd::GraphDrawer<T>::POINTS, rdp.np);
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#38abe0' ps 1.3 ",
             h_S, rd::GraphDrawer<T>::POINTS, rdp.ns);
        gDrawer.endGraph();
        os.clear();
        os.str(std::string());
    }
    
    //---------------------------------------------------
    //               GPU CHOOSE 
    //---------------------------------------------------

    // testChooseMixedOrder(rdp, d_P, d_S, h_S);
    // testChooseRowMajorOrder(rdp, d_S, h_S, h_P);
    // testChooseColMajorOrder(rdp, d_P, d_S, h_S);
    // testChooseSOA(rdp, h_P, h_S);

    testChooseMixedOrderNewApi(rdp, d_P, d_S, h_S);
    testChooseRowMajorOrderNewApi(rdp, d_S, h_S, h_P);
    testChooseColMajorOrderNewApi(rdp, d_P, d_S, h_S);

    // clean-up
    delete[] h_P;
    delete[] h_S;

    checkCudaErrors(cudaFree(d_P));
    checkCudaErrors(cudaFree(d_S));
}

