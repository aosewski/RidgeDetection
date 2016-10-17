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

#include "rd/cpu/brute_force/choose.hpp"
#include "rd/cpu/brute_force/decimate.hpp"

#include "rd/gpu/device/brute_force/decimate.cuh"
#include "rd/gpu/device/device_decimate.cuh"
#include "rd/gpu/device/brute_force/rd_globals.cuh"
#include "rd/gpu/device/samples_generator.cuh"

#include "rd/gpu/util/data_order_traits.hpp"
#include "rd/gpu/util/dev_memcpy.cuh"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/rd_samples.cuh"
#include "rd/utils/rd_params.hpp"

#include "cub/test_util.h"
#include "cub/util_device.cuh"



static const int TEST_DIM = 2;

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
    std::cout << "DOUBLE: " << std::endl;
    testDecimateKernel<double>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;

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

    if (rdp.verbose)
    {
        os << typeid(T).name() << "_" << TEST_DIM << "D_ref_chosen_set";
        gDrawer.startGraph(os.str(), TEST_DIM);
        gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#d64f4f' ps 0.5 ",
             P, rd::GraphDrawer<T>::POINTS, rdp.np);
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#38abe0' ps 1.3 ",
             S, rd::GraphDrawer<T>::POINTS, rdp.ns);
        gDrawer.endGraph();
        os.clear();
        os.str(std::string());
    }

    rd::decimate(S, csList, rdp.ns, TEST_DIM, rdp.r2);

    std::cout << "Decimate count: " << rdp.ns << std::endl;

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

template <typename T>
void testDecimateRowMajorOrder(
    rd::RDParams<T> const &rdp,
    T *d_S,
    T *h_chosenS,
    int h_chosenCount,
    T const *S_gold)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testDecimateRowMajorOrder:" << std::endl;

    T *S_gpu;
    int *d_ns, h_ns;

    checkCudaErrors(cudaMemset(d_S, 0, rdp.np * TEST_DIM * sizeof(T)));

    // get chosen samples to device memory properly ordered
    checkCudaErrors(cudaMemcpy(d_S, h_chosenS, rdp.np * TEST_DIM * sizeof(T), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    rd::gpu::bruteForce::DeviceDecimate::decimate<TEST_DIM, rd::ROW_MAJOR>(d_S, d_ns, rdp.r2);
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
void testDecimateColMajorOrder(
    rd::RDParams<T> const &rdp,
    T *d_S,
    T const *h_chosenS,
    int h_chosenCount,
    T const *S_gold)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testDecimateColMajorOrder:" << std::endl;

    T *S_gpu;
    int *d_ns, h_ns;
    size_t sPitch;
    T *d_S2;

    checkCudaErrors(cudaMallocPitch(&d_S2, &sPitch, h_chosenCount * sizeof(T), TEST_DIM));
    checkCudaErrors(cudaMemset2D(d_S2, sPitch, 0, h_chosenCount * sizeof(T), TEST_DIM));

    rd::gpu::rdMemcpy2D<rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(
        d_S2, h_chosenS, TEST_DIM, h_chosenCount, sPitch, TEST_DIM * sizeof(T));

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    rd::gpu::bruteForce::DeviceDecimate::decimate<TEST_DIM, rd::COL_MAJOR>(d_S2, d_ns, rdp.r2, 
        sPitch / sizeof(T), nullptr, true);
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
        S_gpu, d_S2, h_chosenCount, TEST_DIM, TEST_DIM * sizeof(T), sPitch);
    checkCudaErrors(cudaDeviceSynchronize());

    if (rdp.verbose)
    {
        std::ostringstream os;
        rd::GraphDrawer<T> gDrawer;
        os << typeid(T).name() << "_" << TEST_DIM << "D_gpu_decimate";
        gDrawer.startGraph(os.str(), TEST_DIM);
        gDrawer.addPlotCmd("'-' w p pt 2 lc rgb '#38abe0' ps 1.3 ",
             S_gpu, rd::GraphDrawer<T>::POINTS, h_ns);
        gDrawer.endGraph();
        os.clear();
        os.str(std::string());
    }

    rd::checkResult(S_gold, S_gpu, h_ns * TEST_DIM, rdp.verbose);

    delete[] S_gpu;
    checkCudaErrors(cudaFree(d_S2));
}


template <typename T>
void testDecimateSOA(
    rd::RDParams<T> const &rdp,
    T const *h_chosenS,
    int h_chosenCount,
    T const *S_gold)
{

    std::cout << rd::HLINE << std::endl;
    std::cout << "testDecimateSOA:" << std::endl;

    typedef rd::ColMajorDeviceSamples<T, TEST_DIM> SamplesDevT;

    SamplesDevT *d_S;
    int *d_ns, h_ns;

    d_S = new SamplesDevT(rdp.np);

    T *h_aux = new T[rdp.np * TEST_DIM];
    rd::copyTable(h_chosenS, h_aux, h_chosenCount * TEST_DIM);
    rd::transposeInPlace(h_aux, h_aux + rdp.np * TEST_DIM, TEST_DIM);
    d_S->copyFromContinuousData(h_aux, rdp.np);

    checkCudaErrors(cudaGetSymbolAddress((void**)&d_ns, rd::gpu::rdBruteForceNs));
    checkCudaErrors(cudaMemcpyToSymbol(rd::gpu::rdBruteForceNs, &h_chosenCount, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    __decimate_kernel_v1<T, 512><<<1, 512>>>(d_S->dSamples, d_ns, rdp.r1, TEST_DIM);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpyFromSymbol(&h_ns, rd::gpu::rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    d_S->copyToContinuousData(h_aux);
    checkCudaErrors(cudaDeviceSynchronize());

    rd::transposeInPlace(h_aux, h_aux + rdp.np * TEST_DIM, rdp.np);

    if ((int)rdp.ns != h_ns)
    {
        std::cout << "[ERROR]Incorrect number of chosen samples!" << std::endl;
        std::cout << "Is: " << h_ns << " and should be: " << rdp.ns << std::endl;
    }

    rd::checkResult(S_gold, h_aux, rdp.ns * TEST_DIM);

    delete[] h_aux;
    delete d_S;
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

    rd::GraphDrawer<T> gDrawer;

    T *d_P, *d_S;
    T *h_P, *h_S, *h_chosenS;

    checkCudaErrors(cudaMalloc((void**)&d_P, rdp.np * TEST_DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_S, rdp.np * TEST_DIM * sizeof(T)));

    checkCudaErrors(cudaMemset(d_P, 0, rdp.np * TEST_DIM * sizeof(T)));

    h_P = new T[rdp.np * TEST_DIM];
    h_S = new T[rdp.np * TEST_DIM];
    h_chosenS = new T[rdp.np * TEST_DIM];

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
    //               REFERENCE DECIMATE 
    //---------------------------------------------------

    int h_chosenCount;
    decimateGold(rdp, h_P, h_S, h_chosenS, h_chosenCount);

    //---------------------------------------------------
    //               GPU DECIMATE 
    //---------------------------------------------------

    rdp.devId = (rdp.devId != -1) ? rdp.devId : 0;

    testDecimateRowMajorOrder(rdp, d_S, h_chosenS, h_chosenCount, h_S);
    testDecimateColMajorOrder(rdp, d_S, h_chosenS, h_chosenCount, h_S);
    // testDecimateSOA(rdp, h_chosenS, h_chosenCount, h_S);

    // clean-up
    delete[] h_P;
    delete[] h_S;
    delete[] h_chosenS;

    checkCudaErrors(cudaFree(d_P));
    checkCudaErrors(cudaFree(d_S));
}

