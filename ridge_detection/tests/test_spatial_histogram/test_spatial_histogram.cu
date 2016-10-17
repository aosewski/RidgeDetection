/**
 * @file test_spatial_histogram.cu
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

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR
#define BLOCK_TILE_LOAD_V4 1

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <functional>
#include <algorithm>
#ifdef RD_USE_OPENMP
#include <omp.h>
#endif

#include <helper_cuda.h>

#include "rd/gpu/device/samples_generator.cuh"
#include "rd/gpu/device/device_spatial_histogram.cuh"
#include "rd/gpu/util/dev_samples_set.cuh" 

#include "rd/utils/bounding_box.hpp"
#include "rd/utils/histogram.hpp"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/memory.h" 
#include "rd/utils/name_traits.hpp" 
#include "rd/utils/rd_params.hpp"

#include "cub/test_util.h"


template <int DIM, typename T>
void test(rd::RDParams<T> &rdp,
          rd::RDSpiralParams<T> &rds);

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
            "\t\t[--a=<spiral param>]\n"
            "\t\t[--b=<spiral param>]\n"
            "\t\t[--s=<spiral noise sigma>]\n"
            "\t\t[--d=<device id>]\n"
            "\t\t[--v <verbose>]\n"
            "\t\t[--f=<file name to load>]\n"
            "\n", argv[0]);
        exit(0);
    }

    if (args.CheckCmdLineFlag("f"))
    {
        args.GetCmdLineArgument("f", fSParams.file);
        fSParams.loadFromFile = true;
    }
    else
    {
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

    checkCudaErrors(deviceInit(fParams.devId));

    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT 2D: " << std::endl;
    test<2>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT 3D: " << std::endl;
    test<3>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT 4D: " << std::endl;
    test<4>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT 5D: " << std::endl;
    test<5>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT 6D: " << std::endl;
    test<6>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;

    std::cout << "DOUBLE 2D: " << std::endl;
    test<2>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE 3D: " << std::endl;
    test<3>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE 4D: " << std::endl;
    test<4>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE 5D: " << std::endl;
    test<5>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE 6D: " << std::endl;
    test<6>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;

    checkCudaErrors(deviceReset());

    std::cout << "END!" << std::endl;
    return 0;
}

template <typename T>
struct HistogramMapFuncGold
{
    rd::BoundingBox<T> const &bb;

    HistogramMapFuncGold(rd::BoundingBox<T> const &bb)
    :
        bb(bb)
    {}

    size_t operator()(
        T const *                     sample,
        std::vector<size_t> const &   binsCnt)
    {
        std::vector<size_t> binIdx(bb.dim, 0);

        // get sample's bin [x,y,z...n] idx
        for (size_t i = 0; i < bb.dim; ++i)
        {
            /*
             * translate each sample coordinate to the common origin (by distracting minimum)
             * then divide shifted coordinate by current dimension bin width and get the 
             * floor of this value (counting from zero!) which is our bin idx we search for.
             */
            if (bb.dist[i] < std::numeric_limits<T>::epsilon())
            {
                binIdx[i] = 0;
                sample++;
                continue;
            }

            T normCord = std::abs(*sample - bb.min(i));
            T step = bb.dist[i] / binsCnt[i];

            if (std::abs(normCord - bb.dist[i]) <= std::numeric_limits<T>::epsilon())
            {
                binIdx[i] = binsCnt[i]-1;
            }
            else
            {
                binIdx[i] = std::floor(normCord / step);
            }
            sample++;        
        }

        /*
         * Calculate global idx value linearizing bin idx
         * idx = k_0 + sum_{i=2}^{dim}{k_i mul_{j=i-1}^{1}bDim_j}
         */
        size_t idx = binIdx[0];
        size_t tmp;
        for (size_t i = 1; i < bb.dim; ++i)
        {
            tmp = 1;
            for (int j = (int)i - 1; j >= 0; --j)
            {
                tmp *= binsCnt[j];
            }
            idx += binIdx[i]*tmp;
        }

        return idx;
    }
};

template <typename T>
void histogramGold(
    rd::RDParams<T> &rdp,
    T const *P,
    std::vector<size_t>const &binsCnt,
    rd::BoundingBox<T>const & bbox,
    rd::Histogram<T> &hist)
{
    HistogramMapFuncGold<T> mapFunc(bbox);

    hist.setBinCnt(binsCnt);
    hist.getHist(P, rdp.np, mapFunc);
}

template <
    int                     DIM,
    rd::DataMemoryLayout    INPUT_MEM_LAYOUT,
    typename                T>
void testDeviceSpatialHistogram(
    rd::RDParams<T> const &rdp,
    T const *d_P,
    int *d_hist,
    int stride,
    std::vector<size_t>const &binsCnt,
    size_t numBins,
    rd::BoundingBox<T> const &bboxGold,
    rd::Histogram<T> const &histGold)
{
    std::cout << rd::HLINE << "\n";
    std::cout << "testDeviceSpatialHistogram:" << "\n";
    std::cout << "DataMemoryLayout: " << 
        rd::DataMemoryLayoutNameTraits<INPUT_MEM_LAYOUT>::name << "\n";

    void *d_tempStorage = NULL;
    unsigned long long int tempStorageBytes = 0;

    typedef int AliasedBinCnt[DIM];
    AliasedBinCnt aliasedBinCnt;

    rd::gpu::BoundingBox<DIM, T> d_bbox;

    // initialize data
    for (int d = 0; d < DIM; ++d)
    {
        d_bbox.bbox[d * 2]      = bboxGold.min(d);
        d_bbox.bbox[d * 2 + 1]  = bboxGold.max(d);
        d_bbox.dist[d]          = bboxGold.dist[d];
        aliasedBinCnt[d]        = binsCnt[d];
    }
    checkCudaErrors(cudaMemset(d_hist, 0, numBins * sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());

    cudaError_t error = rd::gpu::DeviceHistogram::spatialHistogram<DIM, INPUT_MEM_LAYOUT>(
        d_tempStorage,
        tempStorageBytes,
        d_P, 
        d_hist, 
        rdp.np,
        aliasedBinCnt,
        d_bbox,
        stride,
        0,
        true);
    checkCudaErrors(error);
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMalloc((void**)&d_tempStorage, tempStorageBytes));

    error = rd::gpu::DeviceHistogram::spatialHistogram<DIM, INPUT_MEM_LAYOUT>(
        d_tempStorage,
        tempStorageBytes,
        d_P, 
        d_hist, 
        rdp.np,
        aliasedBinCnt,
        d_bbox,
        stride,
        0,
        true);
    checkCudaErrors(error);
    checkCudaErrors(cudaDeviceSynchronize());

    int *h_hist = new int[numBins];
    std::vector<int> histGoldValues;
    for (size_t v : histGold.hist)
    {
        histGoldValues.push_back((int)v);
    }

    checkCudaErrors(cudaMemcpy(h_hist, d_hist, numBins * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    bool result = rd::checkResult(histGoldValues.data(), h_hist, numBins);
    if (result)
    {
        std::cout << ">>>> SUCCESS!\n";
    }

    delete[] h_hist;
    checkCudaErrors(cudaFree(d_tempStorage));
}

template <int DIM, typename T>
void test(rd::RDParams<T> &rdp,
          rd::RDSpiralParams<T> &sp)
{
    std::vector<std::string> samplesDir{"../../examples/data/nd_segments/", 
        "../../examples/data/spirals/"};
    rd::gpu::Samples<T> d_samplesSet(rdp, sp, samplesDir, DIM);

    std::cout << "Samples: " << std::endl;
    std::cout <<  "\t dimension: " << DIM << std::endl;
    std::cout <<  "\t n_samples: " << rdp.np << std::endl;

    std::cout << "Spiral params: " << std::endl;
    std::cout <<  "\t a: " << sp.a << std::endl;
    std::cout <<  "\t b: " << sp.b << std::endl;
    std::cout <<  "\t sigma: " << sp.sigma << std::endl; 

    rd::GraphDrawer<T> gDrawer;
    // std::vector<size_t> binsCnt{1,1,1};
    std::vector<size_t> binsCnt(DIM);
    for (int d = 0; d < DIM; ++d)
    {
        binsCnt[d] = 2 << d;
    }
    size_t numBins = std::accumulate(binsCnt.begin(), binsCnt.end(),
                         1, std::multiplies<size_t>());

    T *d_PRowMajor, *d_PColMajor;
    T *h_P;
    int *d_hist;

    // allocate containers
    checkCudaErrors(cudaMalloc((void**)&d_PRowMajor, rdp.np * DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_PColMajor, rdp.np * DIM * sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&d_hist, numBins * sizeof(int)));
    h_P = new T[rdp.np * DIM];

    // initialize data
    checkCudaErrors(cudaMemcpy(d_PRowMajor, d_samplesSet.samples_, rdp.np * DIM * sizeof(T), 
        cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(h_P, d_samplesSet.samples_, rdp.np * DIM * sizeof(T), 
        cudaMemcpyDeviceToHost));

    T * tmp = new T[rdp.np * DIM];
    rd::transposeTable(h_P, tmp, rdp.np, DIM);
    checkCudaErrors(cudaMemcpy(d_PColMajor, tmp, rdp.np * DIM * sizeof(T), 
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
    delete[] tmp;
    tmp = nullptr;

    // draw test samples set if verbose
    std::ostringstream os;
    if (rdp.verbose && DIM <= 3)
    {
        os << typeid(T).name() << "_" << DIM;
        os << "D_initial_samples_set_";
        gDrawer.showPoints(os.str(), h_P, rdp.np, DIM);
        os.clear();
        os.str(std::string());
    }

    //---------------------------------------------------
    //               REFERENCE HISTOGRAM 
    //---------------------------------------------------

    rd::Histogram<T> histGold;
    rd::BoundingBox<T> bboxGold(h_P, rdp.np, rdp.dim);
    bboxGold.calcDistances();
    histogramGold(rdp, h_P, binsCnt, bboxGold, histGold);

    if (rdp.verbose)
    {
        bboxGold.print();
        std::cout << "hist: [";
        for (size_t h : histGold.hist)
        {
            std::cout << ", " << h;
        }
        std::cout << "]\n";
    }
    //---------------------------------------------------
    //               GPU HISTOGRAM 
    //---------------------------------------------------

    testDeviceSpatialHistogram<DIM, rd::ROW_MAJOR>(rdp, d_PRowMajor, d_hist, 1, binsCnt, numBins, 
        bboxGold, histGold);
    testDeviceSpatialHistogram<DIM, rd::COL_MAJOR>(rdp, d_PColMajor, d_hist, rdp.np, binsCnt, 
        numBins, bboxGold, histGold);


    // clean-up
    delete[] h_P;

    checkCudaErrors(cudaFree(d_PRowMajor));
    checkCudaErrors(cudaFree(d_PColMajor));
    checkCudaErrors(cudaFree(d_hist));
}

