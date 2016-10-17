

#include "rd/gpu/device/samples_generator.cuh"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"
#include "rd/utils/samples_set.hpp"
#include "cub/test_util.h"

#include "rd/utils/rd_params.hpp"

#include <helper_cuda.h>

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>


template <typename T>
void generateSamples(rd::RDParams<T> &rdp,
                    rd::RDSpiralParams<T> const &rds)
{
    T *d_samplesSet;
    rd::Samples<T> h_samplesSet;
    h_samplesSet.dim_ = rdp.dim;
    h_samplesSet.size_ = rdp.np;
    h_samplesSet.samples_ = new T[rdp.np * rdp.dim];

    checkCudaErrors(cudaMalloc((void**)&d_samplesSet, rdp.np * rdp.dim * sizeof(T)));

    switch(rdp.dim)
    {
        case 2:
            rd::gpu::SamplesGenerator<T>::template spiral2D<rd::ROW_MAJOR>(
                rdp.np, rds.a, rds.b, rds.sigma, d_samplesSet);
            break;
        case 3:
            rd::gpu::SamplesGenerator<T>::template spiral3D<rd::ROW_MAJOR>(
                rdp.np, rds.a, rds.b, rds.sigma, d_samplesSet);
            break;
        default:
            rd::gpu::SamplesGenerator<T>::segmentND(rdp.np, rdp.dim, rds.sigma, rds.a, d_samplesSet);
            break;
    }

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_samplesSet.samples_, d_samplesSet, rdp.np * rdp.dim * sizeof(T), 
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    std::ostringstream comment;
    std::ostringstream fileName;
    if (rdp.dim == 2 || rdp.dim == 3)
    {
        comment << " --dim=" << rdp.dim << " --np=" << rdp.np << " --a=" << rds.a << " --b=" << rds.b << "  --s=" << rds.sigma << "\n";
        fileName << typeid(T).name();
        fileName << "_spiral"<< rdp.dim << "D_np" << rdp.np << "_a"<< rds.a<< "_b"<< rds.b<< "_s"<< rds.sigma;
    }
    else
    {
        comment << " --dim=" << rdp.dim << " --np=" << rdp.np << " --a=" << rds.a << " --b=" << rds.b << " --s=" << rds.sigma << "\n";
        fileName << typeid(T).name();
        fileName << "_segment" << rdp.dim << "D_np" << rdp.np << "_l" << rds.a <<  "_s" << rds.sigma;

    }
    
    h_samplesSet.saveToFile(fileName.str(), comment.str());

    if (rdp.dim == 2 || rdp.dim == 3)
    {
        rd::GraphDrawer<T> gDrawer;
        gDrawer.showPoints(fileName.str(), h_samplesSet.samples_, rdp.np, rdp.dim);
    }

    checkCudaErrors(cudaFree(d_samplesSet));
}

int main(int argc, char const **argv)
{

    rd::RDParams<double> dParams;
    rd::RDSpiralParams<double> dSParams;

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
            "\t\t[--dim=<data dimension>]\n"
            "\t\t[--d=<device id>]\n"
            "\t\t[--g=<generates range of samples sets>]\n"
            "\n", argv[0]);
        exit(0);
    }

    if (args.CheckCmdLineFlag("d")) 
    {
        args.GetCmdLineArgument("d", dParams.devId);
    }

    checkCudaErrors(deviceInit(dParams.devId));

    if (!args.CheckCmdLineFlag("g"))
    {
        if (args.CheckCmdLineFlag("np"))
        {
            args.GetCmdLineArgument("np", dParams.np);
        }
        if (args.CheckCmdLineFlag("a")) 
        {
            args.GetCmdLineArgument("a", dSParams.a);
        }
        if (args.CheckCmdLineFlag("b")) 
        {
            args.GetCmdLineArgument("b", dSParams.b);
        }
        if (args.CheckCmdLineFlag("s")) 
        {
            args.GetCmdLineArgument("s", dSParams.sigma);
        }
        if (args.CheckCmdLineFlag("dim")) 
        {
            args.GetCmdLineArgument("dim", dParams.dim);
        }

        generateSamples(dParams, dSParams);
    }
    else
    {
        /**
         *  3D, np=50k, a=15, b=8, s=4, double
         */
        dParams.dim = 3;
        dParams.np = 50000;
        dSParams.a = 15;
        dSParams.b = 8;
        dSParams.sigma = 4;
        generateSamples(dParams, dSParams);
        /**
         *  3D, np=100k, a=17, b=8, s=4, double
         */
        dParams.np = 100000;
        dSParams.a = 17;
        generateSamples(dParams, dSParams);
        /**
         *  3D, np=500k, a=18, b=8, s=4, double
         */
        dParams.np = 500000;
        dSParams.a = 18;
        generateSamples(dParams, dSParams);
        /**
         *  3D, np=1M, a=22, b=10, s=4, double
         */
        dParams.np = 1000000;
        dSParams.a = 22;
        dSParams.b = 10;
        generateSamples(dParams, dSParams);

    }

    checkCudaErrors(deviceReset());

    std::cout << "END!" << std::endl;
    return 0;
}