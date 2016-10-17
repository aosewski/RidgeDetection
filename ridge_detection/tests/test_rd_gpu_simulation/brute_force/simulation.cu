/**
 *  @file simulation.cu
 *  @author Adam Rogowiec
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

#include "rd/gpu/device/brute_force/simulation.cuh"
#include "rd/gpu/device/samples_generator.cuh"
#include "rd/gpu/util/dev_memcpy.cuh"

#include "rd/utils/samples_set.hpp"
#include "rd/utils/graph_drawer.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/utils/utilities.hpp"

#include "cub/test_util.h"
#include "cub/util_arch.cuh"

#include "rd/utils/rd_params.hpp"

#include <helper_cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <typeinfo>
#include <stdexcept>
#include <string>

template <int DIM, typename T>
void simulation(rd::RDParams<T> &rdp,
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

    fParams.dim = 2;
    dParams.dim = 2;

    checkCudaErrors(deviceInit(fParams.devId));

    std::cout << rd::HLINE << std::endl;
    std::cout << "2D: " << std::endl;
    std::cout << rd::HLINE << std::endl;

    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT: " << std::endl;
    simulation<2>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE: " << std::endl;
    simulation<2>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;

    std::cout << rd::HLINE << std::endl;
    std::cout << "3D: " << std::endl;
    std::cout << rd::HLINE << std::endl;

    fParams.dim = 3;
    dParams.dim = 3;

    std::cout << rd::HLINE << std::endl;
    std::cout << "FLOAT: " << std::endl;
    simulation<3>(fParams, fSParams);
    std::cout << rd::HLINE << std::endl;
    std::cout << "DOUBLE: " << std::endl;
    simulation<3>(dParams, dSParams);
    std::cout << rd::HLINE << std::endl;

    checkCudaErrors(deviceReset());

    std::cout << "END!" << std::endl;
 
    return 0;
}

template <int DIM, typename T>
void simulation(rd::RDParams<T> &rdp,
                rd::RDSpiralParams<T> const &rds)
{
    std::cout << "Samples: " << std::endl;
    std::cout <<  "\t dimension: " << rdp.dim << std::endl;
    std::cout <<  "\t n_samples: " << rdp.np << std::endl;
    std::cout <<  "\t r1: " << rdp.r1 << std::endl;
    std::cout <<  "\t r2: " << rdp.r2 << std::endl;

    std::cout << "Spiral params: " << std::endl;
    std::cout <<  "\t a: " << rds.a << std::endl;
    std::cout <<  "\t b: " << rds.b << std::endl;
    std::cout <<  "\t sigma: " << rds.sigma << std::endl; 

    rd::GraphDrawer<T> gDrawer; 
    rd::CpuTimer timer;
    rd::Samples<T> h_P;

    switch(DIM)    
    {
        case 2: h_P.genSpiral2D(rdp.np, rds.a, rds.b, rds.sigma); break;
        case 3: h_P.genSpiral3D(rdp.np, rds.a, rds.b, rds.sigma); break;
        default:
            throw std::logic_error("Not supported dimension!");
    }

    std::ostringstream os;
    if (rdp.verbose)
    {
        os << typeid(T).name() << "_" << DIM;
        os << "D_initial_samples_set_";
        gDrawer.showPoints(os.str(), h_P.samples_, rdp.np, DIM);
        os.clear();
        os.str(std::string());
    }

    T *h_S = nullptr;

    rd::gpu::bruteForce::RidgeDetection<T, DIM, rd::ROW_MAJOR, rd::ROW_MAJOR> rr_rdGpu(rdp.np, rdp.r1, rdp.r2, rdp.verbose);
    rd::gpu::bruteForce::RidgeDetection<T, DIM, rd::COL_MAJOR, rd::ROW_MAJOR> cr_rdGpu(rdp.np, rdp.r1, rdp.r2, rdp.verbose);
    rd::gpu::bruteForce::RidgeDetection<T, DIM, rd::COL_MAJOR, rd::COL_MAJOR> cc_rdGpu(rdp.np, rdp.r1, rdp.r2, rdp.verbose);

    rd::gpu::rdMemcpy<DIM, rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(rr_rdGpu.dP_, h_P.samples_, rdp.np, DIM, DIM);
    rd::gpu::rdMemcpy<DIM, rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(cr_rdGpu.dP_, h_P.samples_, rdp.np, rdp.np, DIM);
    rd::gpu::rdMemcpy<DIM, rd::COL_MAJOR, rd::ROW_MAJOR, cudaMemcpyHostToDevice>(cc_rdGpu.dP_, h_P.samples_, rdp.np, rdp.np, DIM);

    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << rd::HLINE << std::endl;
    std::cout << "GPU:" << std::endl;
    std::cout << "<<<<< ROW_MAJOR - ROW_MAJOR" << std::endl;

    timer.start();
    rr_rdGpu.ridgeDetection();
    checkCudaErrors(cudaDeviceSynchronize());
    timer.stop();
    timer.elapsedMillis(0, true);

    rr_rdGpu.getChosenSamplesCount();
    std::cout << "Chosen count: " << rr_rdGpu.ns_ << std::endl;

    if (rdp.verbose)
    {
        h_S = new T[rr_rdGpu.ns_ * DIM];
        rd::gpu::rdMemcpy<DIM, rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyDeviceToHost>(h_S, rr_rdGpu.dS_, rr_rdGpu.ns_);
        os << typeid(T).name() << "_" << DIM;
        os << "D_row_row_gpu_detected_ridge";
        gDrawer.showPoints(os.str(), h_S, rr_rdGpu.ns_, DIM);
        os.clear();
        os.str(std::string());
        delete[] h_S;
    }

    std::cout << rd::HLINE << std::endl;
    std::cout << "<<<<< COL_MAJOR - ROW_MAJOR" << std::endl;
    
    timer.start();
    cr_rdGpu.ridgeDetection();
    checkCudaErrors(cudaDeviceSynchronize());
    timer.stop();
    timer.elapsedMillis(0, true);

    cr_rdGpu.getChosenSamplesCount();
    std::cout << "Chosen count: " << cr_rdGpu.ns_ << std::endl;

    if (rdp.verbose)
    {
        h_S = new T[cr_rdGpu.ns_ * DIM];
        rd::gpu::rdMemcpy<DIM, rd::ROW_MAJOR, rd::ROW_MAJOR, cudaMemcpyDeviceToHost>(h_S, cr_rdGpu.dS_, cr_rdGpu.ns_);
        os << typeid(T).name() << "_" << DIM;
        os << "D_col_row_gpu_detected_ridge";
        gDrawer.showPoints(os.str(), h_S, cr_rdGpu.ns_, DIM);
        os.clear();
        os.str(std::string());
        delete[] h_S;
    }

    std::cout << rd::HLINE << std::endl;
    std::cout << "<<<<< COL_MAJOR - COL_MAJOR" << std::endl;

    timer.start();
    cc_rdGpu.ridgeDetection();
    checkCudaErrors(cudaDeviceSynchronize());
    timer.stop();
    timer.elapsedMillis(0, true);

    cc_rdGpu.getChosenSamplesCount();
    std::cout << "Chosen count: " << cc_rdGpu.ns_ << std::endl;

    if (rdp.verbose)
    {
        h_S = new T[cc_rdGpu.np_ * DIM];
        rd::gpu::rdMemcpy<DIM, rd::ROW_MAJOR, rd::COL_MAJOR, cudaMemcpyDeviceToHost>(h_S, cc_rdGpu.dS_, cc_rdGpu.np_, DIM, cc_rdGpu.np_);
        os << typeid(T).name() << "_" << DIM;
        os << "D_col_col_gpu_detected_ridge";
        gDrawer.showPoints(os.str(), h_S, cc_rdGpu.ns_, DIM);
        os.clear();
        os.str(std::string());
        delete[] h_S;
    }

}