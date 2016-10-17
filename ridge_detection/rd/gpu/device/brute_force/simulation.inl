/**
 * @file simulation.inl
 * @author Adam Rogowiec
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


#include "rd/gpu/util/dev_memcpy.cuh"
#include "rd/gpu/device/device_choose.cuh"
#include "rd/gpu/device/device_decimate.cuh"
#include "rd/gpu/device/device_evolve.cuh"
#include "rd/gpu/device/brute_force/rd_dp.cuh"

#include "rd/utils/samples_set.hpp"
#include "rd/utils/utilities.hpp"

#include "cub/util_arch.cuh"

namespace rd 
{
namespace gpu
{
namespace bruteForce
{

template <
    typename T,
    int DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT>
RidgeDetection<T, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>::RidgeDetection(
        size_t np,
        T r1,
        T r2,
        bool verbose)
    :   
        np_(np), r1_(r1), r2_(r2), dCordSums_(nullptr), dSpherePointCnt_(nullptr),
        dDistMtx_(nullptr), dPtsMask_(nullptr), verbose_(verbose)
{
    // set device shared memory configuration
    if (sizeof(T) >= 8)
    {
        checkCudaErrors(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    }

    // set used kernels cache configurations
    DeviceChoose::setCacheConfig<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, T>();
    DeviceEvolve::setCacheConfig<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT, T>();
    #ifdef CUB_CDP
    DeviceDecimate::setDecimateDistMtxCacheConfig<DIM, OUT_MEM_LAYOUT, T>();
    #else
    DeviceDecimate::setDecimateCacheConfig<DIM, OUT_MEM_LAYOUT, T>();
    #endif

    checkCudaErrors(cudaGetSymbolAddress((void**)&dNs_, rdBruteForceNs));
    if (IN_MEM_LAYOUT == COL_MAJOR)
    {
        checkCudaErrors(cudaMallocPitch(&dP_, &pPitch_, np_ * sizeof(T), 
            DIM));
    }
    else
    {
        checkCudaErrors(cudaMalloc((void**)&dP_, np_ * DIM * sizeof(T)));
    }

    if (OUT_MEM_LAYOUT == COL_MAJOR)
    {
        checkCudaErrors(cudaMallocPitch(&dS_, &sPitch_, np_ * sizeof(T), 
            DIM));
    }
    else
    {
        checkCudaErrors(cudaMalloc((void**)&dS_, np_ * DIM * sizeof(T)));
    }

    dCordSums_ = nullptr;
    dSpherePointCnt_ = nullptr;

    ns_ = 0;
    checkCudaErrors(cudaMemcpyToSymbol(rdBruteForceNs, &ns_, sizeof(int)));
}


template <
    typename T,
    int DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT>
RidgeDetection<T, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>::~RidgeDetection() 
{
    if (dP_ != nullptr)
        checkCudaErrors(cudaFree(dP_));
    if (dS_ != nullptr)
        checkCudaErrors(cudaFree(dS_));
    if (dCordSums_ != nullptr)
        checkCudaErrors(cudaFree(dCordSums_));
    if (dSpherePointCnt_ != nullptr)
        checkCudaErrors(cudaFree(dSpherePointCnt_));
    if (dDistMtx_ != nullptr)
        checkCudaErrors(cudaFree(dDistMtx_));
    if (dPtsMask_ != nullptr)
        checkCudaErrors(cudaFree(dPtsMask_));
}

template <
    typename T,
    int DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT>
void RidgeDetection<T, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>::allocTemporaries() 
{
    if (IN_MEM_LAYOUT == COL_MAJOR)
    {
        checkCudaErrors(cudaMallocPitch(&dCordSums_, &csPitch_, ns_ * sizeof(T),
            DIM));
    }
    else
    {
        checkCudaErrors(cudaMalloc(&dCordSums_, ns_ * DIM * sizeof(T)));
    }
    checkCudaErrors(cudaMalloc(&dSpherePointCnt_, ns_ * sizeof(int)));
    #ifdef CUB_CDP
    checkCudaErrors(cudaMalloc(&dPtsMask_, ns_ * sizeof(char)));
    checkCudaErrors(cudaMallocPitch(&dDistMtx_, &distMtxPitch_, ns_ * sizeof(T), ns_));
    #endif
}

template <
    typename T,
    int DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT>
void RidgeDetection<T, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>::freeTemporaries()
{
    if (dCordSums_ != nullptr) 
    {
        checkCudaErrors(cudaFree(dCordSums_));
        dCordSums_ = nullptr;
    }
    if (dSpherePointCnt_ != nullptr) 
    {
        checkCudaErrors(cudaFree(dSpherePointCnt_));
        dSpherePointCnt_ = nullptr;
    }
    if (dDistMtx_ != nullptr) 
    {
        checkCudaErrors(cudaFree(dDistMtx_));
        dDistMtx_ = nullptr;
    }
    if (dPtsMask_ != nullptr) 
    {
        checkCudaErrors(cudaFree(dPtsMask_));
        dPtsMask_ = nullptr;
    }
}


template <
    typename T,
    int DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT>
void RidgeDetection<T, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>::getChosenSamplesCount()
{
    checkCudaErrors(cudaMemcpyFromSymbol(&ns_, rdBruteForceNs, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());
}

template <
    typename T,
    int DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT>
void RidgeDetection<T, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>::ridgeDetection() 
{

    Samples<T> hS;
    hS.dim_ = DIM;
    T *hP = nullptr;

    doChoose();
    if (verbose_)    
    {
        checkCudaErrors(cudaDeviceSynchronize());
        std::cout << "Wybrano reprezentantów, liczność zbioru S: " <<
                ns_ << std::endl;
        hS.samples_ = new T[np_ * DIM];
        hP = new T[np_ * DIM];
        if (IN_MEM_LAYOUT == COL_MAJOR)
        {
            rdMemcpy2D<ROW_MAJOR, COL_MAJOR, cudaMemcpyDeviceToHost>(
                hP, dP_, np_, DIM, DIM * sizeof(T), pPitch_);
        }
        else
        {
            rdMemcpy<ROW_MAJOR, ROW_MAJOR, cudaMemcpyDeviceToHost>(
                hP, dP_, DIM, np_, DIM, DIM);
        }

        if (OUT_MEM_LAYOUT == COL_MAJOR)
        {
            rdMemcpy2D<ROW_MAJOR, COL_MAJOR, cudaMemcpyDeviceToHost>(
                hS.samples_, dS_, ns_, DIM, DIM * sizeof(T), sPitch_);
        }
        else
        {
            rdMemcpy<ROW_MAJOR, ROW_MAJOR, cudaMemcpyDeviceToHost>(hS.samples_, 
                dS_, DIM, ns_, DIM, DIM);
        }
        checkCudaErrors(cudaDeviceSynchronize());

        std::ostringstream os;
        os << typeid(T).name();
        os << "_";
        os << DIM;
        os << "D_gpu_choosen_set";
        if (DIM <= 3)
        {
            rd::GraphDrawer<T> gDrawer;
            gDrawer.showPoints(os.str(), hS.samples_, ns_, DIM);
        }
        std::cout << HLINE << std::endl;

        std::ostringstream comment;
        comment << "ns: " << ns_;
        comment << ", r1: " << r1_  << ", r2: " << r2_;

        hS.size_ = ns_;
        hS.saveToFile(os.str(), comment.str());
    }

#ifdef CUB_CDP
    __rd_simulation_dp<T, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT><<<1,1>>>(
        dP_, dS_, dCordSums_, dSpherePointCnt_, dDistMtx_, dPtsMask_, 
        r1_, r2_, np_, 
        (IN_MEM_LAYOUT == COL_MAJOR) ? pPitch_ / sizeof(T) : DIM,
        (OUT_MEM_LAYOUT == COL_MAJOR) ? sPitch_ / sizeof(T) : DIM,
        (IN_MEM_LAYOUT == COL_MAJOR) ? csPitch_ / sizeof(T) : DIM,
        distMtxPitch_ / sizeof(T));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    getChosenSamplesCount();
#else

    size_t oldCount = 0;
    int iter = 0;

    /*
     *  Repeat untill the count of chosen samples won't
     *  change in two consecutive iterations.
     */
    while (oldCount != ns_)    
    {
        oldCount = ns_;

        doEvolve();
        if (verbose_)        
        {
            checkCudaErrors(cudaDeviceSynchronize());
            std::cout << "Ewolucja nr: " << iter << std::endl;
            
            if (OUT_MEM_LAYOUT == COL_MAJOR)
            {
                rdMemcpy2D<ROW_MAJOR, COL_MAJOR, cudaMemcpyDeviceToHost>(
                    hS.samples_, dS_, ns_, DIM, DIM * sizeof(T), sPitch_);
            }
            else
            {
                rdMemcpy<ROW_MAJOR, ROW_MAJOR, cudaMemcpyDeviceToHost>(
                    hS.samples_, dS_, DIM, ns_, DIM, DIM);
            }
            checkCudaErrors(cudaDeviceSynchronize());

            std::ostringstream os;
            os << typeid(T).name();
            os << "_gpu_";
            os << DIM;
            os << "D_iter_";
            os << iter;
            os << "_an_evolution";
            if (DIM <= 3)
            {
                rd::GraphDrawer<T> gDrawer;
                if (DIM == 3)
                {
                    gDrawer.showPoints(os.str(), hS.samples_, ns_, DIM);
                }
                else
                {
                    if (np_ > 50000)
                    {
                        gDrawer.showCircles(os.str(), 0, np_, hS.samples_,
                                ns_, r1_, DIM);
                    }
                    else
                    {
                        gDrawer.showCircles(os.str(), hP, np_, hS.samples_,
                                ns_, r1_, DIM);
                    }
                }
            }

            std::ostringstream comment;
            comment << "ns: " << ns_;
            comment << ", r1: " << r1_  << ", r2: " << r2_;

            hS.size_ = ns_;
            hS.saveToFile(os.str(), comment.str());
        }

        doDecimate();
        getChosenSamplesCount();
        if (verbose_)        
        {
            std::cout << "Decymacja nr: " << iter << ", ns: " << ns_
                    << std::endl;
            if (OUT_MEM_LAYOUT == COL_MAJOR)
            {
                rdMemcpy2D<ROW_MAJOR, COL_MAJOR, cudaMemcpyDeviceToHost>(
                    hS.samples_, dS_, ns_, DIM, DIM * sizeof(T), sPitch_);
            }
            else
            {
                rdMemcpy<ROW_MAJOR, ROW_MAJOR, cudaMemcpyDeviceToHost>(hS.samples_, 
                    dS_, DIM, ns_, DIM, DIM);
            }
            checkCudaErrors(cudaDeviceSynchronize());

            std::ostringstream os;
            os << typeid(T).name();
            os << "_gpu_";
            os << DIM;
            os << "D_iter_";
            os << iter;
            os << "_decimation";
            if (DIM <= 3)
            {
                rd::GraphDrawer<T> gDrawer;
                if (DIM == 3)          
                {
                    gDrawer.showPoints(os.str(), hS.samples_, ns_, DIM);
                }
                else       
                {
                    if (np_ > 10000)     
                    {
                        gDrawer.showCircles(os.str(), 0, np_, hS.samples_,
                                ns_, r2_, DIM);
                    }
                    else
                    {
                        gDrawer.showCircles(os.str(), hP, np_, hS.samples_,
                                ns_, r2_, DIM);
                    }
                }
            }

            std::ostringstream comment;
            comment << "ns: " << ns_;
            comment << ", r1: " << r1_  << ", r2: " << r2_;

            hS.size_ = ns_;
            hS.saveToFile(os.str(), comment.str());
        }
        iter++;
    }

#endif

    if(hP != nullptr)
        delete[] hP;

    freeTemporaries();

}

template <
    typename T,
    int DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT>
void RidgeDetection<T, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>::doChoose() 
{
    err_ = DeviceChoose::choose<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
        dP_, dS_, np_, dNs_, r1_, 
        (IN_MEM_LAYOUT == COL_MAJOR) ? pPitch_ / sizeof(T) : DIM,
        (OUT_MEM_LAYOUT == COL_MAJOR) ? sPitch_ / sizeof(T) : DIM,
        nullptr, verbose_);
    checkCudaErrors(err_);
    checkCudaErrors(cudaDeviceSynchronize());
    getChosenSamplesCount();

    allocTemporaries();
}

template <
    typename T,
    int DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT>
void RidgeDetection<T, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>::doEvolve() 
{
    err_ = DeviceEvolve::evolve<DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>(
        dP_, dS_, dCordSums_, dSpherePointCnt_, np_, ns_, r1_, 
        (IN_MEM_LAYOUT == COL_MAJOR) ? pPitch_ / sizeof(T) : DIM,
        (OUT_MEM_LAYOUT == COL_MAJOR) ? sPitch_ / sizeof(T) : DIM,
        (IN_MEM_LAYOUT == COL_MAJOR) ? csPitch_ / sizeof(T) : DIM,
        nullptr, verbose_);
    checkCudaErrors(err_);
}

template <
    typename T,
    int DIM,
    DataMemoryLayout    IN_MEM_LAYOUT,
    DataMemoryLayout    OUT_MEM_LAYOUT>
void RidgeDetection<T, DIM, IN_MEM_LAYOUT, OUT_MEM_LAYOUT>::doDecimate() 
{
    err_ = DeviceDecimate::decimate<DIM, OUT_MEM_LAYOUT>(
        dS_, dNs_, r2_, (OUT_MEM_LAYOUT == COL_MAJOR) ? sPitch_ / sizeof(T) : DIM,
        nullptr, verbose_);

    checkCudaErrors(err_);
    checkCudaErrors(cudaDeviceSynchronize());
}

}   // end namespace bruteForce
}   // end namespace gpu
}   // end namespace rd
