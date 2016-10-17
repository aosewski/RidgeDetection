/**
 * @file test_util.hpp
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering, Faculty of Electronics and Information
 * Technology, Warsaw University of Technology 2016
 */

#pragma once

#include "rd/cpu/samples_generator.hpp"
#include "rd/utils/assessment_quality.hpp"
#include "rd/utils/utilities.hpp"

#if defined(__cplusplus) && defined(__CUDACC__)
#include <helper_cuda.h>
#endif

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <cmath>
#include <type_traits>
#include <algorithm>
#include <stdexcept>
#include <iomanip>
#include <ctime>

#ifdef RD_USE_OPENMP
#include <omp.h>
#endif


//-----------------------------------------------------------------------------
//  Helper routines for list comparison and display
//-----------------------------------------------------------------------------


/**
 * Compares the equivalence of two arrays
 */
template <
    typename T, 
    typename OffsetT,
    typename std::enable_if<
            std::is_integral<T>::value
        >::type* = nullptr>
int CompareResult(
    T const *   computed, 
    T const *   reference, 
    OffsetT     len, 
    bool        verbose = true)
{
    int result = 0;
    #ifdef RD_USE_OPENMP
    int threadsNum = omp_get_num_procs();
    #pragma omp parallel for schedule(static), num_threads(threadsNum), shared(result)
    #endif
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference[i])
        {
            if (verbose)
            {
                std::cout << "INCORRECT: [" << i << "]: "
                << "(computed) " << computed[i] << " != "
                << reference[i];
            }
            #ifdef RD_USE_OPENMP
            #pragma omp atomic write
            result = 1;
            #else
            result = 1;
            #endif
        }

        if (result)
        {
            #ifdef RD_USE_OPENMP
            #pragma omp cancel for
            #else
            break;
            #endif
        }

        #ifdef RD_USE_OPENMP
        #pragma omp cancellation point for
        #endif
    }

    return result;
}

/**
 * Compare with scalar value
 */
template <
    typename T, 
    typename OffsetT,
    typename std::enable_if<
            std::is_integral<T>::value
        >::type* = nullptr>
int CompareResult(
    T const * computed, 
    T reference, 
    OffsetT len, 
    bool verbose = true)
{
    int result = 0;
    #ifdef RD_USE_OPENMP
    int threadsNum = omp_get_num_procs();
    #pragma omp parallel for schedule(static), num_threads(threadsNum), shared(result)
    #endif
    for (OffsetT i = 0; i < len; i++)
    {
        if (computed[i] != reference)
        {
            if (verbose)
            {
                std::cout << "INCORRECT: [" << i << "]: "
                << "(computed) " << computed[i] << " != "
                << reference;
            }
            #ifdef RD_USE_OPENMP
            #pragma omp atomic write
            result = 1;
            #else
            result = 1;
            #endif
        }

        if (result)
        {
            #ifdef RD_USE_OPENMP
            #pragma omp cancel for
            #else
            break;
            #endif
        }

        #ifdef RD_USE_OPENMP
        #pragma omp cancellation point for
        #endif
    }
    return result;
}

/**
 * Compares the equivalence of two arrays
 */
template <
    typename T, 
    typename OffsetT,
    typename std::enable_if<
            std::is_floating_point<T>::value
        >::type* = nullptr>
int CompareResult(
    T const *   computed, 
    T const *   reference, 
    OffsetT     len, 
    bool        verbose = true)
{
    int result = 0;
    #ifdef RD_USE_OPENMP
    int threadsNum = omp_get_num_procs();
    #pragma omp parallel for schedule(static), num_threads(threadsNum), shared(result)
    #endif
    for (OffsetT i = 0; i < len; i++)
    {
        if (!rd::almostEqual(computed[i], reference[i]))
        {
            T difference = std::abs(computed[i]-reference[i]);
            T fraction = difference / std::abs(reference[i]);

            if (fraction > 0.0001)
            {
                if (verbose) 
                {
                    std::cout << "INCORRECT: [" << i << "]: "
                    << "(computed) " << computed[i] << " != "
                    << reference[i] << " (difference:" << difference << ", fraction: " << fraction << ")";

                    // std::cout << "\nReference:\t\t\tComputed:\n";
                    // for (OffsetT k = 0; k < len; k++)
                    // {
                    //     std::cout <<"["<<k<<"]: " << reference[k] << "\t\t" << computed[k] << "\n";
                    // }
                    // std::cout << "\n\n";
                }
                #ifdef RD_USE_OPENMP
                #pragma omp atomic write
                result = 1;
                #else
                result = 1;
                #endif
            }
        }

        if (result)
        {
            #ifdef RD_USE_OPENMP
            #pragma omp cancel for
            #else
            break;
            #endif
        }

        #ifdef RD_USE_OPENMP
        #pragma omp cancellation point for
        #endif
    }
    return result;
}

/**
 * Compares with scalar value
 */
template <
    typename T, 
    typename OffsetT,
    typename std::enable_if<
            std::is_floating_point<T>::value
        >::type* = nullptr>
int CompareResult(
    T const *   computed, 
    T           reference, 
    OffsetT     len, 
    bool        verbose = true)
{
    int result = 0;
    #ifdef RD_USE_OPENMP
    int threadsNum = omp_get_num_procs();
    #pragma omp parallel for schedule(static), num_threads(threadsNum), shared(result)
    #endif
    for (OffsetT i = 0; i < len; i++)
    {
        if (!rd::almostEqual(computed[i], reference))
        {
            T difference = std::abs(computed[i]-reference);
            T fraction = difference / std::abs(reference);

            if (fraction > 0.0001)
            {
                if (verbose) 
                {
                    std::cout << "INCORRECT: [" << i << "]: "
                    << "(computed) " << computed[i] << " != "
                    << reference << " (difference:" << difference << ", fraction: " << fraction << ")";

                    // std::cout << "\nReference:\t\t\tComputed:\n";
                    // for (OffsetT k = 0; k < len; k++)
                    // {
                    //     std::cout <<"["<<k<<"]: " << reference << "\t\t" << computed[k] << "\n";
                    // }
                    // std::cout << "\n\n";
                }
                #ifdef RD_USE_OPENMP
                #pragma omp atomic write
                result = 1;
                #else
                result = 1;
                #endif
            }
        }

        if (result)
        {
            #ifdef RD_USE_OPENMP
            #pragma omp cancel for
            #else
            break;
            #endif
        }
        
        #ifdef RD_USE_OPENMP
        #pragma omp cancellation point for
        #endif
    }
    return result;
}

#if defined(__cplusplus) && defined(__CUDACC__)
/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename T>
int CompareDeviceResults(
    T const *h_reference,
    T const *d_data,
    size_t num_items,
    bool verbose = true,
    bool sort_dev_results = false,
    bool display_data = false)
{
    // Allocate array on host
    T *h_data = new T[num_items];

    // Copy data back
    checkCudaErrors(cudaMemcpy(h_data, d_data, sizeof(T) * num_items, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    if (sort_dev_results)
    {
        std::sort(h_data, h_data + num_items);
    }

    // Display data
    if (display_data) {
        std::cout << "\nReference:\t\t\tComputed:\n";
        for (size_t i = 0; i < num_items; i++)
        {
            std::cout <<"["<<i<<"]: " << h_reference[i] << "\t\t" << h_data[i] << "\n";
        }
        std::cout << "\n\n";
    }

    // Check
    int retval = CompareResult(h_data, h_reference, num_items, verbose);

    // Cleanup
    if (h_data) delete[] h_data;

    return retval;
}

/**
 * Verify the contents of a device array match those
 * of a host array
 */
template <typename T>
int CompareDeviceResults(
    T h_reference,
    T const *d_data,
    size_t num_items,
    bool verbose = true,
    bool sort_dev_results = false,
    bool display_data = false)
{
    // Allocate array on host
    T *h_data = new T[num_items];

    // Copy data back
    checkCudaErrors(cudaMemcpy(h_data, d_data, sizeof(T) * num_items, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    if (sort_dev_results)
    {
        std::sort(h_data, h_data + num_items);
    }

    // Display data
    if (display_data) {
        std::cout << "\nReference:\t\t\tComputed:\n";
        for (size_t i = 0; i < num_items; i++)
        {
            std::cout <<"["<<i<<"]: " << h_reference << "\t\t" << h_data[i] << "\n";
        }
        std::cout << "\n\n";
    }

    // Check
    int retval = CompareResult(h_data, h_reference, num_items, verbose);

    // Cleanup
    if (h_data) delete[] h_data;

    return retval;
}

/**
 * Verify the contents of a device array match those
 * of a device array
 */
template <typename T>
int CompareDeviceDeviceResults(
    T const *d_reference,
    T const *d_data,
    size_t num_items,
    bool verbose = true,
    bool sort_dev_results = false,
    bool display_data = false)
{
    // Allocate array on host
    T *h_reference = new T[num_items];
    T *h_data = new T[num_items];

    // Copy data back
    checkCudaErrors(cudaMemcpy(h_reference, d_reference, sizeof(T) * num_items, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_data, d_data, sizeof(T) * num_items, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    if (sort_dev_results)
    {
        std::sort(h_reference, h_reference + num_items);
        std::sort(h_data, h_data + num_items);
    }

    // Display data
    if (display_data) {
        std::cout << "Reference:\t\t\tComputed:\n";
        for (size_t i = 0; i < num_items; i++)
        {
            std::cout <<"["<<i<<"]: " << h_reference[i] << "\t\t" << h_data[i] << "\n";
        }
        std::cout << "\n\n";
    }

    // Check
    int retval = CompareResult(h_data, h_reference, num_items, verbose);

    // Cleanup
    if (h_reference) delete[] h_reference;
    if (h_data) delete[] h_data;

    return retval;
}

#endif // defined(__cplusplus) && defined(__CUDACC__)

//------------------------------------------------------------
//  Data generation
//------------------------------------------------------------

template <typename T>
struct PointCloud 
{
    int pointCnt_;
    int dim_;
    T stddev_;

    std::vector<T> points_;

    PointCloud(int pointCnt, int dim, T stddev)
    :
        pointCnt_(pointCnt), dim_(dim), stddev_(stddev)
    {}

    PointCloud(
        std::string const & filePath, 
        int pointCnt,
        int dim, 
        T stddev)
    :
        pointCnt_(pointCnt), 
        dim_(dim), 
        stddev_(stddev)
    {
        points_ = readFile(filePath, pointCnt_, dim_);
    }

    virtual void initializeData() = 0;
    
    virtual rd::RDAssessmentQuality<T> * getQualityMeasurer(int dim)  const = 0;
    virtual void getCloudParameters(T & a, T & b) const = 0;
    virtual ~PointCloud() {};

    /**
     * @brief      Extract part of points from a set (and) or extract only part of dimensions
     *
     * @param      out          Output point set.
     * @param[in]  outPointCnt  Number of points to extract
     * @param[in]  outDim       Dimension of extracted points
     *
     * @tparam     T     Data type
     */
    std::vector<T> extractPart(
        int outPointCnt,
        int outDim) const
    {
        if (outDim > dim_)
        {
            std::string errstr = std::string("Output dimension is larger than input! ") + __FILE__ + std::string(" ") + std::to_string(__LINE__);
            throw std::logic_error(errstr.c_str());
        }

        std::vector<T> out(outPointCnt * outDim);

        #ifdef RD_USE_OPENMP
        int threadsNum = omp_get_num_procs();
        #pragma omp parallel for schedule(static), num_threads(threadsNum)
        #endif
        for (int k = 0; k < outPointCnt; ++k)
        {
            for (int d = 0; d < outDim; ++d)
            {
                out[k * outDim + d] = points_[k * dim_ + d];
            }
        }

        return out;
    }

    std::vector<T> readFile(
        std::string const & filePath, 
        int                 npoints, 
        int                 dim) 
    {
        std::ifstream inFile(filePath, std::ios::binary | std::ios::ate);
        if (inFile.fail())
        {
            throw std::logic_error("Couldn't open file for reading in binary mode: " + filePath);
        }
        size_t size = inFile.tellg();
        size_t requestedBytes = npoints * dim * sizeof(T);
        if (size < requestedBytes)
        {
            throw std::logic_error("Cannot read " + std::to_string(requestedBytes) + " bytes from"
                " file of size " + std::to_string(size) + " bytes!");
        }
        inFile.seekg(0);

        // read data into row major order
        std::vector<T> data(npoints * dim);
        inFile.read(reinterpret_cast<char*>(data.data()), requestedBytes);
        if (inFile.fail())
        {
            throw std::logic_error("Error occured while reading file: " + filePath);
        }

        return data;
    }

    void writeFile(
        std::string const &     filePath, 
        std::vector<T> const &  data, 
        int                     npoints, 
        int                     dim)
    {
        std::ofstream outFile(filePath, std::ios::binary | std::ios::ate);
        if (outFile.fail())
        {
            throw std::logic_error("Couldn't open file for writing in binary mode: " + filePath);
        }

        outFile.write(reinterpret_cast<char const *>(data.data()), npoints * dim * sizeof(T));
        if (outFile.fail())
        {
            throw std::logic_error("Error occured while writing file: " + filePath);
        }
    }

};

template <typename T>
struct SpiralPointCloud : PointCloud<T>
{
    T a_, b_;

    SpiralPointCloud(T a, T b, int pointCnt, int dim, T stddev)
    :
        PointCloud<T>(pointCnt, dim, stddev), a_(a), b_(b)
    {}

    SpiralPointCloud(std::string const & filePath, T a, T b, 
        int pointCnt, int dim, T stddev)
    :
        PointCloud<T>(filePath, pointCnt, dim, stddev), a_(a), b_(b)
    {}

    virtual ~SpiralPointCloud() {};

    virtual void initializeData() override final 
    {
        this->points_.clear();
        this->points_.resize(this->pointCnt_ * this->dim_);

        switch (this->dim_)
        {
            case 3: rd::genSpiral3D(this->pointCnt_, a_, b_, this->stddev_, this->points_.data()); break;
            case 2: rd::genSpiral2D(this->pointCnt_, a_, b_, this->stddev_, this->points_.data()); break;
            default: 
            {
                std::string errstr = std::string("Unsupported dimension [") + 
                    std::to_string(this->dim_) + std::string("]! Can't initialize Spiral! ") + 
                    __FILE__ + "(" + std::to_string(__LINE__) + ")";
                    throw std::logic_error(errstr.c_str());
            }
        }
    }

    virtual rd::RDSpiralAssessmentQuality<T> * getQualityMeasurer(int dim) const override final
    {
        return new rd::RDSpiralAssessmentQuality<T>(
            static_cast<size_t>(std::max(std::ceil(0.1f * this->pointCnt_), 100.f)), dim, a_, b_);
    }

    virtual void getCloudParameters(T & a, T & b) const override final
    {
        a = a_;
        b = b_;
    }
};

template <typename T>
struct SegmentPointCloud : PointCloud<T>
{
    T length_;

    SegmentPointCloud(T length, int pointCnt, int dim, T stddev)
    :
        PointCloud<T>(pointCnt, dim, stddev), length_(length)
    {}

    SegmentPointCloud(std::string const & filePath, T length, 
        int pointCnt, int dim, T stddev)
    :
        PointCloud<T>(filePath, pointCnt, dim, stddev), length_(length)
    {}

    virtual ~SegmentPointCloud() {};

    virtual void initializeData() override final 
    {
        this->points_.clear();
        this->points_.resize(this->pointCnt_ * this->dim_);
        rd::genSegmentND(this->pointCnt_, this->dim_, this->stddev_, this->points_.data(), length_);
    }

    // virtual rd::RDAssessmentQuality<T> getQualityMeasurer(int dim) const override final
    virtual rd::RDSegmentAssessmentQuality<T> * getQualityMeasurer(int dim) const override final
    {
        return new rd::RDSegmentAssessmentQuality<T>(
            static_cast<size_t>(std::max(std::ceil(0.1f * this->pointCnt_), 100.f)), dim);
    }

    virtual void getCloudParameters(T & a, T & b) const override final
    {
        a = length_;
        b = 0;
    }
};

template <typename T>
struct RdData
{
    /// samples dimension
    size_t dim;
    /// cardinality of samples set
    size_t np;
    /// cardinality of initial choosen samples set
    size_t ns;
    /**
     * @var r1_ Algorithm parameter. Radius used for choosing samples and in
     * evolve phase.
     */
    T r1;
    /**
     * @var r2_ Algorithm parameter. Radius used for decimation phase.
     */
    T r2;
    /// generated samples parameters (s - standard deviation)
    T a, b, s;
    /// table containing samples
    T const * P;
    /// table containing choosen samples
    std::vector<T> S;

    RdData()
    :
        dim(0),
        np(0),
        ns(0),
        r1(0), r2(0),
        a(0), b(0), s(0),
        P(nullptr), S()
    {
    }  
};

//--------------------------------------------------------------
//  Static for loop (host)
//--------------------------------------------------------------

template <int COUNT, int MAX, typename Lambda, int STEP = 1>
struct StaticFor 
{
    template <typename... Args>
    static void impl(Args&&... args) 
    {
        Lambda::impl(std::integral_constant<int, COUNT>{}, std::forward<Args>(args)...);
        StaticFor<COUNT + STEP, MAX, Lambda, STEP>::impl(std::forward<Args>(args)...);
    }
};

template <int N, typename Lambda, int STEP>
struct StaticFor<N, N, Lambda, STEP> 
{
    template <typename... Args>
    static void impl(Args&&... args) 
    {
        Lambda::impl(std::integral_constant<int, N>{}, std::forward<Args>(args)...);
    }
};

//---------------------------------------------------------------
// Date & time generation
//---------------------------------------------------------------

std::string getCurrDateAndTime()
{
    auto time = std::time(nullptr);
    auto currLocalTime = *std::localtime(&time); 
    char buff[20];
    strftime(buff, 20, "%F-%H-%M-%S", &currLocalTime);

    return std::string(buff);
}

std::string getCurrDate()
{
    auto time = std::time(nullptr);
    auto currLocalTime = *std::localtime(&time); 
    char buff[12];
    strftime(buff, 12, "%F", &currLocalTime);

    return std::string(buff);
}

//---------------------------------------------------------------
//  extract bin suffix for log purposes
//---------------------------------------------------------------

std::string getBinConfSuffix()
{
    std::string suffix;
    #ifdef CUB_CDP
    suffix += "cdp_";
    #else
    suffix += "nocdp_";
    #endif
    #ifdef RD_DOUBLE_PRECISION
    suffix += "dprec_";
    #else
    suffix += "sprec_";
    #endif
    #ifdef RD_USE_OPENMP
    suffix += "omp_";
    #endif

    return suffix;
}

//---------------------------------------------------------------
//  Converting vector to string
//---------------------------------------------------------------

template <typename T>
std::string rdToString(std::vector<T> const & v)
{
    std::string result;
    char comma[3] = {'\0', ' ', '\0'};
    result.reserve(v.size()*2+2);
    result += "[";
    for (const auto & e : v)
    {
        result += comma + std::to_string(e);
        comma[0] = ',';
    }
    result += "]";

    return result;
}

//---------------------------------------------------------------
//  Reporting kernel resource usage
//---------------------------------------------------------------

#if defined(__cplusplus) &&  defined(__CUDACC__)

struct KernelResourceUsage
{
    size_t sharedSizeBytes;
    size_t localSizeBytes;
    int numRegsPerThr;
    int numRegsPerBlock;
    int smOccupancy;

    KernelResourceUsage() = default;

    template <typename KernelPtr>
    KernelResourceUsage(KernelPtr kernelPtr, int blockThreads = 0)
    {
        cudaFuncAttributes kernelAttrs;
        checkCudaErrors(cudaFuncGetAttributes(&kernelAttrs, kernelPtr));
        sharedSizeBytes = kernelAttrs.sharedSizeBytes;
        localSizeBytes = kernelAttrs.localSizeBytes;
        numRegsPerThr = kernelAttrs.numRegs;
        numRegsPerBlock = numRegsPerThr * blockThreads;

        int deviceOrdinal;
        checkCudaErrors(cudaGetDevice(&deviceOrdinal));
        // get SM occupancy
        checkCudaErrors(cub::MaxSmOccupancy(smOccupancy, kernelPtr,
            blockThreads));
    }

    // returns values in KB
    std::string prettyPrint() const 
    {
        std::ostringstream oss;
        oss << "(" << numRegsPerThr << "," 
            << numRegsPerBlock << ","
            << smOccupancy << ","
            << localSizeBytes/1024 << ","
            << sharedSizeBytes/1024 << ")";
        return oss.str();
    }
};

#endif // defined(__cplusplus) &&  defined(__CUDACC__)

//---------------------------------------------------------------
//  nicely formatted logging values to output stream
//---------------------------------------------------------------

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void logValue(std::ostream & stream, T const & value, int w = 12, int p = 3) 
{
    stream << std::fixed << std::right << std::setw(w) << std::setprecision(p) << value << " ";
}

template <
    typename T,
    typename std::enable_if<!std::is_fundamental<T>::value>::type* = nullptr> 
static void logValue(std::ostream & stream, T const & value, int w = 16) 
{
    stream << std::right << std::setw(w) << value << " ";
}

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr> 
static void logValue(std::ostream & stream, T const & value, int w = 10) 
{
    stream << std::right << std::setw(w) << value << " ";
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, bool>::value>::type* = nullptr>
static void logValue(std::ostream & stream, bool const & value) 
{
    stream << std::right << std::boolalpha << std::setw(6) << value << " ";
}

//-----------------------------------------------------------------------------
// LOG MULTIPLE VALUES parameter pack end case
//-----------------------------------------------------------------------------

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void logValues(std::ostream & stream, T const & value)
{
    logValue(stream, value);
}

template <
    typename T,
    typename std::enable_if<!std::is_fundamental<T>::value>::type* = nullptr> 
static void logValues(std::ostream & stream, T const & value)
{
    logValue(stream, value);
}

template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr> 
static void logValues(std::ostream & stream, T const & value)
{
    logValue(stream, value);
}

template <
    typename T,
    typename std::enable_if<std::is_same<T, bool>::value>::type* = nullptr>
static void logValues(std::ostream & stream, T const & value)
{
    logValue(stream, value);
}


//-----------------------------------------------------------------------------
// LOG MULTIPLE VALUES inductive case
//-----------------------------------------------------------------------------


// ---------- forward declarations, so each version sees others----------------
template <
    typename T,
    typename... Args,
    typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
static void logValues(std::ostream & stream, T const & value, Args&&... args);

template <
    typename T,
    typename... Args,
    typename std::enable_if<!std::is_fundamental<T>::value>::type* = nullptr> 
static void logValues(std::ostream & stream, T const & value, Args&&... args);

template <
    typename T,
    typename... Args,
    typename std::enable_if<std::is_integral<T>::value>::type* = nullptr> 
static void logValues(std::ostream & stream, T const & value, Args&&... args);

template <
    typename T,
    typename... Args,
    typename std::enable_if<std::is_same<T, bool>::value>::type* = nullptr>
static void logValues(std::ostream & stream, T const & value, Args&&... args);

//-----------------------------------------------------------------------------

template <
    typename T,
    typename... Args,
    typename std::enable_if<std::is_floating_point<T>::value>::type*>
static void logValues(std::ostream & stream, T const & value, Args&&... args)
{
    logValue(stream, value);
    logValues(stream, std::forward<Args>(args)...);
}

template <
    typename T,
    typename... Args,
    typename std::enable_if<!std::is_fundamental<T>::value>::type*> 
static void logValues(std::ostream & stream, T const & value, Args&&... args)
{
    logValue(stream, value);
    logValues(stream, std::forward<Args>(args)...);
}

template <
    typename T,
    typename... Args,
    typename std::enable_if<std::is_integral<T>::value>::type*> 
static void logValues(std::ostream & stream, T const & value, Args&&... args)
{
    logValue(stream, value);
    logValues(stream, std::forward<Args>(args)...);
}

template <
    typename T,
    typename... Args,
    typename std::enable_if<std::is_same<T, bool>::value>::type*>
static void logValues(std::ostream & stream, T const & value, Args&&... args)
{
    logValue(stream, value);
    logValues(stream, std::forward<Args>(args)...);
}