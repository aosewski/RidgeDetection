/**
 * @file dev_samples_set.cuh
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled: 
 * "Design of the parallel version of the ridge detection algorithm for the multidimensional 
 * random variable density function and its implementation in CUDA technology",
 * which is conducted under the supervision of prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and Information
 * Technology Warsaw University of Technology 2016
 */

#ifndef CPU_SAMPLES_SET_HPP_
#define CPU_SAMPLES_SET_HPP_

#include "rd/utils/utilities.hpp"
#include "rd/utils/rd_params.hpp"
#include "rd/utils/cmd_line_parser.hpp"
#include "rd/gpu/device/samples_generator.cuh"

#include <iomanip>
#include <iterator>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <vector>
#include <algorithm>


#include <cuda_runtime_api.h>

namespace rd {
namespace gpu {

/**
 * @brief      Class representing samples set.
 * 
 * It can generate samples on GPU.
 *
 * @tparam     T     Point's coordinate data type.
 */
template<typename T>
class Samples {

public:

    size_t dim_;
    size_t size_;
    T *samples_;

    Samples() 
    :
        samples_(nullptr),
        dim_(0),
        size_(0)
    {
    }

    /**
     * @brief      Generates or loads samples.
     *
     * @param      rdParams        RD parameters.
     * @param      rdsParams       Spiral parameters.
     * @param[in]  dstFolder       Paths to folders where search the file to
     *                             load, if available.
     * @param[in]  downgradeToDim  Downgrades loaded samples to given dimension,
     *                             if their dimension is higher.
     */
    Samples(RDParams<T> &rdParams,
            RDSpiralParams<T> &rdsParams,
            std::vector<std::string> const &dstFolder = std::vector<std::string>(),
            size_t downgradeToDim = 0)
    :
        samples_(nullptr),
        dim_(0),
        size_(0)
    {
        if (rdsParams.loadFromFile)
        {
            loadGeneratedSamples(rdParams, rdsParams, dstFolder, downgradeToDim);
        }
        else
        {
            switch (rdParams.dim)
            {
                case 3: genSpiral3D(rdParams.np, rdsParams.a, rdsParams.b, rdsParams.sigma); break;
                case 2: genSpiral2D(rdParams.np, rdsParams.a, rdsParams.b, rdsParams.sigma); break;
                default: genSegmentND(rdParams.np, rdParams.dim, rdsParams.sigma, rdsParams.a); break;
            }
        }
    }

    virtual ~Samples() {
        clear();
    }

    void clear() {
        dim_ = 0;
        size_ = 0;
        if (samples_) 
        {
            checkCudaErrors(cudaFree(samples_));
            samples_ = nullptr;
        }
    }

    /**
     * @brief      Loads generated samples from given file.
     *
     * @note       This function assumes file has descriptor in first line as a
     *             comment starting from '%'.
     *
     * @param      rdParams   RD parameters.
     * @param      rdsParams  Spiral parameters.
     * @param[in]  dstFolder  Path to folder containing @p filename relative to
     *                        binary file.
     *
     * @return     Pointer to memory with read data.
     */
    T * loadGeneratedSamples(RDParams<T> &rdParams,
                             RDSpiralParams<T> &rdsParams,
                             std::vector<std::string> const &dstFolder = std::vector<std::string>(),
                             size_t downgradeToDim = 0)
    {
        clear();
        std::ifstream aFile;
        for (std::string p : dstFolder)
        {
            std::string filePath = findPath(p, rdsParams.file);
            aFile.open(filePath);
            if (aFile.is_open())
            {
                break;
            }
            else
            {
                // clear unsuccessful attempt to open a file state
                aFile.clear();
            }
            
        }
        if (!aFile.is_open())
        {
            throw std::invalid_argument(std::string("Can't find file: ") + rdsParams.file);
        }

        // read file descriptor in first line
        std::string line = "";
        std::getline(aFile, line, '\n');        

        std::vector<char const *> argvCstr(1);
        std::vector<std::string> argvStr(1);
        for (size_t pos = 0;;)
        {
            size_t start = line.find_first_of('-', pos);
            size_t end = line.find_first_of(' ', start);
            std::string value = line.substr(start, end - start);
            argvStr.push_back(value);
            argvCstr.push_back(argvStr.back().c_str());
            pos = end + 1;
            if (end == std::string::npos)
                break;
        }

        rd::CommandLineArgs args((int)argvCstr.size(), argvCstr.data());
        if (args.CheckCmdLineFlag("dim"))
        {
            args.GetCmdLineArgument("dim", rdParams.dim);
            if (downgradeToDim && downgradeToDim < rdParams.dim)
                rdParams.dim = downgradeToDim;
        }
        else
        {
            throw std::logic_error("Missing \"dim\" parameter in descriptor!");
        }
        if (args.CheckCmdLineFlag("np"))
        {
            args.GetCmdLineArgument("np", rdParams.np);
        }
        else
        {
            throw std::logic_error("Missing \"np\" parameter in descriptor!");
        }
        if (args.CheckCmdLineFlag("a"))
        {
            args.GetCmdLineArgument("a", rdsParams.a);
        }
        else
        {
            throw std::logic_error("Missing \"a\" parameter in descriptor!");
        }
        if (rdParams.dim == 2 || rdParams.dim == 3)
        {
            if (args.CheckCmdLineFlag("b"))
            {
                args.GetCmdLineArgument("b", rdsParams.b);
            }
            else
            {
                throw std::logic_error("Missing \"b\" parameter in descriptor!");
            }
        }
        if (args.CheckCmdLineFlag("s"))
        {
            args.GetCmdLineArgument("s", rdsParams.sigma);
        }
        else
        {
            throw std::logic_error("Missing \"s\" parameter in descriptor!");
        }

        dim_ = rdParams.dim;
        size_ = rdParams.np;
        allocateMemory();
        T * aux = new T[size_ * dim_];
        // load data tu axuiliary table
        readStream(aFile, aux, size_);
        checkCudaErrors(cudaMemcpy(samples_, aux, size_ * dim_ * sizeof(T), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaDeviceSynchronize());
        delete[] aux;
        return samples_;
    }

    /**
     * @note This function assumes that file contains solely samples.
     *
     * @param      filename   Source file name.
     * @param[in]  dstFolder  Path to folder containing @a filename relative to
     *                        binary file.
     *
     * @return     Pointer to memory with read data.
     */
    T * loadFromFile(std::string filename,
                     std::string dstFolder = std::string())
    {

        clear();
        std::string filePath = findPath(dstFolder, filename);
        std::ifstream aFile(filePath);
        if (aFile.fail()) {
            throw std::invalid_argument(filename);
        }

        // get number of points in file
        size_ = std::count(std::istreambuf_iterator<char>(aFile),
                std::istreambuf_iterator<char>(), '\n');
        aFile.seekg(aFile.beg);

        std::string line = "";
        std::getline(aFile, line, '\n');

        // get points dimension
        std::vector<T> aux;
        std::istringstream iss_(line);
        while(true)
        {
            T value;
            iss_ >> value;
            if (iss_.eof())
                break;
            else
                aux.push_back(value);
        }
        
        dim_ = aux.size();
        samples_ = new T[size_ * dim_];
        aFile.seekg(aFile.beg);

        // read values from file
        readStream(aFile, samples_, size_);

        return samples_;
    }


    /**
     * @brief      Loads input data from input stream.
     *
     * @return     Pointer to input loaded data.
     */
    T * loadFromInputStream()
    {
        clear();
                // get number of samples in stream
        size_ = std::count(std::istreambuf_iterator<char>(std::cin),
                std::istreambuf_iterator<char>(), '\n');
        std::cin.seekg(std::cin.beg);

        std::string line = "";
        std::getline(std::cin, line, '\n');

        // get samples dimension
        std::vector<T> aux;
        std::istringstream iss_(line);
        while(true)
        {
            T value;
            iss_ >> value;
            if (iss_.eof())
                break;
            else
                aux.push_back(value);
        }
        
        dim_ = aux.size();
        samples_ = new T[size_ * dim_];
        std::cin.seekg(std::cin.beg);

        // read values from stream
        readStream(std::cin, samples_, size_);
        return samples_;
    }


    /**
     * @brief      Saves stored samples to @p fileName in binary file directory.
     *
     * @param      fileName  File name to store samples.
     * @param      comment   Comment to add at the beggining of file.
     */
    void saveToFile(std::string const &fileName,
                    std::string const &comment = "")
    {
        if (samples_ == nullptr)
        {
            std::cerr << "There are no samples to save!" << std::endl;
            return;
        }

        std::string dstFilePath = findPath("", "", true) + fileName + ".txt";

        std::ofstream dstFile(dstFilePath.c_str(), std::ios::out | std::ios::trunc);
        if (dstFile.fail())
            throw std::logic_error("Couldn't open file: " + fileName);

        if (!comment.empty())
            dstFile << "%" << comment << "\n";

        T * src;
        src = new T[size_ * dim_];
        checkCudaErrors(cudaMemcpy(src, samples_, size_ * dim_ * sizeof(T), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());

        dstFile.precision(4);
        for (std::size_t i = 0; i < size_; ++i)
        {
            for (std::size_t d = 0; d < dim_; ++d)
            {
                dstFile << std::right << std::fixed << std::setw(10) << src[i * dim_ + d];
                dstFile << "  ";
            }
            dstFile << "\n";
        }

        dstFile.close();

        delete[] src;
    }

    /**
     * @brief      Generates some nice parametric spiral.
     *
     * @param[in]  n      Number of samples to generate.
     * @param[in]  a      Spiral parameter max value.
     * @param[in]  b      Scaling width parameter
     * @param[in]  sigma  Standard deviation.
     *
     * @return     Pointer to generated samples.
     */
    T * genSpiral2D(size_t n, T a, T b, T sigma)
    {
        clear();
        dim_ = 2;
        size_ = n;
        allocateMemory();
        rd::gpu::SamplesGenerator<T>::spiral2D(size_, a, b, sigma, samples_);
        return samples_;
    }

    /**
     * @brief      Generates some nice parametric spiral.
     *
     * @param[in]  n      Number of samples to generate.
     * @param[in]  a      Spiral parameter max value.
     * @param[in]  b      Scaling parameter.
     * @param[in]  sigma  Standard deviation.
     *
     * @return     Pointer to generated samples.
     */
    T * genSpiral3D(size_t n, T a, T b, T sigma)
    {
        clear();
        dim_ = 3;
        size_ = n;
        allocateMemory();
        rd::gpu::SamplesGenerator<T>::spiral3D(size_, a, b, sigma, samples_);
        return samples_;
    }


    /*----------------------------------------------------------------------*//**
     * @brief      Generates Dim-dimensional set of N samples with normally
     *             distributed noise.
     *
     * @param      n       - number of samples
     * @param      dim     - dimension
     * @param      sigma   - standard deviation
     * @param[in]  length  The length of generated segment and range of generated values.
     *
     * @return     Pointer to generated segment data
     */
    T * genSegmentND(size_t n, size_t dim, T sigma, size_t length = 1)
    {
        clear();
        dim_ = dim;
        size_ = n;
        allocateMemory();
        rd::gpu::SamplesGenerator<T>::segmentND(size_, dim_, sigma, length, samples_);
        return samples_;
    }

private:

    void allocateMemory()
    {
        checkCudaErrors(cudaMalloc((void**)&samples_, size_ * dim_ * sizeof(T)));
    }

    template <typename InputStream>
    void readStream(InputStream &input, T * dst, size_t size)
    {
        size_t index = 0;
        std::string line;
        while (std::getline(input, line, '\n'))
        {
            // skip empty lines
            if (line.empty() || line == std::string("\n")) continue;
            std::istringstream iss(line);
            for (size_t d = 0; d < dim_; ++d)
            {
                if (!(iss >> dst[index * dim_ + d]))
                    break;
            }
            index++;
        }
        if (index != size)
            throw std::logic_error("Incorrect points count read!");
    }
};


/*----------------------------------------------------------------------*//**
 * @brief      Saves stored samples to @a fileName in binary file directory.
 * @note       Specialization for double precision floating point numbers.
 *
 * @param      fileName  File name to store samples.
 * @param      comment   Comment to add at the beggining of file.
 */
template <>
void Samples<double>::saveToFile(std::string const &fileName,
                std::string const &comment)
{
    if (samples_ == nullptr)
    {
        std::cerr << "There are no samples to save!" << std::endl;
        return;
    }

    std::string dstFilePath = findPath("", "", true) + fileName + ".txt";

    std::ofstream dstFile(dstFilePath.c_str(), std::ios::out | std::ios::trunc);
    if (dstFile.fail())
        throw std::logic_error("Couldn't open file: " + fileName);

    if (!comment.empty())
        dstFile << "%" << comment << "\n";

    dstFile.precision(12);
    for (std::size_t i = 0; i < size_; ++i)
    {
        for (std::size_t d = 0; d < dim_; ++d)
        {
            dstFile << std::right << std::fixed << std::setw(16) << samples_[i * dim_ + d];
            dstFile << "  ";
        }
        dstFile << "\n";
    }

    dstFile.close();
}

/* **************************************************************************/

} // end namespace gpu
} // end namespace rd

#endif /* CPU_SAMPLES_SET_HPP_ */
