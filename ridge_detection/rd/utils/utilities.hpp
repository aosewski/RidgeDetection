/**
 * @file utilities.hpp
 * @author: Adam Rogowiec
 *
 * Plik jest integralną częścią pracy dyplomowej magisterskiej pod tytułem:
 * "Opracowanie równoległej wersji algorytmu estymacji grani funkcji gęstości
 *	wielowymiarowej zmiennej losowej i jej implementacja w środowisku CUDA"
 * prowadzonej pod opieką prof. dr hab. inż. Marka Nałęcza
 *
 * IAIS wydział Elektroniki i Technik Informacyjnych,
 * Politechniki Warszawska 2015
 */

#ifndef UTILITIES_HPP_
#define UTILITIES_HPP_

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cmath>
#include <fenv.h>
#include <string>
#include <fstream>
#if __cplusplus >= 201103L
#   include <utility>
#   include <random>
#endif
#include <algorithm>

#include <type_traits>
#include <iomanip>
#include <limits>
#include <vector>
#include <stdexcept>

#ifdef RD_USE_OPENMP
#include <omp.h>
#endif

using std::size_t;

namespace rd
{

static const char* HLINE = "************************************************";

/////////////////////////////////////////////////////////////////////////////
//
// Funkcje mierzące czas
//
////////////////////////////////////////////////////////////////////////////

#if defined(_WIN32) || defined(_WIN64)
    #include <windows.h>
    #undef small            // Windows is terrible for polluting macro namespace
#else
	#include <sys/time.h>
#endif

struct CpuTimer
{
#if defined(_WIN32) || defined(_WIN64)

    LARGE_INTEGER ll_freq;
    LARGE_INTEGER ll_start;
    LARGE_INTEGER ll_stop;

    CpuTimer()
    {
        QueryPerformanceFrequency(&ll_freq);
    }

    void Start()
    {
        QueryPerformanceCounter(&ll_start);
    }

    void Stop()
    {
        QueryPerformanceCounter(&ll_stop);
    }

    float ElapsedMillis()
    {
        double start = double(ll_start.QuadPart) / double(ll_freq.QuadPart);
        double stop  = double(ll_stop.QuadPart) / double(ll_freq.QuadPart);

        return float((stop - start) * 1000);
    }

#else

	typedef struct timeval timerType;
	timerType start_, stop_;

    void start() {
    	gettimeofday(&start_, 0);
    }

    void stop() {
    	gettimeofday(&stop_, 0);
    }

    float elapsedMillis(double flops = 0, bool displayTime = false) {

    	double etime;
    	etime = 1000.0 * (stop_.tv_sec - start_.tv_sec)
    			+ 0.001 * (stop_.tv_usec - start_.tv_usec);
        if (displayTime)
        {
        	std::cout << "CPU time = " << etime << "ms ";
        	if (flops) 
                std::cout << "(" << 1e-6 * flops / etime << " GFLOP/s)";
            std::cout << "\n";
        }

        return etime;
    }

#endif
};

/**********************************************************/

/////////////////////////////////////////////////////////////////////////////
//
// Pozostałe przydatne funkcje
//
////////////////////////////////////////////////////////////////////////////

/**
 * \brief Allows for the treatment of an integral constant as a type at compile-time (e.g., to achieve static call dispatch based on constant integral values)
 */
template <int A>
struct Int2Type
{
   enum {VALUE = A};
};

/**
 * @brief Poszukuje pliku lub folderu.
 * @param dst_folder - ścieżka do docelowego folderu
 *  w odniesieniu do położenia pliku wykonywalnego
 * @param filename - nazwa pliku
 * @param folder - flaga czy szukać jedynie folderu
 * @return pełna ścieżka do pliku
 */
static std::string findPath(
    const std::string dst_folder,
	const std::string filename = "",
    bool folder = false)
{

    const int PATH_LENGTH = 1024;
    char searchPath_[PATH_LENGTH];
    for (int i = 0; i < PATH_LENGTH; i++) {
    	searchPath_[i] = 0;
    }
    int res = readlink("/proc/self/exe", searchPath_, PATH_LENGTH);
    if (res == -1) {
    	throw std::logic_error("Error while searching executable path!");
    }
	std::string path(searchPath_);

    // Linux & OSX path delimiter
    size_t delimiter_pos = path.find_last_of('/');
    path.erase(delimiter_pos+1, path.size());

	// cofamy się i przechodzimy do docelowego folderu
	path.append(dst_folder);

	if (!folder) {
		// Test if the file exists
		path.append(filename);
		std::fstream fh(path.c_str(), std::fstream::in);

		if (fh.bad()) {
			return std::string();
		}
	}

	return path;
}

/**
 * @fn getRandIndex
 * @param ub - upper bound for index
 * @return value in range [0,ub)
 */
inline static size_t getRandIndex(size_t ub) {
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, ub);
    return dist(gen);
}

////////////////////////////////////////////////////////////////////////////
//
// Useful table routines (create, fill, copy, transpose.. )
//
////////////////////////////////////////////////////////////////////////////

/**
 * @fn printTable
 * @brief Wypisuje na standardowe wyjście dane tabeli
 *
 * @param table - wskaźnik na tabelę do wydruku
 * @param size - rozmiar tabeli
 * @param tabName - nazwa tabeli
 */
template <typename T>
static void printTable(T* table, size_t w, size_t h, std::string tabName) {

	std::cout << "\n//=========================================================================\n";
	std::cout << tabName << ":\n";
	for (size_t i = 0; i < h; i++) 
    {
        for (size_t j = 0; j < w; j++) 
        {
            std::cout << std::left << std::setw(10) << std::setprecision(7) 
                << table[i * w + j] << " ";
        }
		std::cout << "\n";
	}
	std::cout << "\n//=========================================================================";
    std::cout << std::endl;
}

/**
 * @fn createTable
 * @brief Zwraca tablicę wypełnioną zadaną liczbą
 * @param size - rozmiar tablicy
 * @param value - wartość jaką ma zostać wypełniona tablica
 * @return wskażnik do tablicy
 */
template <typename T>
static T* createTable(size_t size, T value) {
	T* tab = new T[size];
	for (size_t i = 0; i < size; i++) {
		tab[i] = value;
	}
	return tab;
}

/**
 * @fn fillRandomDataTable
 * @param tab - tablica, którą wypełniamy danymi
 * @param size - liczba elementów w tablicy
 * @param lb - dolna granica losowanych wartości
 * @param ub - górna granica losowanych wartości
 */
template <
    typename T,
    typename std::enable_if<std::is_integral<T>::value, T>::type* = nullptr>
static void fillRandomDataTable(T *tab, size_t size, T lb = T(0), T ub = T(1))
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(lb, ub);
	
    for (size_t i = 0; i < size; i++) {
        tab[i] = dist(gen);
	}
}

template <
    typename T,
    typename std::enable_if<std::is_floating_point<T>::value, T>::type* = nullptr>
static void fillRandomDataTable(T *tab, size_t size, T lb = T(0), T ub = T(1))
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(lb, ub);

    for (size_t i = 0; i < size; i++) {
        tab[i] = dist(gen);             
	}
}

/**
 * @fn fillTable
 * @brief fill table with specified value
 * @param src
 * @param value
 * @param n - number of entries in table
 */
template <typename T>
static void fillTable(T *src, T value, size_t n) {
    for (size_t i = 0; i < n; ++i)
    {
        src[i] = value;
    }
}

template <typename T>
static void fillTable_omp(T *src, T value, size_t n) {
	#pragma omp for schedule (static)
	for (size_t i = 0; i < n; ++i)
    {
		src[i] = value;
	}
}

/**
 * @fn createRandomDataTable
 * @brief Zwraca tablicę wypełnioną pseudolosowymi danymi
 * @param size - rozmiar tablicy
 * @return wskaźnik do tablicy
 */
template <typename T>
static T* createRandomDataTable(size_t size, T lb = T(0), T ub = T(1)) {
    T* tab = new T[size];
    fillRandomDataTable(tab, size, lb, ub);
    return tab;
}

/**
 * @fn copyTable
 * @brief copy data from @p src to @p dst
 * @param src
 * @param dst
 * @param move - decides whether perform memory move insted of copy
 * @param n - number of elements to copy
 */
template <typename T>
static void copyTable(T const * const src, T *dst, size_t n, bool move = false) {
    if (move)
    {
        memmove(dst, src, n * sizeof(T));
    }
    else
    {
        memcpy(dst, src, n * sizeof(T));

    }
}

template <typename T>
static void copyTable_omp(T const * const src, T *dst, size_t n, bool move = false) {
	if (move)
    {
		memmove(dst, src, n * sizeof(T));
	}
    else
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i)
        {
            dst[i] = src[i];
        }
	}
}

/**
 * @fn transposeTable
 * @brief Performs tansposition of @p src and stores result in @p dst
 * @note @p src must be different memory location than @p dst !
 * @param src
 * @param dst
 * @param n_elem - number of elements in @p src table
 * @param dim - data dimension
 */
template <typename T>
static void transposeTable(T const * src, T *dst, size_t nelem, size_t dim) 
{
	for (size_t k = 0; k < nelem; ++k) 
    {
		for (size_t d = 0; d < dim; ++d) 
        {
			dst[d * nelem + k] = src[k * dim + d];
		}
	}
}

template <typename T>
static void transposeMatrix(T const * __restrict__ src, T * __restrict__ dst, 
    size_t w, size_t h, size_t srcStride, size_t dstStride) 
{
    for (size_t j = 0; j < h; ++j) 
    {
        for (size_t i = 0; i < w; ++i) 
        {
            dst[i * dstStride + j] = src[j * srcStride + i];
        }
    }
}

template <typename T>
static void transposeMatrix_omp(T const * __restrict__ src, T * __restrict__ dst,
     size_t w, size_t h, size_t srcStride, size_t dstStride) 
{
    #ifdef RD_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t j = 0; j < h; ++j) 
    {
        for (size_t i = 0; i < w; ++i) 
        {
            dst[i * dstStride + j] = src[j * srcStride + i];
        }
    }
}

template <typename T>
static void transposeTable_omp(T const * src, T *dst, size_t n_elem, size_t dim) 
{
    #ifdef RD_USE_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t k = 0; k < n_elem; ++k) 
    {
        for (size_t d = 0; d < dim; ++d) 
        {
            dst[d * n_elem + k] = src[k * dim + d];
        }
    }
}

/**
 * @brief Performs in-place transposition of data.
 * @param first
 * @param last
 * @param width - actual element count in row
 */
template <class RandomIterator>
void transposeInPlace(RandomIterator first, RandomIterator last, int width)
{
    const std::ptrdiff_t mn1 = (last - first - 1);
    const int n   = (last - first) / width;
    std::vector<bool> visited(last - first);
    RandomIterator cycle = first;
    while (++cycle != last) {
        if (visited[cycle - first])
            continue;
        std::ptrdiff_t a = cycle - first;
        do  {
            a = a == mn1 ? mn1 : (n * a) % mn1;
            std::swap(*(first + a), *cycle);
            visited[a] = true;
        } while ((first + a) != cycle);
    }
}

/////////////////////////////////////////////////////////////////////////////
//
// Funkcje sprawdzające poprawność wyników
//
////////////////////////////////////////////////////////////////////////////

template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type
    almostEqual(T x, T y, int ulp = 2)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::abs(x-y) < std::numeric_limits<T>::epsilon() * std::abs(x+y) * ulp
    // unless the result is subnormal
           || std::abs(x-y) < std::numeric_limits<T>::min();
}

template <typename T>
typename std::enable_if<std::is_integral<T>::value, bool>::type
    almostEqual(T x, T y)
{
    return x == y;
}

/**
 * @fn         checkResult
 *
 * @brief      funkcja porównująca poprawność wyników
 *
 * @param      CPU      - wskaźnik do wyników z CPU
 * @param      GPU      - wskaźnik do wyników z GPU
 * @param      size     - liczba elemntów w tablicy
 * @param[in]  verbose  - flaga sterująca wyświetlaniem różniacych się wyników
 * @param      nIter    - w przypadku kerneli gdzie wynik jest akumulowany należy go podzielć przez
 *                      liczbę wykonanych iteracji podczas testów
 *
 * @return     prawda jeśli maksymalna wartość błędu nie przekracza 1e-5, w p.p fałsz
 */
template <typename T>
static double checkResult(T const *CPU, T const *GPU, size_t size,
    bool verbose = false, T nIter = T(1))
{
	size_t k;
	double d, e = -1.0;
    size_t incorrect = 0;

    // remember current precision to restore it in the end
    int def_prec = std::cout.precision();
    int coutPrec, coutWidth = 10;
    if (std::is_floating_point<T>::value)
    {
        if (std::is_same<float, T>::value)
        {
            coutPrec = 6;
            coutWidth = 13;
        }
        else 
        {
            coutPrec = 11;
            coutWidth = 18;
        }
        std::cout.precision(coutPrec);
    }

    #ifdef RD_USE_OPENMP
    #pragma omp parallel for schedule(static) private(d) reduction(max:e) reduction(+:incorrect)
    #endif
	for (k = 0; k < size; k++) 
    {
        T const gpuVal = GPU[k] / nIter;
        if (!almostEqual(gpuVal, CPU[k]))
        {
            incorrect++;
            d = std::abs(gpuVal - CPU[k]);
            
            if (verbose)
            {
                #ifdef RD_USE_OPENMP
                #pragma omp critical
                {
                #endif
                std::cout << "cpu[" << std::setw(4) << k << "]:" 
                    << std::fixed << std::right << std::setw(coutWidth) << CPU[k] 
                    << ", gpu[" << std::setw(4) << k << "]: " 
                    << std::fixed << std::right << std::setw(coutWidth) << gpuVal 
                    << ",\tdiff: " << d << std::endl; 
                #ifdef RD_USE_OPENMP
                }
                #endif
            }
            if (d > e)
            {
                e = d;
            }
        }
	}

    std::cout.precision(def_prec);
    std::cout << "max. abs. err. = \t" << e << ", " 
        << (1.f - float(incorrect) / float(size)) * 100.f << "% correct results" << std::endl;

	return e;
}

/**
 * @fn checkValues
 * @param t - tablica z danymi do sprawdzenia
 * @param size - liczba elementów tablicy
 * @return - prawda jeśli dane mają "sensowne" wartości
 */
template <typename T>
static bool checkValues(T *t, size_t size) {

	T v;
	double max = 1.0e+200;
	double min = 1.0e-200;
	for (size_t i = 0; i < size; ++i) {
		v = t[i];
		if (v > 0) {
			if (v > max || v < min) {
				std::cout << "t[" << i << "]: " << t[i] << std::endl;
				return false;	// sth is wrong..
			}
		} else if (v < 0) {
			if (v < -max || v > -min) {
				std::cout << "t[" << i << "]: " << t[i] << std::endl;
				return false;	// sth is wrong..
			}
		}
	}
	return true;
}

#define checkNaN(val)           check_nan( (val), __FILE__, __LINE__ )

template <typename T>
void check_nan(T * val, const char *const file, int const line) {
	#if defined(RD_USE_OPENMP)
	#pragma omp critical
	{
	#endif
	if (std::isnan(*val)) {
		std::cout << "ERROR! value: " << *val << " is NaN!, file: " << file << ", line: " << line << std::endl;
		std::cout.flush();
		exit(1);
	}
	#if defined(RD_USE_OPENMP)
	}
	#endif
}

} // end namespace rd

#endif /* UTILITIES_HPP_ */
