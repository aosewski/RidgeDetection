/**
 * @file choose2.cuh
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is conducted under the supervision of prof. dr hab. inż. Marka
 * Nałęcza.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */

#ifndef CHOOSE2_CUH_
#define CHOOSE2_CUH_

#include "rd/gpu/thread/thread_sqr_euclidean_dist.cuh"
#include "rd/gpu/warp/warp_functions.cuh"
#ifdef DEBUG
#include "cub/util_debug.cuh"
#endif

#include <device_launch_parameters.h>
#include <device_functions.h>


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300

/*
 *  TODO: sprawdzić czy warto zrobić prefetching danych i przetwarzanie
 *      potokowe.
 */


/**
 * @brief      Choose initial subset of points from the set @p P.
 *
 * @note       This version assumes that P has column major order and S has row
 *             major order data layout.
 *
 * @paragraph The function chooses subset of @p P set of points, where each two
 * of them are R-separated. This means that thera are no two different points
 * closer than R.
 *
 * @param[in]  P     Set of points forming a cloud.
 * @param[out] S     Initial chosen subset of @p P.
 * @param[in]  np    Number of points in the set @p P
 * @param[in]  r     Sphere radius. Min distance between two points.
 * @param[out] ns    Number of chosen samples. Pointer to device (global)
 *                   memory.
 */
template <
    typename    T,
    int         VAL_PER_THREAD,
    int         TILE,
    int         BLOCK_SIZE,
    int         DIM>
static __global__
void __choose_kernel_v3(T const * __restrict__ P, 
                        volatile T *S,
                        int np,
                        T r,
                        int *ns)
{

    // 1. każdy wątek w bloku czyta do rejestrów po 1 punkcie (wybraniec) początkowym
    // 
    // |-------------------|-------------------|-------------------|     |-------------------|
    // | pkt "startowe" b1 | pkt "startowe" b2 | pkt "startowe" b3 | ... | pkt "startowe" bn |
    // |-------------------|-------------------|-------------------|     |-------------------|
    // 

    T inSamples[DIM * VAL_PER_THREAD];

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    T rSqr = r * r;

    #pragma unroll
    for (int d = 0; d < DIM; ++d)
    {
        // reading in col-major order, writing in row-major order
        inSamples[d] = P[d * np + tidx];
    }

    if ((threadIdx.x & 0x1f) == 0)
    {
        (void)atomicAdd(ns, warpSize);
    }

    #pragma unroll
    for (int d = 0; d < DIM; ++d)
    {
        S[tidx * DIM + d] = inSamples[d];
    }
    tidx += gridDim.x * blockDim.x;
    
    // Ensure that all writes to S, are visible after this instruction.
    __threadfence();

    // 3. Czytamy do pamięci współdzielonej kolejny kafelek punktów.
    // 
    // I kafelkujemy przez zbiór "wybrańców" --> każdy wątek sprawdza K swoich punktów 
    // wczytanych do rejstrów, czy nie kolidują ze zbiorem "wybrańców".
    // Zapis do GMEM tak samo jak w wstępnej fazie algorytmu.
    // 
    // |-------------------|-------------------|-------------------|     |-------------------|
    // | pkt "startowe" bn | pkt "testowe" b1  | pkt "testowe" b2  | ... | pkt "testowe" bn  | ...
    // |-------------------|-------------------|-------------------|     |-------------------|

    // round up to the closest block size multiply
    int nnp = (np + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

    __shared__ T shmem[TILE * DIM];

    while (tidx < nnp)
    {
        // load test points to registers
        #pragma unroll
        for (int k = 0; k < VAL_PER_THREAD; ++k)
        {
            #pragma unroll
            for (int d = 0; d < DIM; ++d)
            {
                // reading in col-major order, writing in row-major order
                if (tidx < np)
                    inSamples[k * DIM + d] = P[d * np + tidx];
            }
            tidx += gridDim.x * blockDim.x;
        }

        int ns_ = *ns;
        int m = (ns_ * DIM + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        int chosenSamplesMask = (1 << VAL_PER_THREAD) - 1;

        // iterate through chosen samples set, buffering it to shmem
        for (int i = threadIdx.x; i < m;)
        {
            // load one tile of chosen samples to shmem
            for (int j = threadIdx.x; j < TILE * DIM; j += BLOCK_SIZE, i += BLOCK_SIZE)
            {
                // FIXME: próbki są porównywane również z zerami na końcu shmem!!!
                shmem[j] = (i < ns_ * DIM) ? S[i] : .0;
            }
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < TILE; ++k)
            {
                #pragma unroll
                for (int l = 0; l < VAL_PER_THREAD; ++l)
                {
                    if (rd::gpu::threadSqrEuclideanDistanceRowMajor(inSamples + l * DIM,
                         shmem + k * DIM, DIM) <= rSqr)
                        chosenSamplesMask &= ~(1 << l);
                }
                
            }
        }

        // Zapisywanie do GMEM: mam globalny indeks pod, którym zapisuję pkty;
        // Każdy wątek jak wczytuje w rejestry nowe punkty to sprawdza czy po pierwsze nie 
        // koliduje on z poprzednimi w rejstrach, a jeśli nie to zwiększa licznik (prywatny)
        // wybrańców. 
        
        // Na koniec, po wczytaniu przez każdy punkt K punktów nastepuje zapis do GMEM:
        // Zapis następuje turowo: tzn najpierw dla każdego warpu zliczamy ile mamy elementów do 
        // zapisania (__ballot()) i o tyle zwiększamy atomowo indeks tablicy wybrańców. Operacja 
        // atomicAdd zwraca nam starą wartość, będąca naszym offsetem. Następnie każdy wątek wylicza
        // sobie swój indeks na podstawie zwróconej maski przez __ballot().
        // 

        #pragma unroll
        for (int k = 0; k < VAL_PER_THREAD; ++k)
        {
            int count = chosenSamplesMask & (1 << k);
            int warpMask = __ballot(count);
            // count how many threads in warp have chose a sample
            count = __popc(warpMask);
            int chosenOffset = 0;
            if ((threadIdx.x & 0x1f) == 0)
            {
                 chosenOffset = atomicAdd(ns, count);
            }
            // broadcast value 
            rd::gpu::broadcast(chosenOffset, 0);

            // count ones before my position
            // warpMask &= (1 << threadIdx.x) - 1;
            warpMask &= (1 << rd::gpu::laneId()) - 1;
            int idx = __popc(warpMask);
        
            if (chosenSamplesMask & (1 << k))
            {
                #pragma unroll
                for (int d = 0; d < DIM; ++d)
                    S[chosenOffset * DIM + idx * DIM + d] = inSamples[k * DIM + d];
            }
        }

        // order writes / reads to S.
        __threadfence();
    }

}

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300

#endif /* CHOOSE2_CUH_ */
