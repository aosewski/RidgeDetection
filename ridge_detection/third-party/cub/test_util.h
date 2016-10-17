/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/


/************************************************************************
 *
 * @author Adam Rogowiec
 * Edited to use only specific features.
 *
 ***********************************************************************/

#pragma once

#include <cuda_runtime.h>

#include <string>
#include <math.h>
#include <float.h>
#include <limits>
 
#include "util_debug.cuh"
#include "util_device.cuh"
#include "util_macro.cuh"


/******************************************************************************
 * Assertion macros
 ******************************************************************************/

/**
 * Assert equals
 */
#define assertEquals(a, b) if ((a) != (b)) { std::cerr << "\n(" << __FILE__ << ": " << __LINE__ << ")\n"; exit(1);}

/******************************************************************************
 * Device initialization
 ******************************************************************************/

/**
 * Initialize device
 */
cudaError_t deviceInit(int dev = 0)
{
    cudaError_t     error = cudaSuccess;
    cudaDeviceProp  deviceProp;
    float           device_giga_bandwidth = 0;

    do
    {
        int deviceCount;
        error = CubDebug(cudaGetDeviceCount(&deviceCount));
        if (error) break;

        if (deviceCount == 0) {
            fprintf(stderr, "No devices supporting CUDA.\n");
            exit(1);
        }
        if ((dev > deviceCount - 1))
        {
            dev = 0;
        }

        error = CubDebug(cudaSetDevice(dev));
        if (error) break;

        size_t free_physmem, total_physmem;
        CubDebugExit(cudaMemGetInfo(&free_physmem, &total_physmem));

        int ptx_version;
        error = CubDebug(cub::PtxVersion(ptx_version));
        if (error) break;

        error = CubDebug(cudaGetDeviceProperties(&deviceProp, dev));
        if (error) break;

        if (deviceProp.major < 1) {
            fprintf(stderr, "Device does not support CUDA.\n");
            exit(1);
        }

        device_giga_bandwidth = float(deviceProp.memoryBusWidth) * deviceProp.memoryClockRate * 2 / 8 / 1000 / 1000;

        printf(
                "Using device %d: %s (PTX version %d, SM%d, %d SMs, "
                "%lld free / %lld total MB physmem, "
                "%.3f GB/s / %d kHz mem clock, ECC %s)\n",
                dev,
                deviceProp.name,
                ptx_version,
                deviceProp.major * 100 + deviceProp.minor * 10,
                deviceProp.multiProcessorCount,
                (unsigned long long) free_physmem / 1024 / 1024,
                (unsigned long long) total_physmem / 1024 / 1024,
                device_giga_bandwidth,
                deviceProp.memoryClockRate,
                (deviceProp.ECCEnabled) ? "on" : "off");
        fflush(stdout);

    } while (0);

    return error;
}

cudaError_t deviceReset() {
    return CubDebug(cudaDeviceReset());
}

/******************************************************************************
 * Timing
 ******************************************************************************/

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        CubDebugExit(cudaEventCreate(&start));
        CubDebugExit(cudaEventCreate(&stop));
    }

    ~GpuTimer()
    {
        CubDebugExit(cudaEventDestroy(start));
        CubDebugExit(cudaEventDestroy(stop));
    }

    void Start()
    {
        CubDebugExit(cudaEventRecord(start, 0));
    }

    void Stop()
    {
        CubDebugExit(cudaEventRecord(stop, 0));
    }

    float ElapsedMillis()
    {
        float elapsed;
        CubDebugExit(cudaEventSynchronize(stop));
        CubDebugExit(cudaEventElapsedTime(&elapsed, start, stop));
        return elapsed;
    }
};
