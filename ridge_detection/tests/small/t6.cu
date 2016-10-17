#include <iostream>
#include <assert.h>

__host__ __device__ __forceinline__ cudaError_t Debug(
    cudaError_t     error,
    const char*     filename,
    int             line)
{
    if (error)
    {
    #if (__CUDA_ARCH__ == 0)
        fprintf(stderr, "CUDA error %d [%s, %d]: %s\n",
            error, filename, line, cudaGetErrorString(error));
        fflush(stderr);
    #elif (__CUDA_ARCH__ >= 200)
        printf("CUDA error %d [block (%3d,%3d,%3d) thread (%3d,%3d,%3d), %s, %d]\n",
            error, blockIdx.z, blockIdx.y, blockIdx.x,
            threadIdx.z, threadIdx.y, threadIdx.x, filename, line);
    #endif
    }
    return error;
}

#ifndef devCheckCall
    #define devCheckCall(e) if ( Debug((e), __FILE__, __LINE__) ) { assert(0); }
#endif

#ifndef checkCudaErrors
    #define checkCudaErrors(e) if ( Debug((e), __FILE__, __LINE__) ) { cudaDeviceReset(); exit(1); }
#endif

//----------------------------------------------------------
//
//----------------------------------------------------------


__global__ void testKernel1(char *  in1, char * in2, char * in3, int size, int value)
{
    if (threadIdx.x == 0)
    {
        cudaStream_t s1;
        devCheckCall(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
        devCheckCall(cudaMemsetAsync(in1, value, size * sizeof(char), s1))
        devCheckCall(cudaStreamDestroy(s1));
    }

    if (threadIdx.x == 1)
    {
        cudaStream_t s2;
        devCheckCall(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
        devCheckCall(cudaMemsetAsync(in2, value, size * sizeof(char), s2))
        devCheckCall(cudaStreamDestroy(s2));
    }

    if (threadIdx.x == 2)
    {
        cudaStream_t s3;
        devCheckCall(cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking));
        devCheckCall(cudaMemsetAsync(in3, value, size * sizeof(char), s3))
        devCheckCall(cudaStreamDestroy(s3));
    }
    __syncthreads();
    devCheckCall(cudaDeviceSynchronize());
    
}


//----------------------------------------------------------
//
//----------------------------------------------------------


static constexpr int POINTS_NUM     = 1 << 16;

//----------------------------------------------------------
//
//----------------------------------------------------------


int main(void)
{

    int value = 1;

    char *h_data = new char[POINTS_NUM];
    char *d_out1 = new char[POINTS_NUM];
    char *d_out2 = new char[POINTS_NUM];
    char *d_out3 = new char[POINTS_NUM];
    char *d_in1, *d_in2, *d_in3;

    checkCudaErrors(cudaSetDevice(0));

    checkCudaErrors(cudaMalloc(&d_in1, POINTS_NUM * sizeof(char)));
    checkCudaErrors(cudaMalloc(&d_in2, POINTS_NUM * sizeof(char)));
    checkCudaErrors(cudaMalloc(&d_in3, POINTS_NUM * sizeof(char)));

    for (int k = 0; k < POINTS_NUM; ++k)
    {
        // h_data[k] = k * 0.2f;
        h_data[k] = char(k & 127);
    }

    std::cout << "char(200): " << char(200) << std::endl;
    std::cout << "sizeof(char): " << sizeof(char) << std::endl;

    checkCudaErrors(cudaMemcpy(d_in1, h_data, POINTS_NUM * sizeof(char),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_in2, h_data, POINTS_NUM * sizeof(char),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_in3, h_data, POINTS_NUM * sizeof(char),
        cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());

    testKernel1<<<1, 32>>>(d_in1, d_in2, d_in3, POINTS_NUM, value);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(d_out1, d_in1, POINTS_NUM * sizeof(char),
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(d_out2, d_in2, POINTS_NUM * sizeof(char),
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(d_out3, d_in3, POINTS_NUM * sizeof(char),
        cudaMemcpyDeviceToHost));

    for (int k = 0; k < POINTS_NUM; ++k)
    {
        if (int(d_out1[k]) != value ||
            int(d_out2[k]) != value ||
            int(d_out3[k]) != value)
        {
            std::cout << "--- ERROR!---- d_out1["<<k<<"]: (" << int(d_out1[k])
                << "), d_out2["<<k<<"]: (" << int(d_out2[k])
                << "), d_out3["<<k<<"]: (" << int(d_out3[k]) << ")" << std::endl;
            break;
        }
    }

    checkCudaErrors(cudaDeviceReset());

    return 0;
}