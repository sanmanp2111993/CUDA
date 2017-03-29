/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <cuda_runtime.h>
#include <stdio.h>
#include <algorithm>

// includes, project
#include "reduction.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
bool runTest(int argc, char **argv);

#define MAX_BLOCK_DIM_SIZE 65535

#ifdef WIN32
#define strcasecmp strcmpi
#endif

extern "C"
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    bool bResult = false;

    bResult = runTest(argc, argv);

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
    printf(bResult ? "Test passed\n" : "Test failed!\n");
}

////////////////////////////////////////////////////////////////////////////////
//! Compute sum reduction on CPU
//! We use Kahan summation for an accurate sum of large arrays.
//! http://en.wikipedia.org/wiki/Kahan_summation_algorithm
//!
//! @param data       pointer to input data
//! @param size       number of input data elements
////////////////////////////////////////////////////////////////////////////////
int reduceCPU(int *data, int size)
{
    int sum = data[0];
    int c = (int)0.0;

    for (int i = 1; i < size; i++)
    {
        int y = data[i] - c;
        int t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    return sum;
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
////////////////////////////////////////////////////////////////////////////////
int benchmarkReduce(int  n,
                  int  numThreads,
                  int  numBlocks,
                  int  whichKernel,
                  int  testIterations,
                  int  cpuFinalThreshold,
                  int *h_odata,
                  int *d_idata,
                  int *d_odata)
{
    int gpu_result = 0;
    bool needReadBack = true;
    cudaError_t err = cudaSuccess;

    for (int i = 0; i < testIterations; ++i)
    {
        gpu_result = 0;

        cudaDeviceSynchronize();

        // execute the kernel
        reduce(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);

        // check if kernel execution generated an error
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{   
	    fprintf(stderr, "Failed to launch matrixMul kernel (error code %s)!\n", cudaGetErrorString(err));
	    exit(EXIT_FAILURE);
	}

        cudaDeviceSynchronize();
    }

    if (needReadBack)
    {
        // copy final sum from device to host
        cudaMemcpy(&gpu_result, d_odata, sizeof(int), cudaMemcpyDeviceToHost);
    }

    return gpu_result;
}

////////////////////////////////////////////////////////////////////////////////
// The main function which runs the reduction test.
////////////////////////////////////////////////////////////////////////////////
bool
runTest(int argc, char **argv)
{
    int size = 1<<10;       // number of elements to reduce
    int numThreads = size;  // number of threads per block
    int whichKernel = atoi(argv[1]);
    int numBlocks = 1;
    int cpuFinalThreshold = 1;

    printf("Kernel %d \n", whichKernel);
    printf("%d elements\n", size);
    printf("%d threads (max)\n", numThreads);

    // create random input data on CPU
    unsigned int bytes = size * sizeof(int);

    int *h_idata = (int *) malloc(bytes);

    for (int i=0; i<size; i++)
    {
        // Keep the numbers small so we don't get truncation error in the sum
        h_idata[i] = (int)(rand() & 0xFF);
    }

    if (numBlocks == 1)
    {
        cpuFinalThreshold = 1;
    }

    // allocate mem for the result on host side
    int *h_odata = (int *) malloc(numBlocks*sizeof(int));

    printf("%d blocks\n\n", numBlocks);

    // allocate device memory and data
    int *d_idata = NULL;
    int *d_odata = NULL;

    (cudaMalloc((void **) &d_idata, bytes));
    (cudaMalloc((void **) &d_odata, numBlocks*sizeof(int)));

    // copy data directly to device memory
    (cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    (cudaMemcpy(d_odata, h_idata, numBlocks*sizeof(int), cudaMemcpyHostToDevice));

    int gpu_result = 0;
    cudaError_t err = cudaSuccess;

    //gpu_result = benchmarkReduce(size, numThreads, numBlocks, 
    //                                 whichKernel, testIterations, 
     //                                cpuFinalThreshold, 
      //                               h_odata, d_idata, d_odata);


    cudaDeviceSynchronize();

    // execute the kernel
    reduce(size, numThreads, numBlocks, whichKernel, d_idata, d_odata);

    // check if kernel execution generated an error
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch matrixMul kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaDeviceSynchronize();

    // copy final sum from device to host
    cudaMemcpy(&gpu_result, d_odata, sizeof(int), cudaMemcpyDeviceToHost);

     // compute reference solution
     int cpu_result = reduceCPU(h_idata, size);

     printf("\nGPU result = %d\n", (int)gpu_result);
     printf("CPU result = %d\n\n", (int)cpu_result);

     // cleanup
     free(h_idata);
     free(h_odata);

     (cudaFree(d_idata));
     (cudaFree(d_odata));

     return (gpu_result == cpu_result);

}
