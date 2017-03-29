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

/*
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <stdio.h>


/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n threads
    - only works for power-of-2 arrays
*/

// Reduction Kernels
/**
 * CUDA Kernel Device code 0
 *
 * Run reduction to add all the elements in g_idata. 
 * Assume that g_idata has 1024 elements.
 * Store the final result of each thread block to g_odata[blockIdx.x]
 */
// FILL HERE: Implement the naive reduction code that uses 1024 threads
//            Even numbered threads will compute addition
//            until all pairs of elements are summed up
// 	      Refer to lecture slide "A Simple Reduction Kernel"
__global__ void reduce0(int *g_idata, int *g_odata)
{
	unsigned int tid = threadIdx.x;
	unsigned int start = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ int partialSum[1024];
	partialSum[tid] = g_idata[start];

	__syncthreads();

	for (unsigned int stride = 1; stride < blockDim.x;  stride *= 2)
   		 {	
			if (tid % (stride*2) == 0)
			partialSum[tid]+= partialSum[tid + stride];
			__syncthreads();
		}
	if( tid == 0 )
	g_odata[blockIdx.x] = partialSum[tid];

}

/**
 * CUDA Kernel Device code 1
 *
 * Run reduction to add all the elements in g_idata. 
 * Assume that g_idata has 1024 elements.
 * Store the final result of each thread block to g_odata[blockIdx.x]
 */
// FILL HERE: Revise the naive kernel to use consecutive threads
//	      This helps reducing warp divergence
//            Refer to lecture slide "A Better Reduction Kernel"
__global__ void reduce1(int *g_idata, int *g_odata)
{
        unsigned int tid = threadIdx.x;
        unsigned int start = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int partialSum[1024];
        partialSum[tid] = g_idata[start];

        __syncthreads();

        for (unsigned int stride = blockDim.x/2;stride > 0 ;stride /= 2)
         { 
                        if (tid < stride)
                        	partialSum[tid]+= partialSum[tid+stride];
                        __syncthreads();
                }
        if( tid == 0 )
        g_odata[blockIdx.x] = partialSum[tid];


}

/**
 * CUDA Kernel Device code 2
 *
 * Run reduction to add all the elements in g_idata. 
 * Assume that g_idata has 1024 elements.
 * Store the final result of each thread block to g_odata[blockIdx.x]
 */
// FILL HERE: Revise the reduction kernel to do loop unrolling in the last six iterations
//            You can reduce performance overhead of __syncthreads and branch operations of loop iteration
//            Refer to lecture slide "Unroll The Last Warp"
__global__ void
reduce2(int *g_idata, int *g_odata)
{
        unsigned int tid = threadIdx.x;
        unsigned int start = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int partialSum[1024];
        partialSum[tid] = g_idata[start]; 

        __syncthreads();

        for (unsigned int stride = blockDim.x/2;stride > 32 ;stride /= 2)
                 {
                        if (tid < stride)
                                partialSum[tid]+= partialSum[tid+stride];
                        __syncthreads();
                }
        
        if( tid < 32 )
		partialSum[tid] += partialSum[tid + 32];
	if( tid < 16 )
		partialSum[tid] += partialSum[tid + 16];
	if( tid < 8 )
		partialSum[tid] += partialSum[tid + 8];
	if( tid < 4 )
		partialSum[tid] += partialSum[tid + 4];
	if( tid < 2 )
		partialSum[tid] += partialSum[tid + 2];
	if( tid < 1 )
		partialSum[tid] += partialSum[tid + 1];

if( tid == 0 )
        g_odata[blockIdx.x] = partialSum[tid];




}


/**
 * CUDA Kernel Device code 3
 *
 * Run reduction to add all the elements in g_idata. 
 * Assume that g_idata has 1024 elements.
 * Store the final result of each thread block to g_odata[blockIdx.x]
 */
// FILL HERE: Revise the reduction kernel to do complete loop unrolling and use warp shuffling
//	      The function should use template to be compiled for a specific block size
//            Refer to lecture slide "Reduction with Warp Shuffles"
template <unsigned int blockSize>
__global__ void
reduce3(int *g_idata, int *g_odata)
{

	unsigned int tid = threadIdx.x;
        unsigned int start = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ int partialSum[1024];

        partialSum[tid] = g_idata[start];

        __syncthreads();


	if(blockSize >= 1024 && tid < 512) 
		partialSum[tid] += partialSum[tid+512]; __syncthreads();
	if(blockSize >= 512 && tid < 256) 
		partialSum[tid] += partialSum[tid+256]; __syncthreads();
	if(blockSize >= 256 && tid < 128) 
		partialSum[tid] += partialSum[tid+128]; __syncthreads();
	if(blockSize >= 128 && tid < 64) 
		partialSum[tid] += partialSum[tid+64]; __syncthreads();
	
	int mySum = partialSum[tid];
	
	if( tid < 32 ) 
	{
		if(blockSize >= 64) mySum += partialSum[tid+32]; 

		for (int offset = warpSize/2; offset > 0; offset /= 2) 
		{ 
			mySum += __shfl_down (mySum, offset); 
		}
}

      
if( tid == 0 )
{
	partialSum[tid]=mySum;       
	g_odata[blockIdx.x] = partialSum[tid];
}

}

extern "C"
bool isPow2(unsigned int x);


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
// This function calls corresponding kernel function
////////////////////////////////////////////////////////////////////////////////
void
reduce(int size, int threads, int blocks,
       int whichKernel, int *d_idata, int *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    int smemSize = (threads <= 32) ? 2 * threads * sizeof(int) : threads * sizeof(int);
    // choose which of the optimized versions of reduction to launch
    switch (whichKernel)
    {
        case 0:
            reduce0<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata);
            break;
        case 1:
            reduce1<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata);
            break;
        case 2:
            reduce2<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata);
            break;
        case 3:
            switch (threads)
            {
                case 1024:
                    reduce3<1024><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata);
                    break;

                case 512:
                    reduce3<512><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
                    break;

                case 256:
                    reduce3<256><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
                    break;

                case 128:
                    reduce3<128><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
                    break;

                case 64:
                    reduce3< 64><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
                    break;

                case 32:
                    reduce3< 32><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
                    break;

                case 16:
                    reduce3< 16><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
                    break;

                case  8:
                    reduce3<  8><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
                    break;

                case  4:
                    reduce3<  4><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
                    break;

                case  2:
                    reduce3<  2><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
                    break;

                case  1:
                    reduce3<  1><<< dimGrid, dimBlock, smemSize>>>(d_idata, d_odata);
                    break;
            }
            break;
    }
}

void
reduce(int size, int threads, int blocks,
            int whichKernel, int *d_idata, int *d_odata);


#endif // #ifndef _REDUCE_KERNEL_H_
