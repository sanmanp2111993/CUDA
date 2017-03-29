// Matrix Multiplication in CUDA

#include <stdio.h>
//#include <string.h>
//#include <assert.h>
//#include <stdlib.h>
#include <cuda_runtime.h>


// includes, project
////////////////////////////////////////////////////////////////////////////////
// declarations, forward

#define WIDTH 32 
extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

// FILL HERE: define constant variable


// MatrixMul kernel
/**
 * CUDA Kernel Device code
 *
 * Computes the matrix multiplication of A and B into C. The 3 matrices have the same
 * number of elements WIDTH*WIDTH.
 */
// FILL HERE: translate C-version matrixMul to CUDA-version kernel code
void 
MatrixMul(float* A, float* B, float* C)
{

    // TODO : Kernel Function
    //        C = A * B
    // --> 
    for(int i = 0; i < WIDTH; i++)
	{
		for(int j = 0; j < WIDTH; j++)
		{
			for(int k = 0; k < WIDTH; k++)
    		{
        		C[i*WIDTH + j] += A[i*WIDTH+k] * B[k*WIDTH + j];
    		}
		}
	}
    // <--    

}

/**
 * Host main routine
 */
int 
main(void) 
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the matrix size to be used, and compute its size
    int size = WIDTH*WIDTH*sizeof(float);
    printf("[MatrixMul of %d x %d elements]\n", WIDTH, WIDTH);

    // Allocate the host input matrix h_A 
    float  *h_A = (float *)malloc(size);

    // Allocate the host input matrix h_B 
    float  *h_B = (float *)malloc(size);

    // Allocate the host input matrix h_C 
    float  *h_C = (float *)malloc(size);

    // Allocate the host matrix for compute check 
    float  *reference = (float *)malloc(size);
    
    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL || reference == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrices!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input matrices
    for (int i = 0; i < WIDTH; ++i)
    {
        for (int j = 0; j < WIDTH; ++j)
        {
            h_A[i*WIDTH + j] = 0.01f;
            h_B[i*WIDTH + j] = 1.0f;
        }
    }
    memset(h_C, 0, size);
    memset(reference, 0, size);

    // compute the matrix multiplication on the CPU for comparison
    computeGold(reference, h_A, h_B, WIDTH, WIDTH, WIDTH);

	// Allocate device input matrices 
	// TODO : Leave/Remove the given cudaMalloc code properly
	// --> 
	float* d_A = NULL;	
    err = cudaMalloc((void**)&d_A, size); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	float* d_B = NULL;	
    err = cudaMalloc((void**)&d_B, size); 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	// <--

	// Allocate the device output matrix
	float* d_C = NULL;
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device matrix C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input matrix A and B in host memory to the device input matrices in
    // device memory
	// TODO : Add proper mem copy APIs according to the memory that matrix A and B will be stored
	// -->
    printf("Copy input data from the host memory to the CUDA device\n");
    err = ;// FILL HERE
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = ;// FILL HERE
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	// <--

	// TODO : Clock Measurements
	//		  Add code to return clock cycles from kernel
	// -->
#ifdef TM
    unsigned long long* d_runtime;
    int r_size = WIDTH*WIDTH*sizeof(unsigned long long);
    unsigned long long* runtime = (unsigned long long*)malloc(r_size);
    memset(runtime, 0, r_size);
    cudaMalloc((void**)&d_runtime, r_size);
#endif
	// <--

    // TODO : Kernel Invocation 
    //        Assign as many threads as the size of matrix in a thread block and
    //        invoke the kernel function.
    // --> 
    int blocksPerGrid = ;// FILL HERE
    int threadsPerBlock = ;// FILL HERE
	printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
	MatrixMul(d_A, d_B, d_C);
    // <--                                                           

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch matrixMul kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	cudaThreadSynchronize();
	
    // Copy the device result matrix in device memory to the host result matrix
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy matrix C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaThreadSynchronize();

    // Verify that the result matrix is correct
    bool res = 1;
	for (int i = 0; i < WIDTH*WIDTH; i++)
	{
		float diff = fabs(reference[i] - h_C[i]);
		if(diff > 0.001f)
		{
			res = 0;
			break;
		}
	}
	printf("Test %s\n", (res == 1) ? "PASSED" : "FAILED");

	// TODO : Get elapsed clock cycles from device to host
	//		  Take the longest time as kernel execution time
	// -->
#ifdef TM
    cudaMemcpy(runtime, d_runtime, r_size, cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    
    unsigned long long elapsed_time = 0;
    for(int i = 0; i < WIDTH*WIDTH; i++)
        if(elapsed_time < runtime[i])
            elapsed_time = runtime[i];
    printf("Kernel Execution Time: %llu cycles\n", elapsed_time);
#endif
	// <--

	// TODO : Free device global memory
	// 		  Leave/Remove the given cudaFree statement according to your data allocation
	// -->
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
#ifdef TM
    cudaFree(d_runtime);
#endif
	// <--

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);
	free(reference);
#ifdef TM
	free(runtime);
#endif

	return 0;
}

void
computeGold(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            double sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = (float)sum;
        }

}
