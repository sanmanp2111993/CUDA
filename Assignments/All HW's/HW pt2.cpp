// CMPE297-6 HW2
// CUDA version Rabin-Karp

#include<stdio.h>
#include<iostream>
#include <cuda_runtime.h>
#include <string.h>

__device__ int compare_pattern(const char *input_compare, const char *pattern_compare, unsigned int comp_pat_len)
{
	unsigned int i;
	const unsigned char *  k = (const unsigned char*)input_compare;
	const unsigned char *  l = (const unsigned char*)pattern_compare;
	for(i = 0;i < comp_pat_len; i++, k++, l++)
	{
		if(*k < *l)
		{
			return -1;
		}
		else if(*k > *l)
		{
			return 1;
		}
	}
	return 0;
}

__global__ void findIfExistsCu(char* input_string, int input_len, char* input_pattern, int pattern_len, int hash_pattern, bool* result, unsigned long long* runtime)
{ 
	unsigned long long start_time = clock64();
	int hash_var,i;
	unsigned int x = blockDim.x*blockIdx.x+threadIdx.x;
	while(x<=(input_len-pattern_len))
	{
		for(i=hash_var=0; i<pattern_len; ++i)
	{
			hash_var = ((hash_var*256)+(input_string[x]%997));
		if(hash_var == hash_pattern && compare_pattern(input_string+x-pattern_len+1,input_pattern,pattern_len)==0)
		{
			result[x-pattern_len+1]=1; 	
		}
	x++;
	}	
}	
	unsigned long long stop_time = clock64();
	runtime[threadIdx.x] = (unsigned long long)(stop_time - start_time);
}

int main()
{
	// host variables

	char input_string[] = "HEABAL"; 	/*Sample Input*/
	char input_pattern[] = "AB"; 		/*Sample Pattern*/
	int hash_pattern = 0; 		/*hash for the input_pattern*/
	bool* result;			            /*Result array*/
	unsigned long long* runtime; 	   /*Exection cycles*/
	int pattern_len = 2;		        /*Pattern Length*/
	int input_len = 6; 		            /*Input Length*/


	// device variables
	char* d_input;
	char* d_pattern;
	bool* d_result;
	unsigned long long* d_runtime;

	// measure the execution time by using clock() api in the kernel as we did in Lab3
	int runtime_size = 32*32*sizeof(unsigned long long);/*FILL CODE HERE*/
	result = (bool*)malloc((input_len-pattern_len+1)*sizeof(bool));
	runtime = (unsigned long long *) malloc(runtime_size);
	memset(runtime, 0, runtime_size);               
	cudaMalloc((void**)&d_runtime, runtime_size);   
	/*Calculate the hash of the input_pattern*/
	for (int i = 0; i < pattern_len; i++)
	{
		hash_pattern = ((hash_pattern*256)+(input_pattern[i]%997));
	}

	/*ADD CODE HERE: Allocate memory on the GPU and copy or set the appropriate values from the HOST*/
	cudaMalloc((void**)&d_input, input_len*sizeof(char));
	cudaMalloc((void**)&d_pattern, pattern_len*sizeof(char));
	cudaMalloc((void**)&d_result, (input_len-pattern_len+1)*sizeof(bool));
	//Copy to GPU memory
	cudaMemcpy(d_input,input_string,input_len*sizeof(char),cudaMemcpyHostToDevice);
	cudaMemcpy(d_pattern,input_pattern,pattern_len*sizeof(char),cudaMemcpyHostToDevice);
	cudaMemset(d_result,false,(input_len-pattern_len+1)*sizeof(bool));  

	/*ADD CODE HERE: Launch the kernel and pass the arguments*/
	int threadsPerBlock = 1;
	int blocksPerGrid = (input_len-pattern_len+1);
	findIfExistsCu<<<blocksPerGrid,threadsPerBlock>>>(d_input,input_len,d_pattern,pattern_len,hash_pattern,d_result,d_runtime);
	
	/*ADD CODE HERE: Copy the execution times from the GPU memory to HOST Code*/		
	cudaMemcpy(runtime,d_runtime,runtime_size,cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();	
	/*RUN TIME calculation*/
	unsigned long long time_req = 0;
	for(int i = 0; i < input_len-pattern_len; i++)
	if(time_req < runtime[i])
	    time_req = runtime[i];
	printf("Total cycles: %llu \n", time_req);
	printf("Searching for a single input_pattern in a single string \n");
	printf("Input String = %s\n",input_string);
	printf("Pattern = %s\n",input_pattern);
	/*ADD CODE HERE: COPY the result and print the result as in the HW description*/
	cudaMemcpy(result,d_result,(input_len-pattern_len+1)*sizeof(bool),cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	int n;
	for(n=0;n<=input_len-pattern_len;n++)
	printf("Pos: %d Result: %d\n",n,result[n]);
	cudaFree(d_input);
	cudaFree(d_pattern);
	cudaFree(d_result);
	cudaFree(d_runtime);
	
return 0;
}

