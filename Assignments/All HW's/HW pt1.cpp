// CMPE297-6 HW2
// CUDA version Rabin-Karp

#include<stdio.h>
#include<iostream>
#include<string.h>

/*ADD CODE HERE: Implement the parallel version of the sequential Rabin-Karp*/
__global__ void findIfExistsCu(char* input_string, int input_length, char * input_pattern_1, char * input_pattern_2, char * input_pattern_3, int pattern_len_1, int pattern_len_2, int pattern_len_3, int hash_pattern_1, int hash_pattern_2, int hash_pattern_3, int * runtime)
{
	unsigned long long start_time = clock64();

	int tid = threadIdx.x;
	int y = 0;
	int check=0;

	for(int k=tid; k<pattern_len_1 + tid;k++)
	{
	  y = (y * 256 + input_string[k]) % 997;
	}

	 if(y==hash_pattern_1)
	   {
	     for(int i=0;i<pattern_len_1;i++)
	        {
		   if(input_string[i+tid] == input_pattern_1[i])
		     {
			check++;
			}
		}
	       if(check == pattern_len_1)
		{
			printf("\n Pattern: \"%s\" was found.",input_pattern_1);
			check =0;
		}
	}

	y = 0;
	for(int k=tid; k<pattern_len_2 + tid;k++)
	{
	  y = (y * 256 + input_string[k]) % 997;
	}
	if(y==hash_pattern_2)
	  {
	   for(int i=0;i<pattern_len_2;i++)
	      {
		if(input_string[i+tid] == input_pattern_2[i])
		{
		  check++;
		 }
	      }
		if(check == pattern_len_2)
		  {
		    printf("\n Pattern: \"%s\" was found.",input_pattern_2);
	            check =0;
	           }
	    }

	  y = 0;
        for(int k=tid; k<pattern_len_3 + tid;k++)
	   {
	     y = (y * 256 + input_string[k]) % 997;
            }
           if(y==hash_pattern_3)
	     {
	       for(int i=0;i<pattern_len_3;i++)
		   {
		      if(input_string[i+tid] == input_pattern_3[i])
			 {
			   check++;
			  }
	            }
		if(check == pattern_len_3)
		  {
		    printf("\n Pattern: \"%s\" was found.\n",input_pattern_3);
		    check =0;
		   }
		}

	unsigned long long stop_time = clock64();
	runtime[tid] = (unsigned long long)(stop_time - start_time);			
			
}

int main()
{
	// host variables
	char input_string[] = "wqrtcybeellomjinktroqw frgeedfmoitloacde fghalabaeh";
	const char * input_pattern[3]={"eello", "oqw frge", "acde fgha"};/*Sample Pattern*/

	int hash_pattern_1=0; 	/*hash for the pattern*/
	int hash_pattern_2=0; 	/*hash for the pattern*/
	int hash_pattern_3=0; 	/*hash for the pattern*/
	int* runtime; 	         /*Exection cycles*/
	int pattern_len_1;	/*Pattern Length*/
	int pattern_len_2;	/*Pattern Length*/
	int pattern_len_3;	/*Pattern Length*/
	int input_length = strlen(input_string); 	/*Input Length*/

	// device variables
	char* d_input_string;
	int * d_runtime;
	char* d_input_pattern_1;
	char* d_input_pattern_2;
	char* d_input_pattern_3;


	// measure the execution time by using clock() api in the kernel as we did in Lab3

	pattern_len_1 = strlen(input_pattern[0]);
	pattern_len_2 = strlen(input_pattern[1]);
	pattern_len_3 = strlen(input_pattern[2]);

	int Input_Size_In_Bytes = sizeof(char) * input_length;
	int runtime_size = sizeof(int) * (input_length - pattern_len_3);
	int Pattern_Size_In_Bytes_1;
	int Pattern_Size_In_Bytes_2;
	int Pattern_Size_In_Bytes_3;

	runtime = (int *) malloc(runtime_size);

	Pattern_Size_In_Bytes_1 = sizeof(char) * pattern_len_1;
	Pattern_Size_In_Bytes_2 = sizeof(char) * pattern_len_2;
	Pattern_Size_In_Bytes_3 = sizeof(char) * pattern_len_3;

	/*Calculate the hash of the pattern*/

	for (int i = 0; i < pattern_len_1; i++)
  	{
    		hash_pattern_1 = (hash_pattern_1 * 256 + input_pattern[0][i]) % 997;
  	}

	for (int i = 0; i < pattern_len_2; i++)
	{
		hash_pattern_2 = (hash_pattern_2 * 256 + input_pattern[1][i]) % 997;
	}

	for (int i = 0; i < pattern_len_3; i++)
	{
		hash_pattern_3 = (hash_pattern_3 * 256 + input_pattern[2][i]) % 997;
	}


	/*ADD CODE HERE: Allocate memory on the GPU and copy or set the appropriate values from the HOST*/

	cudaMalloc(&d_runtime, runtime_size);
	cudaMalloc(&d_input_string, Input_Size_In_Bytes);
	cudaMalloc(&d_input_pattern_1, Pattern_Size_In_Bytes_1);
	cudaMalloc(&d_input_pattern_2, Pattern_Size_In_Bytes_2);
	cudaMalloc(&d_input_pattern_3, Pattern_Size_In_Bytes_3);

	cudaMemcpy(d_runtime, runtime, runtime_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input_string, input_string, Input_Size_In_Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input_pattern_1, input_pattern[0], Pattern_Size_In_Bytes_1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input_pattern_2, input_pattern[1], Pattern_Size_In_Bytes_2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input_pattern_3, input_pattern[2], Pattern_Size_In_Bytes_3, cudaMemcpyHostToDevice);

	printf("\nSample Output for Part2:");
	printf("\nSearching for multiple patterns in the input sequence.");

	/*ADD CODE HERE: Launch the kernel and pass the arguments*/
	dim3 gridDim(1,1,1);
	dim3 blockDim((input_length - pattern_len_3 + 1),1,1);

	findIfExistsCu<<<gridDim, blockDim>>>(d_input_string, input_length, d_input_pattern_1, d_input_pattern_2, d_input_pattern_3, pattern_len_1, pattern_len_2, pattern_len_3, hash_pattern_1, hash_pattern_2, hash_pattern_3, d_runtime);


	cudaMemcpy(runtime, d_runtime, runtime_size, cudaMemcpyDeviceToHost);

	/*RUN TIME calculation*/
    unsigned long long elapsed_time = 0;
    for(int i = 0; i < input_length - pattern_len_3 + 1; i++)
        if(elapsed_time < runtime[i])
            elapsed_time = runtime[i];


	printf("\n");

	/*ADD CODE HERE: COPY the result and print the result as in the HW description*/

	cudaFree(d_input_string);
	cudaFree(d_runtime);
	cudaFree(d_input_pattern_1);
	cudaFree(d_input_pattern_2);
	cudaFree(d_input_pattern_3);

	return 0;
}
