#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

/*
 * compute string value, length should be small than strlen
 */
int compute_value(char *input, int length, int d, int q)
{
	int i = 0;
	int patHash = 0;

	for (i = 0; i < length; ++i) {
		patHash = (d * patHash + (input[i] )) % q;
	}

	return patHash;

}

int rk_matcher(char *input, char *pattern, int d, int q)
{
	int i = 0, j = 0;
	int input_length = strlen(input);
	int pattern_length = strlen(pattern);
	int patHash = 0;
	int ts[input_length];

	
	//hash value of the pattern
	patHash = compute_value(pattern, pattern_length, d, q);

	//hash value of the first char
	ts[0] = compute_value(input, pattern_length, d, q);

	//p does not change, calculate once
	int p = pow(d, pattern_length - 1);
	for (i = 1; i < input_length - pattern_length + 1; i++) {
		ts[i] = ((input[i + pattern_length - 1]) * p
			 + (ts[i - 1] - (input[i - 1])) / d) % q;
		
	}



	for (i = 0; i <= input_length - pattern_length + 1; ++i) {
		if (ts[i] == patHash) {
			for (j = 0; j < pattern_length; ++j) {
				if (pattern[j] != input[i + j]) {
					break;
				} else if (j == pattern_length - 1) {
					printf("%d\n", i);
				}
			}
		}
	}

	return 0;

}

__global__ void findIfExistsCu(char *d_input, int d_len, int *d_result,int pattern_length, int d, int p)
{
	int i = 0;
	int ind = d_len * threadIdx.x;
	d_result += ind;
	d_input += ind;
	d_result[0] = 0;
        char pattern[] = "AB";

	int pw = 1, inph, path;
	for (path=i=0; i < pattern_length; i++)
	 {
		//d_result[0] += pw * (d_input[i]);
		//pw *= d;
                path = (path * 256 + pattern[i]) %997;
	}


	for (i = 1; i < d_len - pattern_length + 1; i++) {
		d_result[i] = ((d_input[i + pattern_length - 1]) * p
			    + (d_result[i - 1] - (d_input[i - 1])) / d); 
	}

}

__global__ void seekPattern(char *d_input, int d_len, int *d_result,
                int pattern_length, char* pattern, int d, int patHash) 
{
	int i = 0;
        int j=0;
	int ind = d_len * threadIdx.x;
	d_result += ind;
	d_input += ind;

	for (i = 0; i < d_len - pattern_length + 1; i++) {
		if (d_result[i] == patHash) {
			for (j = 0; j < pattern_length; j++) {
				if (pattern[j] != d_input[i + j]) {
					break;
				} else if (j == pattern_length - 1) {

			
					printf("pos:%d\n", threadIdx.x*(d_len-pattern_length+1)+i-pattern_length+1);
				}
			}
		}
	}

}
int main(int argc, char *argv[])
{
	int i = 0;
	int j = 0;
	char input[] = "HEABAL";
	char pattern[] = "AB";
	int d = 3;
	int num_cores = 8;
        int input_length = 6;
	int pattern_length = 2;
	int chunk_len = (int)ceil((float)input_length / num_cores);
	int padding_len = chunk_len * num_cores - input_length;
	int el_chunk_len = chunk_len + pattern_length - 1;

	//matrix on host which holds the characters, each row will go to a core
	char css[num_cores][el_chunk_len];
	int iss[num_cores][el_chunk_len];
	//on the device
	char *d_input;
        char *d_pattern;
	//hashes on the device
	int *d_result;
	int nchars = num_cores * el_chunk_len;
	

	cudaMalloc((char **)&d_input, nchars * sizeof(char));
	cudaMalloc((int **)&d_result, nchars * sizeof(int));
        cudaMalloc((char **)&d_pattern, pattern_length*sizeof(char));

	//initial zeroes
	for (i = 0; i < pattern_length - 1; i++)
		css[0][i] = 0;

	//first n-1 cores' characters
	for (i = 0; i < num_cores - 1; i++)
		for (j = 0; j < chunk_len; j++)
			css[i][j + pattern_length - 1] = input[i * chunk_len + j];

	//last core's characters
	for (i = (num_cores - 1) * chunk_len, j = 0; i < input_length; i++, j++)
		css[num_cores - 1][j + pattern_length - 1] = input[i];

	//last n-1 cores' padding characters
	for (i = 1; i < num_cores; i++)
		for (j = 0; j < pattern_length - 1; j++)
			css[i][j] = css[i - 1][j + chunk_len];

	//last core's last paddings
	for (i = 0; i < padding_len; i++)
		css[num_cores - 1][el_chunk_len - i - 1] = 0;

	//transfer css to device
	cudaMemcpy(d_input, css, nchars, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input, css, nchars, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pattern, pattern, pattern_length, cudaMemcpyHostToDevice);

	dim3 block(num_cores);	
	
	int p = pow(d, pattern_length - 1);
	findIfExistsCu <<< 1, num_cores >>> (d_input, el_chunk_len, d_result,
					 pattern_length, d, p);

        //find the hash of the pattern
        int pw = 1;
        int patHash=0;
        for (i=0; i < pattern_length; i++) 
        {
            patHash += pw * (pattern[i]);
            pw *= d;
        }
	
        
        seekPattern<<<1, num_cores>>>(d_input, el_chunk_len, d_result,
                pattern_length, d_pattern, d, patHash); 

	
	cudaFree(d_result);
	cudaFree(d_input);

	
	return 0;
}
