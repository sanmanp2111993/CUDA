#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>

void readImageDimension( FILE * fin, int * width, int * height )
{

    int fstatus;
    fstatus = fread( width, 4, 1, fin );
    assert( fstatus );
    fstatus = fread( height, 4, 1, fin );
    assert( fstatus );

    printf("\n Image WxH = %dx%d\n", *width, *height );
}

void readImagePixels( FILE * fin,  char pixels[], int width, int height )
{
    size_t count = 0;
    const size_t MAX_PIXEL_COUNT = width * height;


    for( ; !feof(fin) && count < MAX_PIXEL_COUNT; ++count ){
        int fstatus = fread( pixels+count*3, 1, 3, fin );
        if( fstatus != 3 ){
            fprintf( stderr, "Cannot read pixel. File could be corrupted!" );
            exit(1);
        }
    }
}

void writeImage( FILE * fout,  char pixels[], int width, int height)
{
    size_t count = 0;
    const size_t MAX_PIXEL_COUNT = width * height;

    int fstatus;

    fstatus = fwrite( &width, 4, 1, fout );
    fstatus = fwrite( &height, 4, 1, fout );

    for( ; count < MAX_PIXEL_COUNT; ++count ){
        fstatus = fwrite( pixels+count*3, 1, 3, fout );
        if( fstatus != 3 ){
            fprintf( stderr, "Cannot write pixel to file");
            exit(1);
        }
    }
}


__global__ 
void GrayFunc( unsigned char * pixels,  unsigned char * Grayscale, int width, int height)
{
     int R,G,B;
     int Gray = 0;
	
	int BlockOffset = ( blockDim.x * blockIdx.x  + threadIdx.x ) *3;		

  
		for(int i=0;i<width;i++)
		{	
			
		int index = BlockOffset + i* width*3 ;	 
		R = int(pixels[index]);
		G = int(pixels[index+1]);
		B = int(pixels[index+2]);
		
		Gray=(R+G+B)/3;
		Grayscale[index/3]=  char(Gray);

		}

 }




__global__ 
void SharedMem_col(unsigned char * Grayscale,unsigned char * Matrix, int width, int height)
{
  __shared__ unsigned char shared_Gray[260*32/2];
  __shared__ unsigned char shared_Mat[260*32/2];
  int blockOffset= blockIdx.x * blockDim.x * width;
 
 
    for(int i=0;i<blockDim.x;i++)
    {
 	for(int j=0;j<width/blockDim.x;j+=4)
	{
	  shared_Gray[threadIdx.x * 4 + 0 + j * blockDim.x+i*width+i*4] = Grayscale[blockOffset + threadIdx.x * 4 +0+j*blockDim.x+i*width];
	  shared_Gray[threadIdx.x * 4 + 1 + j * blockDim.x+i*width+i*4] = Grayscale[blockOffset + threadIdx.x * 4 +1+j*blockDim.x+i*width];
	  shared_Gray[threadIdx.x * 4 + 2 + j * blockDim.x+i*width+i*4] = Grayscale[blockOffset + threadIdx.x * 4 +2+j*blockDim.x+i*width];
	  shared_Gray[threadIdx.x * 4 + 3 + j * blockDim.x+i*width+i*4] = Grayscale[blockOffset + threadIdx.x * 4 +3+j*blockDim.x+i*width];
	   
	}
    }
	__syncthreads();

	int step=1;
	int threadOffset=threadIdx.x * width + threadIdx.x*4;
	int columnMax=width;

	while(columnMax>1)
	{
	int loopId=0;
		for(int columnId=0;columnId<width;columnId+=step*2)
		{
			shared_Mat[threadOffset+(columnMax/2)+loopId]=
			(shared_Gray[threadOffset+columnId] -  shared_Gray[threadOffset+columnId+step])/2;
 			shared_Gray[threadOffset + columnId]=(shared_Gray[threadOffset+columnId] + shared_Gray[threadOffset+columnId+step])/2;
			loopId++;
		}
		step*=2; columnMax = columnMax/2;
	}
		shared_Mat[threadOffset]=shared_Gray[threadOffset];		





    for(int i=0;i<blockDim.x;i++)
    {
        for(int j=0;j<width/blockDim.x;j+=4)
	{
	  Matrix[blockOffset + threadIdx.x * 4 +0+j*blockDim.x+i*width]= shared_Mat[threadIdx.x * 4 + 0 + j * blockDim.x+i*width+i*4];
	  Matrix[blockOffset + threadIdx.x * 4 +1+j*blockDim.x+i*width]= shared_Mat[threadIdx.x * 4 + 1 + j * blockDim.x+i*width+i*4];
	  Matrix[blockOffset + threadIdx.x * 4 +2+j*blockDim.x+i*width]= shared_Mat[threadIdx.x * 4 + 2 + j * blockDim.x+i*width+i*4];
	  Matrix[blockOffset + threadIdx.x * 4 +3+j*blockDim.x+i*width]= shared_Mat[threadIdx.x * 4 + 3 + j * blockDim.x+i*width+i*4];
	  
	   
	}
    }

}



__global__ 
void SharedMem_row(unsigned char * Grayscale,unsigned char * Matrix, int width, int height)
{
  __shared__ unsigned char shared_Gray[260*32/2];
  __shared__ unsigned char shared_Mat[260*32/2];

 
 
	int columnId = threadIdx.x + blockIdx.x * blockDim.x;
	int threadOffset = threadIdx.x * (height + 4);
	for(int i=0; i<height; i++)
	{
		shared_Gray[threadOffset + i] = Grayscale[columnId + i * width];
	}
	//
 	int step=1;
	int rowMax=height;

	while(rowMax>1)
	{
	int loopId=0;
		for(int rowId=0;rowId<height;rowId+=step*2)
		{
			shared_Mat[threadOffset+(rowMax/2)+loopId]=
			(shared_Gray[threadOffset+rowId] -  shared_Gray[threadOffset+rowId+step])/2;
 			shared_Gray[threadOffset + rowId]=(shared_Gray[threadOffset+rowId] + shared_Gray[threadOffset+rowId+step])/2;
			loopId++;
		}
		step*=2; rowMax = rowMax/2;
	}
		shared_Mat[threadOffset]=shared_Gray[threadOffset];		
	//
	for(int i=0; i<height; i++)
	{
		 Matrix[columnId + i * width]=shared_Mat[threadOffset + i];
	}

}


__global__ 
void SharedMem_rerow( unsigned char * Grayscale, unsigned char * Matrix, int width, int height)
{
  __shared__  unsigned char shared_Gray[260*32/2];
  __shared__  unsigned char shared_Mat[260*32/2];

	int columnId = threadIdx.x + blockIdx.x * blockDim.x;
	int threadOffset = threadIdx.x * (height + 4);
	for(int i=0; i<height; i++)
	{
		shared_Gray[threadOffset + i] = Grayscale[columnId + i * width];
	}
	//
	shared_Mat[threadOffset]=shared_Gray[threadOffset];
	int step=height/2;
	int no=1;
  	int rowId=1;

	while(rowId<height)
	{
	int target=0;
	for(int i=0;i<no;i++)
	{	
	 shared_Mat[threadOffset+target+step]=shared_Mat[threadOffset+target]-shared_Gray[threadOffset+rowId];
	 shared_Mat[threadOffset+target]+=shared_Gray[threadOffset+rowId];
	 target+=(step*2);
	 rowId++;
	}
	step=step/2;
	no=no *2;
	}
	//
	for(int i=0; i<height; i++)
	{	 
	
		Matrix[columnId + i * width]=shared_Mat[threadOffset + i];
	}
}


__global__ 
void SharedMem_recol( unsigned char * Grayscale, unsigned char * Matrix, int width, int height)
{
  __shared__  char shared_Gray[260*32/2];
  __shared__  char shared_Mat[260*32/2];
 
  int blockOffset= blockIdx.x * blockDim.x * width;
  int threadOffset=threadIdx.x * width + threadIdx.x*4;
 
    for(int i=0;i<blockDim.x;i++)
    {
 	for(int j=0;j<width/blockDim.x;j+=4)
	{
	  shared_Gray[threadIdx.x * 4 + 0 + j * blockDim.x+i*width+i*4] = Grayscale[blockOffset + threadIdx.x * 4 +0+j*blockDim.x+i*width];
	  shared_Gray[threadIdx.x * 4 + 1 + j * blockDim.x+i*width+i*4] = Grayscale[blockOffset + threadIdx.x * 4 +1+j*blockDim.x+i*width];
	  shared_Gray[threadIdx.x * 4 + 2 + j * blockDim.x+i*width+i*4] = Grayscale[blockOffset + threadIdx.x * 4 +2+j*blockDim.x+i*width];
	  shared_Gray[threadIdx.x * 4 + 3 + j * blockDim.x+i*width+i*4] = Grayscale[blockOffset + threadIdx.x * 4 +3+j*blockDim.x+i*width];
	   
	}
    }
	__syncthreads();
	//
	shared_Mat[threadOffset]=shared_Gray[threadOffset];
	int step=width/2;
	int no=1;
  	int columnId=1;

	while(columnId<width)
	{
	int target=0;
	for(int i=0;i<no;i++)
	{	
	 shared_Mat[threadOffset+target+step]=shared_Mat[threadOffset+target]-shared_Gray[threadOffset+columnId];
	 shared_Mat[threadOffset+target]+=shared_Gray[threadOffset+columnId];
	 target+=(step*2);
	 columnId++;
	}
	step=step/2;
	no=no *2;
	}
	//



    for(int i=0;i<blockDim.x;i++)
    {
        for(int j=0;j<width/blockDim.x;j+=4)
	{
	  Matrix[blockOffset + threadIdx.x * 4 +0+j*blockDim.x+i*width]= shared_Mat[threadIdx.x * 4 + 0 + j * blockDim.x+i*width+i*4];
	  Matrix[blockOffset + threadIdx.x * 4 +1+j*blockDim.x+i*width]= shared_Mat[threadIdx.x * 4 + 1 + j * blockDim.x+i*width+i*4];
	  Matrix[blockOffset + threadIdx.x * 4 +2+j*blockDim.x+i*width]= shared_Mat[threadIdx.x * 4 + 2 + j * blockDim.x+i*width+i*4];
	  Matrix[blockOffset + threadIdx.x * 4 +3+j*blockDim.x+i*width]= shared_Mat[threadIdx.x * 4 + 3 + j * blockDim.x+i*width+i*4];
	  
	   
	}
    }

}



int main(int argc, char** argv){
    switch(argc){
    case 4:
        break;
    default:
        fprintf(stderr, "Usage: <file-in> <file-out> <file-out1>\n");
        fprintf(stderr, "\n\n");
        return 1;
    }
    char * filein = argv[1];
    char * fileout = argv[2];
    char * fileout1 = argv[3];

     char * pixels = NULL;
    int width, height;

    //Read pixels from input file.
    FILE * fin = fopen( filein, "rb");
    readImageDimension(fin, &width, &height);
    pixels = ( char*)malloc(width*height*3);
    assert( pixels );
    readImagePixels(fin, pixels, width, height);
    fclose( fin );
    //------------------------------------------------
	
    unsigned char Grayscale[width*height];
    unsigned char Matrix[width*height];
    unsigned char * pixelsArray;
    unsigned char * GrayscaleArray;
    unsigned char * MatrixArray;   

    cudaMalloc(&pixelsArray,  3*width*height); 					// Memory Allocation for Device
    cudaMalloc(&GrayscaleArray, width*height);			        	// Memory Allocation for Device
    cudaMalloc(&MatrixArray,  width*height);


    cudaMemcpy(pixelsArray, pixels, 3*width*height, cudaMemcpyHostToDevice);	//memcopy from host to device
    
    dim3 gridDim(16);        
    dim3 blockDim(16);

    clock_t begin = clock();

    GrayFunc<<< gridDim,  blockDim >>>(pixelsArray,GrayscaleArray,width,height);

    clock_t end = clock();
    
    double time_spent = (double) (end - begin)/(CLOCKS_PER_SEC/1000);

    printf("\n Execution Time For Grayscale Conversion :      %1.3f ms",time_spent);
    
    cudaMemcpy(Grayscale,GrayscaleArray, width*height, cudaMemcpyDeviceToHost); 


  for(int y=0;y<width;y++)
    {
	for(int x=0;x<height;x++)
	{
		Matrix[(x+y*width)] = Grayscale[(x+y*width)];
	}

   }
 
    cudaMemcpy(GrayscaleArray, Grayscale, width*height, cudaMemcpyHostToDevice);


    begin = clock();

    SharedMem_col<<< gridDim, blockDim >>>(GrayscaleArray,MatrixArray,width,height);

    end = clock();

    time_spent = (double) (end - begin)/(CLOCKS_PER_SEC/1000);

    printf("\n Execution Time For Column Wise Compression :   %1.3f ms",time_spent);
 
   cudaMemcpy(Matrix,MatrixArray, width*height, cudaMemcpyDeviceToHost);

   for(int y=0;y<width;y++)
    {
	for(int x=0;x<height;x++)
	{
		Grayscale[(x+y*width)] = Matrix[(x+y*width)];
	}

   }

   cudaMemcpy(GrayscaleArray, Grayscale, width*height, cudaMemcpyHostToDevice);

   begin = clock();

   SharedMem_row<<< gridDim, blockDim >>>(GrayscaleArray,MatrixArray,width,height);

   end = clock();

   time_spent = (double) (end - begin)/(CLOCKS_PER_SEC/1000);

    printf("\n Execution Time For Row Wise Compression :      %1.3f ms",time_spent);
 
   cudaMemcpy(Matrix,MatrixArray, width*height, cudaMemcpyDeviceToHost);
   
   for(int y=0;y<width;y++)
    {
	for(int x=0;x<height;x++)
	{
		Grayscale[(x+y*width)] = Matrix[(x+y*width)];
	}

   }


    for(int y=0;y<width*height;y++)
    {

		pixels[3*y] = Grayscale[y];
		pixels[3*y+1] = Grayscale[y];
		pixels[3*y+2] = Grayscale[y];

    }


    //Write pixels to output file.
    FILE * fout = fopen(fileout1, "wb");
    writeImage( fout, pixels, width, height );
    fclose( fout );


   cudaMemcpy(GrayscaleArray, Grayscale, width*height, cudaMemcpyHostToDevice);

   begin = clock();

   SharedMem_rerow<<< gridDim, blockDim >>>(GrayscaleArray,MatrixArray,width,height);

   end = clock();

   time_spent = (double) (end - begin)/(CLOCKS_PER_SEC/1000);

   printf("\n Execution Time For Row Wise Decompression :    %1.3f ms",time_spent);
 
   cudaMemcpy(Matrix,MatrixArray, width*height, cudaMemcpyDeviceToHost);

   for(int y=0;y<width;y++)
    {
	for(int x=0;x<height;x++)
	{
		Grayscale[(x+y*width)] = Matrix[(x+y*width)];
	}

   }


  cudaMemcpy(GrayscaleArray, Grayscale, width*height, cudaMemcpyHostToDevice);

   begin = clock();

   SharedMem_recol<<< gridDim, blockDim >>>(GrayscaleArray,MatrixArray,width,height);

   end = clock();

   time_spent = (double) (end - begin)/(CLOCKS_PER_SEC/1000);

   printf("\n Execution Time For Column Wise Decompression : %1.3f ms\n\n",time_spent);
 
   cudaMemcpy(Matrix,MatrixArray, width*height, cudaMemcpyDeviceToHost);

   for(int y=0;y<width;y++)
    {
	for(int x=0;x<height;x++)
	{
		Grayscale[(x+y*width)] = Matrix[(x+y*width)];
	}

   }


  for(int y=0;y<width;y++)
    {
	for(int x=0;x<height;x++)
	{
		pixels[3*(x+y*width)] = Matrix[(x+y*width)];
		pixels[3*(x+y*width)+1] = Matrix[(x+y*width)];
		pixels[3*(x+y*width)+2] = Matrix[(x+y*width)];
	}

   }

    cudaFree(& GrayscaleArray);
    cudaFree(& pixelsArray);
    cudaFree(& MatrixArray);

    //Write pixels to output file.
    fout = fopen(fileout, "wb");
    writeImage( fout, pixels, width, height );
    fclose( fout );
    free( pixels );
}
