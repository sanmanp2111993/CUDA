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

void readImagePixels( FILE * fin, unsigned char pixels[], int width, int height )
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

void writeImage( FILE * fout, unsigned char pixels[], int width, int height)
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
void GrayFunc(unsigned char * pixels, unsigned char * Grayscale, int width, int height)
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
void ColtransF(unsigned char * Grayscale,unsigned char * Matrix, int width, int height)
{

   int columnId,elementId;
   int rowId= threadIdx.x + blockIdx.x * blockDim.x;
   int elementOffset = rowId * width;
   int columnMax = width;
  
while(columnMax>1)
	{ 
  	for(columnId=0;columnId<columnMax;columnId=columnId+2)//working on each row
	{
		elementId=rowId * width + columnId;
		Matrix[elementOffset+(columnId+columnMax)/2]= (Grayscale[elementId]-Grayscale[elementId+1])/2;
		Matrix[elementOffset + columnId/2] = (Grayscale[elementId] + Grayscale[elementId+1])/2;	
	}
		columnMax = columnMax /2;

	//copying back matrix into grayscale otherwise the above loop keeps working on the orig. img
	for(columnId=0;columnId<columnMax;columnId=columnId+1)
	{
		Grayscale[elementOffset+columnId]= Matrix[elementOffset+columnId];
	}
	
 }
		
}


__global__ 
void ColtransF_Recon(unsigned char * Grayscale,unsigned char * Matrix, int width, int height)
{
   int columnId;
   int rowId= threadIdx.x + blockIdx.x * blockDim.x;
   int elementOffset = rowId * width;
   int columnMax = 2;
  
while(columnMax<width + 1)
	{ 
  	for(columnId=0;columnId<columnMax;columnId=columnId+2)//working on each row
	{

		Matrix[elementOffset + columnId] = (Grayscale[(elementOffset+(columnId/2))] + Grayscale[(elementOffset+(columnId/2+columnMax/2))]);

		Matrix[elementOffset + (columnId+1)]= (Grayscale[(elementOffset+(columnId/2))] - Grayscale[(elementOffset+(columnId/2+columnMax/2))]);
	}

		

	//copying back matrix into grayscale otherwise the above loop keeps working on the orig. img
	for(columnId=0;columnId<columnMax;columnId=columnId+1)
	{
		Grayscale[elementOffset+columnId]= Matrix[elementOffset+columnId];
	}

	columnMax = columnMax  * 2;
	
 }
		
}
 

__global__ 
void RowtransF(unsigned char * Matrix,unsigned char * Matrix1, int width, int height)
{
 int rowId,elementId;
 int columnId=threadIdx.x + blockIdx.x * blockDim.x;
 int elementOffset = columnId;
 int rowMax=height;


while(rowMax>1)
{
 	for(rowId=0;rowId<rowMax;rowId=rowId+2)
	{
 		elementId=rowId * width + columnId;
		Matrix1[elementOffset+width*(rowId/2)+(rowMax/2)*width] = (Matrix[elementId]-Matrix[elementId + width])/2; 
		Matrix1[elementOffset+width*(rowId/2)] = (Matrix[elementId]+Matrix[elementId + width])/2;
	}
		
	//copying back matrix into grayscale otherwise the aboe loop keeps working on the orig. img
	for(rowId=0;rowId<rowMax;rowId++)
	{
		 Matrix[elementOffset+width*rowId]= Matrix1[elementOffset+width*rowId];
	}
	rowMax=rowMax/2;
}

}


__global__ 
void RowtransF_Recon(unsigned char * grayscale,unsigned char * Matrix, int width, int height)
{
 int rowId;
 int columnId=threadIdx.x + blockIdx.x * blockDim.x;
 int elementOffset = columnId;
 int rowMax=2;

 while(rowMax < height+1)
 {
 	for(rowId=0;rowId<rowMax;rowId=rowId+2)
	{
		Matrix[elementOffset+width*(rowId)] = (grayscale[(rowId/2) * width + columnId] + grayscale[((rowId/2) * width + columnId) + (width * (rowMax/2))]);

		Matrix[elementOffset+width*(rowId+1)] = (grayscale[(rowId/2) * width + columnId] - grayscale[((rowId/2) * width + columnId) + (width * (rowMax/2))]);
	}
	
	//copying back matrix into grayscale otherwise the aboe loop keeps working on the orig. img	

	for(rowId=0;rowId<rowMax;rowId++)
	{
		 grayscale[elementOffset+width*rowId]= Matrix[elementOffset+width*rowId];
	}
	rowMax=rowMax * 2;
	
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

    unsigned char * pixels = NULL;
    int width, height;

    //Read pixels from input file.
    FILE * fin = fopen( filein, "rb");
    readImageDimension(fin, &width, &height);
    pixels = (unsigned char*)malloc(width*height*3);
    assert( pixels );
    readImagePixels(fin, pixels, width, height);
    fclose( fin );
    //------------------------------------------------
	
    unsigned char Grayscale[width*height];
    unsigned char Matrix[width*height];
    unsigned char * pixelsArray;
    unsigned char * GrayscaleArray;
    unsigned char * MatrixArray;   
    unsigned char * MatrixArray1; 

    cudaMalloc(&pixelsArray,  3*width*height); 				// Memory Allocation for Device
    cudaMalloc(&GrayscaleArray, width*height);			        // Memory Allocation for Device
    cudaMalloc(&MatrixArray,  width*height);
    cudaMalloc(&MatrixArray1,  width*height);


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
		pixels[3*(x+y*width)] = Grayscale[(x+y*width)];
		pixels[3*(x+y*width)+1] = Grayscale[(x+y*width)];
		pixels[3*(x+y*width)+2] = Grayscale[(x+y*width)];
	}

   }

   cudaMemcpy(GrayscaleArray, Grayscale, width*height, cudaMemcpyHostToDevice);

   begin = clock();

   ColtransF<<< gridDim, blockDim >>>(GrayscaleArray,MatrixArray,width,height);

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
   
   RowtransF<<< gridDim, blockDim >>>(GrayscaleArray,MatrixArray,width,height);

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
   
   RowtransF_Recon<<< gridDim, blockDim >>>(GrayscaleArray,MatrixArray,width,height);

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
   
    for(int y=0;y<width;y++)
    {
	for(int x=0;x<height;x++)
	{
		pixels[3*(x+y*width)] = Matrix[(x+y*width)];
		pixels[3*(x+y*width)+1] = Matrix[(x+y*width)];
		pixels[3*(x+y*width)+2] = Matrix[(x+y*width)];
	}

   }




   cudaMemcpy(GrayscaleArray, Grayscale, width*height, cudaMemcpyHostToDevice);

   begin = clock();
   
   ColtransF_Recon<<< gridDim, blockDim >>>(MatrixArray,MatrixArray1,width,height);

   end = clock();

   time_spent = (double) (end - begin)/(CLOCKS_PER_SEC/1000);

   printf("\n Execution Time For Column Wise Decompression : %1.3f ms\n\n",time_spent);

   cudaMemcpy(Matrix,MatrixArray, width*height, cudaMemcpyDeviceToHost);
   
    for(int y=0;y<width;y++)
    {
	for(int x=0;x<height;x++)
	{
		pixels[3*(x+y*width)] = Matrix[(x+y*width)];
		pixels[3*(x+y*width)+1] = Matrix[(x+y*width)];
		pixels[3*(x+y*width)+2] = Matrix[(x+y*width)];
	}

   }

////////


    cudaFree(& GrayscaleArray);
    cudaFree(& pixelsArray);
    cudaFree(& MatrixArray);
    cudaFree(& MatrixArray1);

    //Write pixels to output file.
    fout = fopen(fileout, "wb");
    writeImage( fout, pixels, width, height );
    fclose( fout );
    free( pixels );
}
