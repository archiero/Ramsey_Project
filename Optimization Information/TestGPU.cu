/* To compile: nvcc TestGPU.cu -o temp -lcudart -run
This is a code to see what kind architure can be set up on the GPU. I want to be as quick and as easy on the memory allocation and transmission as possible so I am passing an array of 0's as unsigned chars, turning each 0 into a 1 and then adding it back up.
If all of the threads are activated, then the sum of A == N.
*/
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#define threadsperblock 256 
//WARNING N/10 MUST BE AN INT FOR MEMORY COPYING PURPOSES
#define N (unsigned long int)(200000000)
unsigned char *A_CPU;// *B_CPU, *C_CPU; 

unsigned char *A_GPU;// *B_GPU, *C_GPU; 

dim3 dimBlock; 
dim3 dimGrid;

void AllocateMemory()
{	
	cudaMalloc(&A_GPU,N*sizeof(unsigned char));
	//cudaMalloc(&B_GPU,N*sizeof(unsigned char));
	//cudaMalloc(&C_GPU,N*sizeof(unsigned char));

	A_CPU = (unsigned char*)malloc(N*sizeof(unsigned char));
	//B_CPU = (unsigned char*)malloc(N*sizeof(unsigned char));
	//C_CPU = (unsigned char*)malloc(N*sizeof(unsigned char));
}

void Innitialize(int blocks)
{
	//dimBlock.x = 1024;
	//int blocks = (N/2+dimBlock.x -1)/dimBlock.x;
	dimGrid.x = (int)pow(blocks,1.0/3.0) + 1;//(int)blocks; 
	dimGrid.y =(int)pow(blocks,1.0/3.0) + 1; //(int)1;//
	dimGrid.z = (int)pow(blocks,1.0/3.0) + 1;//(int)1;//
	int i;
	
	for(i = 0; i < N; i++)
	{		
		A_CPU[i] = (unsigned char)0;	
		//B_CPU[i] = (unsigned char)1;
		//C_CPU[i] = (unsigned char)0;
	}
}

unsigned long int Additup(unsigned char *A_CPU)
{
	unsigned long int temp = 0;
	for(int i =0; i<N; i++)
	{
		temp += A_CPU[i];
	}
	return(temp);
}

void CleanUp(unsigned char *A_CPU,unsigned char*A_GPU)//,unsigned char *C_CPU,unsigned char*C_GPU,unsigned char *B_CPU,unsigned char *B_GPU)  //free
{
	free(A_CPU);// free(B_CPU); free(C_CPU);
	cudaFree(A_GPU);// cudaFree(B_GPU); cudaFree(C_GPU);
}

__global__ void Addition(unsigned char *A)//, unsigned char *B, unsigned char *C)
{
	//Here I have define Addition to mean Dot Product
	//This fills up the x dimension first, then the y Dimension then the z dimension. It was a quick way to see how threads I could get and probably isn't the best way of doing it.
	//Also, recall that I am only putting threads along the x dimension to make the id simpler
	unsigned int id = threadIdx.x + blockDim.x*blockIdx.x + blockDim.x*gridDim.x*blockIdx.y + blockDim.x*gridDim.x*blockDim.y*gridDim.y*blockIdx.z;
	if(id < N)
	{
		A[id] = (unsigned char)1;
		//A[id+N/2] = A[id + N/2]*2;
	}

}

int main()
{
	unsigned long int total;
	int i;
	timeval start, end;
	cudaError_t err;
	
	AllocateMemory();

	dimBlock.x = threadsperblock;
	int blocks = (N+dimBlock.x-1)/dimBlock.x;
	Innitialize(blocks);
	
	gettimeofday(&start, NULL);
	//There is size limit on how much memory that can be copied over at time. This code helps prevent that from being an issue
	/*int reps = 2;
	for(i =0; i<2; i++)
	//{
		cudaMemcpyAsync(A_GPU + i*N/reps, A_CPU + i*N/reps, (N/reps)*sizeof(unsigned char), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(B_GPU + i*N/reps, B_CPU + i*N/reps, (N/reps)*sizeof(unsigned char), cudaMemcpyHostToDevice);
	}*/
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(unsigned char), cudaMemcpyHostToDevice);
	//cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(unsigned char), cudaMemcpyHostToDevice);	
	Addition<<<dimGrid,dimBlock>>>(A_GPU);//, B_GPU, C_GPU);
	/*for(i =0; i<2; i++)
	//{
		cudaMemcpyAsync(C_CPU + i*N/reps, C_GPU + i*N/reps, (N/reps)*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	}*/
	cudaMemcpyAsync(A_CPU, A_GPU, N*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	total = Additup(A_CPU);

	gettimeofday(&end, NULL);

	float time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
	
	printf("Time in milliseconds= %.15f\n", (time/1000.0));	
	
	for(i = 0; i < N; i++)		
	{		
		//clearprintf("C[%d] = %d", i, C_CPU[i]);
	}


	printf("Blocks total: %d\nBlocks in x dim: %d\nBlocks in y dim: %d\nBlocks in z dim: %d\nAnswer is: %li\nShould be: %li\n", dimGrid.x*dimGrid.y*dimGrid.z ,dimGrid.x, dimGrid.y, dimGrid.z,total, N);
	
	CleanUp(A_CPU,A_CPU);//,C_CPU,C_GPU,B_CPU,B_GPU);	
	
	return(0);
}
