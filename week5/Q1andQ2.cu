#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#define THREADS_PER_BLOCK 256

__global__ void addVectors(int* A, int* B, int* C, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void addVectors_BlockSizeAsN(int* A, int* B,int* C,int N)
{ int idx=threadIdx.x;
  if(idx<N)
  {
   C[idx]=A[idx]+B[idx];
  }
}  

__global__ void addVectors_NThreads(int* A,int* B,int* C,int N)
{
  int idx=blockIdx.x;
  if(idx<N)
  {
   C[idx]=A[idx]+B[idx];
  }
}

int main()
{
 int N=1024;
 int *A,*B,*C;
 int *d_A,*d_B,*d_C;
 
 A=(int*)malloc(N*sizeof(int));
 B=(int*)malloc(N*sizeof(int));
 C=(int*)malloc(N*sizeof(int));
 
 for(int i=0;i<N;i++)
 {
  A[i]=i+1;
  B[i]=(i+1)*2;
 }
 
 cudaMalloc((void**)&d_A,N*sizeof(int));
 cudaMalloc((void**)&d_B,N*sizeof(int));
 cudaMalloc((void**)&d_C,N*sizeof(int));
 
 cudaMemcpy(d_A,A,N*sizeof(int),cudaMemcpyHostToDevice);
 cudaMemcpy(d_B,B,N*sizeof(int),cudaMemcpyHostToDevice);
 addVectors_BlockSizeAsN<<<1,N>>>(d_A,d_B,d_C,N);//1st kernel call
 cudaDeviceSynchronize();
 cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
 printf("Result (Block size N): ");
 for (int i = 0; i < N; i++)
 {
   printf("%d ", C[i]);
 }
 printf("\n");
    
 addVectors_NThreads<<<N,1>>>(d_A,d_B,d_C,N);//2nd kernel call
 cudaDeviceSynchronize();
 cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
 printf("Result (N Threads): ");
 for (int i = 0; i < N; i++)
 {
   printf("%d ", C[i]);
 }
 printf("\n");
 

  dim3 dimBlock(256, 1, 1);  
  dim3 dimGrid(ceil(N / 256.0), 1, 1); 
  addVectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);//3rd kernel call 
  cudaDeviceSynchronize();
  cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
  printf("Result for normal vector addition: ");
  for (int i = 0; i < N; i++) 
  {
    printf("%d ", C[i]);
  }
  printf("\n");


 free(A);
 free(B);
 free(C);
 cudaFree(d_A);
 cudaFree(d_B);
 cudaFree(d_C);
 return 0;
}
