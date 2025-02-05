#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>  // for ceil()

#define THREADS_PER_BLOCK 256

__global__ void addVectors(int* A, int* B, int* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1024; 
    int *A, *B, *C; 
    int *d_A, *d_B, *d_C; 

    // Allocate memory on host
    A = (int*)malloc(N * sizeof(int));
    B = (int*)malloc(N * sizeof(int));
    C = (int*)malloc(N * sizeof(int));

    // Initialize input vectors
    for (int i = 0; i < N; i++) {
        A[i] = i + 1; 
        B[i] = (i + 1) * 2; 
    }

    // Allocate memory on device
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Calculate grid and block dimensions using dim3
    dim3 dimBlock(256, 1, 1);  // Set block size to 256 threads along x-axis
    dim3 dimGrid(ceil(N / 256.0), 1, 1);  // Set grid size using ceil for number of blocks

    // Launch kernel
    addVectors<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Synchronize device
    cudaDeviceSynchronize();

    // Copy result back from device to host
    cudaMemcpy(C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    printf("Result: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", C[i]);
    }
    printf("\n");

    // Free memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

