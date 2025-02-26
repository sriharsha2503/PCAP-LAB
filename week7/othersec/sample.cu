#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define N 1024

__global__ void CUDACount(char* A, unsigned int *d_count, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < length) {
        if (A[i] == 'a') {
            atomicAdd(d_count, 1);
        }
    }
}

int main() {
    char A[N];
    char *d_A;
    unsigned int *d_count, count = 0, *result;

    // Take user input
    printf("Enter a string: ");
    fgets(A, N, stdin);  // Using fgets instead of gets for safety
    A[strcspn(A, "\n")] = 0;  // Remove newline character from fgets input

    int length = strlen(A);

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, length * sizeof(char));
    cudaMalloc((void**)&d_count, sizeof(unsigned int));
    cudaMallocHost((void**)&result, sizeof(unsigned int));

    // Initialize count to 0 on the host
    cudaMemcpy(d_count, &count, sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Copy input string to the device
    cudaMemcpy(d_A, A, length * sizeof(char), cudaMemcpyHostToDevice);

    // CUDA Event setup for measuring time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch kernel with enough threads to cover the entire string
    int threads_per_block = 256;
    int blocks_per_grid = (length + threads_per_block - 1) / threads_per_block;
    CUDACount<<<blocks_per_grid, threads_per_block>>>(d_A, d_count, length);

    // Error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
    }

    // Wait for kernel to finish and record stop time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Copy the result back to the host
    cudaMemcpy(result, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Measure elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Print result
    printf("Total occurrences of 'a': %u\n", *result);
    printf("Time Taken: %f ms\n", elapsedTime);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_count);
    cudaFreeHost(result);

    return 0;
}
