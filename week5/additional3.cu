#include <stdio.h>
#include <cuda_runtime.h>

#define N 16 // Size of the array

// CUDA kernel to perform Odd-Even Transposition Sort
__global__ void oddEvenSortKernel(int *arr, int n, bool isOddPhase) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread index

    if (idx < n) {
        // Odd phase: Compare and swap if idx is odd
        if (isOddPhase) {
            if (idx % 2 == 1 && idx < n - 1) {
                if (arr[idx] > arr[idx + 1]) {
                    // Swap
                    int temp = arr[idx];
                    arr[idx] = arr[idx + 1];
                    arr[idx + 1] = temp;
                }
            }
        }
        // Even phase: Compare and swap if idx is even
        else {
            if (idx % 2 == 0 && idx < n - 1) {
                if (arr[idx] > arr[idx + 1]) {
                    // Swap
                    int temp = arr[idx];
                    arr[idx] = arr[idx + 1];
                    arr[idx + 1] = temp;
                }
            }
        }
    }
}

// Function to print the array
void printArray(int *arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int main() {
    int h_arr[N] = {5, 3, 8, 6, 2, 7, 4, 1, 9, 10, 11, 13, 12, 14, 15, 16};
    int *d_arr;

    // Allocate memory on the device
    cudaMalloc((void**)&d_arr, N * sizeof(int));

    // Copy the array from host to device
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    // Number of blocks and threads
    int numBlocks = (N + 255) / 256;
    int numThreads = 256;

    // Perform Odd-Even Transposition Sort
    for (int phase = 0; phase < N; phase++) {
        bool isOddPhase = (phase % 2 == 1); // Odd phase or even phase
        
        // Launch kernel for each phase
        oddEvenSortKernel<<<numBlocks, numThreads>>>(d_arr, N, isOddPhase);

        // Wait for GPU to finish before moving to the next phase
        cudaDeviceSynchronize();
    }

    // Copy the sorted array back to host
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sorted array
    printf("Sorted Array:\n");
    printArray(h_arr, N);

    // Free the device memory
    cudaFree(d_arr);

    return 0;
}
//Sorted Array:
//1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 
