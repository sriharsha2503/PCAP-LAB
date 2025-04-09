#include <stdio.h>
#include <cuda.h>
#include <cstdlib>

#define N 16           // Smaller input array size for easy verification
#define KERNEL_SIZE 5   // Smaller kernel size
#define BLOCK_SIZE 16

// Constant memory for the kernel
__constant__ float d_kernel[KERNEL_SIZE];

__global__ void convolution1D(const float *d_input, float *d_output, int data_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int radius = kernel_size / 2;

    if (idx < data_size) {
        float sum = 0.0f;

        // Apply convolution using constant memory
        for (int i = -radius; i <= radius; i++) {
            int neighbor_index = idx + i;

            if (neighbor_index >= 0 && neighbor_index < data_size) {
                sum += d_input[neighbor_index] * d_kernel[i + radius];
            }
        }
        d_output[idx] = sum;
    }
}

void inputArray(float *array, int size, const char *name) {
    printf("Enter %d elements for %s:\n", size, name);
    for (int i = 0; i < size; i++) {
        printf("%s[%d]: ", name, i);
        scanf("%f", &array[i]);
    }
}


void printArray(const float *array, int size) {
    for (int i = 0; i < size; i++) {
        printf("%0.2f ", array[i]);
        if ((i + 1) % 8 == 0) printf("\n");  // Print 8 elements per line for readability
    }
    printf("\n");
}

int main() {
    size_t bytes_data = N * sizeof(float);
    size_t bytes_kernel = KERNEL_SIZE * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(bytes_data);
    float *h_output = (float *)malloc(bytes_data);
    float h_kernel[KERNEL_SIZE];

    // Take input from user
    inputArray(h_input, N, "Input Array");
    inputArray(h_kernel, KERNEL_SIZE, "Kernel");

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, bytes_data);
    cudaMalloc((void **)&d_output, bytes_data);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, bytes_data, cudaMemcpyHostToDevice);

    // Copy kernel to constant memory
    cudaMemcpyToSymbol(d_kernel, h_kernel, bytes_kernel);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    convolution1D<<<gridDim, blockDim>>>(d_input, d_output, N, KERNEL_SIZE);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, bytes_data, cudaMemcpyDeviceToHost);

    // Print results
    printf("\nInput Array:\n");
    printArray(h_input, N);

    printf("\nKernel:\n");
    printArray(h_kernel, KERNEL_SIZE);

    printf("\nOutput Array:\n");
    printArray(h_output, N);

    // Free device and host memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
/*
Enter 16 elements for Input Array:
Input Array[0]: 2
Input Array[1]: 3
Input Array[2]: 4
Input Array[3]: 5
Input Array[4]: 6
Input Array[5]: 3
Input Array[6]: 
2
Input Array[7]: 1
Input Array[8]: 3
Input Array[9]: 4
Input Array[10]: 65
Input Array[11]: 6
Input Array[12]: 4
Input Array[13]: 3
Input Array[14]: 5
Input Array[15]: 2
Enter 5 elements for Kernel:
Kernel[0]: 5
Kernel[1]: 3
Kernel[2]: 2
Kernel[3]: 1
Kernel[4]: 6

Input Array:
2.00 3.00 4.00 5.00 6.00 3.00 2.00 1.00 
3.00 4.00 65.00 6.00 4.00 3.00 5.00 2.00 


Kernel:
5.00 3.00 2.00 1.00 6.00 

Output Array:
31.00 46.00 68.00 61.00 62.00 57.00 62.00 50.00 
413.00 123.00 187.00 249.00 384.00 65.00 41.00 34.00 
**/
