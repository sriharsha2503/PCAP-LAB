#include <stdio.h>
#include <cuda_runtime.h>

// Kernel to convert integers to their corresponding octal values
__global__ void int_to_octal_kernel(int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int num = input[idx];
        int octal = 0, place = 1;

        // Convert integer to octal
        while (num > 0) {
            octal += (num % 8) * place;
            num /= 8;
            place *= 10;
        }

        output[idx] = octal;
    }
}

int main() {
    int n;

    // Take number of integers as input
    printf("Enter the number of integers: ");
    scanf("%d", &n);

    int *h_input = (int *)malloc(n * sizeof(int));    // Host input array
    int *h_output = (int *)malloc(n * sizeof(int));   // Host output array

    // Take input integers
    printf("Enter the integers: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &h_input[i]);
    }

    int *d_input, *d_output;

    // Allocate memory on the device
    cudaMalloc((void**)&d_input, n * sizeof(int));
    cudaMalloc((void**)&d_output, n * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;  // Number of threads per block
    int numBlocks = (n + blockSize - 1) / blockSize;  // Number of blocks

    // Launch the kernel to convert integers to octal in parallel
    int_to_octal_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back from device to host
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the original integers and their corresponding octal values
    printf("\nOriginal Integers and their Corresponding Octal Values:\n");
    for (int i = 0; i < n; i++) {
        printf("%d in decimal = %d in octal\n", h_input[i], h_output[i]);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
