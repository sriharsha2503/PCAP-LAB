#include <stdio.h>
#include <cuda_runtime.h>
 
// CUDA kernel to compute one's complement
__global__ void ones_complement_kernel(int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (idx < n) {
        // Compute one's complement: invert all the bits
        output[idx] = ~input[idx];
    }
}
 
void computeOnesComplement(int *h_input, int *h_output, int n) {
    int *d_input, *d_output;
 
    // Allocate memory on the device
    cudaMalloc((void**)&d_input, n * sizeof(int));
    cudaMalloc((void**)&d_output, n * sizeof(int));
 
    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);
 
    // Define the block size and number of blocks
    int blockSize = 256;  // Number of threads per block
    int numBlocks = (n + blockSize - 1) / blockSize;
 
    // Launch the kernel to compute one's complement
    ones_complement_kernel<<<numBlocks, blockSize>>>(d_input, d_output, n);
 
    // Synchronize to ensure kernel execution is completed
    cudaDeviceSynchronize();
 
    // Copy result from device to host
    cudaMemcpy(h_output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
 
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}
 
int main() {
    int h_input[] = {0b1101, 0b1010, 0b0111, 0b1000, 0b1111}; // Example binary numbers
    int n = sizeof(h_input) / sizeof(h_input[0]);    // Number of elements
    int h_output[n];
 
    printf("Original binary numbers: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_input[i]);
    }
    printf("\n");
 
    // Compute one's complement using CUDA
    computeOnesComplement(h_input, h_output, n);
 
    printf("One's complement: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
 
    return 0;
}