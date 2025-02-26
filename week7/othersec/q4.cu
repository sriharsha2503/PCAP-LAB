#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

// CUDA kernel to copy characters from Sin to Sout N times
__global__ void concatenateStringKernel(char *d_input, char *d_output, int input_len, int N, int output_len) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_len) {
        // Copy each character from input string Sin to N positions in output string Sout
        for (int i = 0; i < N; i++) {
            d_output[idx * N + i] = d_input[idx];
        }
    }
}

int main() {
    // Input string Sin and integer N
    const char *input_str = "Hello";
    int N = 3;  // Number of times to concatenate Sin

    int input_len = strlen(input_str);  // Length of input string
    int output_len = input_len * N;    // Length of output string

    // Allocate memory for input and output strings on the host
    char *h_input = (char *)malloc(input_len + 1); // +1 for null terminator
    char *h_output = (char *)malloc(output_len + 1); // +1 for null terminator

    // Copy input string to host array
    strcpy(h_input, input_str);

    // Allocate memory on the device
    char *d_input, *d_output;
    cudaMalloc((void**)&d_input, input_len + 1);
    cudaMalloc((void**)&d_output, output_len + 1);

    // Copy input string from host to device
    cudaMemcpy(d_input, h_input, input_len + 1, cudaMemcpyHostToDevice);

    // Launch the kernel
    int threads_per_block = 256;
    int blocks_per_grid = (input_len + threads_per_block - 1) / threads_per_block;
    concatenateStringKernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, input_len, N, output_len);

    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, output_len + 1, cudaMemcpyDeviceToHost);

    // Add null terminator to the output string
    h_output[output_len] = '\0';

    // Output the original and concatenated string
    printf("Input string: %s\n", input_str);
    printf("Output string: %s\n", h_output);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}
