#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

// Kernel to populate output string T
__global__ void generateStringKernel(char *d_input, char *d_output, int input_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_len) {
        // Start position in output string T
        int start_idx = (idx * (idx + 1)) / 2;  // Sum of first idx elements

        // Write the character from input string to output string T
        for (int i = 0; i <= idx; i++) {
            d_output[start_idx + i] = d_input[idx];
        }
    }
}

void generateString(char *input_str, char *output_str) {
    int str_len = strlen(input_str);
    int output_len = (str_len * (str_len + 1)) / 2;  // Total length of the output string T

    // Allocate memory for input and output strings on the device
    char *d_input, *d_output;
    cudaMalloc((void**)&d_input, str_len + 1);
    cudaMalloc((void**)&d_output, output_len + 1);

    // Copy input string to the device
    cudaMemcpy(d_input, input_str, str_len + 1, cudaMemcpyHostToDevice);

    // Launch kernel to generate string T
    int threads_per_block = 256;
    int blocks_per_grid = (str_len + threads_per_block - 1) / threads_per_block;
    generateStringKernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, str_len);

    // Copy the result back to host
    cudaMemcpy(output_str, d_output, output_len + 1, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    // Input string Sin
    char input_str[] = "Hai";

    // Output string T where the generated string will be stored
    int str_len = strlen(input_str);
    int output_len = (str_len * (str_len + 1)) / 2;  // Length of output string T
    char *output_str = (char *)malloc(output_len + 1);

    // Generate the output string using CUDA
    generateString(input_str, output_str);

    // Output the original and generated string
    printf("Input string: %s\n", input_str);
    printf("Generated string: %s\n", output_str);

    // Free the host memory for output string
    free(output_str);

    return 0;
}
