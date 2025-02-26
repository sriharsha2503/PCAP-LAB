#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel to reverse each word in parallel
__global__ void reverseWordsKernel(char *d_input, char *d_output, int *d_word_indices, int num_words, int input_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_words) {
        // Start and end index of the word to reverse
        int start_idx = d_word_indices[idx];
        int end_idx = (idx == num_words - 1) ? input_len : d_word_indices[idx + 1] - 1;

        // Reverse the word in the output string
        int j = 0;
        for (int i = end_idx; i >= start_idx; i--) {
            d_output[start_idx + j] = d_input[i];
            j++;
        }
    }
}

void reverseWordsInString(char *input_str, char *output_str) {
    int str_len = strlen(input_str);
    int num_words = 0;
    int *word_indices = (int *)malloc(sizeof(int) * (str_len / 2 + 1));  // To hold word start indices

    // Find start indices of each word and count number of words
    int word_start = 0;
    for (int i = 0; i <= str_len; i++) {
        if (input_str[i] == ' ' || input_str[i] == '\0') {
            word_indices[num_words++] = word_start;
            word_start = i + 1;
        }
    }

    // Allocate memory on the device
    char *d_input, *d_output;
    int *d_word_indices;
    cudaMalloc((void**)&d_input, str_len + 1);
    cudaMalloc((void**)&d_output, str_len + 1);
    cudaMalloc((void**)&d_word_indices, sizeof(int) * num_words);

    // Copy input string and word indices to the device
    cudaMemcpy(d_input, input_str, str_len + 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_word_indices, word_indices, sizeof(int) * num_words, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads_per_block = 256;
    int blocks_per_grid = (num_words + threads_per_block - 1) / threads_per_block;
    reverseWordsKernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, d_word_indices, num_words, str_len);

    // Copy the result back to host
    cudaMemcpy(output_str, d_output, str_len + 1, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_word_indices);
    free(word_indices);
}

int main() {
    // Input string with N words
    char input_str[] = "Hello world from CUDA";

    // Output string where the reversed words will be stored
    int str_len = strlen(input_str);
    char *output_str = (char *)malloc(str_len + 1);

    // Reverse the words in parallel using CUDA
    reverseWordsInString(input_str, output_str);

    // Output the original and reversed string
    printf("Original string: %s\n", input_str);
    printf("Reversed words string: %s\n", output_str);

    // Free the host memory for output string
    free(output_str);

    return 0;
}
