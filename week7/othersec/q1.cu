#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string.h>

#define N 1024  // Max length of the input sentence

// Kernel to count occurrences of the word in the sentence
__global__ void countWordKernel(const char *sentence, const char *word, unsigned int *d_count, int sentence_len, int word_len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if index is within bounds of the sentence
    if (idx <= sentence_len - word_len) {
        bool match = true;
        
        // Check if the substring starting at idx matches the word
        for (int i = 0; i < word_len; ++i) {
            if (sentence[idx + i] != word[i]) {
                match = false;
                break;
            }
        }
        
        // If a match is found, increment the count atomically
        if (match) {
            atomicAdd(d_count, 1);
        }
    }
}

// Function to check for CUDA errors
void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        printf("CUDA error (%s): %s\n", msg, cudaGetErrorString(error));
        exit(-1);
    }
}

int main() {
    char sentence[N];
    char word[50];  // Maximum length of the word
    char *d_sentence, *d_word;
    unsigned int *d_count, count = 0;
    unsigned int *result;

    printf("Enter a sentence: ");
    fgets(sentence, N, stdin);  // Using fgets instead of gets to avoid buffer overflow
    printf("Enter the word to search for: ");
    fgets(word, 50, stdin);  // Using fgets instead of gets to avoid buffer overflow

    // Remove the newline character that fgets adds
    sentence[strcspn(sentence, "\n")] = 0;
    word[strcspn(word, "\n")] = 0;

    int sentence_len = strlen(sentence);
    int word_len = strlen(word);

    // CUDA events to measure time elapsed
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Allocate memory on the device
    checkCudaError(cudaMalloc((void**)&d_sentence, (sentence_len + 1) * sizeof(char)), "Allocating memory for d_sentence");
    checkCudaError(cudaMalloc((void**)&d_word, (word_len + 1) * sizeof(char)), "Allocating memory for d_word");
    checkCudaError(cudaMalloc((void**)&d_count, sizeof(unsigned int)), "Allocating memory for d_count");
    checkCudaError(cudaMalloc((void**)&result, sizeof(unsigned int)), "Allocating memory for result");

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_sentence, sentence, (sentence_len + 1) * sizeof(char), cudaMemcpyHostToDevice), "Copying sentence to device");
    checkCudaError(cudaMemcpy(d_word, word, (word_len + 1) * sizeof(char), cudaMemcpyHostToDevice), "Copying word to device");
    checkCudaError(cudaMemcpy(d_count, &count, sizeof(unsigned int), cudaMemcpyHostToDevice), "Copying initial count to device");

    // Launch the kernel with enough threads to cover the sentence length
    int blockSize = 256;
    int numBlocks = (sentence_len + blockSize - 1) / blockSize;

    // Call the kernel
    countWordKernel<<<numBlocks, blockSize>>>(d_sentence, d_word, d_count, sentence_len, word_len);

    // Check for any kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Wait for kernel to finish
    checkCudaError(cudaDeviceSynchronize(), "CUDA device synchronization failed");

    // Record stop time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // Copy the result back to the host
    checkCudaError(cudaMemcpy(&count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost), "Copying count back to host");

    // Print the result and elapsed time
    printf("Total occurrences of the word '%s': %u\n", word, count);
    printf("Time taken: %f milliseconds\n", elapsedTime);

    // Free device memory
    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);
    cudaFree(result);

    return 0;
}

