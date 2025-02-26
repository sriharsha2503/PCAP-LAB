#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string.h>
#define N 1024  

__global__ void countWordKernel(const char *sentence, const char *word, unsigned int *d_count, int sentence_len, int word_len) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx <= sentence_len - word_len) 
    {
        bool match = true;
        for (int i = 0; i < word_len; ++i) 
        {
            if (sentence[idx + i] != word[i]) 
            {
                match = false;
                break;
            }
        }
        if (match) 
        {
            atomicAdd(d_count, 1);
        }
    }
}

void checkCudaError(cudaError_t error, const char* msg) 
{
    if (error != cudaSuccess) 
    {
        printf("CUDA error (%s): %s\n", msg, cudaGetErrorString(error));
        exit(-1);
    }
}

int main() {
    char sentence[N];
    char word[50];
    char *d_sentence, *d_word;
    unsigned int *d_count;
    unsigned int *count;  
    count = (unsigned int*)malloc(sizeof(unsigned int));
    *count = 0; 
    printf("Enter a sentence: ");
    fgets(sentence, N, stdin);  
    printf("Enter the word to search for: ");
    fgets(word, 50, stdin); 
    sentence[strcspn(sentence, "\n")] = 0;
    word[strcspn(word, "\n")] = 0;
    
    int sentence_len = strlen(sentence);
    int word_len = strlen(word);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    checkCudaError(cudaMalloc((void**)&d_sentence, (sentence_len + 1) * sizeof(char)), "Allocating memory for d_sentence");
    checkCudaError(cudaMalloc((void**)&d_word, (word_len + 1) * sizeof(char)), "Allocating memory for d_word");
    checkCudaError(cudaMalloc((void**)&d_count, sizeof(unsigned int)), "Allocating memory for d_count");
    checkCudaError(cudaMemcpy(d_sentence, sentence, (sentence_len + 1) * sizeof(char), cudaMemcpyHostToDevice), "Copying sentence to device");
    checkCudaError(cudaMemcpy(d_word, word, (word_len + 1) * sizeof(char), cudaMemcpyHostToDevice), "Copying word to device");
    checkCudaError(cudaMemcpy(d_count, count, sizeof(unsigned int), cudaMemcpyHostToDevice), "Copying initial count to device");
    dim3 dimGrid((sentence_len + 256 - 1) / 256, 1, 1);
    dim3 dimBlock(256, 1, 1);
    countWordKernel<<<dimGrid, dimBlock>>>(d_sentence, d_word, d_count, sentence_len, word_len);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "CUDA device synchronization failed");
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);   
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    checkCudaError(cudaMemcpy(count, d_count, sizeof(unsigned int), cudaMemcpyDeviceToHost), "Copying count back to host");
    printf("Total occurrences of the word '%s': %u\n", word, *count);
    printf("Time taken: %f milliseconds\n", elapsedTime);
    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);
    free(count); 
    return 0;
}

//Enter a sentence: i am agood person with very good 
//Enter the word to search for: good
//Total occurrences of the word 'good': 2
//Time taken: 1.804288 milliseconds
