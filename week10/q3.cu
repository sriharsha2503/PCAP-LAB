#include <stdio.h>
#include <cuda.h>

#define N 16           // Smaller array size for easy verification
#define BLOCK_SIZE 16   // Block size matching the array size

// Inclusive scan kernel
__global__ void inclusiveScan(float *d_input, float *d_output, int n) {
    __shared__ float temp[BLOCK_SIZE];  // Shared memory for intra-block scan

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Load data into shared memory
    if (tid < n) {
        temp[threadIdx.x] = d_input[tid];
    } else {
        temp[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Inclusive scan using sequential addressing
    for (int offset = 1; offset < BLOCK_SIZE; offset *= 2) {
        float val = 0.0f;
        if (threadIdx.x >= offset) {
            val = temp[threadIdx.x - offset];
        }
        __syncthreads();
        temp[threadIdx.x] += val;
        __syncthreads();
    }

    // Write result back to global memory
    if (tid < n) {
        d_output[tid] = temp[threadIdx.x];
    }
}

void printArray(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%0.2f ", arr[i]);
        if ((i + 1) % 8 == 0) printf("\n");  // Print 8 elements per line for readability
    }
    printf("\n");
}

int main() {
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    // Initialize input array with sample values
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i + 1);  // Values 1, 2, 3, ..., N
    }

    printf("Input Array:\n");
    printArray(h_input, N);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, size);
    cudaMalloc((void **)&d_output, size);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    inclusiveScan<<<gridDim, blockDim>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("\nInclusive Scan Result:\n");
    printArray(h_output, N);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
