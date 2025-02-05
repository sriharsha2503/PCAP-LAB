#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void linearAlgebraOperation(float *x, float *y, float alpha, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread index

    if (idx < N) {
        y[idx] = alpha * x[idx] + y[idx];  // Perform the operation y = alpha * x + y
    }
}

int main(void) {
    int N = 1000;  // Size of the vectors
    float alpha = 2.0f;  // Scalar value
    int size = N * sizeof(float);

    float *h_x = (float *)malloc(size);
    float *h_y = (float *)malloc(size);
    float *d_x, *d_y;

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)(i);  // x = 0, 1, 2, 3, ...
        h_y[i] = (float)(i * 2);  // y = 0, 2, 4, 6, ...
    }

    // Allocate device memory
    cudaMalloc((void **)&d_x, size);
    cudaMalloc((void **)&d_y, size);

    // Copy data from host to device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    // Define block size and number of blocks
    int THREADS_PER_BLOCK = 256;
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch the kernel
    linearAlgebraOperation<<<numBlocks, THREADS_PER_BLOCK>>>(d_x, d_y, alpha, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    // Print the first 10 results for verification
    printf("First 10 results of y = alpha * x + y:\n");
    for (int i = 0; i < (N < 10 ? N : 10); i++) {
        printf("y[%d] = %f\n", i, h_y[i]);
    }

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);

    // Free host memory
    free(h_x);
    free(h_y);

    return 0;
}

//nvcc additional1.cu -o ad1
//./ad1
//First 10 results of y = alpha * x + y:
//y[0] = 0.000000
//y[1] = 4.000000
//y[2] = 8.000000
//y[3] = 12.000000
//y[4] = 16.000000
//y[5] = 20.000000
//y[6] = 24.000000
//y[7] = 28.000000
//y[8] = 32.000000
//y[9] = 36.000000

