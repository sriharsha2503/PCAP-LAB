#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                   \
    {                                                                      \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

// Kernel to apply row-wise transformation
__global__ void transform_matrix(float *matrix, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < n) {
        int index = row * n + col;
        matrix[index] = powf(matrix[index], row + 1);
    }
}

int main() {
    int m, n;

    // Matrix dimensions
    printf("Enter number of rows (m): ");
    scanf("%d", &m);
    printf("Enter number of columns (n): ");
    scanf("%d", &n);

    size_t size = m * n * sizeof(float);

    // Allocate host memory
    float *h_matrix = (float *)malloc(size);

    // Initialize matrix
    printf("\nEnter matrix elements:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("A[%d][%d]: ", i, j);
            scanf("%f", &h_matrix[i * n + j]);
        }
    }

    // Allocate device memory
    float *d_matrix;
    CUDA_CHECK(cudaMalloc((void **)&d_matrix, size));

    // Copy matrix to device
    CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice));

    // Kernel configuration
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    transform_matrix<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, m, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost));

    // Print the transformed matrix
    printf("\nTransformed Matrix:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.2f ", h_matrix[i * n + j]);
        }
        printf("\n");
    }

    // Free device and host memory
    cudaFree(d_matrix);
    free(h_matrix);

    return 0;
}
