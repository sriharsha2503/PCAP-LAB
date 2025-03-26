#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

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

// CUDA Kernel to replace non-border elements with 1's complement
__global__ void ones_complement_kernel(int *A, int *B, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        int index = row * N + col;

        // Check for border elements
        if (row == 0 || row == M - 1 || col == 0 || col == N - 1) {
            B[index] = A[index];  // Keep border elements the same
        } else {
            B[index] = ~A[index];  // 1's complement of non-border elements
        }
    }
}

int main() {
    int M, N;

    // Matrix dimensions
    printf("Enter number of rows (M): ");
    scanf("%d", &M);
    printf("Enter number of columns (N): ");
    scanf("%d", &N);

    size_t size = M * N * sizeof(int);

    // Allocate host memory
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);

    // Initialize matrix A
    printf("\nEnter matrix elements:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("A[%d][%d]: ", i, j);
            scanf("%d", &h_A[i * N + j]);
        }
    }

    // Allocate device memory
    int *d_A, *d_B;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));

    // Copy matrix A to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    // Kernel configuration
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    ones_complement_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, M, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost));

    // Print the output matrix B
    printf("\nOutput Matrix B:\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", h_B[i * N + j]);
        }
        printf("\n");
    }

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}
