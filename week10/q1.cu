#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define TILE_SIZE 16  // Block size (16x16)

__global__ void matrixMul(const float *A, const float *B, float *C, int N) {
    // Row and column indices for C
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    if (row < N && col < N) {
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void matrixMultiplicationHost(float *A, float *B, float *C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // Allocate memory on the device
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    matrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    // Copy the result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void printMatrix(float *M, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%6.2f ", M[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    int N;
    printf("Enter the size of the matrix (N x N): ");
    scanf("%d", &N);

    size_t size = N * N * sizeof(float);

    // Allocate memory for matrices
    float *A = (float *)malloc(size);
    float *B = (float *)malloc(size);
    float *C = (float *)malloc(size);

    // Input Matrix A
    printf("\nEnter elements of Matrix A (%d x %d):\n", N, N);
    for (int i = 0; i < N * N; i++) {
        printf("A[%d][%d]: ", i / N, i % N);
        scanf("%f", &A[i]);
    }

    // Input Matrix B
    printf("\nEnter elements of Matrix B (%d x %d):\n", N, N);
    for (int i = 0; i < N * N; i++) {
        printf("B[%d][%d]: ", i / N, i % N);
        scanf("%f", &B[i]);
    }

    printf("\nMatrix A:\n");
    printMatrix(A, N);

    printf("\nMatrix B:\n");
    printMatrix(B, N);

    // Perform matrix multiplication
    matrixMultiplicationHost(A, B, C, N);

    printf("\nMatrix C (Result):\n");
    printMatrix(C, N);

    // Free memory
    free(A);
    free(B);
    free(C);

    return 0;
}
/*
Enter the size of the matrix (N x N): 3

Enter elements of Matrix A (3 x 3):
A[0][0]: 2
A[0][1]: 3
A[0][2]: 4
A[1][0]: 5
A[1][1]: 6
A[1][2]: 7
A[2][0]: 8
A[2][1]: 9
A[2][2]: 5

Enter elements of Matrix B (3 x 3):
B[0][0]: 4
B[0][1]: 3
B[0][2]: 4
B[1][0]: 6
B[1][1]: 8
B[1][2]: 6
B[2][0]: 45
B[2][1]: 3
B[2][2]: 4

Matrix A:
  2.00   3.00   4.00 
  5.00   6.00   7.00 
  8.00   9.00   5.00 

Matrix B:
  4.00   3.00   4.00 
  6.00   8.00   6.00 
 45.00   3.00   4.00 

Matrix C (Result):
206.00  42.00  42.00 
371.00  84.00  84.00 
311.00 111.00 106.00
**/

