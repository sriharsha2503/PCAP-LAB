#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define M 3 
#define N 3  
#define K 3  

__global__ void multiplyMatricesRowWise(int *A, int *B, int *C, int numColsA, int numColsB) {
    int row = blockIdx.x;  
    if (row < M) {
        for (int col = 0; col < numColsB; col++) {
            int sum = 0;
            for (int k = 0; k < numColsA; k++) {
                sum += A[row * numColsA + k] * B[k * numColsB + col];
            }
            C[row * numColsB + col] = sum;
        }
    }
}

__global__ void multiplyMatricesColWise(int *A, int *B, int *C, int numRowsA, int numColsB) {
    int col = blockIdx.x;
    if (col < N) {
        for (int row = 0; row < numRowsA; row++) {
            int sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

__global__ void multiplyMatricesElementWise(int *A, int *B, int *C, int numColsA, int numColsB) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < M && col < N) {
        int sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * numColsA + k] * B[k * numColsB + col];
        }
        C[row * numColsB + col] = sum;
    }
}

void printMatrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    int *A, *B, *C;
    int *d_A, *d_B, *d_C;

    size_t sizeA = M * K * sizeof(int);
    size_t sizeB = K * N * sizeof(int);
    size_t sizeC = M * N * sizeof(int);
    
    A = (int*)malloc(sizeA);
    B = (int*)malloc(sizeB);
    C = (int*)malloc(sizeC);

    // Taking input from the user for matrix A
    printf("Enter the elements of matrix A (3x3):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("A[%d][%d]: ", i, j);
            scanf("%d", &A[i * K + j]);
        }
    }

    // Taking input from the user for matrix B
    printf("Enter the elements of matrix B (3x3):\n");
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            printf("B[%d][%d]: ", i, j);
            scanf("%d", &B[i * N + j]);
        }
    }

    // Allocating memory on the device
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // Copy data from host to device
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    // Row-wise multiplication (Approach a)
    multiplyMatricesRowWise<<<M, 1>>>(d_A, d_B, d_C, K, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    printf("Resultant Matrix (Approach a - Row-wise):\n");
    printMatrix(C, M, N);

    // Column-wise multiplication (Approach b)
    multiplyMatricesColWise<<<N, 1>>>(d_A, d_B, d_C, M, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    printf("\nResultant Matrix (Approach b - Column-wise):\n");
    printMatrix(C, M, N);

    // Element-wise multiplication (Approach c)
    multiplyMatricesElementWise<<<M, N>>>(d_A, d_B, d_C, K, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    printf("\nResultant Matrix (Approach c - Element-wise):\n");
    printMatrix(C, M, N);

    // Free allocated memory
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

