#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define M 3
#define N 3

__global__ void addMatricesRowWise(int *A, int *B, int *C, int numCols) {
    int row = blockIdx.x; 
    if (row < M) {
        for (int col = 0; col < numCols; col++) {
            C[row * numCols + col] = A[row * numCols + col] + B[row * numCols + col];
        }
    }
}

__global__ void addMatricesColWise(int *A, int *B, int *C, int numRows) {
    int col = blockIdx.x; 
    if (col < N) {
        for (int row = 0; row < numRows; row++) {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }
}

__global__ void addMatricesElementWise(int *A, int *B, int *C) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row < M && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
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
    size_t size = M * N * sizeof(int);
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);
    printf("Enter elements of matrix A (%dx%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("A[%d][%d]: ", i, j);
            scanf("%d", &A[i * N + j]);
        }
    }
    printf("Enter elements of matrix B (%dx%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("B[%d][%d]: ", i, j);
            scanf("%d", &B[i * N + j]);
        }
    }
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    addMatricesRowWise<<<M, 1>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("\nResultant Matrix (Approach a - Row-wise):\n");
    printMatrix(C, M, N);
    addMatricesColWise<<<N, 1>>>(d_A, d_B, d_C, M);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("\nResultant Matrix (Approach b - Column-wise):\n");
    printMatrix(C, M, N);
    addMatricesElementWise<<<M, N>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("\nResultant Matrix (Approach c - Element-wise):\n");
    printMatrix(C, M, N);
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

/*
Enter elements of matrix A (3x3):
A[0][0]: 1
A[0][1]: 2
A[0][2]: 3
A[1][0]: 4
A[1][1]: 5
A[1][2]: 6
A[2][0]: 7
A[2][1]: 8
A[2][2]: 9
Enter elements of matrix B (3x3):
B[0][0]: 1
B[0][1]: 2
B[0][2]: 3
B[1][0]: 4
B[1][1]: 5
B[1][2]: 6
B[2][0]: 7
B[2][1]: 8
B[2][2]: 9

Resultant Matrix (Approach a - Row-wise):
2 4 6 
8 10 12 
14 16 18 

Resultant Matrix (Approach b - Column-wise):
2 4 6 
8 10 12 
14 16 18 

Resultant Matrix (Approach c - Element-wise):
2 4 6 
8 10 12 
14 16 18 
*/




