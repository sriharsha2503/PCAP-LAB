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

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i + j;
            B[i * N + j] = i - j;
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
    printf("Resultant Matrix (Approach a - Row-wise):\n");
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
