#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel 1: Each thread computes one entire row of C.
__global__ void matMulRowKernel(int *A, int *B, int *C, int h_a, int w_a, int w_b) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < h_a) {
        for (int col = 0; col < w_b; col++) {
            int sum = 0;
            for (int k = 0; k < w_a; k++) {
                sum += A[row * w_a + k] * B[k * w_b + col];
            }
            C[row * w_b + col] = sum;
        }
    }
}

// Kernel 2: Each thread computes one entire column of C.
__global__ void matMulColKernel(int *A, int *B, int *C, int h_a, int w_a, int w_b) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < w_b) {
        for (int row = 0; row < h_a; row++) {
            int sum = 0;
            for (int k = 0; k < w_a; k++) {
                sum += A[row * w_a + k] * B[k * w_b + col];
            }
            C[row * w_b + col] = sum;
        }
    }
}

// Kernel 3: Each thread computes one element of C.
__global__ void matMulElementKernel(int *A, int *B, int *C, int h_a, int w_a, int w_b) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < h_a && col < w_b) {
        int sum = 0;
        for (int k = 0; k < w_a; k++) {
            sum += A[row * w_a + k] * B[k * w_b + col];
        }
        C[row * w_b + col] = sum;
    }
}

int main() {
    int h_a, w_a, h_b, w_b;

    // Input dimensions for Matrix A.
    printf("Enter height of Matrix A: ");
    scanf("%d", &h_a);
    printf("Enter width of Matrix A: ");
    scanf("%d", &w_a);

    // Input dimensions for Matrix B.
    printf("Enter height of Matrix B: ");
    scanf("%d", &h_b);
    printf("Enter width of Matrix B: ");
    scanf("%d", &w_b);

    // Check condition for matrix multiplication: width of A must equal height of B.
    if (w_a != h_b) {
        printf("Error: For matrix multiplication, width of Matrix A must equal height of Matrix B.\n");
        return -1;
    }

    // Allocate host memory for matrices.
    // Matrix A: h_a x w_a, Matrix B: h_b x w_b, Result C: h_a x w_b.
    int sizeA = h_a * w_a * sizeof(int);
    int sizeB = h_b * w_b * sizeof(int);
    int sizeC = h_a * w_b * sizeof(int);

    int *h_A = (int *)malloc(sizeA);
    int *h_B = (int *)malloc(sizeB);
    int *h_C = (int *)malloc(sizeC);

    // Input matrix A.
    printf("Enter elements of Matrix A (%d elements):\n", h_a * w_a);
    for (int i = 0; i < h_a; i++) {
        for (int j = 0; j < w_a; j++) {
            scanf("%d", &h_A[i * w_a + j]);
        }
    }

    // Input matrix B.
    printf("Enter elements of Matrix B (%d elements):\n", h_b * w_b);
    for (int i = 0; i < h_b; i++) {
        for (int j = 0; j < w_b; j++) {
            scanf("%d", &h_B[i * w_b + j]);
        }
    }

    // Allocate device memory.
    int *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    // Copy matrices from host to device.
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // -------------------------------------------------------------------
    // Kernel 1: Each row of C computed by one thread.
    int blockSize1 = 256;
    int gridSize1 = (h_a + blockSize1 - 1) / blockSize1;
    matMulRowKernel<<<gridSize1, blockSize1>>>(d_A, d_B, d_C, h_a, w_a, w_b);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    printf("\nResult using Kernel 1 (each row computed by one thread):\n");
    for (int i = 0; i < h_a; i++) {
        for (int j = 0; j < w_b; j++) {
            printf("%d ", h_C[i * w_b + j]);
        }
        printf("\n");
    }

    // -------------------------------------------------------------------
    // Kernel 2: Each column of C computed by one thread.
    int blockSize2 = 256;
    int gridSize2 = (w_b + blockSize2 - 1) / blockSize2;
    matMulColKernel<<<gridSize2, blockSize2>>>(d_A, d_B, d_C, h_a, w_a, w_b);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    printf("\nResult using Kernel 2 (each column computed by one thread):\n");
    for (int i = 0; i < h_a; i++) {
        for (int j = 0; j < w_b; j++) {
            printf("%d ", h_C[i * w_b + j]);
        }
        printf("\n");
    }

    // -------------------------------------------------------------------
    // Kernel 3: Each element of C computed by one thread.
    dim3 blockSize3(16, 16);
    dim3 gridSize3((w_b + blockSize3.x - 1) / blockSize3.x, (h_a + blockSize3.y - 1) / blockSize3.y);
    matMulElementKernel<<<gridSize3, blockSize3>>>(d_A, d_B, d_C, h_a, w_a, w_b);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    printf("\nResult using Kernel 3 (each element computed by one thread):\n");
    for (int i = 0; i < h_a; i++) {
        for (int j = 0; j < w_b; j++) {
            printf("%d ", h_C[i * w_b + j]);
        }
        printf("\n");
    }

    // Free device and host memory.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
