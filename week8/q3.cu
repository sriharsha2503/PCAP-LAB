#include <stdio.h>
#include <cuda_runtime.h>

// Kernel function to process the matrix
__global__ void processMatrix(int *A, int *B, int M, int N, int *rowSum, int *colSum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Flattened 2D index
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    if (idx < M && idy < N) {
        int element = A[idx * N + idy];

        // If the element is even, replace with row sum
        if (element % 2 == 0) {
            B[idx * N + idy] = rowSum[idx];
        }
        // If the element is odd, replace with column sum
        else {
            B[idx * N + idy] = colSum[idy];
        }
    }
}

int main() {
    int M, N;

    // Get matrix dimensions from user
    printf("Enter the number of rows (M): ");
    scanf("%d", &M);
    printf("Enter the number of columns (N): ");
    scanf("%d", &N);

    // Allocate memory for the input and output matrices
    int *h_A = (int *)malloc(M * N * sizeof(int));
    int *h_B = (int *)malloc(M * N * sizeof(int));

    // Row and column sums arrays
    int *h_rowSum = (int *)malloc(M * sizeof(int));
    int *h_colSum = (int *)malloc(N * sizeof(int));

    // Get matrix elements from the user
    printf("Enter the elements of the matrix A (%d x %d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("A[%d][%d]: ", i, j);
            scanf("%d", &h_A[i * N + j]);
        }
    }

    // Calculate row sums and column sums on the host
    for (int i = 0; i < M; i++) {
        h_rowSum[i] = 0;
        for (int j = 0; j < N; j++) {
            h_rowSum[i] += h_A[i * N + j];
        }
    }

    for (int j = 0; j < N; j++) {
        h_colSum[j] = 0;
        for (int i = 0; i < M; i++) {
            h_colSum[j] += h_A[i * N + j];
        }
    }

    // Device pointers for A, B, rowSum, and colSum
    int *d_A, *d_B, *d_rowSum, *d_colSum;

    // Allocate memory on the device
    cudaMalloc(&d_A, M * N * sizeof(int));
    cudaMalloc(&d_B, M * N * sizeof(int));
    cudaMalloc(&d_rowSum, M * sizeof(int));
    cudaMalloc(&d_colSum, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowSum, h_rowSum, M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colSum, h_colSum, N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for the kernel launch
    dim3 blockDim(16, 16); // 16x16 block of threads
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y); // Grid size

    // Launch the kernel
    processMatrix<<<gridDim, blockDim>>>(d_A, d_B, M, N, d_rowSum, d_colSum);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy the result matrix B back from device to host
    cudaMemcpy(h_B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the resultant matrix B
    printf("Resultant matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_B[i * N + j]);
        }
        printf("\n");
    }

    // Free memory on the device and host
    free(h_A);
    free(h_B);
    free(h_rowSum);
    free(h_colSum);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_rowSum);
    cudaFree(d_colSum);

    return 0;
}


