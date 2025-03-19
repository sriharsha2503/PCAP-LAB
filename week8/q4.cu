#include <stdio.h>
#include <cuda_runtime.h>

// Function to compute the factorial of a number
__device__ int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Function to compute the sum of digits of a number
__device__ int sum_of_digits(int n) {
    int sum = 0;
    while (n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return sum;
}

// Kernel to process the matrix
__global__ void processMatrix(int *A, int *B, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // Row index (flattened)
    int idy = threadIdx.y + blockIdx.y * blockDim.y; // Column index (flattened)

    if (idx < N && idy < N) {
        if (idx == idy) {
            // Principal diagonal elements become zero
            B[idx * N + idy] = 0;
        } else if (idx < idy) {
            // Elements above the principal diagonal replaced by factorial
            B[idx * N + idy] = factorial(A[idx * N + idy]);
        } else {
            // Elements below the principal diagonal replaced by sum of digits
            B[idx * N + idy] = sum_of_digits(A[idx * N + idy]);
        }
    }
}

int main() {
    int N;

    // Get matrix size from the user
    printf("Enter the size of the matrix (N x N): ");
    scanf("%d", &N);

    // Allocate memory for the input and output matrices
    int *h_A = (int *)malloc(N * N * sizeof(int));
    int *h_B = (int *)malloc(N * N * sizeof(int));

    // Get matrix elements from the user
    printf("Enter the elements of the matrix A (%d x %d):\n", N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("A[%d][%d]: ", i, j);
            scanf("%d", &h_A[i * N + j]);
        }
    }

    // Device pointers for A, B
    int *d_A, *d_B;

    // Allocate memory on the device
    cudaMalloc(&d_A, N * N * sizeof(int));
    cudaMalloc(&d_B, N * N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions for the kernel launch
    dim3 blockDim(16, 16); // 16x16 block of threads
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y); // Grid size

    // Launch the kernel
    processMatrix<<<gridDim, blockDim>>>(d_A, d_B, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy the result matrix B back from device to host
    cudaMemcpy(h_B, d_B, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Output the resultant matrix B
    printf("Resultant matrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_B[i * N + j]);
        }
        printf("\n");
    }

    // Free memory on the device and host
    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
