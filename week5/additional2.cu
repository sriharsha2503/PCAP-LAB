#include <stdio.h>
#include <cuda_runtime.h>

#define N 5  // Number of rows
#define M 6  // Number of columns

// CUDA kernel for selection sort on each row
__global__ void selectionSortKernel(int* matrix, int numCols) {
    int row = blockIdx.x;  // Each block corresponds to one row
    if (row < N) {
        int i, j, min_idx, temp;
        // Selection sort on the row
        for (i = 0; i < numCols - 1; i++) {
            min_idx = i;
            for (j = i + 1; j < numCols; j++) {
                if (matrix[row * numCols + j] < matrix[row * numCols + min_idx]) {
                    min_idx = j;
                }
            }
            // Swap the found minimum element with the first element
            if (min_idx != i) {
                temp = matrix[row * numCols + i];
                matrix[row * numCols + i] = matrix[row * numCols + min_idx];
                matrix[row * numCols + min_idx] = temp;
            }
        }
    }
}

// Function to print the matrix
void printMatrix(int* matrix, int numRows, int numCols) {
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            printf("%d ", matrix[i * numCols + j]);
        }
        printf("\n");
    }
}

int main() {
    int h_matrix[N][M] = {
        {64, 34, 25, 12, 22, 11},
        {90, 55, 23, 67, 42, 81},
        {10, 9, 8, 7, 6, 5},
        {50, 10, 20, 40, 60, 30},
        {5, 7, 3, 8, 2, 1}
    };

    int* d_matrix;
    size_t matrix_size = N * M * sizeof(int);

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_matrix, matrix_size);

    // Copy the matrix from host to device
    cudaMemcpy(d_matrix, h_matrix, matrix_size, cudaMemcpyHostToDevice);

    // Launch kernel with N blocks (one per row) and 1 thread per block
    selectionSortKernel<<<N, 1>>>(d_matrix, M);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy the sorted matrix back to host
    cudaMemcpy(h_matrix, d_matrix, matrix_size, cudaMemcpyDeviceToHost);

    // Print the sorted matrix
    printf("Sorted Matrix:\n");
    printMatrix(reinterpret_cast<int*>(h_matrix), N, M);

    // Free the GPU memory
    cudaFree(d_matrix);

    return 0;
}
//Sorted Matrix:
//11 12 22 25 34 64 
//23 42 55 67 81 90 
//5 6 7 8 9 10 
//10 20 30 40 50 60 
//1 2 3 5 7 8 
