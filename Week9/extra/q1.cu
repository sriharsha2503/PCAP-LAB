#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
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
__global__ void spmv_csr_kernel(int *row_ptr, int *col_idx, float *values, float *x, float *y, int num_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int j = row_start; j < row_end; ++j) {
            sum += values[j] * x[col_idx[j]];
        }
        y[row] = sum;
    }
}
void generate_sparse_matrix_csr(int num_rows, int num_cols, int nnz, 
                                int **row_ptr, int **col_idx, float **values) {
    *row_ptr = (int *)malloc((num_rows + 1) * sizeof(int));
    *col_idx = (int *)malloc(nnz * sizeof(int));
    *values = (float *)malloc(nnz * sizeof(float));

    srand(time(NULL));

    int nnz_count = 0;
    (*row_ptr)[0] = 0;

    for (int i = 0; i < num_rows; ++i) {
        int row_nnz = rand() % (num_cols / 2) + 1;  // Random number of nonzeros per row
        if (nnz_count + row_nnz > nnz) row_nnz = nnz - nnz_count;

        for (int j = 0; j < row_nnz; ++j) {
            (*col_idx)[nnz_count + j] = rand() % num_cols;  // Random column index
            (*values)[nnz_count + j] = (float)(rand() % 10 + 1);
        }

        nnz_count += row_nnz;
        (*row_ptr)[i + 1] = nnz_count;

        if (nnz_count >= nnz) break;
    }
}
int main() {
    int num_rows = 5;
    int num_cols = 5;
    int nnz = 12;  
    int *h_row_ptr, *h_col_idx;
    float *h_values, *h_x, *h_y;

    generate_sparse_matrix_csr(num_rows, num_cols, nnz, &h_row_ptr, &h_col_idx, &h_values);

    h_x = (float *)malloc(num_cols * sizeof(float));
    h_y = (float *)malloc(num_rows * sizeof(float));

    for (int i = 0; i < num_cols; ++i) {
        h_x[i] = (float)(i + 1);
    }
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void **)&d_row_ptr, (num_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_values, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_x, num_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_y, num_rows * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, h_row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, h_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice));
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_rows + threadsPerBlock - 1) / threadsPerBlock;
    spmv_csr_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_row_ptr, d_col_idx, d_values, d_x, d_y, num_rows);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Sparse Matrix-Vector Multiplication Result (y):\n");
    for (int i = 0; i < num_rows; ++i) {
        printf("%.2f ", h_y[i]);
    }
    printf("\n");
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_row_ptr);
    free(h_col_idx);
    free(h_values);
    free(h_x);
    free(h_y);

    return 0;
}
