#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel to perform Odd-Even Transposition Sort
__global__ void odd_even_transposition_sort(int *arr, int n, bool phase) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n - 1) {
        if (phase) {
            if (idx % 2 == 1 && arr[idx] > arr[idx + 1]) 
            {
                int temp = arr[idx];
                arr[idx] = arr[idx + 1];
                arr[idx + 1] = temp;
            }
        }
        else {
            if (idx % 2 == 0 && arr[idx] > arr[idx + 1]) {
                int temp = arr[idx];
                arr[idx] = arr[idx + 1];
                arr[idx + 1] = temp;
            }
        }
    }
}
void odd_even_sort(int *arr, int n) {
    int *d_arr;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);
    dim3 DimBlock(256,1,1); 
    dim3 DimGrid((n + block_size.x - 1) / block_size.x,1,1);
    for (int phase = 0; phase < n; phase++) 
    {
        odd_even_transposition_sort<<<grid_size, block_size>>>(d_arr, n, phase % 2);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main() {
    int arr[] = {29, 10, 14, 37, 13, 35, 55, 22, 90, 2};
    int n = sizeof(arr) / sizeof(arr[0]);

    printf("Original Array: ");
    for (int i = 0; i < n; i++) 
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
    odd_even_sort(arr, n);
    printf("Sorted Array: ");
    for (int i = 0; i < n; i++) 
    {
        printf("%d ", arr[i]);
    }
    printf("\n");

    return 0;
}
//Original Array: 29 10 14 37 13 35 55 22 90 2 
//Sorted Array: 2 10 13 14 22 29 35 37 55 90 

