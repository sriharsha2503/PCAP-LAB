#include <stdio.h>
#include <cuda.h>

#define TILE_SIZE 16
#define MASK_WIDTH 5
#define RADIUS (MASK_WIDTH / 2)

__constant__ float d_mask[MASK_WIDTH];

// Kernel: Shared memory tiling with constant memory mask
__global__ void tiled1DConvolution(float *d_input, float *d_output, int width) {
    __shared__ float tile[TILE_SIZE + 2 * RADIUS];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    int shared_idx = tid + RADIUS;

    // Center
    tile[shared_idx] = (global_idx < width) ? d_input[global_idx] : 0.0f;

    // Left halo
    if (tid < RADIUS) {
        int left_idx = global_idx - RADIUS;
        tile[shared_idx - RADIUS] = (left_idx >= 0) ? d_input[left_idx] : 0.0f;
    }

    // Right halo
    if (tid >= blockDim.x - RADIUS) {
        int right_idx = global_idx + RADIUS;
        tile[shared_idx + RADIUS] = (right_idx < width) ? d_input[right_idx] : 0.0f;
    }

    __syncthreads();

    // Perform convolution
    if (global_idx < width) {
        float result = 0.0f;
        for (int j = 0; j < MASK_WIDTH; j++) {
            result += tile[shared_idx - RADIUS + j] * d_mask[j];
        }
        d_output[global_idx] = result;
    }
}

// Helper to input array values
void inputArray(float *arr, int size, const char *name) {
    printf("Enter %d values for %s:\n", size, name);
    for (int i = 0; i < size; i++) {
        printf("%s[%d] = ", name, i);
        scanf("%f", &arr[i]);
    }
}

int main() {
    int width;
    printf("Enter width of input array: ");
    scanf("%d", &width);

    size_t size = width * sizeof(float);

    // Host memory
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    float h_mask[MASK_WIDTH];

    // User inputs
    inputArray(h_input, width, "Input Array");
    inputArray(h_mask, MASK_WIDTH, "Mask");

    // Device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Copy to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, h_mask, MASK_WIDTH * sizeof(float));

    // Launch kernel
    int blockSize = TILE_SIZE;
    int gridSize = (width + blockSize - 1) / blockSize;
    tiled1DConvolution<<<gridSize, blockSize>>>(d_input, d_output, width);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("\nConvolved Output:\n");
    for (int i = 0; i < width; i++) {
        printf("%0.2f ", h_output[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
/*
Enter width of input array: 7
Enter 7 values for Input Array:
Input Array[0] = 1
Input Array[1] = 2
Input Array[2] = 3
Input Array[3] = 4
Input Array[4] = 5
Input Array[5] = 6
Input Array[6] = 7
Enter 5 values for Mask:
Mask[0] = 2
Mask[1] = 3
Mask[2] = 4
Mask[3] = 5
Mask[4] = 6

Convolved Output:
32.00 50.00 70.00 90.00 110.00 82.00 56.00
**/
