#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void computeSine(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    if (idx < N) {
        output[idx] = sin(input[idx]);  
    }
}

int main() {
    int N = 5;  
    
    float *h_input, *h_output;

    // Allocate memory on host
    h_input = (float*)malloc(N * sizeof(float));
    h_output = (float*)malloc(N * sizeof(float));

    // Initialize input values (angles in radians)
    h_input[0] = 0.0f;               
    h_input[1] = M_PI / 2.0f;         
    h_input[2] = M_PI;               
    h_input[3] = 3.0f * M_PI / 2.0f;  
    h_input[4] = 2.0f * M_PI;        

    float *d_input, *d_output;

    // Allocate memory on device
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Specify block size and grid size using dim3
    dim3 dimBlock(256, 1, 1);  // Set block size to 256 threads along the x-axis
    dim3 dimGrid(ceil(N / 256.0), 1, 1);  // Calculate number of blocks using ceil

    // Launch kernel
    computeSine<<<dimGrid, dimBlock>>>(d_input, d_output, N);
    
    // Synchronize to ensure kernel finishes before proceeding
    cudaDeviceSynchronize();

    // Copy output data from device to host
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print the results
    printf("Sine values of the angles in radians:\n");
    for (int i = 0; i < N; i++) {
        printf("sin(%.2f) = %.4f\n", h_input[i], h_output[i]);
    }

    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}

//output:
//Sine values of the angles in radians:
//sin(0.00) = 0.0000
//sin(1.57) = 1.0000
//sin(3.14) = -0.0000
//sin(4.71) = -1.0000
//sin(6.28) = 0.0000

