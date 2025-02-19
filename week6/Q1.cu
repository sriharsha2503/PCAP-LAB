#include <stdio.h>
#include <cuda_runtime.h>

__global__ void conv_1d_kernel(float *input, float *mask, float *output, int width, int mask_width)
{
    int half_mask = mask_width / 2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < width)
    {
        float result = 0.0f;
        for (int i = 0; i < mask_width; i++)
        {
            int input_idx = tid + i - half_mask;
            if (input_idx >= 0 && input_idx < width)
            {
                result += input[input_idx] * mask[i];
            }
        }
        output[tid] = result;
    }
}

void convolution_id(float *input, float *mask, float *output, int width, int mask_width)
{
    float *d_input, *d_mask, *d_output;
    cudaMalloc(&d_input, width * sizeof(float));
    cudaMalloc(&d_mask, mask_width * sizeof(float));
    cudaMalloc(&d_output, width * sizeof(float));
    cudaMemcpy(d_input, input, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, mask_width * sizeof(float), cudaMemcpyHostToDevice);
    int block_size = 256;
    int grid_size = (width + block_size - 1) / block_size; 
    dim3 dimBlock(block_size, 1, 1);
    dim3 dimGrid(grid_size, 1, 1); 
    conv_1d_kernel<<<dimGrid, dimBlock>>>(d_input, d_mask, d_output, width, mask_width);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);
}

int main()
{
    int width = 10;
    int mask_width = 3;
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float mask[] = {0.2f, 0.5f, 0.2f};
    float output[width];

    convolution_id(input, mask, output, width, mask_width);

    printf("Output after convolution:\n");
    for (int i = 0; i < width; i++)
    {
        printf("%f ", output[i]);
    }
    printf("\n");

    return 0;
}
//Output after convolution:
//0.900000 1.800000 2.700000 3.600000 4.500000 5.400000 6.300000 7.200000 8.100000 6.800000

