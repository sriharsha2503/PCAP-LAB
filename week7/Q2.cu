#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string.h>
#define MAX_LEN 1024
__global__ void genPatterns(const char *in, char *out, int len, int num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < num) {
        int sub_len = len - idx;
        for (int i = 0; i < sub_len; ++i) {
            out[idx * len + i] = in[i];
        }
    }
}
int main() {
    char in[MAX_LEN];
    char *out;
    char *d_in, *d_out;
    printf("Enter string: ");
    fgets(in, MAX_LEN, stdin);
    in[strcspn(in, "\n")] = 0;
    int len = strlen(in);
    int num = len;
    cudaMalloc((void**)&d_in, (len + 1) * sizeof(char));
    cudaMalloc((void**)&d_out, (len * num + 1) * sizeof(char));
    cudaMemcpy(d_in, in, (len + 1) * sizeof(char), cudaMemcpyHostToDevice);
    int blkSize = 256;
    int blkCount = (num + blkSize - 1) / blkSize;
    genPatterns<<<blkCount, blkSize>>>(d_in, d_out, len, num);
    cudaDeviceSynchronize();
    out = (char *)malloc(len * num * sizeof(char));
    cudaMemcpy(out, d_out, (len * num + 1) * sizeof(char), cudaMemcpyDeviceToHost);
    printf("Input: %s\n", in);
    printf("Pattern:");
    for (int i = 0; i < num; ++i) {
        int sub_len = len - i;
        printf("%.*s", sub_len, &out[i * len]);
    }
    printf("\n");
    cudaFree(d_in);
    cudaFree(d_out);
    free(out);
    return 0;
}
//Enter string: paragraph
//Input: paragraph
//Pattern:paragraphparagrapparagraparagrparagparaparpap
