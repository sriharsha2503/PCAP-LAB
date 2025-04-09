#include <stdio.h>
#include <cuda.h>

#define NUM_ITEMS 5
#define MAX_FRIENDS 100

// Device function to compute total purchase for each friend
__global__ void calculateTotalPurchase(int *quantities, float *prices, float *totals, int numFriends) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < numFriends) {
        float total = 0.0f;
        for (int i = 0; i < NUM_ITEMS; i++) {
            total += quantities[tid * NUM_ITEMS + i] * prices[i];
        }
        totals[tid] = total;
    }
}

// Helper function to display the item menu
void displayMenu(const char *items[], float prices[]) {
    printf("🛒 Shopping Mall Menu:\n");
    for (int i = 0; i < NUM_ITEMS; i++) {
        printf("%d. %s - $%.2f\n", i + 1, items[i], prices[i]);
    }
    printf("\n");
}

int main() {
    const char *items[NUM_ITEMS] = {"Shirt", "Jeans", "Shoes", "Watch", "Sunglasses"};
    float h_prices[NUM_ITEMS] = {25.0f, 50.0f, 75.0f, 100.0f, 40.0f};

    int numFriends;
    printf("Enter number of friends: ");
    scanf("%d", &numFriends);

    if (numFriends > MAX_FRIENDS) {
        printf("Error: Number of friends exceeds maximum allowed (%d).\n", MAX_FRIENDS);
        return 1;
    }

    // Allocate memory for quantities and totals
    int h_quantities[MAX_FRIENDS * NUM_ITEMS] = {0};
    float h_totals[MAX_FRIENDS] = {0.0f};

    displayMenu(items, h_prices);

    // Input quantities for each friend
    for (int f = 0; f < numFriends; f++) {
        printf("Friend %d:\n", f + 1);
        for (int i = 0; i < NUM_ITEMS; i++) {
            printf("  Enter quantity of %s: ", items[i]);
            scanf("%d", &h_quantities[f * NUM_ITEMS + i]);
        }
        printf("\n");
    }

    // Device memory allocation
    int *d_quantities;
    float *d_prices, *d_totals;

    cudaMalloc((void **)&d_quantities, MAX_FRIENDS * NUM_ITEMS * sizeof(int));
    cudaMalloc((void **)&d_prices, NUM_ITEMS * sizeof(float));
    cudaMalloc((void **)&d_totals, MAX_FRIENDS * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_quantities, h_quantities, MAX_FRIENDS * NUM_ITEMS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prices, h_prices, NUM_ITEMS * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 32;
    int gridSize = (numFriends + blockSize - 1) / blockSize;
    calculateTotalPurchase<<<gridSize, blockSize>>>(d_quantities, d_prices, d_totals, numFriends);
    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_totals, d_totals, MAX_FRIENDS * sizeof(float), cudaMemcpyDeviceToHost);

    // Print individual totals and overall total
    float grandTotal = 0.0f;
    for (int f = 0; f < numFriends; f++) {
        printf("Friend %d total purchase: $%.2f\n", f + 1, h_totals[f]);
        grandTotal += h_totals[f];
    }

    printf("\n💰 Total purchase by all friends: $%.2f\n", grandTotal);

    // Free device memory
    cudaFree(d_quantities);
    cudaFree(d_prices);
    cudaFree(d_totals);

    return 0;
}
