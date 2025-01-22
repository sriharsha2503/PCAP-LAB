#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[]) 
{
    int rank, size, num,A[100],fact, i, result,all_facts[100]; 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) 
    {
        printf("Enter %d values\n", size);
        for (int i = 0; i < size; i++)
            scanf("%d", &A[i]);
    }
    MPI_Scatter(A, 1, MPI_INT, &num, 1, MPI_INT, 0, MPI_COMM_WORLD);
    fact = 1;
    for (i = num; i >= 1; i--) 
    {
        fact *= i;
    }
    printf("Received %d in process %d and its factorial is %d\n", num, rank, fact);
    MPI_Gather(&fact, 1, MPI_INT, all_facts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) 
    {
        result = 0;
        for (i = 0; i < size; i++) 
        {
            result += all_facts[i];
        }
        printf("The sum of all factorials is: %d\n", result);
    }
    MPI_Finalize();
    return 0;
}
