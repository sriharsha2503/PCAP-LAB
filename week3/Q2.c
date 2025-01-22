#include "mpi.h"
#include <stdio.h>
#define MAX_SIZE 1000 
int main(int argc, char *argv[]) 
{
    int rank, size, m, A[MAX_SIZE], C[MAX_SIZE], i;
    float avg = 0;
    float result = 0;
    float all_avgs[MAX_SIZE];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) 
    {
        printf("Enter M: ");
        scanf("%d", &m);
    }
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) 
    {
        printf("Enter %d values:\n", size * m);
        for (i = 0; i < size * m; i++) 
        {
            scanf("%d", &A[i]);
        }
    }
    MPI_Scatter(A, m, MPI_INT, C, m, MPI_INT, 0, MPI_COMM_WORLD);
    for (i = 0; i < m; i++)
    {
        avg += C[i];
    }
    avg = avg / m;
    printf("I have received %d numbers in process %d and my local average is %f\n", m, rank, avg);
    MPI_Gather(&avg, 1, MPI_FLOAT, all_avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (rank == 0) 
    {
        for (i = 0; i < size; i++) 
        {
            result += all_avgs[i];
        }
        printf("The average of all numbers is: %f\n", result / size);
    }
    MPI_Finalize();
    return 0;
}


