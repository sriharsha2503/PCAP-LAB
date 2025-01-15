#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size, n;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int buffer_size = 1024;
    int buffer[buffer_size];
    MPI_Buffer_attach(buffer, buffer_size * sizeof(int));
    int arr[size];
    if (rank == 0) 
    {
        printf("Enter %d elements into the array: ", size);
        for (int i = 0; i < size; i++) 
        {
            scanf("%d", &arr[i]);
            printf("process: %d received : %d cube: %d\n",rank,n,n*n*n);
        }
        for (int i = 1; i < size; i++) {
            MPI_Bsend(&arr[i], 1, MPI_INT, i, i, MPI_COMM_WORLD);
        }
    } 
    else
    {
        MPI_Recv(&n, 1, MPI_INT, 0, rank, MPI_COMM_WORLD, &status);
        if (rank % 2 == 0)
            printf("process: %d received: %d square: %d\n", rank, n, n * n);
        else
            printf("process: %d received: %d cube: %d\n", rank, n, n * n * n);
    }
    MPI_Buffer_detach(&buffer, &buffer_size);
    MPI_Finalize();
    return 0;
}

