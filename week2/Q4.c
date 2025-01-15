#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc,char *argv[]) 
{
	int rank, size, num;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status status;
	if(rank == 0) 
	{
		printf("Enter a number\n");
		scanf("%d", &num);
		MPI_Ssend(&num,1,MPI_INT,1,1,MPI_COMM_WORLD);
		MPI_Recv(&num,1,MPI_INT,size-1,1,MPI_COMM_WORLD,&status);
		printf("[%d]: Recieved: %d\n", rank, num);
	}
	else if(rank < size-1)
	{
		MPI_Recv(&num,1,MPI_INT,rank-1,1,MPI_COMM_WORLD,&status);
		printf("[%d]: Recieved: %d\n", rank, num);
		num++;
		MPI_Ssend(&num,1,MPI_INT,rank+1,1,MPI_COMM_WORLD);
	}
	else 
	{
		MPI_Recv(&num,1,MPI_INT,rank-1,1,MPI_COMM_WORLD,&status);
		printf(" process %d Recieved %d\n", rank, num);
		num++;
		MPI_Ssend(&num,1,MPI_INT,0,1,MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}
