#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc, char* argv[]) 
{
	MPI_Init(&argc, &argv);
	int rank, size, i;
	char str[50], temp[50], temp2[50], ans[200], ch;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Status stat;
	if(rank == 0) 
	{
		printf("\nEnter a string of length %d:\n", size);
		scanf("%s",str);
	}
	MPI_Scatter(str, 1, MPI_CHAR, &ch, 1, MPI_CHAR, 0, MPI_COMM_WORLD);
	if(rank != 0) 
	{
		for(i = 0; i <= rank; i++) 
		{
			temp[i] = ch;
		}
		MPI_Send(temp, rank + 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
	}
	if(rank == 0) 
	{
		ans[0] = str[0]; ans[1] = '\0';
		printf("%s\n",ans);
		for(int i = 1; i < size; i++) 
		{
			MPI_Recv(temp2, i+1, MPI_CHAR, i, 1, MPI_COMM_WORLD, &stat);
			strcat(ans, temp2);
		}
		printf("Final Word:\t%s\n", ans);
	}
	MPI_Finalize();
	exit(0);
}
//Enter a string of length 4:
//PCAP
//Final Word:	PCCAAAPPPP

