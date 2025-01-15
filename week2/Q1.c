#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main(int argc,char *argv[]) 
{
	int rank, size, length;
	char *word;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status status;
	if(rank == 0) 
	{
		printf("Enter your word\n");
		scanf("%s", word);
		length = strlen(word);
		MPI_Ssend(&length,1,MPI_INT,1,1,MPI_COMM_WORLD);
		MPI_Ssend(word,length,MPI_CHAR,1,1,MPI_COMM_WORLD);
		MPI_Recv(word,length,MPI_CHAR,1,1,MPI_COMM_WORLD, &status);
		printf("Recieved by process 0: %s\n", word);
	}
	else 
	{
		MPI_Recv(&length,1,MPI_INT,0,1,MPI_COMM_WORLD, &status);
		MPI_Recv(word,length,MPI_CHAR,0,1,MPI_COMM_WORLD, &status);
		for(int i=0; i<length; i++) 
		{
			if(word[i] >= 65 && word[i] <= 90)
				word[i] += 32;
			else
				word[i]-=32;
		printf(" %c was modified by the process %d\n",word[i],rank);
		}
		MPI_Ssend(word, length, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}

