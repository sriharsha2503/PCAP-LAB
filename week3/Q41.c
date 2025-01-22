#include "mpi.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
	int rank, size;
	char word1[100], word2[100];
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	if(rank == 0) {
		printf("Enter your string1: ");
		scanf("%s", word1);
		printf("Enter your string2: ");
		scanf("%s", word2);
	}
	int l = strlen(word1);
	MPI_Bcast(&l,1,MPI_INT,0,MPI_COMM_WORLD);
	char subword1[l/size], subword2[l/size];
	MPI_Scatter(word1,l/size,MPI_CHAR,subword1,l/size,MPI_CHAR,0,MPI_COMM_WORLD);
	MPI_Scatter(word2,l/size,MPI_CHAR,subword2,l/size,MPI_CHAR,0,MPI_COMM_WORLD);
	char res[2*l];
	int index = 0;
	for(int i=0; i<l/size; i++) {
		res[index++] += subword1[i];
		res[index++] += subword2[i];
	}
	res[index] = '\0';
	char finalresult[200];
	MPI_Gather(res, l/size * 2, MPI_CHAR, finalresult, l/size * 2, MPI_CHAR, 0, MPI_COMM_WORLD);
	finalresult[2*l] = '\0';
	if(rank == 0)
		printf("Result: %s\n", finalresult);
	MPI_Finalize();
	return 0;
}

