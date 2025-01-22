#include "mpi.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char word1[100], word2[100];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0) {
        printf("Enter your string1: ");
        scanf("%s", word1);
        printf("Enter your string2: ");
        scanf("%s", word2);
    }

    // Length of the first string (both strings are assumed to be of the same length)
    int l = strlen(word1);
    
    // Broadcast the length of the string to all processes
    MPI_Bcast(&l, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the number of characters each process will handle
    int chunk_size = l / size;
    int remainder = l % size;

    // Buffer for each process to receive part of the words
    char subword1[chunk_size + (rank < remainder ? 1 : 0)];
    char subword2[chunk_size + (rank < remainder ? 1 : 0)];

    // Scatter the parts of both words to each process
    MPI_Scatterv(word1, /*send counts*/
                 NULL, /*displacements (not needed for equal size)*/
                 chunk_size, MPI_CHAR,
                 subword1, chunk_size, MPI_CHAR, 
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(word2, NULL, chunk_size, MPI_CHAR, subword2, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Allocate space for the result for each process
    char res[2 * (chunk_size + (rank < remainder ? 1 : 0))];

    // Interleave the characters of both words
    for (int i = 0; i < (chunk_size + (rank < remainder ? 1 : 0)); i++) {
        res[2 * i] = subword1[i];
        res[2 * i + 1] = subword2[i];
    }

    // Gather the results from all processes
    char finalresult[2 * l];
    MPI_Gather(res, 2 * (chunk_size + (rank < remainder ? 1 : 0)), MPI_CHAR, 
               finalresult, 2 * (chunk_size + (rank < remainder ? 1 : 0)), MPI_CHAR, 0, MPI_COMM_WORLD);

    // Final result on root process
    if (rank == 0) {
        finalresult[2 * l] = '\0'; // Null-terminate the final string
        printf("Result: %s\n", finalresult);
    }

    MPI_Finalize();
    return 0;
}

