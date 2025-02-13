//Q1
#include "mpi.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char word1[100], word2[100];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int l1 = 0, l2 = 0; // Length of word1 and word2

    if (rank == 0) {
        printf("Enter your string1: ");
        scanf("%s", word1);
        printf("Enter your string2: ");
        scanf("%s", word2);

        l1 = strlen(word1);
        l2 = strlen(word2);

        if (l1 != l2) {
            printf("Error: Both strings must have the same length.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast the length of the strings to all processes
    MPI_Bcast(&l1, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = l1 / size;
    int remainder = l1 % size;

    char subword1[chunk_size + 1];
    char subword2[chunk_size + 1];
    subword1[chunk_size] = '\0';
    subword2[chunk_size] = '\0';

    // Scatter word1 and word2 to all processes
    MPI_Scatter(word1, chunk_size, MPI_CHAR, subword1, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(word2, chunk_size, MPI_CHAR, subword2, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    printf("Process %d received substring1: %s\n", rank, subword1);
    printf("Process %d received substring2: %s\n", rank, subword2);

    char res[2 * chunk_size + 1];
    int index = 0;
    for (int i = 0; i < chunk_size; i++) {
        res[index++] = subword1[i];
        res[index++] = subword2[i];
    }
    res[index] = '\0';

    char finalresult[2 * l1 + 1];

    // Gather interleaved chunks from all processes
    MPI_Gather(res, 2 * chunk_size, MPI_CHAR, finalresult, 2 * chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0 && remainder > 0) {
        for (int i = 0; i < remainder; i++) {
            finalresult[2 * chunk_size * size + 2 * i] = word1[chunk_size * size + i];
            finalresult[2 * chunk_size * size + 2 * i + 1] = word2[chunk_size * size + i];
        }
        finalresult[2 * l1] = '\0';

        printf("Result: %s\n", finalresult);
    }

    MPI_Finalize();
    return 0;
}
