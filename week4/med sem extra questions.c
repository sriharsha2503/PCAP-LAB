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
============================================================================================================
    //string reversal
    #include "mpi.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char input[100], reversed[100];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int length = 0;

    if (rank == 0) {
        printf("Enter a string: ");
        scanf("%s", input);
        length = strlen(input);
    }

    // Broadcast the length to all processes
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = length / size;
    int remainder = length % size;

    char subword[chunk_size + 1];
    subword[chunk_size] = '\0';

    // Scatter the string to all processes
    MPI_Scatter(input, chunk_size, MPI_CHAR, subword, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Reverse the local substring
    for (int i = 0; i < chunk_size / 2; i++) {
        char temp = subword[i];
        subword[i] = subword[chunk_size - i - 1];
        subword[chunk_size - i - 1] = temp;
    }

    // Gather the reversed substrings
    MPI_Gather(subword, chunk_size, MPI_CHAR, reversed, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0 && remainder > 0) {
        for (int i = 0; i < remainder; i++) {
            reversed[length - remainder + i] = input[length - 1 - i];
        }
    }

    if (rank == 0) {
        reversed[length] = '\0';
        printf("Reversed String: %s\n", reversed);
    }

    MPI_Finalize();
    return 0;
}
===========================================================================================================
//string sort
#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Function to compare characters in descending order
int compare_desc(const void *a, const void *b) {
    return (*(char*)b - *(char*)a);
}

int main(int argc, char *argv[]) {
    int rank, size;
    char input[100], sorted[100];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int length = 0;

    if (rank == 0) {
        printf("Enter a string: ");
        scanf("%s", input);
        length = strlen(input);
    }

    // Broadcast the length to all processes
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = length / size;
    int remainder = length % size;

    char subword[chunk_size + 1];
    subword[chunk_size] = '\0';

    // Scatter the string to all processes
    MPI_Scatter(input, chunk_size, MPI_CHAR, subword, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Sort the local substring in descending order
    qsort(subword, chunk_size, sizeof(char), compare_desc);

    // Gather the sorted substrings
    MPI_Gather(subword, chunk_size, MPI_CHAR, sorted, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0 && remainder > 0) {
        for (int i = 0; i < remainder; i++) {
            sorted[length - remainder + i] = input[length - 1 - i];
        }
    }

    if (rank == 0) {
        // Sort the entire collected result in descending order
        qsort(sorted, length, sizeof(char), compare_desc);
        sorted[length] = '\0';
        printf("Sorted String in Descending Order: %s\n", sorted);
    }

    MPI_Finalize();
    return 0;
}
======================================================================================    
