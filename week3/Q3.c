#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int rank, size, l;
    char word[100];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(rank == 0) 
    {
        printf("Enter your string: ");
        scanf("%s", word);
    }
    l = strlen(word);
    MPI_Bcast(&l, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int chars_per_process = l / size;
    char subword[chars_per_process + 1];
    MPI_Scatter(word, chars_per_process, MPI_CHAR, subword, chars_per_process, MPI_CHAR, 0, MPI_COMM_WORLD);
    subword[chars_per_process] = '\0';
    int count = 0;
    for(int i = 0; i < chars_per_process; i++) 
    {
        if(subword[i] != 'a' && subword[i] != 'e' && subword[i] != 'i' && subword[i] != 'o' && subword[i] != 'u' &&
           subword[i] != 'A' && subword[i] != 'E' && subword[i] != 'I' && subword[i] != 'O' && subword[i] != 'U') 
        {
            count++;
        }
    }
    int B[size];
    MPI_Gather(&count, 1, MPI_INT, B, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0) 
    {
        int total_count = 0;
        for(int i = 0; i < size; i++) 
        {
            printf("Process %d counted %d non-vowel characters.\n", i, B[i]);
            total_count += B[i];
        }
        printf("Total non-vowel characters: %d\n", total_count);
    }
    MPI_Finalize();
    return 0;
}

