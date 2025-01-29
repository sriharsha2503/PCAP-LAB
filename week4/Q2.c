#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char* argv[])
{   int rank, size, search;
    MPI_Status stat;
    int occ = 0, sum_occ = 0;
    int mat[3][3];
    int temp[3];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(rank==0)
    {
     printf("Enter the nine elements:\n");
      for(int i=0;i<3;i++)
       { 
         for(int j=0;j<3;j++)
         {
           scanf("%d",&mat[i][j]);
         }
       }
       printf("enter the number to be searched in the array\n");
       scanf("%d",&search);
     }
    MPI_Bcast(&search, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(mat, 3, MPI_INT, temp, 3, MPI_INT, 0, MPI_COMM_WORLD);
    for(int i = 0; i < 3; i++) 
    {
        if(search == temp[i]) 
        {
             
            occ++;
            printf("the occured in the process %d for %d time\n",rank,occ);
        }
    }
    MPI_Reduce(&occ, &sum_occ, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if(rank == 0) 
    {
        printf("Total Number of Occurences: %d\n", sum_occ);
    }
    MPI_Finalize();
    exit(0);
} 
//Enter the nine elements:
//4 5 6 7 3 3 5 3 8
//enter the number to be searched in the array/
//3
//Total Number of Occurences: 3
//the occured in the process 1 and for the 1th time
//the occured in the process 1 and for the 2th time
//the occured in the process 2 and for the 1th time

