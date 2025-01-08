#include <mpi.h>
#include <stdio.h>
#include <math.h>
int main(int argc, char *argv[])
{
int rank,size,x=3,result;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
result=pow(x,rank);
printf("My rank is %d and the result for the pow(3,%d) is  %d . \n", rank,rank, result);
MPI_Finalize();
return 0;
}
