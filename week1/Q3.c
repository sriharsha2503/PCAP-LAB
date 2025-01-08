#include "mpi.h"
#include <stdio.h>
int main(int argc, char *argv[])
{
int rank,size,x=5,y=6;
MPI_Init(&argc,&argv);

MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
if(rank==0)
{
printf("the output of the simple calculator\n");
}
switch (rank)
{
case 0:printf(" the sum of the values %d\n",x+y);break;
case 1:printf(" the difference of the values %d\n",x-y);break;
case 2:printf(" the product of the values %d\n",x*y);break;
case 3:printf(" the division of the values %d\n",x/y);break;
}

MPI_Finalize();
return 0;
}
