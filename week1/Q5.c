#include "mpi.h"
#include <stdio.h>
int fact(int n)
{
int p=1;
for(int i=1;i<=n;i++)
{p*=i;}
return p;
}

int fib(int n)
{
if(n==0)
{return 0;}
else if (n==1)
{return 1;}
else
{return fib(n-1)+fib(n-2);}
}

int main(int argc, char *argv[])
{
int rank,size;
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
if(rank%2==0)
{
printf("for the factoral of rank %d is %d\n",rank,fact(rank));
}
else
{
printf("for the fibonacci of rank %d is %d \n",rank,fib(rank));
}
MPI_Finalize();
return 0;
}
