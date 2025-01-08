#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>
int main(int argc, char *argv[])
{
int rank,size;
char str[]="HELLO";
MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
if(size==5)//it should be 5
{if(isupper(str[rank]))
{str[rank]=tolower(str[rank]);}
else if(islower(str[rank]))
{str[rank]=toupper(str[rank]);}
printf("process %d toggled:%c\n",rank,str[rank]);
}

MPI_Finalize();
return 0;
}
