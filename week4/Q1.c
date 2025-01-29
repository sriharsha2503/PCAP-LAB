#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
void ErrorHandler(int ecode, const char *error_desc)
{   if (ecode != MPI_SUCCESS)
    {   char err_str[BUFSIZ];
        int strlen, err_class;
        MPI_Error_class(ecode, &err_class);           
        MPI_Error_string(ecode, err_str, &strlen);    
        printf("Error Description: %s\n", error_desc);
        printf("MPI Error Code: %d\n", ecode);  
        printf("MPI Error Class: %d\n", err_class); 
        printf("MPI Error Message: %s\n", err_str);  
    }
}
int main(int argc,char*argv[])
{
   int rank,size,fact=1,factsum,i;
   int ecode = MPI_Init(&argc, &argv); // First call
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Init failed");
        MPI_Finalize();
        return -1;
    }
    ecode = MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Errhandler_set failed");
        MPI_Finalize();
        return -1;
    }
    ecode = MPI_Comm_rank(MPI_COMM_NULL, &rank);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Comm_rank failed with MPI_COMM_NULL");
    }
    ecode = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Comm_rank failed");
        MPI_Finalize();
        return -1;
    }
   MPI_Comm_size(MPI_COMM_WORLD,&size);
   for(i=1;i<=rank+1;i++)
     fact=fact*i;
   int sum=MPI_Scan(&fact,&factsum,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
      printf("sum of all the factorial at process %d is=%d\n",rank,factsum);
   MPI_Finalize();
   exit(0);
 }
