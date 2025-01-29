#include "mpi.h"
#include <stdio.h>

// Error handling function
void ErrorHandler(int ecode, const char *error_desc)
{
    if (ecode != MPI_SUCCESS)
    {
        char err_str[BUFSIZ];
        int strlen, err_class;

        // Get the error class and the error message string
        MPI_Error_class(ecode, &err_class);           // Get the error class
        MPI_Error_string(ecode, err_str, &strlen);    // Get the error message

        // Print the error description, error code, class, and message
        printf("Error Description: %s\n", error_desc);
        printf("MPI Error Code: %d\n", ecode);  // Print error code
        printf("MPI Error Class: %d\n", err_class); // Print error class
        printf("MPI Error Message: %s\n", err_str);  // Print error message
    }
}

// Function to calculate factorial of a number
int factorial(int n) {
    int fact = 1;
    for (int i = 1; i <= n; i++) {
        fact *= i;
    }
    return fact;
}

int main(int argc, char *argv[])
{
    int rank, size, fact, factsum, total_factsum;
    int ecode;

    // Initialize MPI
    ecode = MPI_Init(&argc, &argv); // First call
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Init failed");
        MPI_Finalize();
        return -1;
    }

    // Set error handler to return errors instead of aborting
    ecode = MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Errhandler_set failed");
        MPI_Finalize();
        return -1;
    }

    // Error 1: Invalid communicator (MPI_COMM_NULL)
    ecode = MPI_Comm_rank(MPI_COMM_NULL, &rank); // Invalid communicator
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Comm_rank failed with MPI_COMM_NULL");
    }

    // Get rank and size for valid operations
    ecode = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Comm_rank failed");
        MPI_Finalize();
        return -1;
    }

    ecode = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Comm_size failed");
        MPI_Finalize();
        return -1;
    }

    // Error 2: Mismatched types in MPI_Send and MPI_Recv
    int send_data = 42;
    float recv_data; // Mismatched type (float instead of int)
    ecode = MPI_Send(&send_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD); // This will work
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Send failed");
    }

    ecode = MPI_Recv(&recv_data, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // Mismatched type (int instead of float)
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Recv failed with mismatched type");
    }

    // Error 3: Sending data to an invalid rank
    ecode = MPI_Send(&send_data, 1, MPI_INT, size, 0, MPI_COMM_WORLD); // Invalid rank (rank >= size)
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Send failed with invalid rank");
    }

    // Calculate factorial for this rank
    fact = factorial(rank + 1);
    printf("Rank %d: Factorial of %d = %d\n", rank, rank + 1, fact);

    // Perform a scan operation (cumulative sum)
    ecode = MPI_Scan(&fact, &factsum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Scan failed");
        MPI_Finalize();
        return -1;
    }

    printf("Rank %d: Factorial = %d, Cumulative Sum of Factorials = %d\n", rank, fact, factsum);

    // Finalize the MPI environment
    ecode = MPI_Finalize();
    if (ecode != MPI_SUCCESS)
    {
        ErrorHandler(ecode, "MPI_Finalize failed");
        return -1;
    }

    return 0;
}
